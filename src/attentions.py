import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from transformers.cache_utils import DynamicCache

# =========================
# Backend 
# =========================
# Attention (Flex or eager)
if torch.cuda.is_available():
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        flex_attention = torch.compile(flex_attention)
        ATTN_IMPL = "flex"
    except Exception:
        ATTN_IMPL = "eager"
else:
    ATTN_IMPL = "eager"


# DeltaNet
try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
except ImportError:
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None

from transformers.models.qwen3_next.modeling_qwen3_next import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

# Conv 
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_causal_conv1d_update,
    )
    causal_conv1d_update = torch_causal_conv1d_update

# =========================
# Cache Diffusion 
# =========================
class KairosCache(DynamicCache):
    """
    KairosCache: unified cache for bidirectional DeltaNet + attention with diffusion-style usage.
    ---- DESIGN PRINCIPLES ----
    Latent cache = state(N) reused for diffusion on M.
    Must clone() each use (no mutation, no accumulation).
    Contains: KV (attention), conv + SSM (DeltaNet).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_caches = []
        self.ssm_caches = []

        self._key_cache = {}
        self._value_cache = {}

        for idx, layer_type in enumerate(config.layers_config):
            if 'l' in layer_type or 'd' in layer_type:  # attention layers
                self._key_cache[idx] = None
                self._value_cache[idx] = None

            self.conv_caches.append(None)
            self.ssm_caches.append(None)

        self.window_size = config.sliding_window_size
        self.layers_config = config.layers_config
        self.past_length = [0 for _ in range(len(config.layers_config))]

    # =========================
    # Attention KV update
    # =========================
    def update(self, k, v, layer_idx):
        """
        Append K/V to attention cache.
        """
        added_len = k.size(1)

        k_cache = self._key_cache[layer_idx]
        v_cache = self._value_cache[layer_idx]

        if k_cache is None:
            k_cache = k
            v_cache = v
        else:
            k_cache = torch.cat([k_cache, k], dim=1)
            v_cache = torch.cat([v_cache, v], dim=1)

        self._key_cache[layer_idx] = k_cache
        self._value_cache[layer_idx] = v_cache
        self.past_length[layer_idx] += added_len

        return k_cache, v_cache

    # =========================
    # Sliding window trim
    # =========================
    def trim(self, layer_idx):
        if 'l' not in self.layers_config[layer_idx]:  # trim SWA only
            return

        window = min(self.window_size, self.config.slw_wsize) if self.config.slw_wsize > 0 else self.window_size

        k = self._key_cache[layer_idx]
        v = self._value_cache[layer_idx]

        if k is not None and k.size(1) > window:
            self._key_cache[layer_idx] = k[:, -window:, ...].contiguous()
            self._value_cache[layer_idx] = v[:, -window:, ...].contiguous()

    # =========================
    # DeltaNet state access
    # =========================
    def get_ssm_cache(self, layer_idx):
        return (
            self.conv_caches[layer_idx],
            self.ssm_caches[layer_idx]
        )

    def get_total_seen(self, layer_idx):
        return self.past_length[layer_idx]

    # =========================
    # CRITICAL: Clone
    # =========================
    def clone(self):
        """
        Deep clone of the cache. REQUIRED for diffusion / iterative inference
        """
        new_cache = KairosCache(self.config)

        new_cache.conv_caches = [
            c.clone() if c is not None else None
            for c in self.conv_caches
        ]

        new_cache.ssm_caches = [
            c.clone() if c is not None else None
            for c in self.ssm_caches
        ]

        new_cache._key_cache = {
            k: v.clone() if v is not None else None
            for k, v in self._key_cache.items()
        }

        new_cache._value_cache = {
            k: v.clone() if v is not None else None
            for k, v in self._value_cache.items()
        }

        new_cache.past_length = self.past_length.copy()

        return new_cache

# =========================
# Rotary
# =========================
class KairosRotaryEmbedding(nn.Module):
    def __init__(self, config, head_dim):
        super().__init__()
        self.config = config

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, position_ids):
        max_pos = self.config.max_position_embeddings
        if max_pos > self.seq_len_cached:
            self.seq_len_cached = max(2 * max_pos, 16)
            t = torch.arange(self.seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().to(torch.bfloat16)
            self.sin_cached = freqs.sin().to(torch.bfloat16)

        cos = self.cos_cached[position_ids][..., None, :]
        sin = self.sin_cached[position_ids][..., None, :]
        return cos, sin


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([
        x1 * cos + x2 * sin,
        x1 * (-sin) + x2 * cos
    ], dim=-1).type_as(x)


# =========================
# Eager attention (CPU / fallback)
# =========================
def eager_attention(q, k, v, window):
    """
    Sliding window attention compatible KV cache.
    q: (B, Lq, H, D)
    k,v: (B, Lk, H_kv, D)
    """

    B, Lq, H, D = q.shape
    Lk = k.shape[1]
    W = 2 * window + 1

    # --- Handle GQA ---
    k = k.repeat_interleave(H // k.size(2), dim=2)
    v = v.repeat_interleave(H // v.size(2), dim=2)

    # --- align K/V to sliding windows ---
    # select relevant KV range per query
    kv_start = max(0, Lk - (Lq + window))
    kv_end = Lk
    k = k[:, kv_start:kv_end]
    v = v[:, kv_start:kv_end]

    # pad if needed
    pad = window
    k_pad = F.pad(k, (0, 0, 0, 0, pad, pad))
    v_pad = F.pad(v, (0, 0, 0, 0, pad, pad))

    # sliding windows
    k_windows = k_pad.unfold(1, W, 1).permute(0, 1, 2, 4, 3)
    v_windows = v_pad.unfold(1, W, 1).permute(0, 1, 2, 4, 3)

    # match Lq
    k_windows = k_windows[:, -Lq:]
    v_windows = v_windows[:, -Lq:]

    # --- compute ---
    q = q.unsqueeze(3)

    scores = (q * k_windows).sum(-1) * (D ** -0.5)
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    out = (attn.unsqueeze(-1) * v_windows).sum(3)

    return out.contiguous()

# =========================
# Flex mask builder (bidir)
# =========================
def build_flex_mask(max_len, window):
    def bidir_window(b, h, q_idx, kv_idx):
        return (kv_idx >= q_idx - window) & (kv_idx <= q_idx + window)

    return create_block_mask(
        bidir_window,
        B=None,
        H=None,
        Q_LEN=max_len,
        KV_LEN=max_len
    )

# =========================
# Kairos Attention (SWA bidirectional)
# =========================
class KairosAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        if layer_idx is None:
            print("Warning: layer_idx should be set for caching")

        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.window = config.sliding_window_size

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)

        self.out = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if ATTN_IMPL == "flex":
            self.block_mask = build_flex_mask(
                config.max_position_embeddings,
                self.window
            )
        
        # RoRE
        self.rope = KairosRotaryEmbedding(config, self.head_dim)


    
    def forward(self, x, position_embeddings=None, cache_params=None):
        B, L, _ = x.shape

        # ---- POSITION IDS
        if cache_params is not None and self.layer_idx is not None:
            offset = cache_params.get_total_seen(self.layer_idx)
        else:
            offset = 0

        pos = torch.arange(offset, offset + L, device=x.device).unsqueeze(0)

        # ---- projection
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim)

        # ---- ROTARY (external vs internal)
        if isinstance(position_embeddings, tuple):
            cos, sin = position_embeddings
        else:
            cos, sin = self.rope(x, pos)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # KV CACHE
        if cache_params is not None:
            k, v = cache_params.update(k, v, self.layer_idx)
            cache_params.trim(self.layer_idx)

        # attention
        if ATTN_IMPL == "flex":
            out = flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=self.block_mask._adjust(q.size(1), k.size(1)),
                scale=self.head_dim ** -0.5,
            ).transpose(1, 2)

        else:
            out = eager_attention(q, k, v, self.window)

        B, L = x.shape[:2]
        out = out.reshape(B, L, self.n_heads * self.head_dim)

        return self.out(out)


# =========================
# Kairos Bidirectional Deltanet
# =========================
class KairosGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if layer_idx is None:
            print("Warning: layer_idx should be set for caching")

        # config param
        self.hidden_size = config.hidden_size
        self.n_kv_heads = config.num_key_value_heads
        self.n_heads = config.num_attention_heads
        self.conv_size = config.linear_conv_kernel_dim

        # calculated param
        self.head_dim = self.hidden_size // self.n_heads
        self.value_dim = 2 * self.head_dim * self.n_heads
        self.n_heads_local = self.n_heads
        self.conv_dim = 4 * self.n_heads * self.head_dim

        # shared param
        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)

        # gating
        self.b_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, 2 * self.head_dim * self.n_heads, bias=False)

        # conv expend
        self.v_expand = nn.Linear(
            self.n_heads * self.head_dim,
            self.n_heads * 2 * self.head_dim,
            bias=False
        )

        # ---- dt init ----
        dt = torch.exp(
            torch.rand(self.n_heads_local) *
            (math.log(config.time_step_max) - math.log(config.time_step_min))
            + math.log(config.time_step_min)
        )
        dt = torch.clamp(dt, min=config.time_step_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # ---- A init ----
        A = torch.empty(self.n_heads_local).uniform_(*config.A_init_range)
        self.A_log = nn.Parameter(torch.log(A))

        # ---- conv ----
        self.qkv_conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_size,
            groups=self.conv_dim,
            padding=self.conv_size - 1
        )

        # ---- kernels ----
        self.causal_conv1d_fn = causal_conv1d_fn

        self.causal_conv1d_update = (
            causal_conv1d_update if causal_conv1d_update is not None else torch_causal_conv1d_update
        )

        self.chunk_gated_delta_rule = (
            chunk_gated_delta_rule if chunk_gated_delta_rule is not None else torch_chunk_gated_delta_rule
        )

        self.recurrent_gated_delta_rule = (
            fused_recurrent_gated_delta_rule
            if fused_recurrent_gated_delta_rule is not None
            else torch_recurrent_gated_delta_rule
        )
        
        # ---- bidirectionnal output merging ----
        self.out_left_right = nn.Linear(2 * self.value_dim, self.hidden_size, bias=False) # intermediate state
        self.out_proj = nn.Linear(self.hidden_size, config.hidden_size, bias=False) # shareable with swa


    def process(self, hidden_states, cache_params=None):
        B, L, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None

        # ---- projections ----
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        b = self.b_proj(hidden_states).view(B, L, self.n_heads)   # beta
        a = self.a_proj(hidden_states).view(B, L, self.n_heads)   # pre-g
        g_out = self.g_proj(hidden_states).view(B, L, self.n_heads, 2 * self.head_dim)

        # ---- reshape ----
        q = rearrange(q, "b l (h d) -> b l h d", h=self.n_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.n_kv_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.n_kv_heads)

        # ---- GQA expand ----
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        # ---- flatten ----
        qf = rearrange(q, "b l h d -> b l (h d)")
        kf = rearrange(k, "b l h d -> b l (h d)")
        vf = rearrange(v, "b l h d -> b l (h d)")

        # ---- V expansion (DeltaNet) 
        vf = self.v_expand(vf) # dv = 2d --> move before conv for expressivity

        # ---- MIX
        mixed_qkv = torch.cat([qf, kf, vf], dim=-1).transpose(1, 2)

        # ---- conv ----
        if use_precomputed_states:
            conv_cache = cache_params.conv_caches[self.layer_idx]
            if conv_cache is None:
                conv_cache = mixed_qkv.new_zeros(B, self.conv_dim, self.conv_size - 1)
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_cache,
                self.qkv_conv1d.weight.squeeze(1),
                self.qkv_conv1d.bias,
                "silu",
            )
            cache_params.conv_caches[self.layer_idx] = mixed_qkv.squeeze(-1)
        else:
            mixed_qkv = F.silu(self.qkv_conv1d(mixed_qkv)[:, :, :L])

        # ---- split ----
        mixed_qkv = mixed_qkv.transpose(1, 2)

        d = self.head_dim
        q_dim = self.n_heads * d
        k_dim = self.n_heads * d
        v_dim = 2 * self.n_heads * d

        q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        # ---- reshape ----
        q = rearrange(q, "b l (h d) -> b l h d", h=self.n_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.n_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.n_heads)

        # ---- gating ----
        beta = b.sigmoid()

        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # ---- DeltaNet ----
        prev_state = cache_params.ssm_caches[self.layer_idx] if use_precomputed_states else None
        if not use_precomputed_states:
            o, ssm_cache = self.chunk_gated_delta_rule(
                q, k, v, g, beta,
                scale=None,
                initial_state=prev_state,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            o, ssm_cache = self.recurrent_gated_delta_rule(
                q, k, v, g, beta,
                initial_state=prev_state,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        
        if use_precomputed_states:
            cache_params.ssm_caches[self.layer_idx] = ssm_cache

        # ---- output gating
        o = o * F.silu(g_out)

        return o

    def forward(self, hidden_states, cache_params=None):        
        # ---- forward (with cache for past memory)----
        out_f = self.process(hidden_states, cache_params)

        # ---- backward pass (intentionally stateless)
        # no cache: future information is unavailable in generation
        x_rev = torch.flip(hidden_states, dims=[1])
        out_b = self.process(x_rev, cache_params=None)
        out_b = torch.flip(out_b, dims=[1])

        # ---- merge and projection ----
        B, L = out_f.shape[:2]
        # concat bidir
        out = torch.cat([out_f, out_b], dim=-1)  # (B, L, H, 2*dv)
        # flatten heads
        out = out.reshape(B, L, -1)  # (B, L, 2 * value_dim)
        # linear
        out = self.out_left_right(out) # concat to swa value dim
        out = self.out_proj(out)  # (B, L, hidden_size) -> shareable
        return out



# =========================
# LiZAttention
# =========================
class KairosLiZAttention2(nn.Module):
    """
    TPTT-inspired (arxiv.org/abs/2506.17671) shared QKV/O projections couple SWA and DeltaNet, 
    enforcing aligned representations while enabling bidirectional (non-causal) modeling.
    Note: Alpha and beta in DeltaNet are not shared as they control directional mixing 
    independently from the shared representation space defined by QKVO.
    Adding : LiZAttention2:
    - Outputs are concatenated (not summed)
    - Then projected back to hidden_size (mixer)
    """


    def __init__(self, config, layer_idx):
        super().__init__()

        self.hidden_size = config.hidden_size

        # SWA
        self.swa = KairosAttention(config, layer_idx)

        # DeltaNet
        self.delta = KairosGatedDeltaNet(config, layer_idx)

        # Shared projection (force alignment & fast convergence)
        self.delta.q_proj = self.swa.q_proj
        self.delta.k_proj = self.swa.k_proj
        self.delta.v_proj = self.swa.v_proj
        self.delta.out_proj = self.swa.out
        
        # Final mixer
        self.out_proj = nn.Linear(
            2 * self.hidden_size,
            self.hidden_size,
            bias=False
        )

    def forward(self, x, position_embeddings=None, cache_params=None):

        # ---- SWA ----
        swa_out = self.swa(
            x,
            position_embeddings,
             cache_params
        )  # (B, L, D)

        # ---- Delta (with cache) ----
        delta_out = self.delta(
            x,
            cache_params=cache_params
        )  # (B, L, D)

        # ---- concat ----
        out = torch.cat([swa_out, delta_out], dim=-1)

        # ---- final projection ----
        out = self.out_proj(out)

        return out
