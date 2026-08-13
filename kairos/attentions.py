import inspect
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

if torch.cuda.is_available():
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        flex_attention = torch.compile(flex_attention)
        ATTN_IMPL = "flex"
    except Exception:  # noqa: BLE001 — flex_attention compile can fail for many backend reasons; must not crash import
        ATTN_IMPL = "eager"
else:
    ATTN_IMPL = "eager"

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

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_causal_conv1d_update,
    )

    causal_conv1d_update = torch_causal_conv1d_update


def _supports_cu_seqlens(fn):
    if fn is None:
        return False
    try:
        return "cu_seqlens" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class KairosNorm(LlamaRMSNorm):
    pass


class KairosRotaryEmbedding(nn.Module):
    """cos/sin cache grows on demand (amortized doubling) so generation past max_position_embeddings never."""

    def __init__(self, config, head_dim):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.cached_dtype = None

    def _build_cache(self, seq_len, device, dtype):
        self.seq_len_cached = seq_len
        self.cached_dtype = dtype
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        self.cos_cached = freqs.cos().to(dtype)
        self.sin_cached = freqs.sin().to(dtype)

    def forward(self, x, position_ids, max_position=None):
        if max_position is None:
            max_position = int(position_ids.max().item()) if position_ids.numel() > 0 else 0
        needed = max_position + 1
        if needed > self.seq_len_cached or self.cos_cached is None or self.cached_dtype != x.dtype:
            new_len = max(needed, self.config.max_position_embeddings, 16, self.seq_len_cached * 2)
            self._build_cache(new_len, x.device, x.dtype)
        cos = self.cos_cached[position_ids][..., None, :]
        sin = self.sin_cached[position_ids][..., None, :]
        return cos, sin


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1).type_as(x)


# Eager attention (CPU / fallback)


def eager_attention(q, k, v, window, key_padding_mask=None):
    _, Lq, H, D = q.shape
    Lk = k.shape[1]
    W = 2 * window + 1
    n_kv = k.size(2)
    assert H % n_kv == 0, f"n_heads ({H}) must be divisible by n_kv_heads ({n_kv}) for GQA repeat"
    k = k.repeat_interleave(H // n_kv, dim=2)
    v = v.repeat_interleave(H // n_kv, dim=2)
    kv_start = max(0, Lk - (Lq + window))
    kv_end = Lk
    k = k[:, kv_start:kv_end]
    v = v[:, kv_start:kv_end]
    mask = key_padding_mask[:, kv_start:kv_end] if key_padding_mask is not None else None
    pad = window
    k_pad = F.pad(k, (0, 0, 0, 0, pad, pad))
    v_pad = F.pad(v, (0, 0, 0, 0, pad, pad))
    mask_pad = F.pad(mask, (pad, pad), value=False) if mask is not None else None
    k_windows = k_pad.unfold(1, W, 1).permute(0, 1, 2, 4, 3)
    v_windows = v_pad.unfold(1, W, 1).permute(0, 1, 2, 4, 3)
    k_windows = k_windows[:, -Lq:]
    v_windows = v_windows[:, -Lq:]
    q = q.unsqueeze(3)
    scores = (q * k_windows).sum(-1) * (D**-0.5)
    if mask_pad is not None:
        mask_windows = mask_pad.unfold(1, W, 1)
        mask_windows = mask_windows[:, -Lq:]
        mask_windows = mask_windows.unsqueeze(2).expand(-1, -1, H, -1)
        scores = scores.masked_fill(~mask_windows, float("-inf"))
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    attn = torch.nan_to_num(attn)
    out = (attn.unsqueeze(-1) * v_windows).sum(3)
    return out.contiguous()


# Flex mask builder (bidir)


def build_flex_mask(max_len, window):
    def bidir_window(b, h, q_idx, kv_idx):
        return (kv_idx >= q_idx - window) & (kv_idx <= q_idx + window)

    return create_block_mask(bidir_window, B=None, H=None, Q_LEN=max_len, KV_LEN=max_len)


def build_flex_mask_padded(window, attention_mask):
    B, Lk = attention_mask.shape

    def bidir_window_padded(b, h, q_idx, kv_idx):
        valid = (kv_idx >= q_idx - window) & (kv_idx <= q_idx + window)
        return valid & attention_mask[b, kv_idx]

    return create_block_mask(bidir_window_padded, B=B, H=None, Q_LEN=Lk, KV_LEN=Lk)


# Kairos Attention (SWA bidirectional)


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
            self.block_mask = build_flex_mask(config.max_position_embeddings, self.window)
        self.rope = KairosRotaryEmbedding(config, self.head_dim)

    def forward(self, x, position_embeddings=None, cache_params=None, attention_mask=None, position_ids=None):
        B, L, _ = x.shape
        if cache_params is not None and self.layer_idx is not None:
            offset = cache_params.get_total_seen(self.layer_idx)
        else:
            offset = 0
        if position_ids is None:
            pos = torch.arange(offset, offset + L, device=x.device).unsqueeze(0).expand(B, -1)
            max_position = offset + L - 1
        else:
            pos = position_ids
            max_position = None
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        if isinstance(position_embeddings, tuple):
            cos, sin = position_embeddings
        else:
            cos, sin = self.rope(x, pos, max_position=max_position)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if cache_params is not None:
            k, v = cache_params.update(k, v, self.layer_idx)
            cache_params.trim(self.layer_idx)
        has_padding = attention_mask is not None and not bool(attention_mask.all())
        if ATTN_IMPL == "flex":
            if has_padding:
                block_mask = build_flex_mask_padded(self.window, attention_mask)
            else:
                block_mask = self.block_mask._adjust(q.size(1), k.size(1))
            out = flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=block_mask,
                scale=self.head_dim**-0.5,
            ).transpose(1, 2)
        else:
            out = eager_attention(q, k, v, self.window, key_padding_mask=attention_mask)
        out = out.reshape(B, L, self.n_heads * self.head_dim)
        return self.out(out)


class KairosGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print("Warning: layer_idx should be set for caching")
        self.hidden_size = config.hidden_size
        self.n_kv_heads = config.num_key_value_heads
        self.n_heads = config.num_attention_heads
        self.conv_size = config.linear_conv_kernel_dim
        self.head_dim = self.hidden_size // self.n_heads
        self.value_dim = 2 * self.head_dim * self.n_heads
        self.n_heads_local = self.n_heads
        self.conv_dim = 4 * self.n_heads * self.head_dim
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for GQA repeat"
        )
        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, 2 * self.head_dim * self.n_heads, bias=False)
        self.v_expand = nn.Linear(self.n_heads * self.head_dim, self.n_heads * 2 * self.head_dim, bias=False)
        dt = torch.exp(
            torch.rand(self.n_heads_local) * (math.log(config.time_step_max) - math.log(config.time_step_min))
            + math.log(config.time_step_min)
        )
        dt = torch.clamp(dt, min=config.time_step_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        A = torch.empty(self.n_heads_local).uniform_(*config.A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.qkv_conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_size,
            groups=self.conv_dim,
            padding=self.conv_size - 1,
        )
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
        self._chunk_supports_varlen = _supports_cu_seqlens(self.chunk_gated_delta_rule)
        self.merge_norm = KairosNorm(2 * self.value_dim)
        self.out_left_right = nn.Linear(2 * self.value_dim, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, config.hidden_size, bias=False)

    def process(self, hidden_states, cache_params=None, attention_mask=None):
        B, L, _ = hidden_states.shape
        has_previous_state = cache_params is not None and cache_params.conv_caches[self.layer_idx] is not None
        has_ssm_state = cache_params is not None and cache_params.ssm_caches[self.layer_idx] is not None
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        b = self.b_proj(hidden_states).view(B, L, self.n_heads)
        a = self.a_proj(hidden_states).view(B, L, self.n_heads)
        g_out = self.g_proj(hidden_states).view(B, L, self.n_heads, 2 * self.head_dim)
        q = rearrange(q, "b l (h d) -> b l h d", h=self.n_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.n_kv_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.n_kv_heads)
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        qf = rearrange(q, "b l h d -> b l (h d)")
        kf = rearrange(k, "b l h d -> b l (h d)")
        vf = rearrange(v, "b l h d -> b l (h d)")
        vf = self.v_expand(vf)
        mixed_qkv = torch.cat([qf, kf, vf], dim=-1)
        if attention_mask is not None:
            mixed_qkv = mixed_qkv * attention_mask[..., None].to(mixed_qkv.dtype)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        if has_previous_state and L == 1:
            conv_state = cache_params.conv_caches[self.layer_idx]
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.qkv_conv1d.weight.squeeze(1),
                self.qkv_conv1d.bias,
                "silu",
            )
        else:
            conv_state = None
            if has_previous_state:
                conv_state = cache_params.conv_caches[self.layer_idx]
            elif cache_params is not None:
                conv_state = mixed_qkv.new_zeros(B, self.conv_dim, self.conv_size - 1)
            if conv_state is not None:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                cache_params.conv_caches[self.layer_idx] = mixed_qkv[:, :, -(self.conv_size - 1) :].clone()
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    mixed_qkv,
                    self.qkv_conv1d.weight.squeeze(1),
                    self.qkv_conv1d.bias,
                    activation="silu",
                )
            else:
                mixed_qkv = F.silu(self.qkv_conv1d(mixed_qkv))
            mixed_qkv = mixed_qkv[:, :, -L:]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        d = self.head_dim
        q_dim = self.n_heads * d
        k_dim = self.n_heads * d
        v_dim = 2 * self.n_heads * d
        q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
        q = rearrange(q, "b l (h d) -> b l h d", h=self.n_heads)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.n_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.n_heads)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        prev_state = cache_params.ssm_caches[self.layer_idx] if has_ssm_state else None
        has_padding = attention_mask is not None and not bool(attention_mask.all())
        # With padding + a real fla
        use_varlen = has_padding and not (has_previous_state and L == 1) and self._chunk_supports_varlen
        if use_varlen:
            flat_idx = attention_mask.nonzero(as_tuple=False)
            bi, li = flat_idx[:, 0], flat_idx[:, 1]
            lengths = attention_mask.sum(dim=1)
            cu_seqlens = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
            q_p = q[bi, li].unsqueeze(0)
            k_p = k[bi, li].unsqueeze(0)
            v_p = v[bi, li].unsqueeze(0)
            g_p = g[bi, li].unsqueeze(0)
            beta_p = beta[bi, li].unsqueeze(0)
            o_p, ssm_cache = self.chunk_gated_delta_rule(
                q_p,
                k_p,
                v_p,
                g_p,
                beta_p,
                scale=None,
                initial_state=prev_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            o = v.new_zeros(v.shape)
            o[bi, li] = o_p.squeeze(0)
        else:
            if has_padding:
                m = attention_mask.to(beta.dtype).unsqueeze(-1)
                beta = beta * m
                g = g * m
            if has_previous_state and L == 1:
                o, ssm_cache = self.recurrent_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=prev_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                o, ssm_cache = self.chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale=None,
                    initial_state=prev_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
        if cache_params is not None:
            cache_params.ssm_caches[self.layer_idx] = ssm_cache
        o = o * F.silu(g_out)
        return o

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        out_f = self.process(hidden_states, cache_params, attention_mask=attention_mask)
        x_rev = torch.flip(hidden_states, dims=[1])
        mask_rev = torch.flip(attention_mask, dims=[1]) if attention_mask is not None else None
        out_b = self.process(x_rev, cache_params=None, attention_mask=mask_rev)
        out_b = torch.flip(out_b, dims=[1])
        B, L = out_f.shape[:2]
        out = torch.cat([out_f, out_b], dim=-1)
        out = out.reshape(B, L, -1)
        out = self.merge_norm(out)
        out = self.out_left_right(out)
        out = self.out_proj(out)
        return out


class KairosLiZAttention2(nn.Module):
    """TPTT-inspired (arxiv.org/abs/2506.17671) shared QKV/O projections couple SWA and DeltaNet."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.swa = KairosAttention(config, layer_idx)
        self.delta = KairosGatedDeltaNet(config, layer_idx)
        del self.delta.q_proj, self.delta.k_proj, self.delta.v_proj, self.delta.out_proj
        self.delta.q_proj = self.swa.q_proj
        self.delta.k_proj = self.swa.k_proj
        self.delta.v_proj = self.swa.v_proj
        self.delta.out_proj = self.swa.out
        self.norm = KairosNorm(2 * self.hidden_size)
        self.out_proj = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x, position_embeddings=None, cache_params=None, attention_mask=None, position_ids=None):
        swa_out = self.swa(
            x,
            position_embeddings=position_embeddings,
            cache_params=cache_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        delta_out = self.delta(x, cache_params=cache_params, attention_mask=attention_mask)
        out = torch.cat([swa_out, delta_out], dim=-1)
        out = self.norm(out)
        out = self.out_proj(out)
        return out
