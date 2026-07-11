import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationMixin,
)

from transformers.cache_utils import DynamicCache

from .attentions import KairosLiZAttention2, KairosNorm, KairosRotaryEmbedding


# =========================
# PretrainedConfig
# =========================
class KairosConfig(PretrainedConfig):
    model_type = "kairos"

    def __init__(
        self,
        d_model=768,
        n_heads=12,
        n_layers=12,
        vocab_size=259,
        intermediate_size=2048,
        window_size=128,
        stride=5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = n_layers
        self.vocab_size = vocab_size

        # FIX (vs original): `num_modalities` was accepted but `modality_scales`
        # was hardcoded to only cover modalities 0/1/2, silently dropping every
        # other modality id from all backbones. We now build a sensible default
        # that covers every modality (extras default to "text-like": scale 0
        # only), while still allowing a fully custom mapping via kwarg.
        self.num_modalities = kwargs.get("num_modalities", 8)
        self.text_modality_id = kwargs.get("text_modality_id", 0)
        self.num_scales = kwargs.get("num_scales", 4)

        default_scales = {0: [0, 1], 1: [1, 2], 2: [2, 3]}
        for m in range(self.num_modalities):
            default_scales.setdefault(m, [0])
        self.modality_scales = kwargs.get("modality_scales", default_scales)

        assert d_model % n_heads == 0, "hidden_size must be divisible by n_heads"

        # Convolutional Byte-Codec
        self.stride = stride

        # SWA full-Attention
        self.sliding_window_size = window_size
        self.num_key_value_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = 0.0
        self.rope_theta = 10000.0
        self.max_position_embeddings = 4096

        # Deltanet Attention
        self.linear_num_value_heads = kwargs.get("linear_num_value_heads", n_heads)
        self.linear_num_key_heads = kwargs.get("linear_num_key_heads", n_heads)
        self.linear_key_head_dim = kwargs.get("linear_key_head_dim", self.head_dim)
        self.linear_value_head_dim = kwargs.get("linear_value_head_dim", self.head_dim)
        self.linear_conv_kernel_dim = kwargs.get("linear_conv_kernel_dim", 4)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)

        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.A_init_range = (1.0, 16.0)

        # FFN / MLP
        self.intermediate_size = intermediate_size

        # MoE
        # FIX (vs original): `num_experts` (the flag that decides dense-FFN vs
        # MoE) used to be a constructor-only argument that was never derived
        # from the config, so all these fields were inert unless the caller
        # remembered to pass it explicitly. `use_moe` now drives that flag by
        # default (see KairosDiffusionLLM.__init__).
        self.num_local_experts = kwargs.get("num_local_experts", 8)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 2)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", intermediate_size)
        self.n_routed_experts = kwargs.get("n_routed_experts", 8)
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)
        self.routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
        self.n_group = kwargs.get("n_group", 1)
        self.topk_group = kwargs.get("topk_group", 1)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", False)
        self.top_k = self.num_experts_per_tok
        self.use_moe = kwargs.get("use_moe", False)

        # Layers config (required by KairosCache)
        self.layers_config = kwargs.get("layers_config", ["ld"] * n_layers)
        self.slw_wsize = kwargs.get("slw_wsize", -1)


# =========================
# Cache Diffusion
# =========================
class KairosCache(DynamicCache):
    """
    Unified cache for bidirectional DeltaNet + attention, built specifically
    for **block diffusion** — read this before touching `clone()`, `update()`,
    or `trim()`, since the intent isn't obvious from the method bodies alone.

    THE MENTAL MODEL
    -----------------
    At inference time the sequence is split in two roles:
      - context N  : already-known tokens (a prompt, or previously finalized
                     blocks). Encoded ONCE, then FROZEN.
      - block M    : the (fixed-size) span currently being denoised, iterated
                     over several diffusion steps.

    A cache instance represents "the state produced by having seen N". It is
    built with exactly one forward pass over N, and from then on is treated as
    read-only ground truth:
      1. cache = KairosCache(config); model(N, cache_params=cache)
      2. for each denoising step on M:
             step_cache = cache.clone()          # <- MUST clone, never reuse
             out = model(xt_M, cache_params=step_cache)
      3. step_cache is discarded after each step; `cache` itself is never
         mutated by a call that passes `xt_M` — only by the call that encoded N.

    WHY `clone()` MUST DEEP-COPY (not share tensors)
    -------------------------------------------------
    Every denoising step re-reads the *same* frozen context N, conditioned on
    a *different* noisy guess for M (`x_{t}`, `x_{t-1}`, ...). If steps shared
    the underlying KV/conv/ssm tensors, `update()`/`trim()` calls from one
    step (appending M's keys, advancing conv state, etc.) would leak into the
    next step's starting point — silently turning "N steps of independent
    denoising against fixed N" into "one long, accidentally-causal sequence
    where each guess of M contaminates the next". `clone()` exists precisely
    to cut that leakage: each step starts from an *identical*, independent
    copy of "having seen N", and whatever it appends while processing M is
    thrown away with it.

    WHY THIS ALSO WORKS FOR (SOME) INFILLING
    ------------------------------------------
    Because SWA and DeltaNet here are bidirectional (see attentions.py), "M"
    doesn't have to sit after N in raw sequence order — a masked gap between
    two already-known spans can be denoised the same way, as long as the
    router feeds the right positions into M and the rest into the frozen
    context. The cache itself is agnostic to *where* M sits; it only cares
    about "what was frozen" vs "what gets re-denoised this step".

    NOTE — this is DIFFERENT from pretraining
    ------------------------------------------
    During pretraining there is no frozen N at all: the *entire* sequence is
    noised and denoised in one shot (see KairosDiffusionTrainer), so a
    KairosCache typically isn't part of that path. This class only encodes
    the block-diffusion *inference* pattern above.

    IMPLEMENTATION NOTE: KV/conv/ssm tensors are stored with their full batch
    dimension intact — the caller must call `update()` (via the model's
    forward) once per scale for the *whole batch at once*, never once per
    example. Calling it per-example silently concatenates different examples'
    K/V into the same tensor — that was the original multi-batch corruption
    bug, and it's a caller contract this class can't enforce by itself.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_caches = []
        self.ssm_caches = []
        self._key_cache = {}
        self._value_cache = {}

        for idx, layer_type in enumerate(config.layers_config):
            if "l" in layer_type or "d" in layer_type:
                self._key_cache[idx] = None
                self._value_cache[idx] = None
            self.conv_caches.append(None)
            self.ssm_caches.append(None)

        self.window_size = config.sliding_window_size
        self.layers_config = config.layers_config
        self.past_length = [0 for _ in range(len(config.layers_config))]

    def update(self, k, v, layer_idx):
        added_len = k.size(1)
        k_cache = self._key_cache[layer_idx]
        v_cache = self._value_cache[layer_idx]

        if k_cache is None:
            k_cache, v_cache = k, v
        else:
            k_cache = torch.cat([k_cache, k], dim=1)
            v_cache = torch.cat([v_cache, v], dim=1)

        self._key_cache[layer_idx] = k_cache
        self._value_cache[layer_idx] = v_cache
        self.past_length[layer_idx] += added_len
        return k_cache, v_cache

    def trim(self, layer_idx):
        if "l" not in self.layers_config[layer_idx]:
            return

        window = min(self.window_size, self.config.slw_wsize) if self.config.slw_wsize > 0 else self.window_size
        k = self._key_cache[layer_idx]
        v = self._value_cache[layer_idx]

        if k is not None and k.size(1) > window:
            self._key_cache[layer_idx] = k[:, -window:, ...].contiguous()
            self._value_cache[layer_idx] = v[:, -window:, ...].contiguous()

    def get_ssm_cache(self, layer_idx):
        return (self.conv_caches[layer_idx], self.ssm_caches[layer_idx])

    def get_total_seen(self, layer_idx):
        return self.past_length[layer_idx]

    def clone(self):
        # Deep-copy every tensor (not just the dict/list structure) — this is
        # what isolates one denoising step of block M from the next. See the
        # class docstring ("WHY clone() MUST DEEP-COPY") for the failure mode
        # this prevents: without it, `update()`/conv-state advances made while
        # denoising one guess of M would bleed into the next step's supposedly
        # fresh copy of "having seen frozen context N".
        new_cache = KairosCache(self.config)
        new_cache.conv_caches = [c.clone() if c is not None else None for c in self.conv_caches]
        new_cache.ssm_caches = [c.clone() if c is not None else None for c in self.ssm_caches]
        new_cache._key_cache = {k: v.clone() if v is not None else None for k, v in self._key_cache.items()}
        new_cache._value_cache = {k: v.clone() if v is not None else None for k, v in self._value_cache.items()}
        new_cache.past_length = self.past_length.copy()
        return new_cache


class KairosMultiCache(DynamicCache):
    """
    One KairosCache per backbone scale (context N is encoded once per scale in
    KairosDiffusionLLM.forward, so each scale needs its own frozen state).
    Same block-diffusion contract as KairosCache: build once against N, then
    `.clone()` before every denoising step on M — never reuse the same
    instance across steps. See KairosCache's docstring for the full rationale.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.caches = [KairosCache(config) for _ in range(config.num_scales)]

    def get(self, idx):
        return self.caches[idx]

    def clone(self):
        out = KairosMultiCache.__new__(KairosMultiCache)
        out.config = self.config
        out.caches = [c.clone() for c in self.caches]  # deep-copy per scale, see KairosCache.clone()
        return out


# =========================
# FeedForward / MoE
# =========================
class KairosFFN(Qwen2MoeMLP):
    """dense KairosFFN (SwiGLU + HF optimisations)."""

    pass


class KairosMoE(DeepseekV3MoE):
    """MoE (routing + scaling + grouping)."""

    pass


# =========================
# Transformer Block
# =========================
class DiffusionBlock(nn.Module):
    def __init__(self, config, layer_idx, use_moe=False):
        super().__init__()

        self.norm1 = KairosNorm(config.hidden_size)
        self.norm2 = KairosNorm(config.hidden_size)

        self.attn = KairosLiZAttention2(config, layer_idx)
        self.ffn = KairosMoE(config) if use_moe else KairosFFN(config)

    def forward(self, x, position_embeddings=None, cache_params=None, attention_mask=None, position_ids=None):
        x = x + self.attn(
            self.norm1(x),
            position_embeddings=position_embeddings,
            cache_params=cache_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = x + self.ffn(self.norm2(x))
        return x


# =========================
# Backbone (with Attention Residual)
# =========================
class KairosCastingNorm(nn.RMSNorm):
    """Cast weight to input dtype on the fly so the fused kernel dispatches under autocast."""

    def forward(self, x):
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, w, self.eps)


class KairosAttnRes(nn.Module):
    """Softmax attention over a list of prior sublayer outputs (arXiv 2603.15031)"""

    def __init__(self, n_embd):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_embd))
        self.key_norm = KairosCastingNorm(n_embd)

    def forward(self, prior_values):
        V = torch.stack(prior_values, dim=0)
        K = self.key_norm(V)
        logits = torch.einsum("d,lbtd->lbt", self.w, K)
        weights = F.softmax(logits, dim=0)
        return (weights.unsqueeze(-1) * V).sum(dim=0)


class KairosDiffusionBackbone(nn.Module):
    def __init__(self, config, use_moe=False):
        super().__init__()
        self.layers = nn.ModuleList([DiffusionBlock(config, i, use_moe) for i in range(config.num_hidden_layers)])
        self.norm = KairosNorm(config.hidden_size)
        self.aggregator = KairosAttnRes(config.hidden_size)

    def forward(self, x, position_embeddings=None, cache_params=None, attention_mask=None, position_ids=None):
        states = [x]
        for layer in self.layers:
            h = self.aggregator(states)
            x = layer(
                h,
                position_embeddings=position_embeddings,
                cache_params=cache_params,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            states.append(x)
        return self.norm(x)


# =========================
# Embedding & Head
# =========================
class KairosEmbedding(nn.Module):
    def __init__(self, vocab_size: int, num_modalities: int, d_model: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.modality_embed = nn.Embedding(num_modalities, d_model)
        self.scale = d_model**0.5

    def forward(self, token_ids: torch.LongTensor, modality_ids: torch.LongTensor):
        h = self.token_embed(token_ids)
        h = h + self.modality_embed(modality_ids)
        h = h * self.scale
        return h


class OutputHead(nn.Module):
    def __init__(self, embedding: KairosEmbedding):
        super().__init__()
        d_model = embedding.token_embed.embedding_dim
        self.vocab_size = embedding.token_embed.num_embeddings
        self.num_modalities = embedding.modality_embed.num_embeddings

        self.token_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.modality_head = nn.Linear(d_model, self.num_modalities, bias=False)

        self.token_head.weight = embedding.token_embed.weight
        self.modality_head.weight = embedding.modality_embed.weight

    def forward(self, h):
        return self.token_head(h), self.modality_head(h)


# =========================
# Codec & Router Scaling
# =========================
class KairosScaleRouter(nn.Module):
    """
    Build a batched active-position mask per scale, then gather/pad/scatter
    around the backbone call.

    FIX (vs original):
    - `build_active_mask` used to loop in Python over every modality *and*
      every batch row to build the mask element-by-element, then forced a
      `.cpu()` sync per (batch, scale) pair to find segments. It's now a
      single vectorized `torch.isin` + pooling call — no Python loop over
      sequence length, no forced sync.
    - The old design called the backbone once per (batch_idx, segment) with
      batch size 1, and reused **the same layer-level cache tensor** across
      those calls — silently concatenating different examples' K/V together
      whenever batch_size > 1. We now gather every active position across the
      *whole batch* into one padded tensor and call the backbone exactly once
      per scale, so the cache only ever sees one real batch dimension.
    """

    def __init__(self, modality_scales):
        super().__init__()
        self.modality_scales = modality_scales

    def build_active_mask(self, modality_ids, scale_len, scale_idx):
        device = modality_ids.device
        allowed = [m for m, scales in self.modality_scales.items() if scale_idx in scales]
        if not allowed:
            return torch.zeros(modality_ids.shape[0], scale_len, dtype=torch.bool, device=device)

        allowed_t = torch.tensor(allowed, device=device)
        active_full = torch.isin(modality_ids, allowed_t)  # (B, T) vectorized
        pooled = F.adaptive_max_pool1d(active_full.float().unsqueeze(1), scale_len).squeeze(1)
        return pooled > 0.5

    @staticmethod
    def gather_active(x, active_mask):
        """
        x: (B, T, D), active_mask: (B, T) bool
        -> gathered (B, max_len, D), pad_mask (B, max_len), positions (B, max_len)

        FIX (vs previous pass): removed the Python loop over the batch entirely.
        Trick: stable-sort each row so active positions come first (in their
        original relative order), keep only the first `max_len` columns, and
        derive `pad_mask` from the per-row active counts with a single
        broadcasted comparison. Everything here is a single vectorized op —
        no `.item()` sync except the one unavoidable `lengths.max()` needed to
        size the output tensor.
        """
        B, T, D = x.shape
        lengths = active_mask.sum(dim=1)  # (B,)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0

        if max_len == 0:
            return None, None, None

        # Stable sort puts False (0) before True... we want actives first, so
        # sort by (~active_mask): active rows get key 0 and land first.
        order = torch.argsort((~active_mask).long(), dim=1, stable=True)  # (B, T)
        positions = order[:, :max_len]  # (B, max_len): absolute index in [0, T)

        gathered = torch.gather(x, 1, positions.unsqueeze(-1).expand(-1, -1, D))

        arange = torch.arange(max_len, device=x.device).unsqueeze(0)  # (1, max_len)
        pad_mask = arange < lengths.unsqueeze(1)  # (B, max_len), vectorized

        return gathered, pad_mask, positions

    @staticmethod
    def scatter_active(output, chunk, pad_mask, positions):
        """
        FIX (vs previous pass): vectorized scatter, no Python loop over B.
        For padded slots (pad_mask == False), `positions` still points at a
        real — just inactive — index in the sequence (since gather_active's
        sort keeps *every* index, active ones first). Writing the chunk value
        there unconditionally would corrupt an untouched position, so we
        substitute the output's own current value at that index for padded
        slots (a no-op write) via `torch.where` before the scatter.

        FIX (this pass): uses the *out-of-place* `scatter` (returns a new
        tensor) instead of `scatter_`. The in-place version mutated `output`
        after `torch.gather(output, ...)` had already saved it for its own
        backward, which autograd flags as "modified by an inplace operation"
        (version mismatch) — `gather`'s saved reference and the later mutation
        pointed at the same tensor object. Returning a fresh tensor avoids the
        aliasing entirely; the caller must reassign (`output = scatter_active(...)`).
        """
        D = output.shape[-1]
        idx = positions.unsqueeze(-1).expand(-1, -1, D)
        current = torch.gather(output, 1, idx)
        values = torch.where(pad_mask.unsqueeze(-1), chunk, current)
        return output.scatter(1, idx, values)


@dataclass
class CodecOutput:
    scales: list[torch.Tensor]
    length: int


class PyramidalConvCodec(nn.Module):
    """Parallel multi-scale convolutional codec with modality routing."""

    def __init__(self, d_model, stride=5, num_scales=4):
        super().__init__()
        self.stride = stride
        self.num_scales = num_scales

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for level in range(num_scales):
            scale_stride = stride ** (level + 1)
            kernel_size = scale_stride // 2
            kernel_size += kernel_size % 2 == 0
            padding = kernel_size // 2

            self.encoders.append(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=scale_stride, padding=padding, groups=d_model)
            )
            self.decoders.append(
                nn.ConvTranspose1d(
                    d_model, d_model, kernel_size=kernel_size, stride=scale_stride, padding=padding,
                    output_padding=max(scale_stride - 1, 0), groups=d_model,
                )
            )

        self.norm = KairosNorm(d_model * num_scales)
        self.fusion = nn.Linear(d_model * num_scales, d_model)

    def encode(self, x):
        h = x.transpose(1, 2)
        scales = [encoder(h).transpose(1, 2) for encoder in self.encoders]
        return CodecOutput(scales=scales, length=x.shape[1])

    def decode(self, encoded):
        scales = encoded.scales
        length = encoded.length
        reconstructed = []

        for scale, decoder in zip(scales, self.decoders):
            h = decoder(scale.transpose(1, 2))
            reconstructed.append(h.transpose(1, 2))

        # FIX (vs original): reconstructed branches were truncated to
        # `min_len` and then sliced again to `length`, with no guarantee that
        # `min_len >= length` for arbitrary (stride, kernel, seq_len)
        # combinations — a silent shape mismatch was possible. We now pad any
        # branch shorter than `length` (edge replication, cheap and stable)
        # before truncating, so the output is always exactly `length` long.
        padded = []
        for r in reconstructed:
            if r.shape[1] < length:
                pad_amount = length - r.shape[1]
                r = F.pad(r.transpose(1, 2), (0, pad_amount), mode="replicate").transpose(1, 2)
            padded.append(r[:, :length])

        h = torch.cat(padded, dim=-1)
        h = self.norm(h)
        return self.fusion(h)


# =========================
# Full Model (standard HF-like)
# =========================
@dataclass
class KairosOutput(CausalLMOutputWithPast):
    encoder_last_hidden_state: torch.FloatTensor | None = None
    modality_logits: torch.FloatTensor | None = None


class KairosDiffusionLLM(PreTrainedModel, DiffusionGemmaGenerationMixin):
    def __init__(self, config, vocab_size=None, use_moe=None):
        super().__init__(config)

        # FIX (vs original): `num_experts` used to be an opaque constructor arg
        # that silently did nothing unless passed explicitly. `use_moe`
        # defaults to `config.use_moe` so the MoE fields in the config are no
        # longer inert by default.
        if use_moe is None:
            use_moe = config.use_moe

        self.codec = PyramidalConvCodec(d_model=config.hidden_size, stride=config.stride, num_scales=config.num_scales)
        self.router = KairosScaleRouter(config.modality_scales)

        if vocab_size is None:
            vocab_size = config.vocab_size

        self.embedding = KairosEmbedding(vocab_size=vocab_size, num_modalities=config.num_modalities, d_model=config.hidden_size)

        self.backbones = nn.ModuleList(
            [KairosDiffusionBackbone(config=config, use_moe=use_moe) for _ in range(self.codec.num_scales)]
        )

        # Shared rotary module used to compute cos/sin from *absolute* position
        # ids at the model level (see forward()), instead of letting each
        # attention layer silently recompute RoPE from a purely local,
        # segment-relative index (that was the "position_embeddings never
        # transmitted" bug from the original code).
        self.rotary = KairosRotaryEmbedding(config, config.head_dim)

        self.norm = KairosNorm(config.hidden_size)
        self.lm_head = OutputHead(self.embedding)

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        modality_ids=None,
        attention_mask=None,
        self_conditioning_logits=None,
        cache_params=None,
        **kwargs,
    ):
        x = decoder_input_ids if decoder_input_ids is not None else input_ids
        if x is None:
            raise ValueError()

        if modality_ids is None:
            modality_ids = torch.full_like(x, self.config.text_modality_id)

        h = self.embedding(token_ids=x, modality_ids=modality_ids)

        if self_conditioning_logits is not None:
            probs = torch.softmax(self_conditioning_logits, dim=-1)
            h = h + (probs @ self.embedding.token_embed.weight)

        encoded = self.codec.encode(h)
        features = []

        for scale_idx, (scale, backbone) in enumerate(zip(encoded.scales, self.backbones)):
            output = scale.clone()
            local_cache = cache_params.get(scale_idx) if cache_params is not None else None

            active_mask = self.router.build_active_mask(modality_ids, scale.shape[1], scale_idx)

            if attention_mask is not None:
                pad_pool = F.adaptive_max_pool1d(attention_mask.float().unsqueeze(1), scale.shape[1]).squeeze(1)
                active_mask = active_mask & (pad_pool > 0.5)

            gathered, pad_mask, positions = self.router.gather_active(scale, active_mask)

            if gathered is not None:
                # absolute position within this scale's timeline = cache offset
                # (from the first attention layer of this backbone) + the
                # gathered index. This replaces the bug where every backbone
                # call implicitly used a purely local, segment-relative index.
                cache_offset = local_cache.get_total_seen(0) if local_cache is not None else 0
                position_ids = positions + cache_offset
                cos, sin = self.rotary(scale, position_ids, max_position=None)

                chunk = backbone(
                    gathered,
                    position_embeddings=(cos, sin),
                    cache_params=local_cache,
                    attention_mask=pad_mask,
                    position_ids=position_ids,
                )
                output = self.router.scatter_active(output, chunk, pad_mask, positions)

            features.append(output)

        decoded = CodecOutput(scales=features, length=encoded.length)
        h = self.codec.decode(decoded)
        h = self.norm(h)
        token_logits, modality_logits = self.lm_head(h)

        return KairosOutput(
            logits=token_logits,
            modality_logits=modality_logits,
            # FIX (vs original): always returned None, breaking HF's
            # generate() which expects the updated cache back from the model.
            past_key_values=cache_params,
        )