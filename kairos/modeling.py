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
    Unified cache for bidirectional DeltaNet + attention with diffusion-style usage.
    KV/conv/ssm tensors are stored with their full batch dimension intact — the
    caller is responsible for calling `update()`/`process()` once per scale for
    the *whole batch at once* (see KairosDiffusionLLM.forward), not once per
    example. That is what actually fixes the batch-corruption bug: this cache
    was never the problem, calling it once per example was.
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
        new_cache = KairosCache(self.config)
        new_cache.conv_caches = [c.clone() if c is not None else None for c in self.conv_caches]
        new_cache.ssm_caches = [c.clone() if c is not None else None for c in self.ssm_caches]
        new_cache._key_cache = {k: v.clone() if v is not None else None for k, v in self._key_cache.items()}
        new_cache._value_cache = {k: v.clone() if v is not None else None for k, v in self._value_cache.items()}
        new_cache.past_length = self.past_length.copy()
        return new_cache


class KairosMultiCache(DynamicCache):
    """One KairosCache per backbone scale."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.caches = [KairosCache(config) for _ in range(config.num_scales)]

    def get(self, idx):
        return self.caches[idx]

    def clone(self):
        out = KairosMultiCache.__new__(KairosMultiCache)
        out.config = self.config
        out.caches = [c.clone() for c in self.caches]
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
        """x: (B, T, D), active_mask: (B, T) bool -> gathered (B, max_len, D), pad_mask (B, max_len), positions (B, max_len)"""
        B, T, D = x.shape
        lengths = active_mask.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0

        if max_len == 0:
            return None, None, None

        gathered = x.new_zeros(B, max_len, D)
        pad_mask = torch.zeros(B, max_len, dtype=torch.bool, device=x.device)
        positions = torch.zeros(B, max_len, dtype=torch.long, device=x.device)

        # Loop bound is batch size (small), not sequence length — the expensive
        # per-timestep loop from the original implementation is gone.
        for b in range(B):
            idx = active_mask[b].nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n == 0:
                continue
            gathered[b, :n] = x[b, idx]
            pad_mask[b, :n] = True
            positions[b, :n] = idx

        return gathered, pad_mask, positions

    @staticmethod
    def scatter_active(output, chunk, pad_mask, positions):
        B = output.shape[0]
        for b in range(B):
            idx = positions[b][pad_mask[b]]
            n = idx.numel()
            if n == 0:
                continue
            output[b, idx] = chunk[b, :n]


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
                self.router.scatter_active(output, chunk, pad_mask, positions)

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