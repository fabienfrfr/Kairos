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

from .attentions import KairosLiZAttention2, KairosNorm


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
        stride=5,  # need to find for multimodality
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = n_layers
        self.vocab_size = vocab_size
        self.num_modalities = kwargs.get("num_modalities", 8)
        self.text_modality_id = kwargs.get("text_modality_id", 0)

        self.modality_scales = {
            0: [0, 1],  # text
            1: [1, 2],  # image
            2: [2, 3],  # video
        }
        self.num_scales = kwargs.get("num_scales", 4)

        # Convolutionnal Byte-Codec
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
        self.linear_conv_kernel_dim = kwargs.get("linear_conv_kernel_dim", 4)  # Qwen3_5
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)

        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.A_init_range = (1.0, 16.0)

        # FFN / MLP
        self.intermediate_size = intermediate_size

        # MoE
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

        # Layers config (required by KairosCache)
        self.layers_config = kwargs.get(
            "layers_config",
            ["ld"] * n_layers,  # default: DeltaNet+SWA layers
        )
        self.slw_wsize = kwargs.get("slw_wsize", -1)

        # warning
        assert d_model % n_heads == 0, "hidden_size must be divisible by n_heads"


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
            if "l" in layer_type or "d" in layer_type:  # attention layers
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
        if "l" not in self.layers_config[layer_idx]:  # trim SWA only
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
        return (self.conv_caches[layer_idx], self.ssm_caches[layer_idx])

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
        new_cache.conv_caches = [c.clone() if c is not None else None for c in self.conv_caches]
        new_cache.ssm_caches = [c.clone() if c is not None else None for c in self.ssm_caches]
        new_cache._key_cache = {k: v.clone() if v is not None else None for k, v in self._key_cache.items()}
        new_cache._value_cache = {k: v.clone() if v is not None else None for k, v in self._value_cache.items()}
        new_cache.past_length = self.past_length.copy()

        return new_cache


class KairosMultiCache(DynamicCache):
    """
    One KairosCache per backbone scale.

    Scales are isolated from each other.
    Segments within the same scale share the same cache/state.
    """    
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.caches = [
            KairosCache(config)
            for _ in range(config.num_scales)  # num_backbone
        ]

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
    def __init__(self, config, layer_idx, num_experts=None):
        super().__init__()

        self.norm1 = KairosNorm(config.hidden_size)
        self.norm2 = KairosNorm(config.hidden_size)

        self.attn = KairosLiZAttention2(config, layer_idx)

        self.ffn = KairosMoE(config) if num_experts is not None else KairosFFN(config)

    def forward(self, x, position_embeddings=None, cache_params=None):
        x = x + self.attn(self.norm1(x), position_embeddings=position_embeddings, cache_params=cache_params)
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
        V = torch.stack(prior_values, dim=0)  # [L, B, T, d]
        K = self.key_norm(V)  # [L, B, T, d]
        logits = torch.einsum("d,lbtd->lbt", self.w, K)  # [L, B, T]
        weights = F.softmax(logits, dim=0)  # over the L source dim
        return (weights.unsqueeze(-1) * V).sum(dim=0)  # [B, T, d]


class KairosDiffusionBackbone(nn.Module):
    def __init__(self, config, num_experts=None):
        super().__init__()

        self.layers = nn.ModuleList([DiffusionBlock(config, i, num_experts) for i in range(config.num_hidden_layers)])

        self.norm = KairosNorm(config.hidden_size)
        self.aggregator = KairosAttnRes(config.hidden_size)

    def forward(self, x, position_embeddings=None, cache_params=None):
        states = [x]

        for layer in self.layers:
            h = self.aggregator(states)
            x = layer(h, position_embeddings=position_embeddings, cache_params=cache_params)
            states.append(x)
        return self.norm(x)


# =========================
# Embedding & Head
# =========================
class KairosEmbedding(nn.Module):
    """Token + modality embeddings."""

    def __init__(
        self,
        vocab_size: int,
        num_modalities: int,
        d_model: int,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.modality_embed = nn.Embedding(num_modalities, d_model)

        self.scale = d_model**0.5

    def forward(
        self,
        token_ids: torch.LongTensor,
        modality_ids: torch.LongTensor,
    ):
        h = self.token_embed(token_ids)
        h = h + self.modality_embed(modality_ids)
        h = h * self.scale

        return h


class OutputHead(nn.Module):
    """Token + modality prediction heads."""

    def __init__(
        self,
        embedding: KairosEmbedding,
    ):
        super().__init__()

        d_model = embedding.token_embed.embedding_dim

        self.vocab_size = embedding.token_embed.num_embeddings
        self.num_modalities = embedding.modality_embed.num_embeddings

        self.token_head = nn.Linear(
            d_model,
            self.vocab_size,
            bias=False,
        )

        self.modality_head = nn.Linear(
            d_model,
            self.num_modalities,
            bias=False,
        )

        self.token_head.weight = embedding.token_embed.weight
        self.modality_head.weight = embedding.modality_embed.weight

    def forward(self, h):
        token_logits = self.token_head(h)
        modality_logits = self.modality_head(h)

        return token_logits, modality_logits


# =========================
# Codec & Router Scaling
# =========================
class KairosScaleRouter(nn.Module):
    """Build routing segments for each scale."""

    def __init__(self, modality_scales):
        super().__init__()
        self.modality_scales = modality_scales

    def _find_segments(self, mask):

        segments = []

        start = None

        for i, active in enumerate(mask.tolist()):
            if active and start is None:
                start = i

            elif not active and start is not None:
                segments.append((start, i))
                start = None

        if start is not None:
            segments.append((start, len(mask)))

        return segments

    def build(
        self,
        modality_ids,
        scales,
    ):
        routing = []

        for scale_idx, scale in enumerate(scales):
            scale_len = scale.shape[1]

            scale_segments = []

            for b in range(modality_ids.shape[0]):
                active = torch.zeros(
                    modality_ids.shape[1],
                    dtype=torch.bool,
                    device=modality_ids.device,
                )

                for modality, allowed_scales in self.modality_scales.items():
                    if scale_idx not in allowed_scales:
                        continue

                    active |= modality_ids[b] == modality

                pooled = F.adaptive_max_pool1d(
                    active.float().view(1, 1, -1),
                    scale_len,
                ).view(-1)

                segments = self._find_segments(pooled.bool().cpu())

                scale_segments.append(segments)

            routing.append(scale_segments)

        return routing


@dataclass
class CodecOutput:
    scales: list[torch.Tensor]
    length: int


class PyramidalConvCodec(nn.Module):
    """Parallel multi-scale convolutional codec with modality routing."""

    def __init__(
        self,
        d_model,
        stride=5,
        num_scales=4,
    ):
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
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    stride=scale_stride,
                    padding=padding,
                    groups=d_model,
                )
            )

            self.decoders.append(
                nn.ConvTranspose1d(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    stride=scale_stride,
                    padding=padding,
                    output_padding=max(
                        scale_stride - 1,
                        0,
                    ),
                    groups=d_model,
                )
            )

        self.norm = KairosNorm(d_model * num_scales)
        self.fusion = nn.Linear(
            d_model * num_scales,
            d_model,
        )

    def encode(self, x):

        h = x.transpose(1, 2)
        scales = []

        for encoder in self.encoders:
            scales.append(encoder(h).transpose(1, 2))

        return CodecOutput(scales=scales, length=x.shape[1])

    def decode(self, encoded):
        scales = encoded.scales
        length = encoded.length

        reconstructed = []

        for scale, decoder in zip(
            scales,
            self.decoders,
        ):
            h = decoder(scale.transpose(1, 2))
            reconstructed.append(h.transpose(1, 2))

        min_len = min(x.shape[1] for x in reconstructed)
        reconstructed = [x[:, :min_len] for x in reconstructed]
        h = torch.cat(reconstructed, dim=-1)

        h = self.norm(h)
        return self.fusion(h)[:, :length]


# =========================
# Full Model (standard HF-like)
# =========================
@dataclass
class KairosOutput(CausalLMOutputWithPast):
    encoder_last_hidden_state: torch.FloatTensor | None = None
    modality_logits: torch.FloatTensor | None = None


class KairosDiffusionLLM(
    PreTrainedModel,
    DiffusionGemmaGenerationMixin,
):
    def __init__(
        self,
        config,
        vocab_size=None,
        num_experts=None,
    ):
        super().__init__(config)

        self.codec = PyramidalConvCodec(
            d_model=config.hidden_size,
            stride=config.stride,
            num_scales=config.num_scales,
        )

        self.router = KairosScaleRouter(config.modality_scales)

        if vocab_size is None:
            vocab_size = config.vocab_size

        self.embedding = KairosEmbedding(
            vocab_size=vocab_size,
            num_modalities=config.num_modalities,
            d_model=config.hidden_size,
        )

        self.backbones = nn.ModuleList(
            [
                KairosDiffusionBackbone(
                    config=config,
                    num_experts=num_experts,
                )
                for _ in range(self.codec.num_scales)
            ]
        )

        self.norm = KairosNorm(config.hidden_size)

        self.lm_head = OutputHead(self.embedding)

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        modality_ids=None,
        self_conditioning_logits=None,
        cache_params=None,
        **kwargs,
    ):

        x = decoder_input_ids if decoder_input_ids is not None else input_ids

        if x is None:
            raise ValueError()

        if modality_ids is None:
            modality_ids = torch.full_like(
                x,
                self.config.text_modality_id,
            )

        h = self.embedding(
            token_ids=x,
            modality_ids=modality_ids,
        )

        if self_conditioning_logits is not None:
            probs = torch.softmax(
                self_conditioning_logits,
                dim=-1,
            )

            h = h + (probs @ self.embedding.token_embed.weight)

        # Encode
        encoded = self.codec.encode(h)
        routing = self.router.build(modality_ids, encoded.scales)

        features = []

        # Gather -> Backbone -> Scatter
        for scale_idx, (scale, backbone) in enumerate(zip(encoded.scales, self.backbones)):
            output = scale.clone()

            for batch_idx, segments in enumerate(routing[scale_idx]):
                for start, end in segments:
                    chunk = scale[batch_idx : batch_idx + 1, start:end]

                    if chunk.shape[1] == 0:
                        continue

                    local_cache = cache_params.get(scale_idx) if cache_params is not None else None

                    chunk = backbone(chunk, cache_params=local_cache)
                    output[batch_idx : batch_idx + 1, start:end] = chunk

            features.append(output)

        # reconstruct
        decoded = CodecOutput(scales=features, length=encoded.length)

        # Decode
        h = self.codec.decode(decoded)
        h = self.norm(h)
        token_logits, modality_logits = self.lm_head(h)

        return KairosOutput(
            logits=token_logits,
            modality_logits=modality_logits,
            past_key_values=None,
        )
