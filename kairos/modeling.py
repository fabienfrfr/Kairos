import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.diffusion_gemma.generation_diffusion_gemma import DiffusionGemmaGenerationMixin


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
        stride = 5, # need to find for multimodality
        **kwargs
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
        }

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
        self.linear_conv_kernel_dim = kwargs.get("linear_conv_kernel_dim", 4) # Qwen3_5
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
            ["ld"] * n_layers  # default: DeltaNet+SWA layers
        )
        self.slw_wsize = kwargs.get("slw_wsize", -1)

        # warning
        assert d_model % n_heads == 0, "hidden_size must be divisible by n_heads"


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

        self.ffn = (
            KairosMoE(config)
            if num_experts is not None
            else KairosFFN(config)
        )

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
        V = torch.stack(prior_values, dim=0)             # [L, B, T, d]
        K = self.key_norm(V)                             # [L, B, T, d]
        logits = torch.einsum("d,lbtd->lbt", self.w, K)  # [L, B, T]
        weights = F.softmax(logits, dim=0)               # over the L source dim
        return (weights.unsqueeze(-1) * V).sum(dim=0)    # [B, T, d]


class KairosDiffusionBackbone(nn.Module):
    def __init__(self, config, num_experts=None):
        super().__init__()

        self.layers = nn.ModuleList([
            DiffusionBlock(config, i, num_experts)
            for i in range(config.num_hidden_layers)
        ])

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
# Embeddings, Codec & Head
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


class PyramidalConvCodec(nn.Module):
    """ Parallel multi-scale convolutional codec. """

    def __init__(
        self,
        d_model: int,
        stride: int = 5,
        num_scales: int = 3,
    ):
        super().__init__()

        self.stride = stride
        self.num_scales = num_scales

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for level in range(num_scales):

            scale_stride = stride ** (level + 1)
            kernel_size = scale_stride // 2
            if kernel_size % 2 == 0:
                kernel_size += 1

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
                    output_padding=max(scale_stride - 1, 0),
                    groups=d_model,
                )
            )
        

        self.norm = KairosNorm(num_scales * d_model)
        self.fusion = nn.Linear(num_scales * d_model,  d_model)


    def encode(self, x):
        h = x.transpose(1, 2)  # [B, D, T]
        scales = []

        for encoder in self.encoders:
            scales.append(
                encoder(h).transpose(1, 2)
            )

        return scales

    def decode(self, scales):
        reconstructed = []

        for scale, decoder in zip(
            scales,
            self.decoders,
        ):

            h = scale.transpose(1, 2)
            h = decoder(h)
            h = h.transpose(1, 2)

            reconstructed.append(h)

        h = torch.cat(reconstructed, dim=-1)
        h = self.norm(h)
        return self.fusion(h)

    def forward(self,  x, mode="encode"):
        if mode == "encode":
            return self.encode(x)

        if mode == "decode":
            return self.decode(x)

        raise ValueError("mode must be 'encode' or 'decode'")


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
            config.hidden_size, stride=config.stride,
        )

        if vocab_size is None:
            vocab_size = config.vocab_size

        # Embedding
        self.embedding = KairosEmbedding(
            vocab_size=vocab_size,
            num_modalities=config.num_modalities,
            d_model=config.hidden_size,
        )

        # One backbone per pyramid scale
        self.backbones = nn.ModuleList([
            KairosDiffusionBackbone(
                config=config,
                num_experts=num_experts,
            )
            for _ in range(self.codec.num_scales)
        ])

        self.norm = KairosNorm(config.hidden_size)
        self.lm_head = OutputHead(self.embedding)

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        modality_ids=None,
        self_conditioning_logits=None,
        past_key_values=None,
        cache_params=None,
        **kwargs,
    ):
        # Input selection
        if decoder_input_ids is not None:
            x = decoder_input_ids
        elif input_ids is not None:
            x = input_ids
        else:
            raise ValueError("You must provide input_ids or decoder_input_ids")

        # Default modality
        if modality_ids is None:
            modality_ids = torch.full_like(x, self.config.text_modality_id)

        # Embedding  h : [B,T,D]
        h = self.embedding(token_ids=x, modality_ids=modality_ids)

        # Diffusion self-conditioning
        if self_conditioning_logits is not None:

            probs = torch.softmax(self_conditioning_logits,dim=-1)
            soft_emb = (probs@ self.embedding.token_embed.weight)
            h = h + soft_emb

        # Pyramid encoding
        scales = self.codec(h, mode="encode")

        # Parallel backbones
        features = []

        for backbone, scale in zip(
            self.backbones,
            scales,
        ):
            feat = backbone(
                scale, cache_params=cache_params,
            )
            features.append(feat)

        # Pyramid decoding / fusion
        h = self.codec(features, mode="decode")

        # Final norm
        h = self.norm(h)

        # Heads
        token_logits, modality_logits = self.lm_head(h)

        return KairosOutput(
            logits=token_logits,
            modality_logits=modality_logits,
            past_key_values=None,
        )