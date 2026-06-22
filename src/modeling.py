import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.diffusion_gemma.generation_diffusion_gemma import DiffusionGemmaGenerationMixin


from .attentions import KairosLiZAttention2

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
        stride = 3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = n_layers
        self.vocab_size = vocab_size

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


        # Convolutionnal Byte-Codec
        self.stride = stride

        # Layers config (required by KairosCache)
        self.layers_config = kwargs.get(
            "layers_config",
            ["ld"] * n_layers  # default: DeltaNet+SWA layers
        )
        self.slw_wsize = kwargs.get("slw_wsize", -1)

        # warning
        assert d_model % n_heads == 0, "hidden_size must be divisible by n_heads"


# =========================
# Normalization
# =========================
class KairosNorm(LlamaRMSNorm):
    """RMS Norm for stabilization"""
    pass


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
    def __init__(self, vocab_size, d_model, codec):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.scale = d_model ** 0.5
        self.codec = codec

    def forward(self, x):
        h = self.embed(x) * self.scale
        return self.codec(h, mode="encode")


class OutputHead(nn.Module):
    def __init__(self, embedding, codec):
        super().__init__()

        d = embedding.embed.embedding_dim
        self.vocab_size = embedding.embed.num_embeddings

        self.codec = codec

        self.lm_head = nn.Linear(d, self.vocab_size, bias=False)
        self.lm_head.weight = embedding.embed.weight

    def forward(self, h):
        h = self.codec(h, mode="decode")
        return self.lm_head(h)


class ConvCodec(nn.Module):
    def __init__(self, d_model, stride=3):
        super().__init__()

        self.stride = stride

        self.enc = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=d_model
        )

        self.dec = nn.ConvTranspose1d(
            d_model,
            d_model,
            kernel_size=5,
            stride=stride,
            padding=2,
            output_padding=stride - 1,
            groups=d_model
        )
        # buffer
        self.orig_len = 0

    def forward(self, x, mode="encode"):
        # x: (B, T, d)
        if mode == "encode":
            self.orig_len = x.shape[1] # to fix : not-thread-safe
            return self.enc(x.transpose(1, 2)).transpose(1, 2)
            
        elif mode == "decode":
            h = self.dec(x.transpose(1, 2)).transpose(1, 2)
            return h[:, :self.orig_len, :]
        else:
            raise ValueError("mode must be 'encode' or 'decode'")


# =========================
# Full Model (standard HF-like)
# =========================
class DiffusionGemmaBlockDiffusionOutputWithPast(CausalLMOutputWithPast):
    encoder_last_hidden_state: torch.FloatTensor | None = None


class KairosDiffusionLLM(PreTrainedModel, DiffusionGemmaGenerationMixin):
    def __init__(
        self,
        config,
        vocab_size=259,
        num_experts=None,
    ):
        super().__init__(config)

        # Codec
        self.codec = ConvCodec(config.hidden_size, stride=config.stride)
        # Embedding (learned tokenizer via conv + stride)
        self.token_embed = KairosEmbedding(vocab_size, config.hidden_size, self.codec)

        # Backbone (SWA / DeltaNet etc.)
        self.backbone = KairosDiffusionBackbone(
            config=config,
            num_experts=num_experts,
        )

        # Output head
        self.norm = KairosNorm(config.hidden_size)
        self.lm_head = OutputHead(self.token_embed, self.codec)


    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        self_conditioning_logits=None,
        past_key_values=None,
        cache_params=None,
        **kwargs
    ):
        # canvas
        if decoder_input_ids is not None:
            x = decoder_input_ids
        elif input_ids is not None:
            x = input_ids
        else:
            raise ValueError("You must provide input_ids or decoder_input_ids")

        # embedding
        h = self.token_embed(x)

        # self-conditioning (diffusion)
        if self_conditioning_logits is not None:
            probs = torch.softmax(self_conditioning_logits, dim=-1)
            soft_emb = probs @ self.token_embed.embed.weight
            h = h + soft_emb

        # backbone
        h = self.backbone(h, cache_params=cache_params)
        h = self.norm(h)

        # projection vocab
        logits = self.lm_head(h)

        # HF-compatible
        return DiffusionGemmaBlockDiffusionOutputWithPast(
            logits=logits,
            past_key_values=None,  # need or not ?
        )
