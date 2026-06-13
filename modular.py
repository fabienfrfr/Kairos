import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# =========================
# Attention
# =========================
class SlidingWindowAttention(MistralAttention):
    def __init__(self, config, layer_idx, window_size):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.window_size = window_size

    def forward(self, hidden_states, position_embeddings, **kwargs):
        B, L, _ = hidden_states.shape
        device = hidden_states.device

        idx = torch.arange(L, device=device)
        dist = idx[None, :] - idx[:, None]
        mask = (dist.abs() > self.window_size)

        mask = mask.float() * float("-inf")
        mask = mask[None, None, :, :]

        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            past_key_values=None
        )

# =========================
# DeltaNet
# =========================
class BidirectionalDeltaNet(Qwen3NextGatedDeltaNet):
    def __init__(self, config, layer_idx=0):
        super().__init__(config, layer_idx)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # forward direction
        out_fwd = super().forward(
            hidden_states=x,
            cache_params=None,
            attention_mask=None
        )

        # backward direction (reverse sequence)
        x_rev = torch.flip(x, dims=[1])
        out_bwd = super().forward(
            hidden_states=x_rev,
            cache_params=None,
            attention_mask=None
        )
        out_bwd = torch.flip(out_bwd, dims=[1])

        # combine (shared weights)
        return self.alpha * (out_fwd + out_bwd)


# =========================
# Shared QKV Projection
# =========================
class SharedQKV(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)



class SharedQKVProjection(nn.Module):
    """
    Replacement for Qwen QKVZ projection with shared QKV.
    Keeps correct expected dimensions.
    """

    def __init__(self, shared_qkv, delta):
        super().__init__()
        self.shared_qkv = shared_qkv

        # adapt to Qwen expected dims
        self.q_proj = nn.Linear(delta.hidden_size, delta.key_dim, bias=False)
        self.k_proj = nn.Linear(delta.hidden_size, delta.key_dim, bias=False)
        self.v_proj = nn.Linear(delta.hidden_size, delta.value_dim, bias=False)
        self.z_proj = nn.Linear(delta.hidden_size, delta.value_dim, bias=False)

    def forward(self, x):
        # shared base
        q, k, v = self.shared_qkv(x)

        # adapt dims
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        z = self.z_proj(x)

        return torch.cat([q, k, v, z], dim=-1)


# =========================
# LiZAttention
# =========================
class LiZAttention(nn.Module):
    def __init__(self, config, layer_idx, window_size):
        super().__init__()

        self.hidden_size = config.hidden_size

        # shared QKV
        self.shared_qkv = SharedQKV(self.hidden_size)

        # SWA
        self.swa = SlidingWindowAttention(config, layer_idx, window_size)
        self.swa.q_proj = self.shared_qkv.q
        self.swa.k_proj = self.shared_qkv.k
        self.swa.v_proj = self.shared_qkv.v

        # DeltaNet
        self.delta = BidirectionalDeltaNet(config, layer_idx)

        self.delta.in_proj_qkvz = SharedQKVProjection(
            self.shared_qkv,
            self.delta
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, position_embeddings=None):
        swa_out, _ = self.swa(
            hidden_states=x,
            position_embeddings=position_embeddings,
        )

        delta_out = self.delta(x)

        return swa_out + self.alpha * delta_out
        

# =========================
# Normalization
# =========================
class RMSNorm(LlamaRMSNorm):
    """RMS Norm for stabilization"""
    pass


# =========================
# FeedForward / MoE
# =========================
class FFN(Qwen2MoeMLP):
    """dense FFN (SwiGLU + optimisations HF)."""
    pass


class MoEBlock(DeepseekV3MoE):
    """MoE (routing + scaling + grouping)."""
    pass



# =========================
# Transformer Block
# =========================

class DiffusionBlock(nn.Module):
    def __init__(self, config, layer_idx, window_size, num_experts=None):
        super().__init__()

        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

        self.attn = LiZAttention(config, layer_idx, window_size)

        self.ffn = (
            MoEBlock(config)
            if num_experts is not None
            else FFN(config)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



# =========================
# Backbone (with Attention Residual)
# =========================
class CastingRMSNorm(nn.RMSNorm):
    """Cast weight to input dtype on the fly so the fused kernel dispatches under autocast."""

    def forward(self, x):
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, w, self.eps)

class AttnResAggregator(nn.Module):
    """Softmax attention over a list of prior sublayer outputs (arXiv 2603.15031)"""
    def __init__(self, n_embd):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_embd))
        self.key_norm = CastingRMSNorm(n_embd)

    def forward(self, prior_values):
        V = torch.stack(prior_values, dim=0)             # [L, B, T, d]
        K = self.key_norm(V)                             # [L, B, T, d]
        logits = torch.einsum("d,lbtd->lbt", self.w, K)  # [L, B, T]
        weights = F.softmax(logits, dim=0)               # over the L source dim
        return (weights.unsqueeze(-1) * V).sum(dim=0)    # [B, T, d]


class DiffusionBackbone(nn.Module):
    def __init__(self, config, n_layers, window_size, num_experts=None):
        super().__init__()

        self.layers = nn.ModuleList([
            DiffusionBlock(config, i, window_size, num_experts)
            for i in range(n_layers)
        ])

        self.norm = RMSNorm(config.hidden_size)
        self.aggregator = AttnResAggregator(config.hidden_size)

    def forward(self, x):
        states = [x]

        for layer in self.layers:
            h = self.aggregator(states)
            x = layer(h)
            states.append(x)
        return self.norm(x)


# =========================
# Embeddings
# =========================
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

    def forward(self, x):
        pass


class TimestepEmbedding(nn.Module):
    def __init__(self, max_timesteps, d_model):
        super().__init__()

    def forward(self, t):
        pass


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()

    def forward(self, x):
        pass



# =========================
# Full Model (standard HF-like)
# =========================
class DiffusionLLM(nn.Module):
    def __init__(
        self,
        vocab_size=260,
        d_model=768,
        n_layers=12,
        n_heads=12,
        window_size=128,
        max_seq_len=2048,
        max_timesteps=1000,
        num_experts=None,
    ):
        super().__init__()

        # Embeddings
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.pos_embed = PositionEmbedding(max_seq_len, d_model)
        self.time_embed = TimestepEmbedding(max_timesteps, d_model)

        # Backbone
        self.backbone = DiffusionBackbone(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size,
            num_experts=num_experts,
        )

        # Head
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x_t, t, mask=None):
        pass


# =========================
# Diffusion Scheduler
# =========================
class DiffusionScheduler(nn.Module):
    def __init__(self, num_timesteps, mask_token_id):
        super().__init__()

    def corrupt(self, x0, t):
        pass


# =========================
# Loss
# =========================
class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, x0, t):
        pass


# =========================
# Sampler
# =========================
class DiffusionSampler(nn.Module):
    def __init__(self, mask_token_id, num_steps):
        super().__init__()

    def sample(self, model, seq_len):
        pass


# =========================
# Trainer Wrapper
# =========================
class DiffusionTrainer(nn.Module):
    def __init__(self, model, scheduler):
        super().__init__()

    def training_step(self, batch):
        pass