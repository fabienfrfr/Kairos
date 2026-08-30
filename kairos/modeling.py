import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP

from .attentions import KairosLiZAttention2, KairosNorm, KairosRotaryEmbedding

try:
    from .generation import KairosDiffusionGenerationMixin
except ImportError:  # transformers < 5.15: training still works, generation does not

    class KairosDiffusionGenerationMixin:
        pass


class KairosConfig(PretrainedConfig):
    """modality_scales defaults every modality id up to num_modalities to scale 0 so."""

    model_type = "kairos"

    def __init__(
        self,
        d_model=768,
        n_heads=12,
        n_layers=12,
        vocab_size=291,  # matches current KairosTokenizer; callers should pass len(tokenizer)
        intermediate_size=2048,
        window_size=128,
        stride=5,
        **kwargs,
    ):
        # a reloaded config.json stores derived HF-style names; read them back first for round-trips
        d_model = kwargs.pop("hidden_size", d_model)
        n_heads = kwargs.pop("num_attention_heads", n_heads)
        n_layers = kwargs.pop("num_hidden_layers", n_layers)
        window_size = kwargs.pop("sliding_window_size", window_size)

        super().__init__(**kwargs)

        assert d_model % n_heads == 0, "hidden_size must be divisible by n_heads"

        # canonical HF names (read by borrowed HF modules) + friendly aliases (model cards, logs)
        self.hidden_size = self.d_model = d_model
        self.num_attention_heads = self.n_heads = n_heads
        self.num_hidden_layers = self.n_layers = n_layers
        self.sliding_window_size = self.window_size = window_size
        self.vocab_size = vocab_size

        self.num_modalities = kwargs.get("num_modalities", 9)
        self.text_modality_id = kwargs.get("text_modality_id", 0)
        self.num_scales = kwargs.get("num_scales", 4)
        self.codec_mode = kwargs.get("codec_mode", "conv")
        # rank of the conv-mode codec's mixer bottleneck; None -> PyramidalCodec default
        self.codec_mixer_rank = kwargs.get("codec_mixer_rank", None)
        self.predict_octet_family = kwargs.get("predict_octet_family", True)
        self.num_octet_families = kwargs.get("num_octet_families", 18)  # match KairosTokenizer

        # id 0 (text) stays finest; higher ids default to coarser scales, never scale 0 alone
        def _clip(scales):
            return sorted({s for s in scales if 0 <= s < self.num_scales}) or [self.num_scales - 1]

        default_scales = {m: _clip([m, m + 1]) for m in range(self.num_modalities)}
        self.modality_scales = kwargs.get("modality_scales", default_scales)

        self.stride = stride

        self.num_key_value_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = 0.0
        self.rope_theta = 10000.0
        self.max_position_embeddings = 4096

        self.linear_conv_kernel_dim = kwargs.get("linear_conv_kernel_dim", 4)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)

        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.A_init_range = (1.0, 16.0)
        self.initializer_range = kwargs.get("initializer_range", 0.02)

        self.intermediate_size = intermediate_size

        # only num_local_experts is real; n_routed_experts is a property alias below
        self.num_local_experts = kwargs.get("num_local_experts", kwargs.get("n_routed_experts", 8))
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 2)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", intermediate_size)
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)
        self.routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
        self.n_group = kwargs.get("n_group", 1)
        self.topk_group = kwargs.get("topk_group", 1)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", False)
        self.use_moe = kwargs.get("use_moe", False)
        self.use_memory_gate = kwargs.get("use_memory_gate", False)

        self.layers_config = kwargs.get("layers_config", ["ld"] * n_layers)
        self.slw_wsize = kwargs.get("slw_wsize", -1)

        # v3 Block-AttnRes: windows prior layer outputs
        self.attnres_block_size = kwargs.get("attnres_block_size", 1)
        self.share_backbones = kwargs.get("share_backbones", False)

        # block-diffusion generation: one canvas of this many tokens per outer loop
        self.canvas_length = kwargs.get("canvas_length", 128)

    @property
    def n_routed_experts(self):
        """Alias for num_local_experts, the field DeepseekV3Experts actually reads."""
        return self.num_local_experts

    @n_routed_experts.setter
    def n_routed_experts(self, value):
        self.num_local_experts = value


class KairosCache(DynamicCache):
    """Cache for block-diffusion inference: `.clone()` before each denoising step to avoid state."""

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


class KairosMemoryGate(nn.Module):
    """Cross-attention gate over a low-rank bottleneck; blends state_t with a memory bank."""

    def __init__(self, state_dim, bottleneck_dim=None):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim or max(8, round(math.sqrt(state_dim) / 4))
        self.down = nn.Linear(state_dim, self.bottleneck_dim)
        self.up = nn.Linear(self.bottleneck_dim, state_dim)
        self.context_attn = nn.MultiheadAttention(self.bottleneck_dim, num_heads=1, batch_first=True)

    def forward(self, state_t, memory=None):
        """state_t: (B, D). memory: (M, D) or None. Returns (B, D)."""
        if memory is None or memory.size(0) == 0:
            return state_t
        if memory.size(0) == 1:
            return memory[0].expand_as(state_t).contiguous()
        q = self.down(state_t).unsqueeze(1)
        kv = torch.cat([q, self.down(memory).unsqueeze(0).expand(state_t.size(0), -1, -1)], dim=1)
        out, _ = self.context_attn(q, kv, kv)
        return self.up(out.squeeze(1))


def gate_memory_bank(model, memory_caches: list, batch_size: int) -> "KairosMultiCache":
    """Gates a zero state_t against memory_caches to seed ssm_caches; no-op layers stay None."""
    new_cache = KairosMultiCache(model.config)
    gate = model.memory_gate
    if gate is None:
        return new_cache
    for scale_idx, backbone in enumerate(model.backbones):
        for layer_idx in backbone.deltanet_layer_indices:
            parts = []
            per_row_shape = None
            for c in memory_caches:
                s = c.caches[scale_idx].ssm_caches[layer_idx]
                if s is not None:
                    per_row_shape = s.shape[1:]
                    parts.append(s.reshape(s.shape[0], -1))
            if not parts:
                continue
            memory = torch.cat(parts, dim=0)
            state_t = memory.new_zeros(batch_size, memory.shape[1])
            blended = gate(state_t, memory)
            new_cache.caches[scale_idx].ssm_caches[layer_idx] = blended.reshape(batch_size, *per_row_shape)
    return new_cache


class KairosFFN(Qwen2MoeMLP):
    pass


class KairosMoE(DeepseekV3MoE):
    """DeepseekV3MoE's expert weights are raw torch.empty(), never initialized; fixed here."""

    def __init__(self, config):
        super().__init__(config)
        std = getattr(config, "initializer_range", 0.02)
        self.experts.gate_up_proj.data.normal_(mean=0.0, std=std)
        self.experts.down_proj.data.normal_(mean=0.0, std=std)
        self.gate.weight.data.normal_(mean=0.0, std=std)  # was torch.zeros() at construction; fine


class DiffusionBlock(nn.Module):
    def __init__(self, config, layer_idx, use_moe=False):
        super().__init__()
        self.norm1 = KairosNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = KairosNorm(config.hidden_size, eps=config.rms_norm_eps)
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


class KairosCastingNorm(nn.RMSNorm):
    def forward(self, x):
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, w, self.eps)


class KairosAttnRes(nn.Module):
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
    """v3 Block-AttnRes: prior layer outputs are windowed into blocks before aggregation."""

    def __init__(self, config, use_moe=False):
        super().__init__()
        self.layers = nn.ModuleList([DiffusionBlock(config, i, use_moe) for i in range(config.num_hidden_layers)])
        self.norm = KairosNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.aggregator = nn.ModuleList([KairosAttnRes(config.hidden_size) for _ in range(config.num_hidden_layers)])
        self.attnres_block_size = max(1, getattr(config, "attnres_block_size", 1))
        self.deltanet_layer_indices = [i for i, lt in enumerate(config.layers_config) if "d" in lt]

    def forward(self, x, position_embeddings=None, cache_params=None, attention_mask=None, position_ids=None):
        emb = x
        completed = []  # finalized block-sums of prior layer
        partial = None  # running sum of the current
        in_block = 0
        S = self.attnres_block_size

        def sources():
            return [emb] + completed + ([partial] if partial is not None else [])

        for layer_idx, layer in enumerate(self.layers):
            h = self.aggregator[layer_idx](sources())
            x = layer(
                h,
                position_embeddings=position_embeddings,
                cache_params=cache_params,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            partial = x if partial is None else partial + x
            in_block += 1
            if in_block == S:
                completed.append(partial)
                partial = None
                in_block = 0

        return self.norm(x)


class KairosEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_octet_families: int = 0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.family_embed = nn.Embedding(num_octet_families, d_model) if num_octet_families > 0 else None
        n_parts = 1 + (1 if self.family_embed is not None else 0)
        self.fusion_proj = nn.Linear(d_model * n_parts, d_model)

    def forward(self, token_ids, family_ids=None):
        parts = [self.token_embed(token_ids)]
        if self.family_embed is not None:
            if family_ids is None:
                family_ids = torch.zeros_like(token_ids)
            parts.append(self.family_embed(family_ids))
        if len(parts) == 1:
            return parts[0]
        return self.fusion_proj(torch.cat(parts, dim=-1))


class OutputHead(nn.Module):
    """Predicts the token stream; optionally an auxiliary octet-family id too (see below)."""

    def __init__(self, embedding: KairosEmbedding, num_octet_families: int = 0):
        super().__init__()
        d_model = embedding.token_embed.embedding_dim
        self.vocab_size = embedding.token_embed.num_embeddings
        self.token_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.token_head.weight = embedding.token_embed.weight
        self.octet_head = nn.Linear(d_model, num_octet_families) if num_octet_families > 0 else None

    def forward(self, h):
        token_logits = self.token_head(h)
        octet_logits = self.octet_head(h) if self.octet_head is not None else None
        return token_logits, octet_logits


class KairosScaleRouter(nn.Module):
    """Gathers active positions per scale into a padded batch, runs the backbone."""

    def __init__(self, modality_scales):
        super().__init__()
        self.modality_scales = modality_scales

    def build_active_mask(self, modality_ids, scale_len, scale_idx):
        device = modality_ids.device
        allowed = [m for m, scales in self.modality_scales.items() if scale_idx in scales]
        if not allowed:
            return torch.zeros(modality_ids.shape[0], scale_len, dtype=torch.bool, device=device)
        allowed_t = torch.tensor(allowed, device=device)
        active_full = torch.isin(modality_ids, allowed_t)
        pooled = F.adaptive_max_pool1d(active_full.float().unsqueeze(1), scale_len).squeeze(1)
        return pooled > 0.5

    @staticmethod
    def gather_active(x, active_mask):
        _, _, D = x.shape
        lengths = active_mask.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_len == 0:
            return None, None, None
        order = torch.argsort((~active_mask).long(), dim=1, stable=True)
        positions = order[:, :max_len]
        gathered = torch.gather(x, 1, positions.unsqueeze(-1).expand(-1, -1, D))
        arange = torch.arange(max_len, device=x.device).unsqueeze(0)
        pad_mask = arange < lengths.unsqueeze(1)
        return gathered, pad_mask, positions

    @staticmethod
    def scatter_active(output, chunk, pad_mask, positions):
        D = output.shape[-1]
        idx = positions.unsqueeze(-1).expand(-1, -1, D)
        current = torch.gather(output, 1, idx)
        values = torch.where(pad_mask.unsqueeze(-1), chunk.to(output.dtype), current)
        return output.scatter(1, idx, values)


@dataclass
class CodecOutput:
    scales: list
    length: int


class PyramidalCodec(nn.Module):
    """Multi-scale encode/decode: mode='patch' (nn.Linear per scale) or 'conv' (cuDNN, faster)."""

    def __init__(self, d_model, stride=5, num_scales=4, mode="patch", norm_eps=1e-6, mixer_rank=None):
        super().__init__()
        assert mode in ("patch", "conv"), f"mode must be 'patch' or 'conv', got {mode!r}"
        self.stride = stride
        self.num_scales = num_scales
        self.mode = mode
        self.patch_sizes = [stride ** (level + 1) for level in range(num_scales)]
        # bottleneck mixer: O(2*r*d) instead of a dense 1x1 conv's O(d^2), same full mixing
        self.mixer_rank = mixer_rank if mixer_rank is not None else max(8, d_model // 4)

        if mode == "patch":
            self._init_patch(d_model)
        else:
            self._init_conv(d_model)

        self.norm = KairosNorm(d_model * num_scales, eps=norm_eps)
        self.fusion = nn.Linear(d_model * num_scales, d_model)

    def _init_patch(self, d_model):
        """Linear(patch*d_model, d_model) per scale; decode reuses the same weight, transposed."""
        self.encode_lin = nn.ModuleList(
            nn.Linear(patch * d_model, d_model) for patch in self.patch_sizes
        )
        self.decode_b = nn.ParameterList(
            nn.Parameter(torch.zeros(patch * d_model)) for patch in self.patch_sizes
        )

    def _make_bottleneck_mixer(self, d_model):
        """Rank-r channel mixer: full mixing at O(2rd) not O(d^2); GELU keeps a real bottleneck."""
        r = self.mixer_rank
        return nn.Sequential(
            nn.Conv1d(d_model, r, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(r, d_model, kernel_size=1, bias=True),
        )

    def _init_conv(self, d_model):
        """Depthwise conv encoder/decoder; one shared pre-mixer feeds context to every scale."""
        self.pre_mixer = self._make_bottleneck_mixer(d_model)
        self.encoders = nn.ModuleList()
        self.mixers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decode_mixers = nn.ModuleList()
        for patch in self.patch_sizes:
            self.encoders.append(
                nn.Conv1d(d_model, d_model, kernel_size=patch, stride=patch, groups=d_model, bias=True)
            )
            self.mixers.append(self._make_bottleneck_mixer(d_model))
            # depthwise transpose conv: O(patch*d_model), not O(patch*d_model^2) for a dense decoder
            self.decoders.append(
                nn.ConvTranspose1d(d_model, d_model, kernel_size=patch, stride=patch, groups=d_model, bias=True)
            )
            self.decode_mixers.append(self._make_bottleneck_mixer(d_model))

    @staticmethod
    def _pad_to_multiple(x, patch):
        length = x.shape[1]
        remainder = length % patch
        return F.pad(x, (0, 0, 0, patch - remainder)) if remainder else x

    def encode(self, x):
        length = x.shape[1]
        if self.mode == "patch":
            return self._encode_patch(x, length)
        return self._encode_conv(x, length)

    def _encode_patch(self, x, length):
        scales = []
        for patch, lin in zip(self.patch_sizes, self.encode_lin):
            padded = self._pad_to_multiple(x, patch)
            batch, padded_len, d_model = padded.shape
            grouped = padded.reshape(batch, padded_len // patch, patch * d_model)
            scales.append(lin(grouped))
        return CodecOutput(scales=scales, length=length)

    def _encode_conv(self, x, length):
        scales = []
        h_pre = self.pre_mixer(x.transpose(1, 2))  # (B,D,L) cross-channel context, computed once
        for patch, encoder, mixer in zip(self.patch_sizes, self.encoders, self.mixers):
            h = self._pad_to_multiple(h_pre.transpose(1, 2), patch).transpose(1, 2)  # (B,D,L')
            h = encoder(h)                     # (B,D,G)
            h = mixer(h)                       # (B,D,G)
            scales.append(h.transpose(1, 2))   # (B,G,D)
        return CodecOutput(scales=scales, length=length)

    def decode(self, encoded):
        length = encoded.length
        if self.mode == "patch":
            return self._decode_patch(encoded, length)
        return self._decode_conv(encoded, length)

    def _decode_patch(self, encoded, length):
        reconstructed = []
        for idx, (scale, patch) in enumerate(zip(encoded.scales, self.patch_sizes)):
            lin, decode_b = self.encode_lin[idx], self.decode_b[idx]
            # tied weights: same Linear, transposed (F.linear -> addmm).
            expanded = F.linear(scale, lin.weight.t(), decode_b)
            expanded = expanded.reshape(expanded.shape[0], -1, lin.in_features // patch)
            if expanded.shape[1] < length:
                expanded = F.pad(expanded, (0, 0, 0, length - expanded.shape[1]))
            reconstructed.append(expanded[:, :length])
        h = torch.cat(reconstructed, dim=-1)
        h = self.norm(h)
        return self.fusion(h)

    def _decode_conv(self, encoded, length):
        reconstructed = []
        for scale, decoder, mixer in zip(encoded.scales, self.decoders, self.decode_mixers):
            h = scale.transpose(1, 2)   # (B,D,G)
            h = decoder(h)              # (B,D,G*patch) — depthwise, expands the length dim only
            h = mixer(h)                # (B,D,L) — bottleneck mixer, O(2*r*d) not O(d^2)
            reconstructed.append(h.transpose(1, 2)[:, :length])  # (B,L,D)
        h = torch.cat(reconstructed, dim=-1)
        h = self.norm(h)
        return self.fusion(h)


@dataclass
class KairosOutput(CausalLMOutputWithPast):
    encoder_last_hidden_state: torch.FloatTensor = None
    octet_logits: torch.FloatTensor = None


class KairosDiffusionFM(PreTrainedModel, KairosDiffusionGenerationMixin):
    def __init__(self, config, vocab_size=None, num_octet_families=None, use_moe=None):
        super().__init__(config)
        if use_moe is None:
            use_moe = config.use_moe
        self.codec = PyramidalCodec(
            d_model=config.hidden_size,
            stride=config.stride,
            num_scales=config.num_scales,
            mode=config.codec_mode,
            norm_eps=config.rms_norm_eps,
            mixer_rank=getattr(config, "codec_mixer_rank", None),
        )
        self.router = KairosScaleRouter(config.modality_scales)
        if vocab_size is None:
            vocab_size = config.vocab_size
        if num_octet_families is None:
            num_octet_families = config.num_octet_families
        n_octet = num_octet_families if config.predict_octet_family else 0
        self.embedding = KairosEmbedding(
            vocab_size=vocab_size, d_model=config.hidden_size, num_octet_families=n_octet
        )
        share = getattr(config, "share_backbones", False)
        if share:
            shared = KairosDiffusionBackbone(config=config, use_moe=use_moe)
            self.backbones = nn.ModuleList([shared] * self.codec.num_scales)
        else:
            self.backbones = nn.ModuleList(
                [KairosDiffusionBackbone(config=config, use_moe=use_moe) for _ in range(self.codec.num_scales)]
            )
        if getattr(config, "use_memory_gate", False):
            head_dim = config.hidden_size // config.num_attention_heads
            state_dim = config.num_attention_heads * head_dim * 2 * head_dim
            self.memory_gate = KairosMemoryGate(state_dim=state_dim)
        else:
            self.memory_gate = None
        self.rotary = KairosRotaryEmbedding(config, config.head_dim)
        self.norm = KairosNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = OutputHead(self.embedding, num_octet_families=n_octet)
        self.post_init()  # triggers _init_weights on every submodule/parameter

    def _init_weights(self, module):
        """Every PreTrainedModel subclass must define this (the base class default is a."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        modality_ids=None,
        family_ids=None,
        attention_mask=None,
        self_conditioning_logits=None,
        cache_params=None,
        logits_mask=None,
        **kwargs,
    ):
        x = decoder_input_ids if decoder_input_ids is not None else input_ids
        if x is None:
            raise ValueError("either input_ids or decoder_input_ids must be provided")
        if modality_ids is None:
            modality_ids = torch.full_like(x, self.config.text_modality_id)
        h = self.embedding(token_ids=x, family_ids=family_ids)
        if self_conditioning_logits is not None:
            probs = torch.softmax(self_conditioning_logits, dim=-1)
            h = h + (probs @ self.embedding.token_embed.weight).to(h.dtype)
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
        # heads cost scales with vocab_size, so restrict to logits_mask positions when given
        h = h[logits_mask] if logits_mask is not None else h
        token_logits, octet_logits = self.lm_head(h)
        return KairosOutput(logits=token_logits, octet_logits=octet_logits, past_key_values=cache_params)