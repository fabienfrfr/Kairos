from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP

try:
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        DiffusionGemmaGenerationMixin,
    )
except ImportError:

    class DiffusionGemmaGenerationMixin:
        pass


from transformers.cache_utils import DynamicCache

from .attentions import KairosLiZAttention2, KairosNorm, KairosRotaryEmbedding


class KairosConfig(PretrainedConfig):
    """modality_scales defaults every modality id up to num_modalities to scale 0 so none get silently dropped."""

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

        self.num_modalities = kwargs.get("num_modalities", 8)
        self.text_modality_id = kwargs.get("text_modality_id", 0)
        self.num_scales = kwargs.get("num_scales", 4)

        default_scales = {0: [0, 1], 1: [1, 2], 2: [2, 3]}
        for m in range(self.num_modalities):
            default_scales.setdefault(m, [0])
        self.modality_scales = kwargs.get("modality_scales", default_scales)

        assert d_model % n_heads == 0, "hidden_size must be divisible by n_heads"

        self.stride = stride

        self.sliding_window_size = window_size
        self.num_key_value_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = 0.0
        self.rope_theta = 10000.0
        self.max_position_embeddings = 4096

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
        self.initializer_range = kwargs.get("initializer_range", 0.02)

        self.intermediate_size = intermediate_size

        # num_local_experts is the only one transformers' DeepseekV3Experts actually reads;
        # n_routed_experts is a property alias below so setting either one can't silently diverge
        self.num_local_experts = kwargs.get("num_local_experts", kwargs.get("n_routed_experts", 8))
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 2)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", intermediate_size)
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)
        self.routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
        self.n_group = kwargs.get("n_group", 1)
        self.topk_group = kwargs.get("topk_group", 1)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", False)
        self.use_moe = kwargs.get("use_moe", False)
        self.use_state_combiner = kwargs.get("use_state_combiner", False)  # combine_deltanet_caches()

        self.layers_config = kwargs.get("layers_config", ["ld"] * n_layers)
        self.slw_wsize = kwargs.get("slw_wsize", -1)

        # v3 Block-AttnRes: windows prior layer outputs into blocks of S so AttnRes sources stay O(N/S); S=1 (default) reproduces the original per-layer graph
        self.attnres_block_size = kwargs.get("attnres_block_size", 1)

    @property
    def n_routed_experts(self):
        """Alias for num_local_experts (the field transformers' DeepseekV3Experts actually reads) so the two names can never silently diverge."""
        return self.num_local_experts

    @n_routed_experts.setter
    def n_routed_experts(self, value):
        self.num_local_experts = value


class KairosCache(DynamicCache):
    """Cache for block-diffusion inference: `.clone()` before each denoising step to avoid state leaking across steps."""

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


class KairosStateCombiner(nn.Module):
    """Learned, size-unbounded combination of K DeltaNet ssm_cache states via self-attention over the set (so scores reflect how states relate, not each in isolation), pooled by weighting the raw states themselves - K=1 is an exact identity, so a lone resumed session is never perturbed."""

    def __init__(self, state_dim, num_heads=4):
        super().__init__()
        self.context_attn = nn.MultiheadAttention(state_dim, num_heads, batch_first=True)
        self.score = nn.Sequential(nn.Linear(state_dim, state_dim), nn.Tanh(), nn.Linear(state_dim, 1))

    def forward(self, states):
        """states: (K, state_dim), K unbounded (including 1). Returns (state_dim,)."""
        if states.size(0) == 1:
            return states[0]
        context, _ = self.context_attn(states.unsqueeze(0), states.unsqueeze(0), states.unsqueeze(0))
        weights = torch.softmax(self.score(context.squeeze(0)).squeeze(-1), dim=0)
        return (weights.unsqueeze(-1) * states).sum(dim=0)


def combine_deltanet_caches(model, caches: list) -> "KairosMultiCache":
    """Combines multiple per-session caches (same batch layout, typically batch_size=1 each) via a learned combiner per (scale, layer), row-by-row; len(caches)==1 leaves that cache unchanged; conv_caches are reset."""
    if not caches:
        raise ValueError("combine_deltanet_caches needs at least one cache")
    new_cache = KairosMultiCache(model.config)
    for scale_idx, backbone in enumerate(model.backbones):
        for layer_idx_str, combiner in backbone.state_combiners.items():
            layer_idx = int(layer_idx_str)
            states = [c.caches[scale_idx].ssm_caches[layer_idx] for c in caches]
            states = [s for s in states if s is not None]
            if not states:
                continue
            batch_size = states[0].shape[0]
            per_row_shape = states[0].shape[1:]
            combined_rows = [
                combiner(torch.stack([s[row].reshape(-1) for s in states], dim=0)) for row in range(batch_size)
            ]
            new_cache.caches[scale_idx].ssm_caches[layer_idx] = torch.stack(combined_rows, dim=0).view(
                batch_size, *per_row_shape
            )
    return new_cache


def pool_batch_states(model, cache) -> "KairosMultiCache":
    """Pools every row of ONE cache (e.g. a training batch's final states) into a single state per (scale, layer), via the same learned combiner; returns a batch_size=1 cache; batch_size==1 leaves it unchanged."""
    new_cache = KairosMultiCache(model.config)
    for scale_idx, backbone in enumerate(model.backbones):
        for layer_idx_str, combiner in backbone.state_combiners.items():
            layer_idx = int(layer_idx_str)
            states = cache.caches[scale_idx].ssm_caches[layer_idx]
            if states is None:
                continue
            per_row_shape = states.shape[1:]
            combined = combiner(states.reshape(states.shape[0], -1))
            new_cache.caches[scale_idx].ssm_caches[layer_idx] = combined.view(1, *per_row_shape)
    return new_cache


class KairosFFN(Qwen2MoeMLP):
    pass


class KairosMoE(DeepseekV3MoE):
    """DeepseekV3MoE's expert weights are raw torch.empty() tensors, never initialized by default; fixed here on self.experts/self.gate (version-stable) instead of via fragile isinstance checks on internal class names."""

    def __init__(self, config):
        super().__init__(config)
        std = getattr(config, "initializer_range", 0.02)
        self.experts.gate_up_proj.data.normal_(mean=0.0, std=std)
        self.experts.down_proj.data.normal_(mean=0.0, std=std)
        self.gate.weight.data.normal_(mean=0.0, std=std)  # was torch.zeros() at construction; fine either way


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
    """v3 Block-AttnRes: prior layer outputs are windowed into blocks of `attnres_block_size` before aggregation, cost O(N/S)."""

    def __init__(self, config, use_moe=False):
        super().__init__()
        self.layers = nn.ModuleList([DiffusionBlock(config, i, use_moe) for i in range(config.num_hidden_layers)])
        self.norm = KairosNorm(config.hidden_size)
        self.aggregator = KairosAttnRes(config.hidden_size)
        self.attnres_block_size = max(1, getattr(config, "attnres_block_size", 1))
        head_dim = config.hidden_size // config.num_attention_heads  # ssm_cache shape: (B, n_heads, head_dim, 2*head_dim)
        n_heads = config.num_attention_heads
        self.state_combiners = nn.ModuleDict(
            {
                str(i): KairosStateCombiner(state_dim=n_heads * head_dim * 2 * head_dim)
                for i, layer_type in enumerate(config.layers_config)
                if "d" in layer_type
            }
            if getattr(config, "use_state_combiner", False)
            else {}
        )

    def forward(self, x, position_embeddings=None, cache_params=None, attention_mask=None, position_ids=None):
        emb = x
        completed = []  # finalized block-sums of prior layer outputs
        partial = None  # running sum of the current (unfinished) block
        in_block = 0
        S = self.attnres_block_size

        def sources():
            return [emb] + completed + ([partial] if partial is not None else [])

        for layer in self.layers:
            h = self.aggregator(sources())
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
    def __init__(self, vocab_size: int, num_modalities: int, d_model: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.modality_embed = nn.Embedding(num_modalities, d_model)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.scale = d_model**0.5

    def forward(self, token_ids, modality_ids):
        tok = self.token_embed(token_ids)
        mod = self.modality_embed(modality_ids)
        h = self.fusion_proj(torch.cat([tok, mod], dim=-1))
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


class KairosScaleRouter(nn.Module):
    """Gathers active positions per scale into a padded batch, runs the backbone once per scale, and scatters back."""

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
        values = torch.where(pad_mask.unsqueeze(-1), chunk, current)
        return output.scatter(1, idx, values)


@dataclass
class CodecOutput:
    scales: list
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
                nn.Conv1d(
                    d_model, d_model, kernel_size=kernel_size, stride=scale_stride, padding=padding, groups=d_model
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
        padded = []
        for r in reconstructed:
            if r.shape[1] < length:
                pad_amount = length - r.shape[1]
                r = F.pad(r.transpose(1, 2), (0, pad_amount), mode="replicate").transpose(1, 2)
            padded.append(r[:, :length])
        h = torch.cat(padded, dim=-1)
        h = self.norm(h)
        return self.fusion(h)


@dataclass
class KairosOutput(CausalLMOutputWithPast):
    encoder_last_hidden_state: torch.FloatTensor = None
    modality_logits: torch.FloatTensor = None


class KairosDiffusionLLM(PreTrainedModel, DiffusionGemmaGenerationMixin):
    def __init__(self, config, vocab_size=None, use_moe=None):
        super().__init__(config)
        if use_moe is None:
            use_moe = config.use_moe
        self.codec = PyramidalConvCodec(d_model=config.hidden_size, stride=config.stride, num_scales=config.num_scales)
        self.router = KairosScaleRouter(config.modality_scales)
        if vocab_size is None:
            vocab_size = config.vocab_size
        self.embedding = KairosEmbedding(
            vocab_size=vocab_size, num_modalities=config.num_modalities, d_model=config.hidden_size
        )
        self.backbones = nn.ModuleList(
            [KairosDiffusionBackbone(config=config, use_moe=use_moe) for _ in range(self.codec.num_scales)]
        )
        self.rotary = KairosRotaryEmbedding(config, config.head_dim)
        self.norm = KairosNorm(config.hidden_size)
        self.lm_head = OutputHead(self.embedding)
        self.post_init()  # triggers _init_weights on every submodule/parameter below

    def _init_weights(self, module):
        """Every PreTrainedModel subclass must define this (the base class default is a no-op). MoE expert/router weights are handled separately in KairosMoE.__init__ itself, not here - see that class for why."""
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
        attention_mask=None,
        self_conditioning_logits=None,
        cache_params=None,
        **kwargs,
    ):
        x = decoder_input_ids if decoder_input_ids is not None else input_ids
        if x is None:
            raise ValueError("either input_ids or decoder_input_ids must be provided")
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
        return KairosOutput(logits=token_logits, modality_logits=modality_logits, past_key_values=cache_params)
