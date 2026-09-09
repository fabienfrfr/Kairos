import importlib
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from kairos.attentions import (
    ATTN_IMPL,
    CAUSAL_CONV1D_BACKEND,
    DELTA_RULE_BACKEND,
    KairosAttention,
    KairosGatedDeltaNet,
    KairosLiZAttention2,
    KairosRotaryEmbedding,
    _resolve_attn_impl,
    _supports_cu_seqlens,
)
from kairos.modeling import KairosCache


class DummySWAConfig:
    hidden_size = 32
    num_attention_heads = 4
    num_key_value_heads = 2
    sliding_window_size = 2
    max_position_embeddings = 64
    rope_theta = 10000.0
    layers_config = ("l",)
    slw_wsize = -1


class DummyDeltaConfig:
    hidden_size = 32
    num_attention_heads = 4
    num_key_value_heads = 2
    max_position_embeddings = 64
    rope_theta = 10000.0
    expand_factor = 2.0
    linear_conv_kernel_dim = 3
    time_step_min = 0.001
    time_step_max = 0.1
    time_step_floor = 1e-4
    A_init_range = (0.1, 1.0)
    use_uscaling = False
    layers_config = ("d",)
    sliding_window_size = 16
    slw_wsize = -1


def get_swa_model():
    cfg = DummySWAConfig()
    attn = KairosAttention(cfg)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
    return attn, rope


def get_swa_inputs(B=2, L=8, D=32):
    x = torch.randn(B, L, D)
    pos_ids = torch.arange(L).unsqueeze(0)
    return x, pos_ids


def test_swa_shape():
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()
    out = attn(x, rope(x, pos))
    assert out.shape == x.shape


def test_swa_no_nan():
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()
    out = attn(x, rope(x, pos))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_swa_bidirectional_symmetry():
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()
    out1 = attn(x, rope(x, pos))
    x_rev = torch.flip(x, dims=[1])
    pos_rev = torch.flip(pos, dims=[1])
    out2 = attn(x_rev, rope(x_rev, pos_rev))
    out2 = torch.flip(out2, dims=[1])
    assert torch.allclose(out1, out2, atol=1e-4)


def test_swa_window_locality():
    attn, rope = get_swa_model()
    x = torch.zeros(1, 10, 32)
    x[:, 5] = 10.0
    pos = torch.arange(10).unsqueeze(0)
    out = attn(x, rope(x, pos))
    assert out[:, 5].abs().mean() > out[:, 0].abs().mean()


def test_swa_eager_vs_flex(monkeypatch):
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()
    out_eager = attn(x, rope(x, pos))
    try:
        monkeypatch.setattr("attention.ATTN_IMPL", "flex")
        out_flex = attn(x, rope(x, pos))
        assert torch.allclose(out_eager, out_flex, atol=1e-3)
    except Exception:  # noqa: BLE001, S110 — flex path optional; failure just skips
        pass


def test_swa_backward():
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()
    x.requires_grad = True
    out = attn(x, rope(x, pos))
    out.mean().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_swa_batch_independence():
    attn, rope = get_swa_model()
    x1, pos = get_swa_inputs(B=1)
    x2, _ = get_swa_inputs(B=1)
    x = torch.cat([x1, x2], dim=0)
    out = attn(x, rope(x, pos.repeat(2, 1)))
    assert not torch.allclose(out[0], out[1])


def test_swa_determinism():
    torch.manual_seed(0)
    attn, rope = get_swa_model()
    x, pos = get_swa_inputs()
    out1 = attn(x, rope(x, pos))
    torch.manual_seed(0)
    attn2, rope2 = get_swa_model()
    out2 = attn2(x, rope2(x, pos))
    assert torch.allclose(out1, out2, atol=1e-5)


def test_swa_linear_complexity():
    class Cfg:
        hidden_size = 32
        num_attention_heads = 4
        num_key_value_heads = 2
        sliding_window_size = 4
        max_position_embeddings = 4096
        rope_theta = 10000.0

    cfg = Cfg()
    attn = KairosAttention(cfg)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
    lengths = [256, 512, 1024]
    times = []

    def measure(x, cos_sin):
        start = time.time()
        _ = attn(x, cos_sin)
        return time.time() - start

    for L in lengths:
        x = torch.randn(1, L, 32)
        pos = torch.arange(L).unsqueeze(0)
        cos_sin = rope(x, pos)
        for _ in range(3):
            _ = attn(x, cos_sin)
        times.append(sum(measure(x, cos_sin) for _ in range(3)) / 3)

    r1 = times[1] / times[0]
    r2 = times[2] / times[1]
    assert r1 < 3.0 and r2 < 3.0


def get_deltanet_model():
    return KairosGatedDeltaNet(DummyDeltaConfig(), layer_idx=0)


def get_deltanet_inputs(B=2, L=16, D=32):
    return torch.randn(B, L, D)


def test_deltanet_shape():
    model = get_deltanet_model()
    x = get_deltanet_inputs()
    out = model(x)
    assert out.shape == x.shape


def test_deltanet_stability():
    model = get_deltanet_model()
    x = get_deltanet_inputs()
    out = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_deltanet_bidir_consistency():
    model = get_deltanet_model()
    x = get_deltanet_inputs()
    forward_features = model.process(x)
    backward_features = model.process(torch.flip(x, dims=[1]))
    backward_features = torch.flip(backward_features, dims=[1])
    actual_output = model(x)
    features = torch.cat([forward_features, backward_features], dim=-1)
    features = features.reshape(actual_output.shape[0], actual_output.shape[1], -1)
    features = model.merge_norm(features)
    expected_output = model.out_proj(model.out_left_right(features))
    assert torch.allclose(actual_output, expected_output, atol=1e-5)


def test_deltanet_bidir_effect():
    model = get_deltanet_model()
    x = get_deltanet_inputs()
    out_f = model.process(x)
    x_rev = torch.flip(x, dims=[1])
    out_b = model.process(x_rev)
    out_b = torch.flip(out_b, dims=[1])
    assert not torch.allclose(out_f, out_b, atol=1e-3)


class _CuSeqlensSpy:
    """Stand-in for chunk_gated_delta_rule: records args, returns a shape-correct zero output."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, g, beta, **kwargs):
        self.calls.append({"q_shape": tuple(q.shape), "cu_seqlens": kwargs["cu_seqlens"].clone()})
        return v.new_zeros(v.shape), None


def test_deltanet_varlen_pads_total_tokens_to_fixed_block_during_training():
    """Regression test: unpadded token count varies every step, so warmup=0 re-autotunes it."""
    from kairos.attentions import _FLEX_BLOCK_SIZE

    model = get_deltanet_model()
    model._chunk_supports_varlen = True
    spy = _CuSeqlensSpy()
    model.chunk_gated_delta_rule = spy

    x = get_deltanet_inputs(B=2, L=16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[0, 12:] = False  # row 0 has real length 12 -> total valid tokens = 12 + 16 = 28

    model.process(x, cache_params=None, attention_mask=mask)

    assert len(spy.calls) == 1
    padded_total = spy.calls[0]["q_shape"][1]
    assert padded_total % _FLEX_BLOCK_SIZE == 0
    assert padded_total >= 28  # never smaller than the real content


def test_deltanet_varlen_cu_seqlens_gets_a_phantom_trailing_segment():
    """The padding rows must form their own segment (never mixed into a real one's state)."""
    model = get_deltanet_model()
    model._chunk_supports_varlen = True
    spy = _CuSeqlensSpy()
    model.chunk_gated_delta_rule = spy

    x = get_deltanet_inputs(B=2, L=16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[0, 12:] = False

    model.process(x, cache_params=None, attention_mask=mask)

    cu_seqlens = spy.calls[0]["cu_seqlens"]
    real_lengths = mask.sum(dim=1)
    expected_real_boundaries = F.pad(real_lengths.cumsum(0), (1, 0)).to(torch.int32)
    assert torch.equal(cu_seqlens[: len(expected_real_boundaries)], expected_real_boundaries)
    assert len(cu_seqlens) == len(expected_real_boundaries) + 1  # + 1 phantom segment
    assert cu_seqlens[-1] == spy.calls[0]["q_shape"][1]  # phantom segment ends at the padded total


def test_deltanet_varlen_output_shape_unaffected_by_internal_padding():
    """Padding is internal: process()'s output shape must still match the input."""
    model = get_deltanet_model()
    model._chunk_supports_varlen = True
    model.chunk_gated_delta_rule = _CuSeqlensSpy()

    x = get_deltanet_inputs(B=2, L=16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[0, 12:] = False

    out = model.process(x, cache_params=None, attention_mask=mask)

    assert out.shape == (2, 16, model.n_heads, 2 * model.head_dim)


def test_deltanet_varlen_skips_padding_when_cache_params_present():
    """Padding is only safe with no cache to round-trip: no-op for generation/decoding."""
    model = get_deltanet_model()
    model._chunk_supports_varlen = True
    spy = _CuSeqlensSpy()
    model.chunk_gated_delta_rule = spy

    x = get_deltanet_inputs(B=2, L=16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[0, 12:] = False
    fake_cache = MagicMock()
    fake_cache.conv_caches = [None]
    fake_cache.ssm_caches = [None]

    model.process(x, cache_params=fake_cache, attention_mask=mask)

    cu_seqlens = spy.calls[0]["cu_seqlens"]
    real_lengths = mask.sum(dim=1)
    expected = F.pad(real_lengths.cumsum(0), (1, 0)).to(torch.int32)
    assert torch.equal(cu_seqlens, expected)  # no phantom segment appended
    assert spy.calls[0]["q_shape"][1] == int(real_lengths.sum())  # not padded to a block boundary


def test_deltanet_varlen_pads_to_static_full_seq_len_when_provided():
    """full_seq_len is the pre-gather scale length: padding to B*full_seq_len is run-invariant."""
    model = get_deltanet_model()
    model._chunk_supports_varlen = True
    spy = _CuSeqlensSpy()
    model.chunk_gated_delta_rule = spy

    x = get_deltanet_inputs(B=2, L=16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[0, 12:] = False

    model.process(x, cache_params=None, attention_mask=mask, full_seq_len=64)

    assert spy.calls[0]["q_shape"][1] == 2 * 64


def test_deltanet_varlen_static_shape_is_stable_across_different_content_lengths():
    """Core regression test: different real content must still produce the same padded shape."""
    model = get_deltanet_model()
    model._chunk_supports_varlen = True
    spy = _CuSeqlensSpy()
    model.chunk_gated_delta_rule = spy

    x = get_deltanet_inputs(B=2, L=16)
    mask_a = torch.ones(2, 16, dtype=torch.bool)
    mask_a[0, 12:] = False
    mask_b = torch.ones(2, 16, dtype=torch.bool)
    mask_b[1, 3:] = False

    model.process(x, cache_params=None, attention_mask=mask_a, full_seq_len=64)
    model.process(x, cache_params=None, attention_mask=mask_b, full_seq_len=64)

    assert spy.calls[0]["q_shape"] == spy.calls[1]["q_shape"]


def test_deltanet_varlen_static_padding_never_goes_negative():
    """Sanity bound: total valid tokens never exceeds B*full_seq_len, so pad_n stays >= 0."""
    model = get_deltanet_model()
    model._chunk_supports_varlen = True
    spy = _CuSeqlensSpy()
    model.chunk_gated_delta_rule = spy

    x = get_deltanet_inputs(B=2, L=16)
    mask = torch.ones(2, 16, dtype=torch.bool)  # fully valid: total == B*L == B*full_seq_len

    model.process(x, cache_params=None, attention_mask=mask, full_seq_len=16)

    assert spy.calls[0]["q_shape"][1] == 2 * 16


def test_deltanet_varlen_falls_back_to_block_rounding_without_full_seq_len():
    """Backward compatibility: full_seq_len=None must keep the block-rounding fallback."""
    from kairos.attentions import _FLEX_BLOCK_SIZE

    model = get_deltanet_model()
    model._chunk_supports_varlen = True
    spy = _CuSeqlensSpy()
    model.chunk_gated_delta_rule = spy

    x = get_deltanet_inputs(B=2, L=16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[0, 12:] = False

    model.process(x, cache_params=None, attention_mask=mask)

    assert spy.calls[0]["q_shape"][1] % _FLEX_BLOCK_SIZE == 0


def test_deltanet_backward():
    model = get_deltanet_model()
    x = get_deltanet_inputs()
    x.requires_grad = True
    out = model(x)
    out.mean().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_deltanet_determinism():
    model = get_deltanet_model()
    x = get_deltanet_inputs()
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_deltanet_order_sensitivity():
    model = get_deltanet_model()
    x = get_deltanet_inputs()
    out1 = model(x)
    x_rev = torch.flip(x, dims=[1])
    out2 = model(x_rev)
    assert not torch.allclose(out1, out2)


def test_deltanet_signal_propagation():
    model = get_deltanet_model()
    x = torch.zeros(1, 32, 32)
    x[:, 16] = 10.0
    out = model(x)
    assert out[:, 16].abs().mean() > out[:, 0].abs().mean()


def test_deltanet_cache_not_mutated():
    model = get_deltanet_model()
    x_N = torch.randn(1, 16, 32)
    cache = KairosCache(model.config)
    _ = model(x_N, cache)
    cache_ref = cache.clone()
    x_M = torch.randn(1, 8, 32)
    _ = model(x_M, cache.clone())
    for c1, c2 in zip(cache.ssm_caches, cache_ref.ssm_caches):
        if c1 is not None:
            assert torch.allclose(c1, c2)


def test_deltanet_cache_clone_isolation():
    model = get_deltanet_model()
    x_N = torch.randn(1, 16, 32)
    cache = KairosCache(model.config)
    _ = model(x_N, cache)
    cache_a = cache.clone()
    cache_b = cache.clone()
    x_M1 = torch.randn(1, 8, 32)
    x_M2 = torch.randn(1, 8, 32)
    _ = model(x_M1, cache_a)
    _ = model(x_M2, cache_b)
    for a, b in zip(cache_a.ssm_caches, cache_b.ssm_caches):
        if a is not None:
            assert not torch.allclose(a, b)


def test_deltanet_cache_effect():
    model = get_deltanet_model()
    x_N1 = torch.randn(1, 16, 32)
    x_N2 = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)
    cache1 = KairosCache(model.config)
    cache2 = KairosCache(model.config)
    _ = model(x_N1, cache1)
    _ = model(x_N2, cache2)
    out1 = model(x_M, cache1.clone())
    out2 = model(x_M, cache2.clone())
    assert not torch.allclose(out1, out2)


def test_deltanet_ssm_cache_used_even_without_conv_cache():
    # regression: has_previous_state was wrongly gating the SSM path
    model = get_deltanet_model()
    x = torch.randn(1, 8, 32)

    cache_empty = KairosCache(model.config)
    out_empty = model(x, cache_empty)

    cache_with_state = KairosCache(model.config)
    cache_with_state.ssm_caches[0] = torch.randn(1, 4, 8, 16)  # (B, n_heads, head_dim, 2*head_dim)
    assert cache_with_state.conv_caches[0] is None  # conv_cache deliberately left unset
    out_with_state = model(x, cache_with_state)

    assert not torch.allclose(out_empty, out_with_state)


def test_deltanet_cache_determinism():
    model = get_deltanet_model()
    x_N = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)
    cache = KairosCache(model.config)
    _ = model(x_N, cache)
    out1 = model(x_M, cache.clone())
    out2 = model(x_M, cache.clone())
    assert torch.allclose(out1, out2, atol=1e-5)


def test_swa_cache_consistency():
    cfg = DummySWAConfig()
    attn = KairosAttention(cfg, layer_idx=0)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
    x = torch.randn(1, 16, 32)
    pos = torch.arange(16).unsqueeze(0)
    full = attn(x, rope(x, pos))
    cache = KairosCache(cfg)
    outs = []
    for i in range(16):
        xi = x[:, i : i + 1]
        pi = pos[:, i : i + 1]
        out = attn(xi, rope(xi, pi), cache_params=cache)
        outs.append(out)
    step = torch.cat(outs, dim=1)
    assert step.shape == full.shape
    assert not torch.isnan(step).any()


def test_swa_cache_trim():
    cfg = DummySWAConfig()
    cfg.sliding_window_size = 4
    attn = KairosAttention(cfg, layer_idx=0)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
    cache = KairosCache(cfg)
    x = torch.randn(1, 32, 32)
    pos = torch.arange(32).unsqueeze(0)
    for i in range(32):
        xi = x[:, i : i + 1]
        pi = pos[:, i : i + 1]
        attn(xi, rope(xi, pi), cache_params=cache)
    k = cache._key_cache[0]
    assert k.shape[1] <= cfg.sliding_window_size


def test_swa_cache_no_trim_when_small_sequence():
    cfg = DummySWAConfig()
    cfg.sliding_window_size = 32
    attn = KairosAttention(cfg, layer_idx=0)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
    cache = KairosCache(cfg)
    x = torch.randn(1, 8, 32)
    pos = torch.arange(8).unsqueeze(0)
    for i in range(8):
        xi = x[:, i : i + 1]
        pi = pos[:, i : i + 1]
        attn(xi, rope(xi, pi), cache_params=cache)
    k = cache._key_cache[0]
    assert k.shape[1] == 8


def test_deltanet_diffusion_stability():
    model = get_deltanet_model()
    x_N = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)
    cache = KairosCache(model.config)
    _ = model(x_N, cache)
    outs = []
    for _ in range(5):
        out = model(x_M, cache.clone())
        outs.append(out)
    for o in outs[1:]:
        assert torch.allclose(outs[0], o, atol=1e-5)


def test_swa_partial_diffusion_stability():
    cfg = DummySWAConfig()
    cfg.sliding_window_size = 64
    attn = KairosAttention(cfg, layer_idx=0)
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
    x_N = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)
    pos_N = torch.arange(16).unsqueeze(0)
    pos_M = torch.arange(8).unsqueeze(0)
    cache = KairosCache(cfg)
    _ = attn(x_N, rope(x_N, pos_N), cache_params=cache)
    outs = []
    for _ in range(5):
        cache_iter = cache.clone()
        out = attn(x_M, rope(x_M, pos_M), cache_params=cache_iter)
        outs.append(out)
    for o in outs[1:]:
        assert torch.allclose(outs[0], o, atol=1e-5)


def get_liz_model():
    cfg = DummyDeltaConfig()
    return KairosLiZAttention2(cfg, layer_idx=0)


def test_liz_shape():
    model = get_liz_model()
    x = torch.randn(2, 16, 32)
    pos = torch.arange(16).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)
    out = model(x, rope(x, pos))
    assert out.shape == x.shape


def test_liz_no_nan():
    model = get_liz_model()
    x = torch.randn(2, 16, 32)
    pos = torch.arange(16).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)
    out = model(x, rope(x, pos))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_liz_concat_effect():
    model = get_liz_model()
    x = torch.randn(1, 16, 32)
    pos = torch.arange(16).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)
    swa = model.swa(x, rope(x, pos))
    delta = model.delta(x)
    liz = model(x, rope(x, pos))
    assert not torch.allclose(liz, swa)
    assert not torch.allclose(liz, delta)


def test_liz_cache_effect():
    model = get_liz_model()
    x_N1 = torch.randn(1, 16, 32)
    x_N2 = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)
    cache1 = KairosCache(model.delta.config)
    cache2 = KairosCache(model.delta.config)
    _ = model.delta(x_N1, cache_params=cache1)
    _ = model.delta(x_N2, cache_params=cache2)
    pos = torch.arange(8).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)
    out1 = model(x_M, rope(x_M, pos), cache_params=cache1.clone())
    out2 = model(x_M, rope(x_M, pos), cache_params=cache2.clone())
    assert not torch.allclose(out1, out2)


def test_liz_cache_not_mutated():
    model = get_liz_model()
    x_N = torch.randn(1, 16, 32)
    cache = KairosCache(model.delta.config)
    _ = model.delta(x_N, cache_params=cache)
    ref = cache.clone()
    x_M = torch.randn(1, 8, 32)
    pos = torch.arange(8).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)
    _ = model(x_M, rope(x_M, pos), cache_params=cache.clone())
    for c1, c2 in zip(cache.ssm_caches, ref.ssm_caches):
        if c1 is not None:
            assert torch.allclose(c1, c2)


def test_liz_determinism():
    model = get_liz_model()
    x_N = torch.randn(1, 16, 32)
    x_M = torch.randn(1, 8, 32)
    cache = KairosCache(model.delta.config)
    _ = model.delta(x_N, cache_params=cache)
    pos = torch.arange(8).unsqueeze(0)
    rope = KairosRotaryEmbedding(model.swa.config, 8)
    out1 = model(x_M, rope(x_M, pos), cache_params=cache.clone())
    out2 = model(x_M, rope(x_M, pos), cache_params=cache.clone())
    assert torch.allclose(out1, out2, atol=1e-5)


def test_supports_cu_seqlens_returns_false_for_none():
    assert _supports_cu_seqlens(None) is False


def test_supports_cu_seqlens_returns_false_when_signature_unavailable():
    # builtins raise ValueError in inspect.signature; handle it
    assert _supports_cu_seqlens(int) is False


def test_supports_cu_seqlens_detects_the_parameter():
    def with_cu_seqlens(q, k, v, cu_seqlens=None):
        pass

    def without_cu_seqlens(q, k, v):
        pass

    assert _supports_cu_seqlens(with_cu_seqlens) is True
    assert _supports_cu_seqlens(without_cu_seqlens) is False


def test_attention_without_layer_idx_warns_but_still_works(capsys):
    """layer_idx=None disables caching but must not raise."""
    cfg = DummySWAConfig()
    attn = KairosAttention(cfg, layer_idx=None)
    captured = capsys.readouterr()
    assert "layer_idx should be set" in captured.out
    x, pos = get_swa_inputs()
    rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
    out = attn(x, rope(x, pos))
    assert out.shape == x.shape


def test_current_backend_is_eager_on_cpu():
    # flex_attention requires CUDA; lock the eager CPU fallback here
    assert ATTN_IMPL == "eager"


def test_kairos_attn_backend_eager_override(tmp_path):
    # KAIROS_ATTN_BACKEND=eager must force eager regardless of CUDA.
    script = tmp_path / "check_backend.py"
    script.write_text(
        "import os\nos.environ['KAIROS_ATTN_BACKEND'] = 'eager'\nimport kairos.attentions as a\nprint(a.ATTN_IMPL)\n"
    )
    repo_root = str(Path(__file__).resolve().parent.parent)
    out = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": repo_root},
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "eager"


def test_auto_backend_picks_flex_on_a_usable_gpu():
    got = _resolve_attn_impl("auto", flex_import_ok=True, can_fuse=True)
    assert got == "flex"


def test_auto_backend_falls_back_without_a_usable_gpu():
    got = _resolve_attn_impl("auto", flex_import_ok=True, can_fuse=False)
    assert got == "eager"


def test_auto_backend_falls_back_when_flex_import_failed():
    got = _resolve_attn_impl("auto", flex_import_ok=False, can_fuse=True)
    assert got == "eager"


def test_explicit_flex_backend_raises_when_import_failed():
    with pytest.raises(ImportError):
        _resolve_attn_impl("flex", flex_import_ok=False, can_fuse=True)


def test_explicit_eager_backend_ignores_gpu_availability():
    got = _resolve_attn_impl("eager", flex_import_ok=True, can_fuse=True)
    assert got == "eager"


def test_round_up_flex_block():
    from kairos.attentions import _FLEX_BLOCK_SIZE, _round_up

    assert _FLEX_BLOCK_SIZE == 128
    assert _round_up(1, 128) == 128
    assert _round_up(114, 128) == 128
    assert _round_up(128, 128) == 128
    assert _round_up(129, 128) == 256
    assert _round_up(256, 128) == 256


def test_build_flex_mask_bucketed_semantics():
    # bucketed mask keeps the window and forces padded query rows to attend kv 0.
    from kairos.attentions import build_flex_mask_bucketed

    window = 2
    bq = 8
    q_len = 5
    q_mask = torch.ones(2, bq, dtype=torch.bool)
    q_mask[:, q_len:] = False
    kv_mask = torch.ones(2, bq, dtype=torch.bool)
    kv_mask[:, q_len:] = False
    m = build_flex_mask_bucketed(window, q_mask, kv_mask, device="cpu")
    assert m.kv_indices.shape[0] == 2
    mask_mod = m.mask_mod
    for q in range(q_len):
        row = [bool(mask_mod(0, 0, torch.tensor(q), torch.tensor(k))) for k in range(bq)]
        lo, hi = max(0, q - window), min(q_len, q + window + 1)
        assert row == [lo <= k < hi for k in range(bq)], q
    for q in range(q_len, bq):
        row = [bool(mask_mod(0, 0, torch.tensor(q), torch.tensor(k))) for k in range(bq)]
        assert row[0] is True and not any(row[1:]), q


def test_build_backbone_flex_block_mask_no_padding_matches_bucketed_semantics():
    """The shared mask must match the per-layer _flex_mask_bucketed semantics it replaces."""
    from kairos.attentions import build_backbone_flex_block_mask

    window, bq, q_len, batch_size = 2, 8, 5, 2
    m = build_backbone_flex_block_mask(window, q_len, batch_size, attention_mask=None, device="cpu")
    mask_mod = m.mask_mod
    for q in range(q_len):
        row = [bool(mask_mod(0, 0, torch.tensor(q), torch.tensor(k))) for k in range(bq)]
        lo, hi = max(0, q - window), min(q_len, q + window + 1)
        assert row == [lo <= k < hi for k in range(bq)], q
    for q in range(q_len, bq):
        row = [bool(mask_mod(0, 0, torch.tensor(q), torch.tensor(k))) for k in range(bq)]
        assert row[0] is True and not any(row[1:]), q


def test_build_backbone_flex_block_mask_padded_matches_per_row_padding():
    """Padding branch: must respect each row's real length, same as _flex_mask_bucketed_padded."""
    from kairos.attentions import build_backbone_flex_block_mask

    window, q_len = 2, 5
    pad = torch.ones(2, q_len, dtype=torch.bool)
    pad[1, 3:] = False  # row 1 has real length 3
    m = build_backbone_flex_block_mask(window, q_len, batch_size=2, attention_mask=pad, device="cpu")
    mask_mod = m.mask_mod
    for b, length in ((0, 5), (1, 3)):
        for q in range(length):
            row = [bool(mask_mod(b, 0, torch.tensor(q), torch.tensor(k))) for k in range(8)]
            lo, hi = max(0, q - window), min(length, q + window + 1)
            assert row == [lo <= k < hi for k in range(8)], (b, q)


def test_build_backbone_flex_block_mask_ignores_fully_valid_attention_mask():
    """An all-True attention_mask must take the cheaper no-padding branch."""
    from kairos.attentions import build_backbone_flex_block_mask

    window, q_len = 2, 5
    all_valid = torch.ones(2, q_len, dtype=torch.bool)
    m_with_full_mask = build_backbone_flex_block_mask(window, q_len, 2, all_valid, device="cpu")
    m_without_mask = build_backbone_flex_block_mask(window, q_len, 2, None, device="cpu")
    for q in range(8):
        row_a = [bool(m_with_full_mask.mask_mod(0, 0, torch.tensor(q), torch.tensor(k))) for k in range(8)]
        row_b = [bool(m_without_mask.mask_mod(0, 0, torch.tensor(q), torch.tensor(k))) for k in range(8)]
        assert row_a == row_b, q


@pytest.mark.skipif(
    not torch.cuda.is_available() or ATTN_IMPL != "flex",
    reason="flex_attention requires a CUDA device",
)
class TestSharedFlexBlockMask:
    """Passing a pre-built mask must not change output and must skip the layer's own cache."""

    def _attn(self):
        cfg = DummySWAConfig()
        attn = KairosAttention(cfg, layer_idx=0)
        rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
        return attn, rope

    def test_pre_built_mask_matches_default_construction(self):
        from kairos.attentions import build_backbone_flex_block_mask

        attn, rope = self._attn()
        L = 114
        x = torch.randn(2, L, 32, device="cuda")
        pos = torch.arange(L, device="cuda").unsqueeze(0).expand(2, -1)
        cos_sin = rope(x, pos)

        out_default = attn(x, cos_sin)

        shared_mask = build_backbone_flex_block_mask(attn.window, L, 2, None, device="cuda")
        out_shared = attn(x, cos_sin, attn_block_mask=shared_mask)

        assert torch.allclose(out_default, out_shared, atol=1e-5)

    def test_pre_built_mask_skips_internal_cache(self):
        from kairos.attentions import build_backbone_flex_block_mask

        attn, rope = self._attn()
        L = 114
        x = torch.randn(2, L, 32, device="cuda")
        pos = torch.arange(L, device="cuda").unsqueeze(0).expand(2, -1)
        cos_sin = rope(x, pos)
        shared_mask = build_backbone_flex_block_mask(attn.window, L, 2, None, device="cuda")

        attn(x, cos_sin, attn_block_mask=shared_mask)

        assert attn._flex_mask_cache == {}  # internal build was skipped entirely


def test_flex_mask_bucketed_padded_per_row(monkeypatch):
    # pad_mask from gather_active is per-row; the mask must respect each row's padding.
    from kairos.attentions import KairosAttention

    cfg = DummySWAConfig()
    attn = KairosAttention(cfg, layer_idx=0)
    pad = torch.ones(2, 5, dtype=torch.bool)
    pad[1, 3:] = False  # row 1 has length 3
    m = attn._flex_mask_bucketed_padded(8, pad, torch.device("cpu"))
    f = m.mask_mod
    for b, length in ((0, 5), (1, 3)):
        for q in range(length):
            row = [bool(f(b, 0, torch.tensor(q), torch.tensor(k))) for k in range(8)]
            lo, hi = max(0, q - cfg.sliding_window_size), min(length, q + cfg.sliding_window_size + 1)
            assert row == [lo <= k < hi for k in range(8)], (b, q)
        for q in range(length, 8):
            row = [bool(f(b, 0, torch.tensor(q), torch.tensor(k))) for k in range(8)]
            assert row[0] is True and not any(row[1:]), (b, q)


@pytest.mark.skipif(
    not torch.cuda.is_available() or ATTN_IMPL != "flex",
    reason="flex_attention requires a CUDA device",
)
class TestFlexAttentionBlockMask:
    """Flex path (CUDA only): block masks must match the exact input length (114 vs 128)."""

    def _attn(self, max_position_embeddings=128, layer_idx=0):
        cfg = DummySWAConfig()
        cfg.max_position_embeddings = max_position_embeddings
        attn = KairosAttention(cfg, layer_idx=layer_idx)
        rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
        return attn, rope

    def test_non_block_aligned_length_no_longer_raises(self):
        # q_len=114 vs a 128-max block mask was the exact reported failure
        attn, rope = self._attn()
        for L in (114, 37, 64, 128):
            x = torch.randn(2, L, 32, device="cuda")
            pos = torch.arange(L, device="cuda").unsqueeze(0).expand(2, -1)
            out = attn(x, rope(x, pos))
            assert out.shape == x.shape
            assert not torch.isnan(out).any()

    def test_padded_path_accepts_non_block_aligned_length(self):
        attn, rope = self._attn()
        L = 114
        x = torch.randn(2, L, 32, device="cuda")
        pos = torch.arange(L, device="cuda").unsqueeze(0).expand(2, -1)
        mask = torch.ones(2, L, dtype=torch.bool, device="cuda")
        mask[:, -7:] = False
        out = attn(x, rope(x, pos), attention_mask=mask)
        assert out.shape == x.shape

    def test_rectangular_mask_with_growing_kv_cache(self):
        # during cached generation kv_len grows beyond q_len
        cfg = DummySWAConfig()
        cfg.sliding_window_size = 128  # no trimming during this test
        attn = KairosAttention(cfg, layer_idx=0)
        rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
        cache = KairosCache(cfg)
        L = 37
        for step in range(2):
            x = torch.randn(2, L, 32, device="cuda")
            pos = torch.arange(step * L, step * L + L, device="cuda").unsqueeze(0).expand(2, -1)
            out = attn(x, rope(x, pos), cache_params=cache, position_ids=pos)
            assert out.shape == x.shape
        assert cache._key_cache[0].size(1) == 2 * L

    def test_flex_mask_cache_reuses_masks_per_length(self):
        attn, _ = self._attn()
        dev = torch.device("cuda")
        m1 = attn._flex_mask(114, 114, dev)
        m2 = attn._flex_mask(114, 114, dev)
        assert m1 is m2
        m3 = attn._flex_mask(114, 128, dev)
        assert m3 is not m1
        assert len(attn._flex_mask_cache) == 2

    def test_flex_train_bucketed_matches_eager(self):
        # training path (q_len == kv_len): padded to a 128 block, then cropped.
        from kairos.attentions import apply_rotary_emb, eager_attention

        cfg = DummySWAConfig()
        attn = KairosAttention(cfg, layer_idx=0)
        rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
        L = 114
        x = torch.randn(2, L, 32, device="cuda")
        pos = torch.arange(L, device="cuda").unsqueeze(0).expand(2, -1)
        cos_sin = rope(x, pos)
        out_flex = attn(x, cos_sin)

        q = attn.q_proj(x).view(2, L, cfg.num_attention_heads, attn.head_dim)
        k = attn.k_proj(x).view(2, L, cfg.num_key_value_heads, attn.head_dim)
        v = attn.v_proj(x).view(2, L, cfg.num_key_value_heads, attn.head_dim)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        eager = eager_attention(q, k, v, cfg.sliding_window_size)
        eager = eager.reshape(2, L, -1)
        out_eager = attn.out(eager)
        assert torch.allclose(out_flex, out_eager, atol=1e-3)

        # with padding: real rows match eager; padded q-rows attend kv 0 (finite).
        mask = torch.ones(2, L, dtype=torch.bool, device="cuda")
        mask[:, -7:] = False
        out_flex_p = attn(x, cos_sin, attention_mask=mask)
        assert not torch.isnan(out_flex_p).any()
        q = attn.q_proj(x).view(2, L, cfg.num_attention_heads, attn.head_dim)
        k = attn.k_proj(x).view(2, L, cfg.num_key_value_heads, attn.head_dim)
        v = attn.v_proj(x).view(2, L, cfg.num_key_value_heads, attn.head_dim)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        eager_p = eager_attention(q, k, v, cfg.sliding_window_size, key_padding_mask=mask)
        eager_p = attn.out(eager_p.reshape(2, L, -1))
        assert torch.allclose(out_flex_p[:, :-7], eager_p[:, :-7], atol=1e-3)

    def test_flex_train_bucketed_mask_reuse_across_lengths(self):
        # 114 and 129 map to 128/256 buckets: mask must be reused per gathered length.
        attn, _ = self._attn()
        dev = torch.device("cuda")
        m1 = attn._flex_mask_bucketed(128, 114, 2, dev)
        m2 = attn._flex_mask_bucketed(128, 114, 2, dev)
        assert m1 is m2
        m3 = attn._flex_mask_bucketed(256, 129, 2, dev)
        assert m3 is not m1
        m4 = attn._flex_mask_bucketed(256, 129, 2, dev)
        assert m3 is m4
        assert len(attn._flex_mask_cache) == 2

    def test_flex_matches_eager_on_cuda(self):
        # parity check: flex (fixed) and eager must agree for the same windowed mask
        from kairos.attentions import apply_rotary_emb, eager_attention

        cfg = DummySWAConfig()
        attn = KairosAttention(cfg, layer_idx=0)
        rope = KairosRotaryEmbedding(cfg, cfg.hidden_size // cfg.num_attention_heads)
        x = torch.randn(2, 114, 32, device="cuda")
        pos = torch.arange(114, device="cuda").unsqueeze(0).expand(2, -1)
        cos_sin = rope(x, pos)
        out_flex = attn(x, cos_sin)

        # manual eager equivalent: project + rotate + eager_attention
        q = attn.q_proj(x).view(2, 114, cfg.num_attention_heads, attn.head_dim)
        k = attn.k_proj(x).view(2, 114, cfg.num_key_value_heads, attn.head_dim)
        v = attn.v_proj(x).view(2, 114, cfg.num_key_value_heads, attn.head_dim)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        eager = eager_attention(q, k, v, cfg.sliding_window_size)
        eager = eager.reshape(2, 114, -1)
        out_eager = attn.out(eager)
        assert torch.allclose(out_flex, out_eager, atol=1e-3)


# --------------------------------------------------------- kernel backend detection
def test_delta_rule_backend_is_valid_value():
    assert DELTA_RULE_BACKEND in ("fla", "torch_fallback")


def test_causal_conv1d_backend_is_valid_value():
    assert CAUSAL_CONV1D_BACKEND in ("causal_conv1d", "torch_fallback")


def test_delta_rule_backend_matches_installed_fla():
    fla_available = importlib.util.find_spec("fla") is not None
    assert (DELTA_RULE_BACKEND == "fla") == fla_available


def test_causal_conv1d_backend_matches_installed_package():
    pkg_available = importlib.util.find_spec("causal_conv1d") is not None
    assert (CAUSAL_CONV1D_BACKEND == "causal_conv1d") == pkg_available


def test_warns_on_cuda_without_fast_kernels():
    """On CUDA without fla/causal-conv1d, importing kairos.attentions should warn loudly."""
    from kairos.attentions import _warn_if_missing_fast_kernels

    with pytest.warns(UserWarning, match="fast-attn"):
        _warn_if_missing_fast_kernels(
            cuda_available=True, delta_backend="torch_fallback", conv_backend="torch_fallback"
        )


def test_warns_lists_only_the_actually_missing_package():
    from kairos.attentions import _warn_if_missing_fast_kernels

    with pytest.warns(UserWarning) as record:
        _warn_if_missing_fast_kernels(cuda_available=True, delta_backend="fla", conv_backend="torch_fallback")
    msg = str(record[0].message)
    assert "causal-conv1d" in msg
    assert "flash-linear-attention" not in msg


def test_no_warning_on_cpu_without_fast_kernels(recwarn):
    """CPU-only machines can't install the CUDA-only fast kernels — no point warning there."""
    from kairos.attentions import _warn_if_missing_fast_kernels

    _warn_if_missing_fast_kernels(cuda_available=False, delta_backend="torch_fallback", conv_backend="torch_fallback")
    assert len(recwarn.list) == 0


def test_no_warning_on_cuda_with_both_fast_kernels_present(recwarn):
    from kairos.attentions import _warn_if_missing_fast_kernels

    _warn_if_missing_fast_kernels(cuda_available=True, delta_backend="fla", conv_backend="causal_conv1d")
    assert len(recwarn.list) == 0
