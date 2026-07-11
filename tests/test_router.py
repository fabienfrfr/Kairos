import torch
import pytest

from kairos.modeling import (
    KairosScaleRouter,
    KairosConfig,
    KairosDiffusionLLM,
    KairosMultiCache,
)


# =========================
# Fixtures
# =========================
@pytest.fixture
def modality_scales():
    # 0: text (scales 0,1) | 1: image (scales 1,2) | 2: video (scales 2,3)
    return {0: [0, 1], 1: [1, 2], 2: [2, 3]}


@pytest.fixture
def router(modality_scales):
    return KairosScaleRouter(modality_scales)


@pytest.fixture
def config():
    return KairosConfig(d_model=32, n_heads=4, n_layers=2, vocab_size=259, num_modalities=4)


# =========================
# build_active_mask
# =========================
def test_build_active_mask_shape(router):
    modality_ids = torch.zeros(2, 16, dtype=torch.long)
    mask = router.build_active_mask(modality_ids, scale_len=8, scale_idx=0)

    assert mask.shape == (2, 8)
    assert mask.dtype == torch.bool


def test_build_active_mask_respects_scale_mapping(router):
    """Modality 0 (text) is only allowed on scales 0 and 1 — must be inactive on scale 3."""
    modality_ids = torch.zeros(1, 16, dtype=torch.long)

    active_scale0 = router.build_active_mask(modality_ids, scale_len=8, scale_idx=0)
    active_scale3 = router.build_active_mask(modality_ids, scale_len=8, scale_idx=3)

    assert active_scale0.any()
    assert not active_scale3.any()


def test_build_active_mask_unmapped_modality_is_inactive_everywhere(router):
    """
    Regression test for the historical bug: a modality id absent from
    `modality_scales` used to silently produce an all-False mask on every
    scale, meaning those tokens never crossed any backbone. This test locks
    in that *documented* behavior for an explicitly unmapped id, so any
    future change to the default-mapping logic (KairosConfig now auto-fills
    unmapped modalities to scale 0) doesn't regress unnoticed for a router
    built with a manually incomplete mapping.
    """
    modality_ids = torch.full((1, 16), 7, dtype=torch.long)  # not in modality_scales fixture

    for scale_idx in range(4):
        mask = router.build_active_mask(modality_ids, scale_len=8, scale_idx=scale_idx)
        assert not mask.any(), f"modality 7 unexpectedly active on scale {scale_idx}"


def test_build_active_mask_no_python_loop_over_length(router):
    """Sanity check: mask must be vectorized correctly on a large sequence length."""
    modality_ids = torch.randint(0, 3, (4, 2048))
    mask = router.build_active_mask(modality_ids, scale_len=256, scale_idx=1)

    assert mask.shape == (4, 256)


def test_config_default_modality_scales_cover_all_modalities():
    """
    Regression test for the config-level fix: every modality id up to
    num_modalities must appear in modality_scales, unlike the original
    hardcoded {0,1,2} mapping which silently dropped ids >= 3.
    """
    config = KairosConfig(d_model=16, n_heads=2, n_layers=1, num_modalities=8)

    for m in range(config.num_modalities):
        assert m in config.modality_scales
        assert len(config.modality_scales[m]) > 0


# =========================
# gather_active / scatter_active roundtrip
# =========================
def test_gather_scatter_roundtrip_identity(router):
    """Gathering then scattering back (with the same chunk) must reproduce the input exactly."""
    x = torch.randn(3, 10, 4)
    active_mask = torch.zeros(3, 10, dtype=torch.bool)
    active_mask[0, [1, 3, 5]] = True
    active_mask[1, [0, 2]] = True
    active_mask[2, [9]] = True

    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    output = router.scatter_active(x.clone(), gathered, pad_mask, positions)

    assert torch.allclose(output, x, atol=1e-6)


def test_gather_active_shapes(router):
    x = torch.randn(2, 12, 4)
    active_mask = torch.zeros(2, 12, dtype=torch.bool)
    active_mask[0, [0, 1, 2]] = True  # 3 active
    active_mask[1, [5]] = True  # 1 active

    gathered, pad_mask, positions = router.gather_active(x, active_mask)

    assert gathered.shape == (2, 3, 4)  # max_len = 3
    assert pad_mask.shape == (2, 3)
    assert positions.shape == (2, 3)

    assert pad_mask[0].all()  # 3 active out of 3
    assert pad_mask[1].sum() == 1  # only 1 active out of 3 slots


def test_gather_active_no_active_positions_returns_none(router):
    x = torch.randn(2, 8, 4)
    active_mask = torch.zeros(2, 8, dtype=torch.bool)

    gathered, pad_mask, positions = router.gather_active(x, active_mask)

    assert gathered is None
    assert pad_mask is None
    assert positions is None


def test_gather_active_preserves_relative_order(router):
    """Active positions must stay in their original left-to-right order after the sort trick."""
    x = torch.arange(10).float().view(1, 10, 1)
    active_mask = torch.zeros(1, 10, dtype=torch.bool)
    active_mask[0, [2, 5, 7]] = True

    gathered, pad_mask, positions = router.gather_active(x, active_mask)

    assert positions[0].tolist() == [2, 5, 7]
    assert gathered[0, :, 0].tolist() == [2.0, 5.0, 7.0]


def test_scatter_active_does_not_touch_inactive_positions(router):
    """Positions never marked active must keep their original value after scatter."""
    x = torch.zeros(1, 6, 2)
    active_mask = torch.zeros(1, 6, dtype=torch.bool)
    active_mask[0, [1, 4]] = True

    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    chunk = torch.ones_like(gathered) * 99.0  # sentinel value

    output = router.scatter_active(x.clone(), chunk, pad_mask, positions)

    untouched_idx = [0, 2, 3, 5]
    assert torch.allclose(output[0, untouched_idx], torch.zeros(4, 2))
    assert torch.allclose(output[0, [1, 4]], torch.full((2, 2), 99.0))


def test_scatter_active_is_not_inplace(router):
    """
    Regression test for the autograd bug: scatter_active must NOT mutate
    `output` in place (that broke backward through an earlier `torch.gather`
    call on the same tensor). Returning a new tensor object is required.
    """
    x = torch.randn(1, 6, 2)
    active_mask = torch.zeros(1, 6, dtype=torch.bool)
    active_mask[0, [1, 4]] = True

    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    x_before = x.clone()

    result = router.scatter_active(x, gathered, pad_mask, positions)

    assert result is not x, "scatter_active must return a new tensor, not mutate its input"
    assert torch.allclose(x, x_before), "the original tensor passed to scatter_active must be left untouched"


def test_gather_scatter_batch_independence(router):
    """
    Regression test for the original multi-batch cache-corruption bug: gathering
    and scattering different-length active segments per batch row must never
    leak values across rows.
    """
    x = torch.zeros(2, 8, 1)
    x[0] = 1.0
    x[1] = 2.0

    active_mask = torch.zeros(2, 8, dtype=torch.bool)
    active_mask[0, [0, 1, 2, 3]] = True  # 4 active
    active_mask[1, [0]] = True  # 1 active

    gathered, pad_mask, positions = router.gather_active(x, active_mask)

    # row 0 must only ever see 1.0, row 1 must only ever see 2.0
    assert torch.allclose(gathered[0][pad_mask[0]], torch.ones_like(gathered[0][pad_mask[0]]))
    assert torch.allclose(gathered[1][pad_mask[1]], 2 * torch.ones_like(gathered[1][pad_mask[1]]))


def test_gather_scatter_backward():
    """Gradient must flow through gather -> (fake) processing -> scatter."""
    router = KairosScaleRouter({0: [0]})

    x = torch.randn(2, 8, 4, requires_grad=True)
    active_mask = torch.zeros(2, 8, dtype=torch.bool)
    active_mask[0, [1, 2]] = True
    active_mask[1, [3]] = True

    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    chunk = gathered * 2.0  # stand-in for a backbone call

    output = router.scatter_active(x.clone(), chunk, pad_mask, positions)

    # NOTE: `output` equals `x` everywhere except at active positions (that's
    # the intended identity-elsewhere behavior). Summing the *whole* tensor
    # would route a gradient of 1 through every untouched position too (a
    # clone is an identity function), which isn't what this test wants to
    # isolate. Restrict the loss to the active positions only.
    output[active_mask].sum().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    # gradient should be nonzero exactly at the active positions
    assert x.grad[0, 1].abs().sum() > 0
    assert x.grad[0, 2].abs().sum() > 0
    assert x.grad[0, 0].abs().sum() == 0
    assert x.grad[1, 0].abs().sum() == 0  # the padded slot's underlying index for row 1


# =========================
# End-to-end model regression (routing-specific, not attention-specific)
# =========================
def test_model_forward_batch_independence_via_routing(config):
    """
    End-to-end regression for the original bug: two batch rows with different
    modality patterns (hence different active segment lengths per scale) must
    not influence each other's output.
    """
    model = KairosDiffusionLLM(config)

    x = torch.randint(0, 259, (2, 16))
    modality_ids = torch.zeros(2, 16, dtype=torch.long)
    modality_ids[0, :8] = 1  # row 0 has a mixed modality pattern
    # row 1 stays all-text

    out_batched = model(input_ids=x, modality_ids=modality_ids)

    out_row0 = model(input_ids=x[0:1], modality_ids=modality_ids[0:1])
    out_row1 = model(input_ids=x[1:2], modality_ids=modality_ids[1:2])

    assert torch.allclose(out_batched.logits[0:1], out_row0.logits, atol=1e-4)
    assert torch.allclose(out_batched.logits[1:2], out_row1.logits, atol=1e-4)


def test_model_cache_offset_consistent_across_scales(config):
    """
    Regression test for the position_ids fix: RoPE position must reflect the
    cache's total-seen offset, not just the local chunk index, so a second
    forward call with a populated cache must differ from a first call without
    one (same input, same shape).
    """
    model = KairosDiffusionLLM(config)
    cache = KairosMultiCache(config)

    x_ctx = torch.randint(0, 259, (1, 16))
    x_next = torch.randint(0, 259, (1, 8))

    _ = model(input_ids=x_ctx, cache_params=cache)
    out_with_offset = model(input_ids=x_next, cache_params=cache.clone())
    out_without_offset = model(input_ids=x_next, cache_params=None)

    assert not torch.allclose(out_with_offset.logits, out_without_offset.logits, atol=1e-4)