import pytest
import torch

from kairos.modeling import (
    KairosConfig,
    KairosDiffusionFM,
    KairosMultiCache,
    KairosScaleRouter,
)


def _mask_first_n(batch, seq_len, n):
    """Active mask with the first n positions of the first row set True."""
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[0, :n] = True
    return mask


@pytest.fixture
def modality_scales():
    return {0: [0, 1], 1: [1, 2], 2: [2, 3]}


@pytest.fixture
def router(modality_scales):
    return KairosScaleRouter(modality_scales)


@pytest.fixture
def config():
    return KairosConfig(d_model=32, n_heads=4, n_layers=2, vocab_size=259, num_modalities=4)


def test_build_active_mask_shape(router):
    modality_ids = torch.zeros(2, 16, dtype=torch.long)
    mask = router.build_active_mask(modality_ids, scale_len=8, scale_idx=0)
    assert mask.shape == (2, 8)
    assert mask.dtype == torch.bool


def test_build_active_mask_respects_scale_mapping(router):
    modality_ids = torch.zeros(1, 16, dtype=torch.long)
    active_scale0 = router.build_active_mask(modality_ids, scale_len=8, scale_idx=0)
    active_scale3 = router.build_active_mask(modality_ids, scale_len=8, scale_idx=3)
    assert active_scale0.any()
    assert not active_scale3.any()


def test_build_active_mask_unmapped_modality_is_inactive_everywhere(router):
    modality_ids = torch.full((1, 16), 7, dtype=torch.long)
    for scale_idx in range(4):
        mask = router.build_active_mask(modality_ids, scale_len=8, scale_idx=scale_idx)
        assert not mask.any(), f"modality 7 unexpectedly active on scale {scale_idx}"


def test_build_active_mask_no_python_loop_over_length(router):
    modality_ids = torch.randint(0, 3, (4, 2048))
    mask = router.build_active_mask(modality_ids, scale_len=256, scale_idx=1)
    assert mask.shape == (4, 256)


def test_config_default_modality_scales_cover_all_modalities():
    config = KairosConfig(d_model=16, n_heads=2, n_layers=1, num_modalities=8)
    for m in range(config.num_modalities):
        assert m in config.modality_scales
        assert len(config.modality_scales[m]) > 0


def test_gather_scatter_roundtrip_identity(router):
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
    active_mask[0, [0, 1, 2]] = True
    active_mask[1, [5]] = True
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    # global raw max active count is 3 -> next power of two (4) floored to _MIN_BUCKET (8)
    assert gathered.shape == (2, 8, 4)
    assert pad_mask.shape == (2, 8)
    assert positions.shape == (2, 8)
    assert pad_mask[0].sum() == 3
    assert pad_mask[1].sum() == 1


def test_gather_active_no_active_positions_returns_none(router):
    x = torch.randn(2, 8, 4)
    active_mask = torch.zeros(2, 8, dtype=torch.bool)
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    assert gathered is None
    assert pad_mask is None
    assert positions is None


def test_gather_active_preserves_relative_order(router):
    x = torch.arange(10).float().view(1, 10, 1)
    active_mask = torch.zeros(1, 10, dtype=torch.bool)
    active_mask[0, [2, 5, 7]] = True
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    # only the first pad_mask.sum() entries are real (rest is bucket padding)
    n_active = int(pad_mask[0].sum())
    assert positions[0, :n_active].tolist() == [2, 5, 7]
    assert gathered[0, :n_active, 0].tolist() == [2.0, 5.0, 7.0]


def test_gather_active_rounds_up_to_min_bucket_floor(router):
    x = torch.randn(1, 64, 4)
    active_mask = torch.zeros(1, 64, dtype=torch.bool)
    active_mask[0, [0, 1, 2]] = True  # 3 active tokens, below the floor
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    assert gathered.shape[1] == KairosScaleRouter._MIN_BUCKET
    assert positions.shape[1] == KairosScaleRouter._MIN_BUCKET
    assert pad_mask.sum() == 3


def test_gather_active_bucket_stays_on_exact_power_of_two(router):
    x = torch.randn(1, 64, 4)
    active_mask = torch.zeros(1, 64, dtype=torch.bool)
    active_mask[0, :32] = True  # already an exact power of two, no rounding needed
    gathered, _pad_mask, _positions = router.gather_active(x, active_mask)
    assert gathered.shape[1] == 32


def test_gather_active_bucket_rounds_up_to_next_power_of_two(router):
    x = torch.randn(1, 96, 4)
    active_mask = torch.zeros(1, 96, dtype=torch.bool)
    active_mask[0, :33] = True  # one token past a power-of-two boundary
    gathered, _pad_mask, _positions = router.gather_active(x, active_mask)
    assert gathered.shape[1] == 64


def test_gather_active_bucket_count_grows_logarithmically_with_seq_len(router):
    # the whole point of power-of-two bucketing: distinct shapes stay ~log2(seq_len),
    # not linear in seq_len, so compiled-shape count stays bounded as models scale up.
    x_small = torch.randn(1, 128, 4)
    x_large = torch.randn(1, 8192, 4)
    seen_small = {
        router.gather_active(x_small, _mask_first_n(1, 128, n))[0].shape[1] for n in (1, 5, 20, 60, 128)
    }
    seen_large = {
        router.gather_active(x_large, _mask_first_n(1, 8192, n))[0].shape[1]
        for n in (1, 5, 20, 60, 500, 3000, 8192)
    }
    assert len(seen_small) <= 5  # log2(128) ~ 7, well short of "one shape per raw length"
    assert len(seen_large) <= 8  # log2(8192) = 13; still small, unlike a linear 32-step bucket


def test_gather_active_bucket_capped_at_sequence_length(router):
    x = torch.randn(1, 6, 4)
    active_mask = torch.zeros(1, 6, dtype=torch.bool)
    active_mask[0, :5] = True  # bucket (8) would exceed the actual seq_len (6)
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    assert gathered.shape[1] == 6
    assert positions.shape[1] == 6
    assert pad_mask.sum() == 5


def test_gather_scatter_roundtrip_identity_with_bucket_padding(router):
    # S=64 so the bucketed length (8, from _MIN_BUCKET) is real padding, not capped by seq_len
    x = torch.randn(2, 64, 4)
    active_mask = torch.zeros(2, 64, dtype=torch.bool)
    active_mask[0, [1, 3, 5]] = True
    active_mask[1, [10]] = True
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    assert gathered.shape[1] == 8  # confirms real bucket padding is exercised here
    output = router.scatter_active(x.clone(), gathered, pad_mask, positions)
    assert torch.allclose(output, x, atol=1e-6)


def test_scatter_active_does_not_touch_inactive_positions(router):
    x = torch.zeros(1, 6, 2)
    active_mask = torch.zeros(1, 6, dtype=torch.bool)
    active_mask[0, [1, 4]] = True
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    chunk = torch.ones_like(gathered) * 99.0
    output = router.scatter_active(x.clone(), chunk, pad_mask, positions)
    untouched_idx = [0, 2, 3, 5]
    assert torch.allclose(output[0, untouched_idx], torch.zeros(4, 2))
    assert torch.allclose(output[0, [1, 4]], torch.full((2, 2), 99.0))


def test_scatter_active_casts_chunk_to_output_dtype(router):
    x = torch.randn(2, 8, 4, dtype=torch.float16)
    active_mask = torch.zeros(2, 8, dtype=torch.bool)
    active_mask[0, [1, 2]] = True
    active_mask[1, [3]] = True
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    chunk = gathered.double()
    output = router.scatter_active(x.clone(), chunk, pad_mask, positions)
    assert output.dtype == torch.float16
    assert torch.allclose(output, x, atol=1e-3)


def test_scatter_active_is_not_inplace(router):
    x = torch.randn(1, 6, 2)
    active_mask = torch.zeros(1, 6, dtype=torch.bool)
    active_mask[0, [1, 4]] = True
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    x_before = x.clone()
    result = router.scatter_active(x, gathered, pad_mask, positions)
    assert result is not x
    assert torch.allclose(x, x_before)


def test_gather_scatter_batch_independence(router):
    x = torch.zeros(2, 8, 1)
    x[0] = 1.0
    x[1] = 2.0
    active_mask = torch.zeros(2, 8, dtype=torch.bool)
    active_mask[0, [0, 1, 2, 3]] = True
    active_mask[1, [0]] = True
    gathered, pad_mask, _positions = router.gather_active(x, active_mask)
    assert torch.allclose(gathered[0][pad_mask[0]], torch.ones_like(gathered[0][pad_mask[0]]))
    assert torch.allclose(gathered[1][pad_mask[1]], 2 * torch.ones_like(gathered[1][pad_mask[1]]))


def test_gather_scatter_backward():
    router = KairosScaleRouter({0: [0]})
    x = torch.randn(2, 8, 4, requires_grad=True)
    active_mask = torch.zeros(2, 8, dtype=torch.bool)
    active_mask[0, [1, 2]] = True
    active_mask[1, [3]] = True
    gathered, pad_mask, positions = router.gather_active(x, active_mask)
    chunk = gathered * 2.0
    output = router.scatter_active(x.clone(), chunk, pad_mask, positions)
    output[active_mask].sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert x.grad[0, 1].abs().sum() > 0
    assert x.grad[0, 2].abs().sum() > 0
    assert x.grad[0, 0].abs().sum() == 0
    assert x.grad[1, 0].abs().sum() == 0


def test_model_forward_batch_independence_via_routing(config):
    model = KairosDiffusionFM(config)
    x = torch.randint(0, 259, (2, 16))
    modality_ids = torch.zeros(2, 16, dtype=torch.long)
    modality_ids[0, :8] = 1
    out_batched = model(input_ids=x, modality_ids=modality_ids)
    out_row0 = model(input_ids=x[0:1], modality_ids=modality_ids[0:1])
    out_row1 = model(input_ids=x[1:2], modality_ids=modality_ids[1:2])
    assert torch.allclose(out_batched.logits[0:1], out_row0.logits, atol=1e-4)
    assert torch.allclose(out_batched.logits[1:2], out_row1.logits, atol=1e-4)


def test_model_cache_offset_consistent_across_scales(config):
    model = KairosDiffusionFM(config)
    cache = KairosMultiCache(config)
    x_ctx = torch.randint(0, 259, (1, 16))
    x_next = torch.randint(0, 259, (1, 8))
    _ = model(input_ids=x_ctx, cache_params=cache)
    out_with_offset = model(input_ids=x_next, cache_params=cache.clone())
    out_without_offset = model(input_ids=x_next, cache_params=None)
    assert not torch.allclose(out_with_offset.logits, out_without_offset.logits, atol=1e-4)
