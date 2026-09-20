"""Tests for scripts/roofline.py, the source of the paper's compute table."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "roofline.py"
_spec = importlib.util.spec_from_file_location("roofline", _SCRIPT)
roofline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roofline)

H100 = roofline.GPUS[2]


def test_target_active_parameters_follow_eq_1():
    """With every parameter in the experts, E=32 and k=4 give exactly 25M active."""
    assert roofline.active_params(200e6, 200e6, 32, 4) == pytest.approx(25e6)


def test_dense_shared_parameters_are_always_active():
    """Dense parameters stay fully active; only the expert share is scaled by k/E."""
    assert roofline.active_params(100e6, 80e6, 8, 2) == pytest.approx(20e6 + 20e6)


@pytest.mark.parametrize(("gpu", "ridge"), [("T4", 203.1), ("A100 40GB", 200.6), ("H100 SXM5", 295.2)])
def test_ridge_points(gpu, ridge):
    """Ridge point is peak FLOP/s over bandwidth."""
    found = next(g for g in roofline.GPUS if g.name == gpu)
    assert found.ridge == pytest.approx(ridge, abs=0.1)


def test_dense_intensity_equals_tokens_per_microbatch():
    """For a dense model the weight-traffic intensity is one FLOP/byte per microbatch token."""
    assert roofline.intensity(50e6, 50e6, 1000) == pytest.approx(1000)


def test_moe_needs_total_over_active_times_the_dense_microbatch():
    """Sparsity multiplies the minimum microbatch by N_total / N_active (here 8x)."""
    dense = roofline.min_microbatch(200e6, 200e6, H100.ridge)
    moe = roofline.min_microbatch(25e6, 200e6, H100.ridge)
    assert moe / dense == pytest.approx(8)


def test_intensity_reaches_ridge_at_min_microbatch():
    """Plugging the minimum microbatch back into the intensity recovers the ridge point."""
    tokens = roofline.min_microbatch(25e6, 200e6, H100.ridge)
    assert roofline.intensity(25e6, 200e6, tokens) == pytest.approx(H100.ridge)


def test_target_training_flops():
    """25M active parameters on 30B tokens is 4.5e18 FLOPs."""
    assert roofline.train_flops(25e6, 30e9) == pytest.approx(4.5e18)


def test_h100_throughput_and_duration_at_assumed_mfu():
    """Compute-bound H100 throughput at 25% MFU and the implied days for 30B tokens."""
    tps = roofline.tokens_per_second(H100, 25e6, 0.25)
    assert tps == pytest.approx(1.648e6, rel=1e-3)
    assert roofline.days(30e9, tps) == pytest.approx(0.21, abs=0.005)


@pytest.mark.parametrize("gpu", roofline.GPUS)
def test_optimizer_overhead_is_small_at_large_step_batches(gpu):
    """A 2**17-token optimizer step keeps the memory-bound update under 3% of compute time."""
    overhead = roofline.optimizer_overhead(gpu, 25e6, 200e6, roofline.OPTIMIZER_BATCH)
    assert overhead < 0.03


def test_latex_tabular_wraps_one_row_per_gpu():
    """The generated tabular has a header, one row per GPU and closes its environment."""
    lines = roofline.latex_tabular().splitlines()
    assert lines[0].startswith(r"\begin{tabular}")
    assert lines[-1] == r"\end{tabular}"
    rows = [line for line in lines if line.endswith(r"\\") and "&" in line]
    assert len(rows) == len(roofline.GPUS) + 1
