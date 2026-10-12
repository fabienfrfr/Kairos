import math

from kairos.diagnostics import REGIMES, format_report, run


def test_run_returns_finite_losses_and_expert_usage_for_each_regime():
    for regime in REGIMES:
        losses, usage = run(regime, bias_update_rate=1e-3, steps=2, num_local_experts=4)

        assert len(losses) == 2
        assert all(math.isfinite(loss_val) for loss_val in losses)
        assert len(usage) == 4


def test_format_report_includes_final_loss_and_usage():
    report = format_report(losses=[1.0, 0.5, 0.25], usage=[3.0, 1.0])

    assert "loss finale: 0.2500" in report
    assert "usage par expert" in report
    assert "tous finis: True" in report


def test_format_report_flags_non_finite_losses():
    report = format_report(losses=[1.0, float("nan")], usage=[0.0])

    assert "tous finis: False" in report
