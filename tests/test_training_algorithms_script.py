import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_training_algorithms", Path(__file__).parents[1] / "scripts" / "run_training_algorithms.py"
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

_OUTPUT = (
    "overfit_test:  50%|#####     | 900/1800\roverfit_test: 1800 steps on 16 examples - loss 8.4608 -> 0.2521 (last 0.0092)\n"
    "overfit_test: reference CE (linear, p_max=1, no reweight) 0.2193\n"
    "overfit_test: sampling accuracy {'dfm_8': 0.5, 'gen_8': 0.75, 'dfm_16': 0.7, 'gen_16': 0.76}\n"
)


def test_parse_output_reads_loss_reference_and_sampling():
    parsed = script.parse_output(_OUTPUT)
    assert parsed["train_last"] == 0.0092 and parsed["ref_ce"] == 0.2193 and parsed["acc"]["gen_16"] == 0.76
    with pytest.raises(ValueError, match="missing"):
        script.parse_output("nothing useful")


def _rec(phase, arm, seed, lr, ref, steps=600):
    acc = {"gen_16": 0.5, "dfm_16": 0.4}
    return {
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "lr": lr,
        "ref_ce": ref,
        "train_last": 0.0,
        "acc": acc,
    }


def test_best_lr_uses_mean_over_seeds_and_breaks_ties_low():
    records = [
        _rec("sweep", "train_adamw", s, lr, ref) for s in (9, 10) for lr, ref in ((5e-4, 0.5), (1e-3, 0.3), (2e-3, 0.3))
    ]
    assert script.best_lr(records, "train_adamw") == 1e-3
    with pytest.raises(ValueError, match="no sweep"):
        script.best_lr(records, "train_muon_hybrid")


def test_final_jobs_use_selected_lr_and_share_the_adamw_rate_with_sampling_arms():
    records = [
        _rec("sweep", a, s, lr, 0.3 if lr == 2e-3 else 0.9)
        for a in script.SWEEP_ARMS
        for s in (9, 10)
        for lr in script.LRS
    ]
    planned = script.jobs("final", records)
    assert len(planned) == len(script.ARMS) * len(script.FINAL_SEEDS)
    assert all(job[4] == 2e-3 and job[3] == script.FINAL_STEPS for job in planned)
    with pytest.raises(SystemExit):
        script.jobs("final", [])


def test_report_pairs_arms_with_the_adamw_control_by_seed():
    records = [
        _rec("final", a, s, 1e-3, 0.1 + 0.01 * i + 0.1 * s, 1800) for i, a in enumerate(script.ARMS) for s in (0, 1)
    ]
    text = script.report(records)
    assert "`train_muon_hybrid`: 0.020 ± 0.000 (lower on 0/2 seeds)" in text


def test_best_lr_never_selects_a_diverged_run():
    records = [_rec("sweep", "train_adamw", s, 2e-3, float("inf")) for s in (9, 10)]
    records += [_rec("sweep", "train_adamw", s, 1e-3, 0.4) for s in (9, 10)]
    assert script.best_lr(records, "train_adamw") == 1e-3


def test_parse_output_reads_the_heldout_reference_when_present():
    assert "heldout_ce" not in script.parse_output(_OUTPUT)
    parsed = script.parse_output(_OUTPUT + "overfit_test: held-out reference CE 3.4992\n")
    assert parsed["heldout_ce"] == 3.4992 and parsed["ref_ce"] == 0.2193
