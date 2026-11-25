import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "run_convergence_experiments", _SCRIPTS / "run_convergence_experiments.py"
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


def test_plans_cover_every_run_of_the_paper():
    assert [len(script.plan(name)) for name in ("tables", "flow", "heldout", "curriculum")] == [26, 12, 12, 4]


def test_commands_match_the_documented_protocols():
    flow = script.command("flow", "flow_dfm_cubic", 2)
    assert (
        "--no-mask-reweight" in flow
        and "2" in flow
        and "1.0" in flow
        and "1800" in flow
        and "--mask-reweight" not in flow
    )
    assert "--mask-reweight" in script.command("flow", "flow_mdlm_control", 0)
    held = script.command("heldout", "flow_heldout_dfm_linear", 1)
    assert "--heldout" in held and "24" in held and "1000" in held and "--no-mask-reweight" in held
    assert "--mask-p-max" not in script.command("curriculum", "flow_mdlm_control", 3)
    tables = script.command("tables", "baseline", 0)
    assert tables[tables.index("--mask-p-max") + 1] == "0.3" and "600" in tables


def test_derive_curve_stats():
    stats = script.derive_curve_stats([5.0, 2.0, 0.9, 0.5] + [0.4] * 100)
    assert stats["min_loss"] == 0.4 and stats["steps_to_1"] == 3 and stats["last100"] == 0.4
    assert script.derive_curve_stats([3.0, 2.0])["steps_to_1"] is None


def test_committed_results_reproduce_the_numbers_quoted_in_the_paper():
    tex = (Path(__file__).parents[1] / "docs" / "paper" / "kairos_paper.tex").read_text(encoding="utf-8")
    quoted = {
        "tables": ["0.273 ± 0.309", "0.336 ± 0.177", "0.054 ± 0.018", "0.024 ± 0.001", "0.017 ± 0.001"],
        "flow": ["0.140 ± 0.099", "0.132 ± 0.137", "0.145 ± 0.078"],
        "heldout": ["3.494 ± 0.026", "3.526 ± 0.025", "3.571 ± 0.030"],
        "curriculum": ["0.126 ± 0.102"],
    }
    for name, values in quoted.items():
        text = script.report(name)
        for value in values:
            assert value in text, (name, value)
            assert value.replace(" ± ", " \\pm ") in tex or value.replace(" ± ", "\\pm") in tex, value


def test_lr_option_changes_the_command_and_the_results_file_only_when_given():
    assert "--lr" not in script.command("flow", "flow_mdlm_control", 0)
    cmd = script.command("flow", "flow_mdlm_control", 0, 0.008)
    assert cmd[-2:] == ["--lr", "0.008"]
    assert script.results_path("flow").name == "flow.jsonl"
    assert script.results_path("flow", 0.008).name == "flow_lr0.008.jsonl"


def _paired(rows, arm, base, key):
    import statistics

    def pick(name):
        return [r[key] for r in sorted((r for r in rows if r["arm"] == name), key=lambda r: r["seed"])]

    diffs = [a - b for a, b in zip(pick(arm), pick(base), strict=True)]
    return f"{statistics.mean(diffs):+.3f}\\pm{statistics.stdev(diffs):.3f}"


def test_results_at_the_tuned_rate_reproduce_the_numbers_quoted_in_the_paper():
    tex = (Path(__file__).parents[1] / "docs" / "paper" / "kairos_paper.tex").read_text(encoding="utf-8")
    quoted = {
        "flow": ["0.082 ± 0.106", "0.073 ± 0.052", "0.034 ± 0.027"],
        "heldout": ["3.591 ± 0.045", "3.675 ± 0.064", "3.805 ± 0.079"],
        "curriculum": ["0.067 ± 0.099"],
    }
    for name, values in quoted.items():
        text = script.report(name, 0.008)
        for value in values:
            spaced, compact = value.replace(" ± ", " \\pm "), value.replace(" ± ", "\\pm")
            assert value in text and (spaced in tex or compact in tex), (name, value)
    flow, held = script.load("flow", 0.008), script.load("heldout", 0.008)
    control, hcontrol = "flow_mdlm_control", "flow_heldout_mdlm_control"
    expected = [
        _paired(flow, "flow_dfm_linear", control, "ref_ce"),
        _paired(flow, "flow_dfm_cubic", control, "ref_ce"),
        _paired(held, "flow_heldout_dfm_linear", hcontrol, "heldout_ce"),
        _paired(held, "flow_heldout_dfm_cubic", hcontrol, "heldout_ce"),
        _paired(held, "flow_heldout_dfm_linear", hcontrol, "ref_ce"),
        _paired(held, "flow_heldout_dfm_cubic", hcontrol, "ref_ce"),
    ]
    assert all(value in tex for value in expected), expected
