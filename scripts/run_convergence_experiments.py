"""Reproduces the convergence experiments of the paper (tables, flow matching, held-out, curriculum).

Usage: python scripts/run_convergence_experiments.py tables|flow|heldout|curriculum|report [--budget-s N]
Results go to docs/experiments/results/<experiment>.jsonl (curves to results/curves/); finished runs are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_training_algorithms import parse_output

ROOT = Path(__file__).parents[1]
OUT = ROOT / "docs" / "experiments" / "results"
TABLE_FLAGS = ["--mask-p-max", "0.3", "--no-mask-reweight"]
FULL_RANGE = {"flow_mdlm_control": ["--mask-reweight"], "flow_dfm_linear": ["--no-mask-reweight"]}
FULL_RANGE["flow_dfm_cubic"] = FULL_RANGE["flow_dfm_linear"]
EXPERIMENTS = {
    "tables": {"steps": 600, "extra": TABLE_FLAGS, "curve": True},
    "flow": {"steps": 1800, "extra": ["--mask-p-max", "1.0"], "curve": False},
    "heldout": {
        "steps": 1000,
        "extra": ["--mask-p-max", "1.0", "--n-examples", "48", "--heldout", "24"],
        "curve": False,
    },
    "curriculum": {"steps": 1800, "extra": [], "curve": False},
}
TABLE_ARMS = {
    "baseline": 5,
    "attn_vanilla": 5,
    "codec_conv_cpg8": 5,
    "codec_conv_dense": 3,
    "codec_patch": 3,
    "liz2_separate_qkv": 5,
}
SEEDS = (0, 1, 2, 3)


def plan(experiment: str) -> list[tuple[str, int]]:
    if experiment == "tables":
        return [(arm, s) for arm, n in TABLE_ARMS.items() for s in range(n)]
    arms = {
        "flow": ("flow_mdlm_control", "flow_dfm_linear", "flow_dfm_cubic"),
        "heldout": ("flow_heldout_mdlm_control", "flow_heldout_dfm_linear", "flow_heldout_dfm_cubic"),
        "curriculum": ("flow_mdlm_control",),
    }[experiment]
    return [(arm, s) for s in SEEDS for arm in arms]


def derive_curve_stats(curve: list[float]) -> dict:
    first = next((i + 1 for i, x in enumerate(curve) if x <= 1.0), None)
    return {"min_loss": min(curve), "last100": statistics.mean(curve[-100:]), "steps_to_1": first}


def command(experiment: str, arm: str, seed: int, lr: float | None = None) -> list[str]:
    spec = EXPERIMENTS[experiment]
    extra = list(spec["extra"])
    if experiment in ("flow", "heldout"):
        extra += FULL_RANGE[arm.replace("heldout_", "")]
    cmd = [
        sys.executable,
        "-m",
        "kairos.cli",
        "overfit",
        f"configs/ablations/{arm}.yaml",
        "--steps",
        str(spec["steps"]),
    ]
    cmd += [*extra, "--seed", str(seed), "--log-every", "0"]
    return cmd + ([] if lr is None else ["--lr", str(lr)])


def results_path(experiment: str, lr: float | None = None) -> Path:
    return OUT / (f"{experiment}.jsonl" if lr is None else f"{experiment}_lr{lr:g}.jsonl")


def load(experiment: str, lr: float | None = None) -> list[dict]:
    path = results_path(experiment, lr)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def run_one(experiment: str, arm: str, seed: int, lr: float | None = None) -> dict:
    cmd = command(experiment, arm, seed, lr)
    curve_path = OUT / "curves" / f"{arm}_{seed}{'' if lr is None else f'_lr{lr:g}'}.json"
    if EXPERIMENTS[experiment]["curve"]:
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--save-curve", str(curve_path)]
    env = {**os.environ, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    rec = {"experiment": experiment, "arm": arm, "seed": seed, "steps": EXPERIMENTS[experiment]["steps"], "lr": lr}
    rec.update(parse_output(out.stdout + out.stderr))
    if EXPERIMENTS[experiment]["curve"]:
        rec.update(derive_curve_stats(json.loads(curve_path.read_text())))
    with results_path(experiment, lr).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def run_phase(experiment: str, budget_s: float, lr: float | None = None) -> None:
    start, longest = time.time(), 0.0
    for arm, seed in plan(experiment):
        if any(r["arm"] == arm and r["seed"] == seed for r in load(experiment, lr)):
            continue
        if time.time() - start + longest > budget_s:
            print("budget reached; rerun the same command to continue")
            return
        t0 = time.time()
        run_one(experiment, arm, seed, lr)
        longest = max(longest, time.time() - t0)
        print(experiment, arm, seed, f"{longest:.0f}s", flush=True)
    print(f"{experiment} complete")


def _fmt(values: list[float]) -> str:
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}" if len(values) > 1 else f"{values[0]:.3f}"


def report(experiment: str, lr: float | None = None) -> str:
    rows = load(experiment, lr)
    lines = []
    for arm in dict.fromkeys(r["arm"] for r in rows):
        mine = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"])
        if experiment == "tables":
            steps = [r["steps_to_1"] for r in mine if r["steps_to_1"] is not None]
            cells = [
                _fmt([r["min_loss"] for r in mine]),
                _fmt([r["last100"] for r in mine]),
                f"{statistics.mean(steps):.0f}",
            ]
        elif experiment == "heldout":
            cells = [_fmt([r["ref_ce"] for r in mine]), _fmt([r["heldout_ce"] for r in mine])]
        else:
            cells = [_fmt([r["ref_ce"] for r in mine]), _fmt([r["acc"]["gen_16"] for r in mine])]
        lines.append(f"| `{arm}` | {len(mine)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*EXPERIMENTS, "report"])
    parser.add_argument("--budget-s", type=float, default=float("inf"))
    parser.add_argument(
        "--lr", type=float, default=None, help="learning rate; results go to <experiment>_lr<value>.jsonl"
    )
    args = parser.parse_args()
    if args.command == "report":
        for name in EXPERIMENTS:
            print(f"## {name}\n{report(name, args.lr)}\n")
    else:
        run_phase(args.command, args.budget_s, args.lr)


if __name__ == "__main__":
    main()
