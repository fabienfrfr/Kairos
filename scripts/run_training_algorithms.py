"""Reproduces the training-algorithm study: lr sweep, final runs and the report tables.

Usage: python scripts/run_training_algorithms.py sweep|final|report [--budget-s N]
Results are appended to docs/experiments/results/training_algorithms.jsonl; finished runs are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "docs" / "experiments" / "results" / "training_algorithms.jsonl"
ARMS = ("train_adamw", "train_adamw_tuned", "train_muon_hybrid", "train_stratified", "train_antithetic")
SWEEP_ARMS = ("train_adamw", "train_adamw_tuned", "train_muon_hybrid")
LRS = (5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2)
SWEEP_SEEDS, SWEEP_STEPS = (9, 10), 600
FINAL_SEEDS, FINAL_STEPS = (0, 1, 2, 3), 1800
LR_OF_ARM = {"train_stratified": "train_adamw", "train_antithetic": "train_adamw"}


def parse_output(text: str) -> dict:
    """Extracts the metrics printed by `kairos overfit` (tqdm carriage returns are tolerated)."""
    text = text.replace("\r", "\n")
    last = re.search(r"overfit_test: \d+ steps on \d+ examples - loss [\d.]+ -> [\d.]+ \(last ([\d.]+)\)", text)
    ref = re.search(r"reference CE [^\n]*? ([\d.]+)\n", text)
    acc = re.search(r"sampling accuracy (\{[^}]*\})", text)
    if not (last and ref and acc):
        raise ValueError("overfit output is missing the loss, reference CE or sampling accuracy line")
    held = re.search(r"held-out reference CE ([\d.]+)", text)
    parsed = {"train_last": float(last[1]), "ref_ce": float(ref[1]), "acc": eval(acc[1], {"__builtins__": {}})}
    return {**parsed, "heldout_ce": float(held[1])} if held else parsed


def load() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]


def best_lr(records: list[dict], arm: str) -> float:
    """Learning rate with the lowest mean reference CE over the sweep seeds (ties go to the lower rate)."""
    by_lr: dict[float, list[float]] = {}
    for r in records:
        if r["phase"] == "sweep" and r["arm"] == arm:
            by_lr.setdefault(r["lr"], []).append(r["ref_ce"])
    if not by_lr:
        raise ValueError(f"no sweep records for {arm}")
    return min(by_lr, key=lambda lr: (statistics.mean(by_lr[lr]), lr))


def run_one(phase: str, arm: str, seed: int, steps: int, lr: float) -> dict:
    cmd = [sys.executable, "-m", "kairos.cli", "overfit", f"configs/ablations/{arm}.yaml", "--steps", str(steps)]
    cmd += ["--mask-p-max", "1.0", "--mask-reweight", "--seed", str(seed), "--lr", str(lr), "--log-every", "0"]
    env = {**os.environ, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}
    base = {"phase": phase, "arm": arm, "seed": seed, "steps": steps, "lr": lr}
    try:
        out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        rec = {**base, **parse_output(out.stdout + out.stderr)}
    except (subprocess.CalledProcessError, ValueError):
        # a diverged run is recorded with an infinite reference CE so it can never be selected
        rec = {**base, "train_last": float("nan"), "ref_ce": float("inf"), "acc": {}, "failed": True}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def jobs(phase: str, records: list[dict]) -> list[tuple[str, str, int, int, float]]:
    if phase == "sweep":
        return [("sweep", a, s, SWEEP_STEPS, lr) for s in SWEEP_SEEDS for a in SWEEP_ARMS for lr in LRS]
    if not all(any(r["phase"] == "sweep" and r["arm"] == a for r in records) for a in SWEEP_ARMS):
        raise SystemExit("run the sweep first")
    return [("final", a, s, FINAL_STEPS, best_lr(records, LR_OF_ARM.get(a, a))) for s in FINAL_SEEDS for a in ARMS]


def run_phase(phase: str, budget_s: float) -> None:
    start, longest = time.time(), 0.0
    for job in jobs(phase, load()):
        done = {(r["phase"], r["arm"], r["seed"], r["steps"], r["lr"]) for r in load()}
        if job in done:
            continue
        if time.time() - start + longest > budget_s:
            print("budget reached; rerun the same command to continue")
            return
        t0 = time.time()
        rec = run_one(*job)
        longest = max(longest, time.time() - t0)
        print(job, f"ref_ce={rec['ref_ce']:.4f}", flush=True)
    print(f"{phase} complete")


def _fmt(values: list[float]) -> str:
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}" if len(values) > 1 else f"{values[0]:.3f}"


def report(records: list[dict]) -> str:
    lines = ["| Arm | lr | seeds | reference CE | generate 16 | dfm_sample 16 |", "|---|---:|---:|---:|---:|---:|"]
    final = [r for r in records if r["phase"] == "final"]
    for arm in ARMS:
        rows = sorted((r for r in final if r["arm"] == arm), key=lambda r: r["seed"])
        if not rows:
            continue
        lines.append(
            f"| `{arm}` | {rows[0]['lr']:g} | {len(rows)} | {_fmt([r['ref_ce'] for r in rows])} "
            f"| {_fmt([r['acc']['gen_16'] for r in rows])} | {_fmt([r['acc']['dfm_16'] for r in rows])} |"
        )
    base = {r["seed"]: r["ref_ce"] for r in final if r["arm"] == "train_adamw"}
    lines += ["", "Reference CE, arm minus `train_adamw`, paired by seed:", ""]
    for arm in ARMS[1:]:
        diffs = [r["ref_ce"] - base[r["seed"]] for r in final if r["arm"] == arm and r["seed"] in base]
        if diffs:
            lines.append(f"- `{arm}`: {_fmt(diffs)} (lower on {sum(d < 0 for d in diffs)}/{len(diffs)} seeds)")
    sweep = [r for r in records if r["phase"] == "sweep"]
    lines += ["", "Sweep (mean reference CE over seeds, 600 steps):", ""]
    for arm in SWEEP_ARMS:
        cells = {lr: [r["ref_ce"] for r in sweep if r["arm"] == arm and r["lr"] == lr] for lr in LRS}
        shown = ", ".join(f"{lr:g}: {statistics.mean(v):.3f}" for lr, v in cells.items() if v)
        if shown:
            lines.append(f"- `{arm}`: {shown}; selected {best_lr(sweep, arm):g}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sweep", "final", "report"])
    parser.add_argument("--budget-s", type=float, default=float("inf"), help="stop starting runs after this time")
    args = parser.parse_args()
    if args.command == "report":
        print(report(load()))
    else:
        run_phase(args.command, args.budget_s)


if __name__ == "__main__":
    main()
