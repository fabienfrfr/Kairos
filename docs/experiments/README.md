# Experiments: how to reproduce them

Reproducible logs and raw results behind the paper (`docs/paper/`). Every experiment has a script that runs it, a
results file with the raw per-run numbers, and a log section with the interpretation. Scripts are resumable: they skip
finished runs, so a command that stops at a time limit is simply run again.

## Environment (the one used for every reported number)

- CPU only, one core (`nproc` = 1), x86_64, Python 3.12.3, torch 2.15.0 (CPU), transformers 5.23.0.
- Install torch with `bash scripts/setup_limited_sandbox_torch.sh && source ~/torch_env.sh` (never the GPU wheels;
  do not install torch with its dependencies). `torch.optim.Muon` is needed for the Muon arm (verified on 2.15.0).
- Run with `HF_HUB_OFFLINE=1` (the scripts set it); no network access is needed.
- Commands are limited to 300 s in that environment: run in the foreground and use `--budget-s` (see below).

## Determinism

`kairos overfit --seed N` seeds model initialization, step order and masking. It reproduced a run bit for bit on this
machine: `train_adamw.yaml`, seed 0, 1800 steps, `--lr 1e-3` gives a reference CE of 0.2368 and sampling accuracy
`{'dfm_8': 0.5234, 'gen_8': 0.7969, 'dfm_16': 0.5195, 'gen_16': 0.8125}` both before and after the optimizer refactor.
Bit-exactness is not claimed across other hardware, thread counts or torch versions. The runs of the original v1 tables
(`ablations_codec_attention.md` §1-§10) predate this fix and are not reproducible; §12 replaces §1 and the 600-step row of §7.

## Experiments

| Experiment | Command | Runs | Per run | Raw results | Paper section | Log |
|---|---|---:|---|---|---|---|
| Codec and attention tables (600 steps, `p_max=0.3`) | `python scripts/run_convergence_experiments.py tables` | 26 | 65-250 s | `results/tables.jsonl`, `results/curves/` | Results, Table 3 | `ablations_codec_attention.md` §12 |
| Training objective: weight and path (1800 steps) | `... flow` | 12 | 207 s | `results/flow.jsonl` | Results, objective | §11a |
| Same on unseen passages (1000 steps) | `... heldout` | 12 | 95 s | `results/heldout.jsonl` | Results, objective | §11b |
| Curriculum vs none (1800 steps) | `... curriculum` | 4 | 207 s | `results/curriculum.jsonl` | Results, curriculum | §13 |
| Same three at the tuned rate 8e-3 | `... flow --lr 0.008`, `... heldout --lr 0.008`, `... curriculum --lr 0.008` | 12 + 12 + 4 | 200-230 s / 115 s / 195 s | `results/*_lr0.008.jsonl` | Results, objective and curriculum | §11c, §13 |
| Optimizer and noise sampling | `python scripts/run_training_algorithms.py sweep`, then `final` | 42 + 20 | 80 s / 207 s | `results/training_algorithms.jsonl` | Results, training algorithm | `training_algorithms.md` |

(`...` is `python scripts/run_convergence_experiments.py`.) Add `--budget-s 250` to stop starting runs before the 300 s
limit; rerun the same command until it prints `<name> complete`. Print the tables with
`python scripts/run_convergence_experiments.py report` and `python scripts/run_training_algorithms.py report`.
To start from scratch, delete the matching `results/*.jsonl` file (and `results/curves/` for `tables`).

The configs are in `configs/ablations/` (`baseline`, `attn_vanilla`, `codec_*`, `liz2_separate_qkv`, `flow_*`,
`flow_heldout_*`, `train_*`). The exact flags of each run are built by `command()` in the scripts, for example
`kairos overfit configs/ablations/flow_dfm_cubic.yaml --steps 1800 --mask-p-max 1.0 --no-mask-reweight --seed 2 --log-every 0`.

## Checks tying the paper to the data

`tests/test_convergence_experiments_script.py` recomputes the quoted means and standard deviations from the committed
raw results and checks that they appear in the paper; `tests/test_training_algorithms_script.py` tests the sweep selection
rule and the report. Run `pytest -m "not integration" tests/test_convergence_experiments_script.py`.

## Known limits of the measurements

- The sampling accuracy printed by `overfit` is measured on one batch drawn from the training subset. The subset holds
  four different windows of the sentence, so the batch composition changes the accuracy (values cluster at 1.0, 0.75 and
  0.51) and depends on the random state consumed by the arm. Compare samplers within a run, not accuracies across arms.
- Not rerun with seeded runs: the single-run sweeps of `ablations_codec_attention.md` §2-§6, and the 3000-step and
  frozen-projection checks of §7.
- The runs without `--lr` use 1e-3, which was not swept; the same experiments were repeated at the swept rate (`--lr 0.008`,
  separate result files) and `train_*` arms use it too.
- Everything is a memorization or tiny-scale test: one sentence (or 96 short passages), a model of 2-8 M parameters.
