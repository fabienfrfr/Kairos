# Training algorithms: optimizer and noise sampling (seeded, tiny scale)

Paper: Results, "Convergence by training algorithm". Environment and rules: `docs/experiments/README.md`.

## Question

Does the training algorithm change how fast the tiny network memorizes one sentence, with the architecture and the
objective fixed (full masking range, `1/p` weight clipped at 10, linear path)? Arms, each a config in `configs/ablations/`:

| Arm | Change |
|---|---|
| `train_adamw` | control: AdamW defaults (betas 0.9/0.999, weight decay 0.01), cosine without warmup, uniform `t` |
| `train_adamw_tuned` | betas (0.9, 0.95), weight decay 0.1, 5% linear warmup |
| `train_muon_hybrid` | `torch.optim.Muon` (`adjust_lr_fn="match_rms_adamw"`) on 2D backbone matrices except routers; AdamW on all other parameters |
| `train_stratified` | one `t` per stratum of width `1/B` (batch of 4 rows) |
| `train_antithetic` | `(t, 1-t)` pairs across rows carrying different sequences (not the complementary-mask scheme of the literature) |

Implementation: `make_optimizer`, `make_scheduler` and `sample_noise_time` in `kairos/trainer.py` (tests in
`tests/test_trainer.py`, `tests/test_pipeline.py`).

## Protocol

1. **Sweep** (`python scripts/run_training_algorithms.py sweep`): AdamW, tuned AdamW and Muon over
   `5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2`, seeds 9 and 10, 600 steps, `--mask-p-max 1.0 --mask-reweight`. The rate with
   the lowest mean reference CE is kept (ties go to the lower rate). The grid was first `5e-4..8e-3`, where every arm chose
   its upper edge, so it was extended once to bracket the optimum (the earlier runs are part of the same file).
2. **Final** (`... final`): five arms, seeds 0-3, 1800 steps, the selected rate (`8e-3` for all three optimizers; the two
   noise-sampling arms reuse the AdamW rate).
3. **Report** (`... report`): reference CE (plain CE, linear path, `p ~ U(0, 1)`, measured on the memorized subset before
   the weights are restored), paired differences to `train_adamw`.

Raw results: `docs/experiments/results/training_algorithms.jsonl`.

## Results

Sweep, mean reference CE over seeds 9 and 10 (600 steps); selected rate 8e-3 for all three:

| Arm | 5e-4 | 1e-3 | 2e-3 | 4e-3 | 8e-3 | 1.6e-2 | 3.2e-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train_adamw` | 1.127 | 0.864 | 0.311 | 0.113 | 0.054 | 0.431 | 0.121 |
| `train_adamw_tuned` | 1.284 | 0.897 | 0.614 | 0.162 | 0.058 | 0.085 | 0.658 |
| `train_muon_hybrid` | 1.205 | 0.535 | 0.148 | 0.133 | 0.050 | 0.441 | 0.125 |

Final runs (1800 steps, 8e-3), reference CE per seed:

| Arm | seed 0 | seed 1 | seed 2 | seed 3 | mean ± sd | paired difference to AdamW |
|---|---:|---:|---:|---:|---:|---|
| `train_adamw` | 0.2378 | 0.0009 | 0.0381 | 0.0510 | 0.082 ± 0.106 | --- |
| `train_adamw_tuned` | 0.0702 | 0.0002 | 0.0015 | 0.1354 | 0.052 ± 0.065 | -0.030 ± 0.105 (lower on 3/4) |
| `train_muon_hybrid` | 0.1186 | 0.0005 | 0.0565 | 0.0338 | 0.052 ± 0.050 | -0.030 ± 0.061 (lower on 3/4) |
| `train_stratified` | 0.0742 | 0.0867 | 0.0043 | 0.1088 | 0.069 ± 0.045 | -0.013 ± 0.112 (lower on 2/4) |
| `train_antithetic` | 0.1809 | 0.2891 | 0.0516 | 0.0211 | 0.136 ± 0.124 | +0.054 ± 0.159 (lower on 2/4) |

- No arm is separated from AdamW: every paired difference is smaller than its standard deviation. The seed spread (reference
  CE near 1e-3 on some seeds, 0.05-0.29 on others for the same arm) is as large as the differences between arms.
- At their best rate the three optimizers are indistinguishable in the sweep (0.050-0.058).
- The sweep is noisy over two seeds (AdamW: 0.121 at 3.2e-2 but 0.431 at 1.6e-2).
- Generation accuracy is **not** compared across arms: the batch it is measured on depends on the random state the arm
  consumed (uniform arms share batches, the stratified and antithetic arms do not), so the values (1.0, 0.75, 0.51 clusters)
  reflect which windows were drawn. A fix would evaluate a fixed set of rows; it was not applied so that the committed
  numbers stay reproducible with the committed code.
- The rate of 1e-3 used for `flow_*`, `flow_heldout_*` and the curriculum control was not swept. At 600 steps AdamW scores
  0.864 at 1e-3 against 0.054 at 8e-3; at 1800 steps on seed 0 it scores 0.2368 at 1e-3 and 0.2378 at 8e-3
  (`kairos overfit configs/ablations/train_adamw.yaml --steps 1800 --mask-p-max 1.0 --mask-reweight --seed 0 --lr 1e-3`).

## Not done

Complementary-mask (MIRROR-style) antithetic masking, weight averaging (EMA), other optimizers (Lion, Schedule-Free,
SOAP, NorMuon), a sweep of weight decay and warmup, repeating the objective and curriculum sections at a tuned rate.

## Context from the literature

Muon: Jordan et al. 2024; at multi-billion scale about 2x AdamW compute efficiency (Liu et al. 2025, arXiv 2502.16982).
Variance reduction for masked diffusion training (arXiv 2511.18159) and clipped noise schedules (BD3-LM, arXiv 2503.09573)
motivate the noise-sampling arms. These results are at tiny scale and do not test those claims.
