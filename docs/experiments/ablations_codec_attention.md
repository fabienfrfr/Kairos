# Codec & Attention Ablations: Tiny-Scale Overfit Sanity Checks

## Reproducibility

Full instructions, commands and the environment are in `docs/experiments/README.md`; this log holds the interpretation.

- **Original runs (§1-§10)**: `kairos 0.4.1.dev46`, `torch 2.14.0+cu130`. They are **not reproducible**: `--seed` then
  only selected the training subset, while initialization, step order and masking used an unseeded global RNG (this, not
  non-deterministic CPU ops, is why `baseline` at seed 0 gave 0.227 in one run and 0.31 in another).
- **Seeded runs (§11-§13 and `training_algorithms.md`)**: `kairos 0.4.1.dev28` (version file of the sandbox), `torch 2.15.0` CPU, Python 3.12.3, transformers 5.23.0, one core.
  `--seed` now also seeds initialization, step order and masking; a repeated run matched bit for bit on this machine.
- Base setup: `d_model=64`, 1 layer unless noted, MoE off unless noted, one sentence split into 4 windows each repeated
  16 times (64 examples; the overfit subset takes 16), 600 steps, fixed mask regime (`mask_p_max=0.3`, no reweighting)
  unless noted. Configs in `configs/ablations/`.

## 1. Seed robustness (v1, 3 unseeded runs; superseded by §12)

| Config | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---:|---:|---:|---:|
| `baseline` (liz2) | 0.227 | 0.258 | 0.576 | 0.354 ± 0.193 |
| `attn_vanilla` | 0.596 | 0.728 | 0.798 | 0.707 ± 0.103 |
| `codec_conv_cpg8` | 0.106 | 0.045 | 0.075 | 0.075 ± 0.031 |
| `codec_conv_dense` | 0.037 | 0.024 | 0.024 | 0.028 ± 0.007 |
| `codec_patch` | 0.019 | 0.016 | 0.018 | 0.018 ± 0.002 |

Historical record: these runs were not reproducible (the seed only chose the subset). The claim that `baseline`
beats `attn_vanilla` on every seed does not survive the seeded re-measurement (§12); the codec ranking does.

```bash
kairos overfit configs/ablations/<name>.yaml --steps 600 --mask-p-max 0.3 --no-mask-reweight --seed <0|1|2>
```

## 2. Depth (`n_layers`, on `baseline`)

| n_layers | min loss | params |
|---:|---:|---:|
| 1 | 0.227 | 1.98M |
| 2 | 0.843 | 3.79M |
| 4 | 0.368 | 7.41M |
| 8* | 1.228 | 14.65M |

\* 300 steps, not 600 (compute budget). **No monotonic trend** — depth doesn't help
memorization here; codec/attention choice dominates, not depth.

## 3. MoE routing: top-1 vs. top-2 (`moe_top1`/`moe_top2`, 7 experts)

| | min loss |
|---|---:|
| top-1 | 0.281 |
| top-2 | 0.272 |

Nearly identical. The notebook's claim that top-1 routing is "slow to converge" is not
supported at this scale — top-1 is essentially as good as top-2 here.

## 4. `mask_reweight_clip` (diffusion regime: `mask_p_max=1.0`, `mask_reweight=True`, on `codec_conv_cpg8`)

| clip | min loss |
|---:|---:|
| 3.0 | 0.139 |
| 10.0 (default) | ~0.08-0.11 (see §1 diffusion-regime runs) |
| 30.0 | 0.249 |

Tighter clipping helps: a looser bound lets rare low-`p` batches' `1/p` reweighting
spike the loss, as hypothesized when the late-run instability was first observed.

## 5. Attention window (`attnres_block_size`, on `n_layers=4`)

Needs `n_layers>1` to have any effect at all — `KairosBackbone`'s block
aggregation windows *across layers*, so it's a structural no-op at `n_layers=1`
(confirmed empirically: identical bit-for-bit output before this was caught).

| block size | min loss |
|---:|---:|
| 4 | 0.386 |
| 8 | 0.386 |
| 16 | 0.386 |

Identical across all three: the effect saturates once `block_size >= n_layers` (4 layers
all fold into one block regardless of how large the block is allowed to be).

## 6. Codec scales (`num_scales`, on `baseline`)

| num_scales | min loss | params |
|---:|---:|---:|
| 2 | 0.430 | 0.96M |
| 3 | 0.308 | 1.44M |
| 4 (default) | 0.227 | 1.98M |

Monotonic: more scales help, at a proportional param cost. No sign of the extra scales
being wasted capacity here.

## 7. Attention: shared vs. separate QKV, and DeltaNet alone

| Variant | min loss (600 steps, 16 sentences) |
|---|---:|
| `liz2` (shared QKV, default) | 0.227-0.369 (run-to-run noise, see §1) |
| `liz2`, `liz2_share_qkv=False` (separate) | 0.610-0.858 |
| `delta_only` (DeltaNet branch alone) | 0.400 |
| `vanilla` (SWA branch alone) | 0.596 |

At first glance, sharing looks like a clear win. **It isn't — this was a correlation,
not a causal architectural effect**, confirmed by 4 follow-up checks
(`liz2_shared_vs_separate_followup_raw.jsonl`, full per-step curves, not just
min/final):

| Check | Shared | Separate | Verdict |
|---|---:|---:|---|
| 600 steps, 16 identical sentences (as above) | 0.369 | 0.610 | shared ahead |
| **3000 steps**, same data | **0.0011** | **0.0011** | **gap gone** - separate was only slower |
| **63 distinct real sentences**, 600 steps | **2.269** | **2.285** | **gap gone** on real data |
| **QKV frozen after step 100** (removes the shared path's double gradient signal) | 0.705 | **0.521** | **reversed** - separate wins |

The apparent advantage was a convergence-speed artifact of a too-easy, too-short toy
setup (shared QKV gets gradient signal from both the SWA and DeltaNet branches every
step, an effective higher learning rate early on) - not evidence that sharing is
architecturally better. With enough steps, diverse data, or the gradient-count confound
controlled for, the gap closes or reverses. Neither branch alone matches the shared
hybrid at 600 steps either, but that comparison inherits the same caveat.

**The same check extends to `liz2` vs. `attn_vanilla` from §1.** At 3000 steps,
`attn_vanilla` alone reaches `min_loss=0.0014`, matching `liz2`'s `0.0011` from the
shared-QKV long run above - the "liz2 beats vanilla" result in §1's table is the same
convergence-speed artifact, not evidence the hybrid has more capacity than SWA alone at
this scale.

## 8. Sequence length (`max_len`, on `baseline`)

| max_len | min loss |
|---:|---:|
| 64 | 1.129 |
| 128 (default) | 0.227 |
| 256 | 0.035 |

More context helps substantially — no sign of this task being long-context-agnostic.

## 9. Batch size (on `baseline`)

| batch_size | min loss |
|---:|---:|
| 1 | 0.890 |
| 4 (default) | 0.227 |
| 16 (all data, one batch) | 0.032 |

Larger batches converge faster in step-count terms here (expected: `n_examples=16`, so
`batch_size=16` sees the full set every step).

## 10. Memory gate (`KairosMemoryGate`, via `pipe.train()`, 3 epochs, not `overfit_test`)

Note: absolute losses aren't comparable to the sections above (different training loop,
far fewer steps - 3 epochs x 4 batches).

| Variant | min loss | final loss |
|---|---:|---:|
| without (`use_memory_gate=False`) | 7.512 | 12.927 |
| with, self (`prev_cache` only) | 7.453 | 12.893 |
| with, random bank | 7.423 | 9.752 |

All three are close; 12 steps total is too few to draw a conclusion either way. Would
need many more steps/epochs to say whether the memory gate's content matters.

## 11. Flow-matching training arms (`flow_*`, `mask_p_max=1.0`)

Arms differ in the corruption path and loss weight: `flow_mdlm_control` (linear, `1/p` clipped at 10),
`flow_dfm_linear` and `flow_dfm_cubic` (plain CE). Training losses are not comparable across arms, so each run reports
common yardsticks computed before the weights are restored: a *reference CE* (plain CE, linear path, `p ~ U(0, 1)`) and a
*sampling accuracy* (exact-token rate over a 64-token continuation of a 32-token prompt, from uniform noise) for the arm's
`dfm_sample` and the entropy-bound `generate()` at 8 and 16 steps. An earlier 600-step batch (6 runs per arm, seeds not yet
reproducible) is superseded: the seed only drove the subset choice; it now also seeds init, step order and masking
(`test_overfit_test_is_reproducible_for_a_fixed_seed`).

### 11a. Memorization of one sentence (1800 steps, seeds 0-3, paired by seed)

| Arm | reference CE | `dfm_sample` 8 | `generate` 8 | `dfm_sample` 16 | `generate` 16 |
|---|---:|---:|---:|---:|---:|
| `flow_mdlm_control` | 0.140 ± 0.099 | 0.812 ± 0.221 | 0.888 ± 0.131 | 0.760 ± 0.196 | 0.892 ± 0.127 |
| `flow_dfm_linear` | 0.132 ± 0.137 | 0.660 ± 0.217 | 0.831 ± 0.188 | 0.700 ± 0.219 | 0.762 ± 0.186 |
| `flow_dfm_cubic` | 0.145 ± 0.078 | 0.797 ± 0.169 | 0.906 ± 0.108 | 0.762 ± 0.178 | 0.935 ± 0.126 |

- Reference CE, arm minus control per seed: linear -0.008 ± 0.065 (lower in 3/4), cubic +0.005 ± 0.085 (lower in 1/4).
  No training-objective effect is detectable.
- `generate()` minus `dfm_sample` over the 12 runs: +0.119 ± 0.147 at 8 steps (higher in 10, tie in 1, lower in 1) and
  +0.122 ± 0.120 at 16 steps (higher in 11, tie in 1, lower in 0).
- Caveats: one memorized sentence (training-set measures), tiny model, 4 seeds; `dfm_sample` has no corrector and no
  time conditioning. `generate()` returns identical accuracy at 8 and 16 steps in some runs (early stopping, not checked).

### 11b. Held-out passages (1000 steps, seeds 0-3, `--n-examples 48 --heldout 24`)

Configs `flow_heldout_*`: 96 distinct documentation passages (<= 100 characters, one window each); training uses 48,
the reference CE is also measured on 24 passages never seen. Nothing is memorized at this budget.

| Arm | train CE | held-out CE | gap | `dfm_sample` 16 | `generate` 16 |
|---|---:|---:|---:|---:|---:|
| `flow_heldout_mdlm_control` | 3.078 ± 0.036 | 3.494 ± 0.026 | 0.416 | 0.062 ± 0.012 | 0.148 ± 0.019 |
| `flow_heldout_dfm_linear` | 3.051 ± 0.030 | 3.526 ± 0.025 | 0.475 | 0.065 ± 0.009 | 0.155 ± 0.019 |
| `flow_heldout_dfm_cubic` | 2.997 ± 0.029 | 3.571 ± 0.030 | 0.574 | 0.064 ± 0.008 | 0.155 ± 0.021 |

- Held-out CE, arm minus control per seed: linear +0.032 ± 0.009 and cubic +0.077 ± 0.010, **worse in 4/4 seeds** for
  both. The DFM arms fit the training set as well or better (cubic lowest) but generalize slightly less.
- Sampling accuracy is near the floor for every arm; `generate()` is again above `dfm_sample` (about 0.15 vs 0.06).
- Trivial references on the same kind of split (random 48/24 passage splits, 4 seeds, characters, no masking, add-one or
  0.1-interpolated smoothing): unigram 3.446 nats, bigram 2.927, uniform over 257 symbols 5.549. All three arms (3.49-3.57)
  are near the unigram level and far from the bigram one. The metrics differ (model CE is measured on masked positions
  with `p ~ U(0, 1)`), so this is indicative only.
- Caveats: the effect is small (0.03-0.08 nats on about 3.5), 4 seeds, one tiny model, short training. The `1/p` weight also
  changes the effective gradient scale (Adam, clip 10), which is not controlled here, so the cause is not established.

```bash
kairos overfit configs/ablations/flow_dfm_linear.yaml --steps 1800 --mask-p-max 1.0 --no-mask-reweight --seed <0..3>
kairos overfit configs/ablations/flow_heldout_dfm_cubic.yaml --steps 1000 --n-examples 48 --heldout 24 \
  --mask-p-max 1.0 --no-mask-reweight --seed <0..3>
```

### 11c. Same experiments at the tuned learning rate (8e-3)

The §11a/§11b runs use `--lr 1e-3`, which was not swept. They were repeated with `--lr 0.008` (the rate selected for AdamW in
`training_algorithms.md`): `python scripts/run_convergence_experiments.py flow --lr 0.008` and `... heldout --lr 0.008`
(results in `results/flow_lr0.008.jsonl`, `results/heldout_lr0.008.jsonl`).

| Arm | memorization CE (1800 steps) | held-out CE | train CE (held-out run) |
|---|---:|---:|---:|
| control `1/p` | 0.082 ± 0.106 | 3.591 ± 0.045 | 3.074 ± 0.034 |
| DFM linear | 0.073 ± 0.052 | 3.675 ± 0.064 | 2.988 ± 0.062 |
| DFM cubic | 0.034 ± 0.027 | 3.805 ± 0.079 | 2.913 ± 0.144 |

- Memorization, arm minus control per seed: linear -0.009 ± 0.072 (lower on 1/4), cubic -0.048 ± 0.091 (lower on 2/4): no
  detectable effect, as at 1e-3.
- Held-out CE, arm minus control: linear +0.084 ± 0.019, cubic +0.214 ± 0.123, **worse on 4/4 seeds**; train CE lower on 4/4
  (-0.086 ± 0.057, -0.161 ± 0.144). Train-held-out gap 0.517 (control), 0.687, 0.892 (0.416, 0.475, 0.574 at 1e-3).
- All three held-out values (3.59-3.81) are above the smoothed unigram reference (3.446); indicative only (metrics differ).
- Sampling accuracy at 8e-3 saturates: `generate` and `dfm_sample` tie in 10 (8 steps) and 11 (16 steps) of 12 runs, so no
  sampler comparison is made at this rate. Accuracy across arms is also batch-dependent (see `README.md`).

## 12. Seeded re-measurement of §1 and §7 (600 steps, `--mask-p-max 0.3 --no-mask-reweight`, `--save-curve`)

Since `--seed` now seeds init, step order and masking, §1 and the 600-step row of §7 were re-run (curves saved with
`--save-curve`). Seeds 0-4, except dense `conv` and `patch` (seeds 0-2, about 250 s per dense run).

| Config | n | min loss (per seed) | min ± sd | last-100 mean | steps to <=1.0 (per seed) |
|---|---:|---|---:|---:|---|
| `baseline` (liz2, depthwise) | 5 | 0.101, 0.100, 0.293, 0.070, 0.803 | 0.273 ± 0.309 | 0.387 | 187, 192, 234, 204, 344 |
| `attn_vanilla` | 5 | 0.219, 0.584, 0.289, 0.144, 0.443 | 0.336 ± 0.177 | 0.607 | 303, 301, 234, 244, 282 |
| `codec_conv_cpg8` | 5 | 0.058, 0.037, 0.070, 0.033, 0.072 | 0.054 ± 0.018 | 0.067 | 185, 175, 156, 170, 203 |
| `codec_conv_dense` | 3 | 0.025, 0.024, 0.023 | 0.024 ± 0.001 | 0.028 | 120, 112, 105 |
| `codec_patch` | 3 | 0.017, 0.018, 0.017 | 0.017 ± 0.001 | 0.019 | 86, 88, 85 |
| `liz2_separate_qkv` | 5 | 0.326, 0.182, 0.199, 0.065, 0.121 | 0.179 ± 0.098 | 0.272 | 268, 260, 193, 182, 204 |

- Codec ranking `patch < dense < cpg8 < depthwise` holds on every seed measured.
- `liz2` vs `vanilla`: min loss lower on 3/5 seeds (0, 1, 3), higher on 2; steps to <=1.0 faster on 3, tie on 1, slower on 1.
  The v1 gap (0.354 vs 0.707, 3 unseeded runs) is not reproduced; means overlap heavily (one `liz2` seed at 0.803).
- Shared vs separate QKV: shared min loss lower on 2/5 seeds only; means 0.273 vs 0.179. The §7 first-row effect
  (0.369 vs 0.610) is not reproduced. Cause of the earlier gap not isolated (unseeded runs, seed drove only the subset).
- Not re-run: §1 thresholds below 1.0 for the non-baseline arms, the 3000-step and frozen-QKV checks of §7, all sweeps of §2-§6.
- Raw curves are not versioned; they are regenerated with `kairos overfit <config> --steps 600 --mask-p-max 0.3
  --no-mask-reweight --seed <n> --save-curve <path>`.

## 13. Curriculum vs no curriculum (1800 steps, seeds 0-3, `flow_mdlm_control`)

Curriculum arm: `kairos overfit configs/ablations/flow_mdlm_control.yaml --steps 1800 --seed <n>` (stages walked, as in the
default). No-curriculum arm: the same with `--mask-p-max 1.0 --mask-reweight` (§11a `flow_mdlm_control`). Reference CE
(plain, linear path, `p ~ U(0, 1)`):

| Arm | seed 0 | seed 1 | seed 2 | seed 3 | mean ± sd |
|---|---:|---:|---:|---:|---:|
| curriculum | 0.219 | 0.015 | 0.065 | 0.204 | 0.126 ± 0.102 |
| no curriculum | 0.237 | 0.052 | 0.059 | 0.213 | 0.140 ± 0.099 |

- Paired difference (curriculum - none): -0.014 ± 0.018, lower on 3/4 seeds: no detectable effect.
- Sampling (16 steps, curriculum arm): `generate()` 0.839 ± 0.115, `dfm_sample` 0.795 ± 0.138.
- No instability without the curriculum was observed, contrary to the informal motivation in the paper. Memorization on one
  sentence only.

Repeated at `--lr 0.008` (`... curriculum --lr 0.008`, `results/curriculum_lr0.008.jsonl`): curriculum 0.067 ± 0.099 against
0.082 ± 0.106 without (the `flow_mdlm_control` run of §11c); paired difference -0.015 ± 0.149, lower with the curriculum on 2/4
seeds: no detectable effect at either rate.

## Takeaways

- **Codec choice dominates**; depth, MoE top-k, and QKV-sharing barely move the needle
  once enough steps or real data are used (§2, §3, §7).
- **§7 is a worked example of correlation vs. causation**: a striking single-run,
  single-config result (shared QKV converging faster) did not survive 3 independent
  controls (more steps, more diverse data, a gradient-signal control) - it reversed
  under the strictest one. Treat any single-run architecture claim from this sandbox as
  a hypothesis to check, not a conclusion.
- **None of this changes production defaults** (`codec_mode="conv"`, `attn_type="liz2"`,
  `codec_conv_channels_per_group=None`). These are tiny memorization sanity checks, not
  generalization benchmarks - `max_len`/`batch_size`/depth results in particular may not
  transfer to real, diverse data at scale.
- Memory gate and `n_layers=8` remain inconclusive (too few steps / too little compute
  budget respectively) - flagged, not resolved.
- Flow-matching arms (§11): no training-objective effect on memorization; on unseen passages the `1/p` control is slightly better in 4/4 seeds; `dfm_sample` is below `generate()` (higher in 11 of 12 runs at 16 steps in 11a; mean 0.06 vs 0.15 in 11b).
- Seeded re-measurement (§12): the `liz2`-vs-`vanilla` and shared-vs-separate QKV gaps of §1/§7 do not replicate; the codec ranking does.
- Curriculum control (§13): no detectable effect of the curriculum over 4 seeds; no instability without it at this scale.
