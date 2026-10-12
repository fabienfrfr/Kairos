# Planned Experiments

All items below have been run. Results are in `ablations_codec_attention.md` (§1-10).
Kept here as the historical record of what was planned and why, and as templates for
future sweeps.

1. Seed robustness (3 seeds x 5 configs) - done, §1.
2. Depth sweep (`n_layers` 1/2/4/8) - done, §2 (`n_layers=8` at reduced steps).
3. MoE routing top-1 vs top-2 - done, §3.
4. `mask_reweight_clip` sensitivity - done, §4.
5. Attention window (`attnres_block_size`) - done, §5 (required `n_layers=4`, not the
   original 1-layer baseline - see note there).
6. Codec scales (`num_scales`) - done, §6.
7. Attention: shared vs. separate QKV, DeltaNet alone - done, §7.
8. Memory gate (`KairosMemoryGate`) - done, §10 (via `pipe.train()`, inconclusive at the
   step budget used).
9. Sequence length (`max_len`) - done, §8.
10. Batch size - done, §9.

## Still open

- `n_layers=8` and the memory gate both need a larger compute/step budget than this
  sandbox allowed to give a conclusive result.
- All non-seed-robustness sections (§2, §4-10) are single-run; §1 shows the noise floor
  at this scale is large enough (+/-0.03 to +/-0.19 depending on config) that any of
  these could shift with a different seed. Worth a 3-seed rerun before citing precise
  numbers.

## Not yet covered by the ablation study (from the paper's old "Anticipated Results")

These three were specified in an earlier draft of `docs/paper/kairos_paper.tex` before
any experiment had been run. They are real gaps, not covered by the tiny-scale ablation
study above, and still worth running:

11. **Per-modality loss** (`pipe.check_per_modality_loss()`) - mean masked CE loss
    broken down by modality (text/image/audio/video/lidar/control). Needs real
    multimodal data; every ablation config so far is text-only. A large, isolated gap
    on one modality would flag a codec-scale or tokenizer mismatch for that stream
    specifically.
12. **Instrumented compute report** (`pipe.summary(benchmark=True)`,
    `pipe.memory_report()`, `pipe.profile()`) - measured tok/s, peak memory, and
    per-module step time on real hardware, as opposed to the roofline model's
    theoretical upper bounds (`docs/paper/kairos_paper.tex`, \S\ref{sec:compute}). **Partially
    done**: on `baseline.yaml` (1.98M params) in this CPU-only sandbox, measured model
    memory 7.9MB, optimizer memory 8.6MB, total 64.1MB, avg step time 96.0ms (eager
    attention, DeltaNet/causal-conv1d on the `torch_fallback` path -- no
    `flash-linear-attention` installed here). Not representative of the target design
    point or of GPU throughput; still needed on A100/H100-class hardware with the fused
    kernels installed.
13. **Per-stage overfit breakdown** - first-loss and tail-mean loss reported separately
    for the MAE / transition / diffusion stages of a single `overfit_test` run (rather
    than only min/final over the whole run, as in the ablation study above). **Done**:
    on `baseline.yaml`, 600 steps, seed 0 (`pipe.curriculum_bounds` gives the exact
    stage boundaries): MAE first=8.663, tail-mean=1.485 (n=200); transition
    first=1.419, tail-mean=2.007 (n=200, the reweighting switch-on bump the paper's
    Methods section describes); diffusion first=1.873, tail-mean=1.555 (n=200). One
    run, not yet repeated across seeds.

## Methodology note: measure steps-to-threshold, not loss-at-fixed-steps

Comparing `min_loss` at a fixed step count (as items 1-10 mostly do) conflates final
capacity with convergence speed - this is exactly what made the shared-vs-separate-QKV
and `liz2`-vs-`vanilla` results look like capacity effects when they were speed effects
(`ablations_codec_attention.md` §7). Where full curves are saved
(`liz2_shared_vs_separate_followup_raw.jsonl`), computing steps-to-reach-threshold
instead is free and more informative: it showed the shared/separate gap is concentrated
in the 0.5-0.1 loss range (+9 to +14% steps) and vanishes elsewhere, rather than being a
uniform gap as the single-step-count comparison implied. Prefer this metric when
re-running any of the single-run sweeps above.

## 14. Classic hyperparameter sweeps (not yet run)

Standard ML sanity checks, absent so far from every sweep in this project:

- **Learning rate** - never swept; `lr` is already a parameter of `pipe.overfit_test()`.
- **Gradient clipping** (`train_config.grad_clip`, default `1.0`) - classic stability
  check, existing config field, easy `--set train.grad_clip=X`.
- **`mask_eps`** - fixed at `1e-3` in every config so far, never swept.
- **Weight decay** - not exposed (`torch.optim.AdamW(..., lr=lr)` in
  `kairos/pipeline.py`, no `weight_decay` kwarg); needs a small code change to test.
- **LR schedule shape** - `overfit_test` hardcodes `CosineAnnealingLR`; comparing
  against a constant LR needs exposing the schedule choice, not just a `--set`.

Deliberately not planned: a classic MoE load-balancing auxiliary loss ablation. This
codebase doesn't use one - routing balance is controlled by a learned bias
(`moe_bias_update_rate`), already explored in `kairos/diagnostics.py` and unrelated to
this sweep's configs.
