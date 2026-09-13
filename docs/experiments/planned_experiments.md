# Planned Experiments (Not Yet Run)

Follow-ups to `ablations_codec_attention.md`, prepared but not executed. Commands use
the same tiny-scale setup (`d_model=64`, 16 repeated sentences) unless noted.

## 1. Seed robustness

All results in `ablations_codec_attention.md` use a single seed. Before citing any gap
(e.g. `liz2` vs `vanilla`) in the paper, rerun each of the 5 existing configs at 3 seeds
and report mean/std:

```bash
for cfg in baseline attn_vanilla codec_conv_cpg8 codec_conv_dense codec_patch; do
  for seed in 0 1 2; do
    kairos overfit configs/ablations/$cfg.yaml --steps 600 --mask-p-max 0.3 \
      --no-mask-reweight --seed $seed
  done
done
```

(`--seed` was just added to the CLI/`overfit_with_progress` for this purpose.)

## 2. Depth sweep

Does depth help at all on this task, or does it plateau immediately (as capacity
ablations so far suggest)?

```bash
kairos overfit configs/ablations/baseline.yaml --steps 600 --mask-p-max 0.3 \
  --no-mask-reweight --set model.n_layers=1   # repeat with 2, 4, 8
```

## 3. MoE routing: top-1 vs top-2

The notebook claims (comment, unverified): *"top-1 routing is slow to converge on tiny
overfit-test runs."* New configs `moe_top1.yaml` / `moe_top2.yaml` (7 experts,
`liz2`/`conv` defaults) are ready — build-validated, not yet trained:

```bash
kairos overfit configs/ablations/moe_top1.yaml --steps 600 --mask-p-max 0.3 --no-mask-reweight
kairos overfit configs/ablations/moe_top2.yaml --steps 600 --mask-p-max 0.3 --no-mask-reweight
```

## 4. `mask_reweight_clip` sensitivity

Candidate explanation for the late-run loss spikes seen in the diffusion regime
(`--mask-p-max 1.0 --mask-reweight`, see `ablations_codec_attention.md`'s original
diffusion-regime runs): rare low-`p` batches blow up the `1/p` reweighting term.

```bash
kairos overfit configs/ablations/codec_conv_cpg8.yaml --steps 600 --mask-p-max 1.0 \
  --mask-reweight --set train.mask_reweight_clip=3.0   # repeat with 10.0 (current), 30.0, none
```

## 5. Local attention window (`attnres_block_size`)

Default is `1` (no windowing). Worth checking it doesn't quietly hurt memorization on
short sequences before ever using it for real:

```bash
kairos overfit configs/ablations/baseline.yaml --steps 600 --mask-p-max 0.3 \
  --no-mask-reweight --set model.attnres_block_size=4   # repeat with 8, 16
```

## 6. Codec scales/stride

Defaults: `num_scales=4`, `stride=5` (patch sizes 5/25/125/625). Does the tiny task need
all 4 scales?

```bash
kairos overfit configs/ablations/baseline.yaml --steps 600 --mask-p-max 0.3 \
  --no-mask-reweight --set model.num_scales=2   # repeat with 3, 4 (default)
```

## 7. Attention: shared vs. separate QKV, and DeltaNet alone

Directly tests why `liz2` beat `vanilla` in `ablations_codec_attention.md`: is the
shared Q/K/V/O between SWA and DeltaNet actually helping, or is one branch doing all
the work? Code added: `attn_type="delta_only"` and `liz2_share_qkv=False` (both
build-validated, not yet trained). New configs `attn_delta_only.yaml` /
`liz2_separate_qkv.yaml`:

```bash
kairos overfit configs/ablations/attn_delta_only.yaml   --steps 600 --mask-p-max 0.3 --no-mask-reweight
kairos overfit configs/ablations/liz2_separate_qkv.yaml --steps 600 --mask-p-max 0.3 --no-mask-reweight
```

Compare all four against `baseline` (liz2, shared) and `attn_vanilla` (SWA only) from
`ablations_codec_attention.md`.

## 8. Memory gate (`KairosMemoryGate`)

Simpler than first thought: `train()` already carries the model's own previous-step
cache forward automatically (`prev_cache = cache_params` each step) whenever
`use_memory_gate=True` — no code change needed for the "real self-memory" case. Use
`pipe.train()` on the tiny dataset (a few epochs), not `overfit_test()` (which has no
memory support), comparing:

- **without**: `use_memory_gate=False` (current baseline)
- **with (self)**: `use_memory_gate=True`, no external `memory_bank` — `prev_cache` alone
- **with (random)**: `use_memory_gate=True`, plus a `memory_bank` built from one forward
  pass on random token IDs (uncorrelated content, same shapes) — isolates whether the
  gate's benefit, if any, comes from real recall or just extra computation

No plumbing gap left to fill; just needs a short `train()`-based script instead of the
`overfit`/`kairos` CLI path (which only wraps `overfit_test()` today).

## 9. Sequence length (`max_len`)

```bash
kairos overfit configs/ablations/baseline.yaml --steps 600 --mask-p-max 0.3 \
  --no-mask-reweight --set data.max_len=64   # repeat with 128 (default), 256
```

## 10. Batch size

```bash
kairos overfit configs/ablations/baseline.yaml --steps 600 --mask-p-max 0.3 \
  --no-mask-reweight --set data.batch_size=1   # repeat with 4 (default), 16
```

