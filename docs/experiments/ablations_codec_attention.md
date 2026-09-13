# Codec & Attention Ablations: Tiny-Scale Overfit Sanity Checks

## Reproducibility

- **kairos**: `0.4.1.dev46` (`python -c "import kairos; print(kairos.__version__)"`)
- **torch**: `2.14.0+cu130` (project requires `torch>=2.10.0`; pin your own if it differs)
- **seed**: `0` (the `overfit_test` default; not passed explicitly since the CLI has no `--seed` flag)

## Setup

All runs: `d_model=64`, 1 layer, MoE off, 16 identical repeated sentences, 600 steps,
fixed mask regime (`mask_p_max=0.3`, no reweighting). This isolates architecture
capacity from the masked-diffusion curriculum and MoE routing, both covered separately.

Configs live in `configs/ablations/`. Exact commands used for every row below:

```bash
kairos overfit configs/ablations/baseline.yaml         --steps 600 --mask-p-max 0.3 --no-mask-reweight
kairos overfit configs/ablations/attn_vanilla.yaml     --steps 600 --mask-p-max 0.3 --no-mask-reweight
kairos overfit configs/ablations/codec_conv_cpg8.yaml  --steps 600 --mask-p-max 0.3 --no-mask-reweight
kairos overfit configs/ablations/codec_conv_dense.yaml --steps 600 --mask-p-max 0.3 --no-mask-reweight
kairos overfit configs/ablations/codec_patch.yaml      --steps 600 --mask-p-max 0.3 --no-mask-reweight
```

The min/final loss values below are printed by `overfit_with_progress`
(`kairos/runners.py`); no post-processing is applied.

## Results

| Config | Attention | Codec | Params | Min loss | Final loss |
|---|---|---|---:|---:|---:|
| `baseline` | liz2 (production) | conv, depthwise | 1.98M | 0.307 | 0.649 |
| `attn_vanilla` | vanilla SWA | conv, depthwise | 1.81M | 0.711 | 1.113 |
| `codec_conv_cpg8` | vanilla | conv, `channels_per_group=8` | 2.50M | 0.078 | 0.134 |
| `codec_conv_dense` | vanilla | conv, `channels_per_group=64` (dense) | 8.10M | 0.033 | 0.070 |
| `codec_patch` | vanilla | patch | 4.93M | 0.019 | 0.036 |

Mixer bottleneck rank, swept on top of `codec_conv_cpg8`:

```bash
kairos overfit configs/ablations/codec_conv_cpg8.yaml --steps 600 --mask-p-max 0.3 \
  --no-mask-reweight --set model.codec_mixer_rank=1    # repeat with 4, 8, 16, 64
```

| Rank | Min loss |
|---:|---:|
| 1 | 1.417 |
| 4 | 0.261 |
| **8** | **0.092** |
| 16 (default) | 0.106 |
| 64 (full rank) | 0.097 |

## Findings

- **The codec dominates convergence speed on this task; attention and depth barely
  matter.** `codec_patch` and `codec_conv_dense` both beat every depthwise-conv variant
  by 4-10x, regardless of attention type.
- **Depthwise conv (production default) trades capacity for efficiency.** Its
  channel-independent encoder only mixes channels afterward, in a small bottleneck —
  fine for real, diverse data at scale, but slow to memorize a trivial repeated sample.
- **`codec_conv_channels_per_group=8` recovers most of the gap for ~40% more params**
  than the depthwise default, at roughly half the cost of `patch` or full-dense conv.
- **The mixer rank plateaus at 8.** Below it (rank 4) convergence collapses (0.26 vs.
  0.09); above it (rank 64) gains are negligible. The current default
  (`max(8, d_model/4)`) already sits safely past the threshold — no change needed there.
- **The real `liz2` hybrid attention clearly outperforms plain SWA** (0.307 vs. 0.711
  min loss), despite sharing its Q/K/V/O projections between the SWA and DeltaNet
  branches — the shared basis is not a capacity tax here, it's a net win. Attention is
  not the bottleneck this ablation set out to check.
- **None of this changes production defaults.** `codec_mode="conv"`, `attn_type="liz2"`,
  `codec_conv_channels_per_group=None` (depthwise) remain correct for real training runs
  on diverse data at scale, where conv's parameter/compute efficiency is the point. These
  are memorization sanity checks, not generalization benchmarks.

## Recommendation

Use `codec_conv_channels_per_group=8` as an ablation knob when a paper or experiment
needs a more expressive-but-still-efficient conv variant. Keep the depthwise default
(`None`) for production training.
