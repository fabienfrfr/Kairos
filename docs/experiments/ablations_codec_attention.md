# Codec & Attention Ablations: Tiny-Scale Overfit Sanity Checks

## Reproducibility

- **kairos**: `0.4.1.dev46` (`python -c "import kairos; print(kairos.__version__)"`)
- **torch**: `2.14.0+cu130` (project requires `torch>=2.10.0`)
- Base setup: `d_model=64`, 1 layer unless noted, MoE off unless noted, 16 identical
  repeated sentences, 600 steps, fixed mask regime (`mask_p_max=0.3`, no reweighting)
  unless noted. Configs in `configs/ablations/`.
- **Known limitation**: even at a fixed seed, results vary run to run (non-deterministic
  CPU ops) — e.g. `baseline` at seed 0 measured 0.227 in one run, 0.31 in another. Single
  numbers below (everything but §1) should be read as indicative, not exact; §1
  quantifies the actual noise floor.

## 1. Seed robustness (3 seeds, `--seed 0/1/2`)

| Config | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---:|---:|---:|---:|
| `baseline` (liz2) | 0.227 | 0.258 | 0.576 | 0.354 ± 0.193 |
| `attn_vanilla` | 0.596 | 0.728 | 0.798 | 0.707 ± 0.103 |
| `codec_conv_cpg8` | 0.106 | 0.045 | 0.075 | 0.075 ± 0.031 |
| `codec_conv_dense` | 0.037 | 0.024 | 0.024 | 0.028 ± 0.007 |
| `codec_patch` | 0.019 | 0.016 | 0.018 | 0.018 ± 0.002 |

`baseline` beats `attn_vanilla` on all 3 individual seeds (paired comparison), despite
overlapping mean±std ranges — the gap is real, not a single-seed fluke. Codec ranking
(`patch` < `conv-dense` < `conv-cpg8`) is stable across all seeds too.

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

Needs `n_layers>1` to have any effect at all — `KairosDiffusionBackbone`'s block
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
