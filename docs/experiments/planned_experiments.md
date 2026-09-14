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
