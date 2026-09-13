Tiny-scale (d_model=64, 1 layer, MoE off unless named) overfit-test configs used to
sanity-check architecture ablations. See docs/experiments/ablations_codec_attention.md
for completed results, and docs/experiments/planned_experiments.md for prepared-but-not-run
follow-ups (seed robustness, depth sweep, MoE routing, mask_reweight_clip, attention
window, codec scales).

Run with e.g.:
    kairos overfit configs/ablations/baseline.yaml --steps 600 --mask-p-max 0.3 --no-mask-reweight
