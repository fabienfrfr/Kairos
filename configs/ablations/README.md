Tiny-scale (d_model=64, 1 layer, MoE off unless named) overfit-test configs used to
sanity-check architecture ablations. See docs/experiments/ablations_codec_attention.md
for full results (all sweeps run) and docs/experiments/planned_experiments.md for the
list of what was covered.

Run with e.g.:
    kairos overfit configs/ablations/baseline.yaml --steps 600 --mask-p-max 0.3 --no-mask-reweight
