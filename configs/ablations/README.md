Tiny-scale (d_model=64, 1 layer, MoE off unless named) overfit-test configs used to
sanity-check architecture ablations. See docs/experiments/ablations_codec_attention.md
for full results (all sweeps run) and docs/experiments/planned_experiments.md for the
list of what was covered.

Flow-matching arms (`flow_*.yaml`, see docs/experiments/flow_matching.md) and training-algorithm arms
(`train_*.yaml`, see docs/experiments/training_algorithms.md) need `--mask-p-max 1.0`. The exact commands of every
experiment, the environment and the results files are listed in docs/experiments/README.md; the scripts
`scripts/run_convergence_experiments.py` and `scripts/run_training_algorithms.py` run them.

Run with e.g.:
    kairos overfit configs/ablations/baseline.yaml --steps 600 --mask-p-max 0.3 --no-mask-reweight
