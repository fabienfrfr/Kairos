# Continue on a PC with an NVIDIA RTX GPU

What was checked on CPU without the datasets is in `pretraining_readiness.md`. This guide is the continuation on a GPU with
network access: audit the real data, measure the GPU, check the learning rate on real rows, then train a sample.
Nothing here has been run on a GPU yet; the CPU numbers it quotes are for comparison only.

## 1. Install (Linux or WSL2)

```bash
git clone <your clone of the project> && cd kairos
uv sync --extra fast-attn --group dev        # CUDA PyTorch from PyPI + fused kernels (README: NVIDIA/CUDA)
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "HF_TOKEN=hf_..." > .env                 # only if the dataset repos are private or gated
```

- Without `flash-linear-attention` (the `fast-attn` extra) the DeltaNet kernel falls back to a pure-PyTorch path that the
  library itself calls "correct but much slower"; the CPU runs here used that path. The extra needs Triton, which the project
  declares for Linux only: use WSL2 on Windows. A native-Windows setup is not covered by the project; if you try it, install a
  CUDA build of PyTorch from `download.pytorch.org/whl/cu<version>` and pass `--no-compile` to the scripts below.
- Precision is chosen automatically: bf16 on GPUs of compute capability 8.0 or higher (RTX 30/40 series), fp16 with a gradient
  scaler otherwise (RTX 20 series). The scripts print it under `device`.

## 2. Download the two datasets

```bash
make fetch-datasets          # python scripts/fetch_datasets.py -> data/keep-it-simple, data/keep-it-simple-multimodal
```

If the analysis sandbox has to see the data too, mirror it to GitHub first: `docs/experiments/share_datasets.md`
(`make mirror-datasets`, then `python scripts/fetch_datasets.py --from-github <user>/<repo>` on the other side).

## 3. Audit the real rows (no training, a few seconds)

```bash
uv run python scripts/check_pretraining_readiness.py data --source hf --max-per-modality 200 --scale-factors '{}'
uv run python scripts/check_pretraining_readiness.py data --source hf --max-per-modality 200      # with the sample scale factors
```

It prints, per modality, the examples, the windows of 1024 tokens one example needs, tokens per example and the padding share.
On the synthetic rows with the builder's sizes an image took 3 133 tokens, audio 4 024, video 4 735, lidar 2 412 and control
16 012 at the default scale factors; check whether the real rows match, and whether `modality_scale_factors` (or a larger
`data.max_len`) is needed so that one example fits a window.

## 4. Measure the GPU (about a minute)

```bash
make gpu-bench               # or: ... bench --source hf --batch-sizes 4,8,16,32 --steps 50 --out results.json
```

It reports seconds per optimizer step, tokens per second and peak GPU memory for each batch size, and `oom` where it does not
fit. The CPU figure to beat is about 1 400-2 300 tokens/s (reference kernel, one core). `docs/readme.md` asks for a measured
tokens/s rather than the theoretical bound: this is that measurement. Choose the largest batch size that fits and set it with
`--set data.batch_size=N`.

## 5. Check the learning rate on real rows

```bash
for lr in 0.003 0.008 0.016; do
  uv run python scripts/check_pretraining_readiness.py train --source hf --max-per-modality 100 --batch-size 8 \
    --lr $lr --out results_lr$lr.json
done
```

Each run trains the three curriculum stages of `configs/pretrain.yaml` on the sampled rows and prints the plain cross-entropy
per modality before and after, on training windows and on the held-out split of the real data. Compare the held-out values and
the `skipped_nonfinite` count (a nonzero count means the rate is too high). The synthetic-row sweep gave 4.98 (3e-3), 4.86 (8e-3)
and 4.86 (1.6e-2) from 5.70; `configs/pretrain.yaml` ships 3e-4, the slowest of those tried.

## 6. Train a sample, then the full run

```bash
make pretrain NAME=sample CONFIG=configs/pretrain_sample.yaml ARGS="--no-push --set train.lr=0.008 --set data.batch_size=16"
make pretrain NAME=d64-e7                    # full run, pushes to the Hub repo of the config
```

`configs/pretrain_sample.yaml` carries lr 8e-3 and the scale factors; set `train.warmup_frac=0.05` as a precaution for a long
run (not tested at this scale). Rerunning the same `NAME` resumes.

## 7. What to bring back

The `results_*.json` files, the stdout of step 3 and the bench JSON. They go next to the other results in
`docs/experiments/results/`, and `pretraining_readiness.md` is updated from them: real tokens per modality, tokens/s and memory
per batch size on the GPU, and the learning rate that converges fastest on real rows.
