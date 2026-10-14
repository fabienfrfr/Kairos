# CLI cheat sheet

Install: `uv add kairos-fm` (add the `fast-attn` extra on NVIDIA for the fused kernels). In a clone of the repo:
`uv sync --group dev`. Run commands with `uv run kairos ...`. `--help` works on every command.

## Configs

| File | Use |
|---|---|
| `minimal` (bundled) | Offline CPU smoke run: 4 sentences, patch codec, 1.6M params, ~1 min |
| `pretrain` (bundled) | Real pretraining: HF datasets, 3.8M params, pushes to `ffurfaro/kairos` |
| `configs/example.yaml` | Local `.pt` data (`data/train.pt`, `data/eval.pt`) |
| `configs/ablations/*.yaml` | Tiny overfit sanity checks (see `docs/experiments/`) |

`CONFIG` is a file path or a bundled name. Export a bundled config to edit it: `kairos init-config pretrain -o my.yaml`.

## One command, full pretraining (CUDA box)

```bash
echo "HF_TOKEN=hf_..." > .env
make pretrain NAME=d64-e7                          # HF data -> train -> push to ffurfaro/kairos/d64-e7/
make pretrain NAME=test ARGS="--no-push"           # train only
make pretrain NAME=dev ARGS="--set hf_data.text_pct=1 --no-push"   # quick dev run
make pretrain CONFIG=minimal ARGS="--no-push"
```

Same thing without make:

```bash
uv run kairos pretrain pretrain --name d64-e7
```

| Option | Effect |
|---|---|
| `--name NAME` | Run name: `checkpoints/kairos-multimodal/NAME` locally, `NAME/` in the Hub repo (letters, digits, `.`, `_`, `-`) |
| `--push / --no-push` | Push the full model to `train.hub_repo_id` at the end (default: push) |
| `--force-restart` | Ignore any checkpoint already in `run_dir` |
| `--set section.key=value` | Override a config value (repeatable; JSON values accepted: `3e-4`, `false`, `null`, `[...]`) |

Rerunning the same `--name` resumes that run; a new name starts from scratch. Without `--name`, the config's `run_dir` is used.

## Other commands

```bash
kairos init-config pretrain -o my.yaml             # export a bundled config to edit
kairos train configs/example.yaml                 # train only, no Hub push of the final model
kairos overfit configs/ablations/baseline.yaml --steps 600 --mask-p-max 0.3 --no-mask-reweight --seed 0
kairos evaluate configs/example.yaml              # one pass over the config's eval_data
kairos generate configs/example.yaml --n-examples 3 --max-new-tokens 64 --denoising-steps 16
kairos moe-bias-check REGIME --rate 0.001 --steps 200   # REGIME: see `kairos moe-bias-check --help`
```

## Useful `--set` overrides

```bash
--set train.lr=3e-4
--set data.batch_size=16 --set data.max_len=512
--set train.compile_model=false          # CPU, or if torch.compile is slow to start
--set train.hub_repo_id=me/my-model --set train.hub_push_every_ckpt=true
--set hf_data.text_pct=1                 # percent of the text dataset to load
```

## Make targets

| Command | What it does |
|---|---|
| `make dev` | Install dev + scripts dependency groups |
| `make test` / `make lint` / `make coverage` | Tests / ruff / coverage |
| `make build-multimodal` | Build and push the multimodal dataset |
| `make notebook` | Open the marimo training notebook |
| `make ddp-smoke` | 2-process DDP smoke test (CPU) |
| `make build` | Build sdist + wheel into `dist/` |
| `make publish v0.X.X` | Manual PyPI publish of a tag (needs `UV_PUBLISH_TOKEN`); see `docs/RELEASING.md` |
| `make clean` | Remove caches and temporary files |
