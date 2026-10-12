# --- Variables ---

ENV_FILE=.env

# `make publish v0.X.X` publishes that tag; without it, the tag exactly on HEAD (never a .dev version)
VERSION_ARG := $(firstword $(filter v%,$(MAKECMDGOALS)))
ifneq ($(VERSION_ARG),)
$(VERSION_ARG): ; @:
endif
TAG := $(or $(VERSION_ARG),$(shell git describe --tags --exact-match 2>/dev/null))

# Loads UV_PUBLISH_TOKEN (PyPI token) from .env if present; never committed
-include $(ENV_FILE)
export

# ROCm (AMD) vs CUDA/CPU: auto-detect at build time.
# Force with: make notebook ROCM=1   (or ROCM=0 for CUDA/CPU machines like Kaggle)
ROCM ?= $(shell command -v rocm-smi >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(ROCM),1)
UV_TORCH_EXTRA := --extra rocm
else
UV_TORCH_EXTRA :=
endif

# --- Feature ---

coverage: ## prod-level > 95%
	uv run pytest --cov=kairos --cov-report=term-missing

test: ## Run the test suite
	uv run pytest -q

converge: ## Converge tests (-s for stdout : take print)
	uv run pytest tests/test_pipeline.py -k "converg" -v -s

lint: ## Check code style
	uv run ruff check .

build: ## Build sdist + wheel into dist/
	rm -rf dist/
	uv build

publish: ## Manual PyPI publish: make publish v0.X.X (or on a tagged HEAD); needs UV_PUBLISH_TOKEN (env or .env)
	@test -n "$(UV_PUBLISH_TOKEN)" || { echo "UV_PUBLISH_TOKEN missing: set it in $(ENV_FILE) (see .env.example)"; exit 1; }
	@test -n "$(TAG)" || { echo "Usage: make publish v0.X.X (HEAD is not on a release tag; try git fetch --tags)"; exit 1; }
	@git rev-parse -q --verify "refs/tags/$(TAG)" >/dev/null || { echo "Tag $(TAG) not found: git fetch --tags"; exit 1; }
	git worktree prune
	rm -rf dist/ .publish-src/
	git worktree add --detach .publish-src $(TAG)
	cd .publish-src && SETUPTOOLS_SCM_PRETEND_VERSION=$(TAG:v%=%) uv build --out-dir ../dist
	git worktree remove --force .publish-src
	@! ls dist | grep -q '\.dev' || { echo "dev version found in dist/: aborting"; exit 1; }
	uv run --with twine twine check dist/*
	uv publish

notebook: ## Working test
	uv run $(UV_TORCH_EXTRA) marimo edit notebook/kairos_pretraining.py

pretrain: ## Full pretraining: HF data -> train -> push to HF (NAME=run-name, CONFIG=..., ARGS="--no-push")
	uv run --extra cli $(UV_TORCH_EXTRA) kairos pretrain $(or $(CONFIG),configs/pretrain.yaml) $(if $(NAME),--name $(NAME),) $(ARGS)

jupyter: ## If you want to use Kaggle T4x2
	uv run $(UV_TORCH_EXTRA) marimo check --fix notebook/kairos_pretraining.py
	uv run $(UV_TORCH_EXTRA) marimo export ipynb notebook/kairos_pretraining.py -o notebook/notebook.ipynb

ddp-smoke: ## 2-process DDP smoke test (gloo/CPU); validates the multi-GPU train path
	KAIROS_DDP_SMOKE=1 uv run pytest -q tests/test_ddp_smoke.py
	

mapper: ## Export full project structure to JSON
	uv run python3 mapper.py --to-json

mapper-lean: ## Export project structure to JSON, excluding scripts/tests/docs
	uv run python3 mapper.py --to-json . project_structure.json --exclude scripts tests docs

build-multimodal: ## Build + push the multimodal dataset (uv sync --group scripts first)
	uv run --group scripts python3 scripts/pretrain/build_keep_it_simple_multimodal.py

dev: ## Install dev + scripts dependency groups (these are uv groups, not extras: `uv sync --group dev`, not `--extra dev`)
	uv sync --group dev --group scripts

format:
	uv run ruff format .


##@ Maintenance
clean: ## Remove python caches and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .venv .ruff_cache .mypy_cache
	@# Remove legacy VS Code Snap environment injections that break devpod/devbox sessions
	-sed -i '/snap\/code/d' ~/.profile ~/.bashrc ~/.bash_aliases 2>/dev/null


#  Automatically collect all targets with descriptions for .PHONY
ALL_TARGETS := $(shell grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | cut -d: -f1)

.PHONY: $(ALL_TARGETS)
