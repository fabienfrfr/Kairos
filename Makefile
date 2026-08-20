# --- Variables ---

ENV_FILE=.env

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

publish: ## Real publishing happens via CI on tag push (see .github/workflows/publish.yml); this is a manual/local fallback only
	uv build
	uv run --with twine twine check dist/*
	uv publish

notebook: ## Working test
	uv run marimo edit notebook/kairos_pretraining.py

jupyter: ## If you want to use Kaggle T4x2
	uv run marimo export ipynb notebook/kairos_pretraining.py -o notebook/notebook.ipynb

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
