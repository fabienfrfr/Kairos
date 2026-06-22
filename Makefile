# --- Variables ---

ENV_FILE=.env

# --- Feature ---

coverage: ## prod-level > 95%
	uv run pytest --cov=src --cov-report=term-missing

edit-notebook: ## Working test
	uv run marimo edit notebook.py

code-map: ## Export project structure to JSON
	uv run python3 mapper.py --to-json

##@ Maintenance
clean: ## Remove python caches and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .venv .ruff_cache .mypy_cache
	@# Remove legacy VS Code Snap environment injections that break devpod/devbox sessions
	-sed -i '/snap\/code/d' ~/.profile ~/.bashrc ~/.bash_aliases 2>/dev/null


#  Automatically collect all targets with descriptions for .PHONY
ALL_TARGETS := $(shell grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | cut -d: -f1)

.PHONY: $(ALL_TARGETS)