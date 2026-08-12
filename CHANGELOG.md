# Changelog

All notable changes to this project are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/), bumped automatically from commit messages (`fix:` → patch, `feat:` → minor, `BREAKING CHANGE` → major).

## [0.5.0]

Pre-release baseline. Architecture not yet validated by a full training run — `v1.0.0` is reserved for the first checkpoint that has actually proven itself; a `BREAKING CHANGE` commit will trigger that bump once training results back the design.

### Added
- `DataConfig(pack=True)`: concatenates samples before chunking so only the final chunk is padded, instead of padding every short example individually.
- `TrainConfig.hub_subfolder`: push checkpoints/model/logs under `repo_id/<subfolder>` so multiple runs or configs can share one Hub repo.
- `pipe.run_config_dict()` / `training_config.json`: the full model/train/data hyperparameters behind a run, persisted at `build()` time and embedded in every checkpoint — previously only `model_config` was saved.
- `pipe.inspect_batch()`: visualizes tokenized input exactly as the model receives it (decoded preview, modality counts, raw ids, out-of-bounds detection, degenerate-repeat detection) — no data leaves the process.
- `pipe.locate_nan_source()` / `pipe.nan_log`: pinpoints which submodule first produced a non-finite value, with a full history of skipped batches.
- `TrainConfig.max_consecutive_nan`: aborts training with a diagnostic instead of looping silently through a run that stopped converging.
- Model card is now a standalone template (`kairos/templates/model_card.md`) instead of an inline string in `pipeline.py`.
- Package metadata completed for a real PyPI release: license, authors, classifiers, urls; `pytest`/`ruff` moved out of runtime dependencies into the `dev` group.
- CI: lint + test on every push/PR to `main`; version-tag-triggered publish workflow (build, `twine check`, PyPI trusted publishing, post-publish install smoke test); auto-tagging workflow that bumps and pushes a semver tag from Conventional Commits on `main`.


### Fixed
- **DeltaNet cache bug**: `has_previous_state` was gated only on `conv_caches`, so an `ssm_cache` set without a matching `conv_cache` (exactly what state-carrying mechanisms do) was silently ignored — any carried/injected state was never actually used by the model.
- **MoE weight initialization**: `KairosDiffusionLLM` never called `self.post_init()`, so `DeepseekV3Experts`' raw `nn.Parameter(torch.empty(...))` weights were left as uninitialized memory (occasionally NaN/Inf at construction, before any data or training). Fixed in `KairosMoE.__init__` directly (not via fragile `isinstance` checks on internal `transformers` class names, which differ across versions).
- `KairosConfig.top_k` collided with `GenerationConfig.top_k` (sampling parameter), silently producing an invalid generation config. Removed the dead alias.
- `n_routed_experts` is now a property alias of `num_local_experts` (the field `transformers` actually reads) — setting either name can no longer silently diverge from the other.
- `save_pretrained()` rejected `LiZAttention2`'s intentional SWA/DeltaNet weight sharing (tied-weight validation). `push_to_hub()` now saves the state dict directly instead.
- Various `ruff`-flagged issues across `scripts/` (blind excepts without justification, a lost executable bit, an unreachable `sorted()[0]`).

### Changed
- Package renamed for PyPI: `kairos` was already taken, published as **`kairos-fm`** (import name stays `kairos`). New tagline: *"KairosFM — less parameters, more signal."*

### Removed
- The `state_carry` / `build_carried_cache` mechanism (random or all-rows cache mixing between batches): superseded by `KairosMemoryBank`, which is a genuinely learnable mechanism rather than a fixed policy.

## [0.1.0]
Initial internal version.
