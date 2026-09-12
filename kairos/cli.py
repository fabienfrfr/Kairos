"""Config-file-driven CLI: `kairos train|overfit|evaluate|generate config.yaml`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .diagnostics import REGIMES
from .diagnostics import format_report as _format_moe_report
from .diagnostics import run as _run_moe_bias_check
from .modeling import KairosConfig
from .pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig
from .runners import evaluate_and_log, overfit_with_progress, run_generation_demo, train_with_progress

app = typer.Typer(add_completion=False, help="Launch KairosFM training/eval/generation from a config file.")


def _load_raw_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Applies `section.key=value` overrides (e.g. `train.lr=1e-4`) onto a loaded config dict."""
    for raw in overrides:
        path, _, value = raw.partition("=")
        section, _, key = path.partition(".")
        cfg.setdefault(section, {})[key] = json.loads(value) if _looks_like_json(value) else value
    return cfg


def _looks_like_json(value: str) -> bool:
    return value.lower() in ("true", "false", "null") or value.lstrip("-").replace(".", "", 1).isdigit()


def build_pipeline(config: Path, overrides: list[str] | None = None) -> KairosMultimodalPipeline:
    """Loads a YAML/JSON config file, applies overrides, builds and returns a ready pipeline."""
    cfg = _apply_overrides(_load_raw_config(config), overrides or [])
    model_config = KairosConfig(**cfg.get("model", {}))
    data_config = DataConfig(**cfg.get("data", {}))
    train_config = TrainConfig(**cfg.get("train", {}))
    eval_cfg = cfg.get("eval_data")
    eval_data_config = DataConfig(**eval_cfg) if eval_cfg else None
    return KairosMultimodalPipeline.from_configs(model_config, data_config, train_config, eval_data_config)


ConfigArg = Annotated[Path, typer.Argument(exists=True, help="Path to a YAML or JSON pipeline config file.")]
OverrideOpt = Annotated[list[str], typer.Option("--set", help="Override e.g. --set train.lr=1e-4 (repeatable).")]


@app.command()
def train(
    config: ConfigArg,
    force_restart: Annotated[bool, typer.Option(help="Ignore any existing checkpoint in run_dir.")] = False,
    set_: OverrideOpt = None,
) -> None:
    """Runs a full training loop, resuming from run_dir/last.pt unless --force-restart is set."""
    pipe = build_pipeline(config, set_ or [])
    train_with_progress(pipe, force_restart=force_restart, mo=None)


@app.command()
def overfit(
    config: ConfigArg,
    n_examples: Annotated[int, typer.Option(help="Number of examples in the tiny memorization subset.")] = 16,
    steps: Annotated[int, typer.Option(help="Number of optimizer steps to run.")] = 200,
    log_every: Annotated[int, typer.Option(help="Print the running loss every N steps (0 disables).")] = 20,
    set_: OverrideOpt = None,
) -> None:
    """Runs pipe.overfit_test() to sanity-check that the model can memorize a tiny subset."""
    pipe = build_pipeline(config, set_ or [])
    overfit_with_progress(pipe, n_examples=n_examples, steps=steps, log_every=log_every, mo=None)


@app.command()
def evaluate(config: ConfigArg, set_: OverrideOpt = None) -> None:
    """Runs one evaluation pass over the config's eval_data section and prints the average loss."""
    pipe = build_pipeline(config, set_ or [])
    if pipe.eval_data_config is None:
        raise typer.BadParameter("config file has no `eval_data` section to evaluate against")
    evaluate_and_log(pipe, pipe.eval_data_config)


@app.command()
def generate(
    config: ConfigArg,
    prompt_tokens: Annotated[int, typer.Option(help="Tokens kept as the prompt before generating.")] = 32,
    max_new_tokens: Annotated[int, typer.Option(help="Tokens to generate after the prompt.")] = 64,
    denoising_steps: Annotated[int, typer.Option(help="Max diffusion denoising steps.")] = 16,
    entropy_bound: Annotated[float, typer.Option(help="Entropy-bound sampler threshold.")] = 0.5,
    t_min: Annotated[float, typer.Option(help="Lower diffusion time bound.")] = 0.05,
    t_max: Annotated[float, typer.Option(help="Upper diffusion time bound.")] = 1.0,
    seed: Annotated[int, typer.Option(help="Base RNG seed; offset per example internally.")] = 0,
    n_examples: Annotated[int, typer.Option(help="Number of prompts to run the smoke test on.")] = 3,
    set_: OverrideOpt = None,
) -> None:
    """Runs a diffusion-generation smoke test on examples drawn from the config's data section."""
    pipe = build_pipeline(config, set_ or [])
    examples = pipe.eval_data_config.multimodal_examples if pipe.eval_data_config else None
    examples = examples or pipe.data_config.multimodal_examples or pipe.data_config.text_examples or []
    run_generation_demo(
        pipe,
        pipe.tokenizer,
        examples,
        examples,
        prompt_tokens,
        max_new_tokens,
        denoising_steps,
        entropy_bound,
        t_min,
        t_max,
        seed,
        n_examples,
    )


@app.command("moe-bias-check")
def moe_bias_check(
    regime: Annotated[str, typer.Argument(help=f"One of: {', '.join(REGIMES)}.")],
    rate: Annotated[float, typer.Option(help="moe_bias_update_rate to test; 0 disables the bias update.")],
    eps: Annotated[float, typer.Option(help="mask_eps floor.")] = 1e-3,
    clip: Annotated[float, typer.Option(help="mask_reweight_clip.")] = 10.0,
    steps: Annotated[int, typer.Option(help="Overfit-test steps to run.")] = 200,
    seed: Annotated[int, typer.Option(help="RNG seed.")] = 0,
) -> None:
    """A/B-checks whether moe_bias_update_rate helps a tiny top-1 MoE overfit-test converge."""
    if regime not in REGIMES:
        raise typer.BadParameter(f"regime must be one of: {', '.join(REGIMES)}")
    losses, usage = _run_moe_bias_check(regime, rate, mask_eps=eps, mask_reweight_clip=clip, steps=steps, seed=seed)
    typer.echo(_format_moe_report(losses, usage))


if __name__ == "__main__":
    app()
