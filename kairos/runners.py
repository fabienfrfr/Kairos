"""Thin orchestration helpers usable from the marimo notebook or a plain CLI/script."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .dataset import KairosPretrainingDataset
from .trainer import stage_name_at
from .utils import make_progress_callback


def _uses_marimo_bar(mo) -> bool:
    return mo is not None and mo.running_in_notebook()


def _print_progress_mode(desc: str, mo) -> None:
    mode = "marimo progress bar" if _uses_marimo_bar(mo) else "tqdm (mo not passed or not in marimo runtime)"
    print(f"{desc}: using {mode}")


def _stage_watcher(pipe):
    """Returns stage_at(step) -> current curriculum stage; prints once whenever it changes."""
    seen = {"stage": None}

    def stage_at(step: int) -> str:
        bounds = pipe.curriculum_bounds
        stage = stage_name_at(step, *bounds) if bounds is not None else "fixed regime"
        if stage != seen["stage"]:
            print(f"step {step}: entering '{stage}' stage")
            seen["stage"] = stage
        return stage

    return stage_at


def train_with_progress(pipe, force_restart: bool = False, mo=None) -> list[dict]:
    """Runs pipe.train() with resume-aware logging; a marimo bar if `mo` is given, else tqdm."""
    resumed = not force_restart and (pipe.ckpt_dir / "last.pt").exists()
    if resumed:
        print(f"found last.pt in {pipe.ckpt_dir} - resuming")
    elif force_restart:
        print("FORCE_RESTART is True - ignoring any existing checkpoint")
    _print_progress_mode("train", mo)
    if _uses_marimo_bar(mo):
        logs = _train_with_marimo_bar(pipe, force_restart, mo)
    else:
        cb = make_progress_callback(stage_fn=_stage_watcher(pipe))
        logs = pipe.train(progress_callback=cb, phase_callback=cb.phase, resume=not force_restart)
    _print_training_summary(pipe, logs)
    return logs


def _train_with_marimo_bar(pipe, force_restart: bool, mo) -> list[dict]:
    # on multi-GPU, pipe.train spawns a torchrun job and replays its steps into this callback
    total_steps = pipe.train_config.epochs * len(pipe.loader)
    stage_at = _stage_watcher(pipe)
    with mo.status.progress_bar(total=total_steps, title="training") as bar:
        state = {"last_step": 0}

        def _on_step(step, total, loss_val):
            bar.update(increment=step - state["last_step"], subtitle=f"loss={loss_val:.4f} stage={stage_at(step)}")
            state["last_step"] = step

        def _on_phase(name):
            bar.update(increment=0, subtitle=name)

        return pipe.train(progress_callback=_on_step, phase_callback=_on_phase, resume=not force_restart)


def _print_training_summary(pipe, logs: list[dict]) -> None:
    print(f"training complete - steps: {len(logs)}  best avg-epoch loss: {pipe.best_loss:.4f}")
    print(f"skipped non-finite batches: {pipe.skipped_nonfinite_steps}")
    if pipe.eval_log_rows:
        print(f"eval points: {len(pipe.eval_log_rows)}  best eval loss: {pipe.best_eval_loss:.4f}")
    print(f"checkpoints: {pipe.ckpt_dir}")


def overfit_with_progress(pipe, n_examples: int, steps: int, log_every: int, mo=None) -> list[dict] | None:
    """Runs pipe.overfit_test() with a marimo bar when available, else a plain call."""
    _print_progress_mode("overfit_test", mo)
    stage_at = _stage_watcher(pipe)
    if _uses_marimo_bar(mo):
        with mo.status.progress_bar(total=steps, title="overfit_test") as bar:
            logs = pipe.overfit_test(
                n_examples=n_examples,
                steps=steps,
                log_every=log_every,
                progress_callback=lambda step, total, loss_val: bar.update(
                    increment=1, subtitle=f"loss={loss_val:.4f} stage={stage_at(step)}"
                ),
            )
    else:
        cb = make_progress_callback(desc="overfit_test", stage_fn=stage_at)
        logs = pipe.overfit_test(n_examples=n_examples, steps=steps, log_every=log_every, progress_callback=cb)
    print(f"overfit_test done: loss {logs[0]['loss']:.4f} -> {logs[-1]['loss']:.4f}")
    return logs


def evaluate_and_log(pipe, eval_data_config) -> float | None:
    """Runs a full pass over eval_data_config, logs the average loss and returns it."""
    if not eval_data_config.multimodal_examples:
        print("no eval examples - skipping")
        return None
    eval_dataset = KairosPretrainingDataset(
        multimodal_examples=eval_data_config.multimodal_examples,
        tokenizer=pipe.tokenizer,
        max_len=eval_data_config.max_len,
        stride=eval_data_config.stride,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=eval_data_config.batch_size, shuffle=False)
    eval_loss = _run_eval_pass(pipe, eval_loader)
    pipe.writer.add_scalar("eval/loss", eval_loss, pipe.global_step)
    print(f"eval loss: {eval_loss:.4f} on {len(eval_dataset)} samples")
    return eval_loss


def _run_eval_pass(pipe, eval_loader) -> float:
    pipe.model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(pipe.device) for k, v in batch.items()}
            with pipe._autocast():
                losses.append(pipe.hf_trainer.compute_loss(pipe.model, batch).item())
    pipe.model.train()
    return sum(losses) / len(losses)


def run_generation_demo(
    pipe,
    tokenizer,
    eval_examples: list[dict],
    text_examples: list[dict],
    prompt_tokens: int,
    max_new_tokens: int,
    denoising_steps: int,
    entropy_bound: float,
    t_min: float,
    t_max: float,
    seed: int,
    n_examples: int,
) -> list[dict]:
    """Runs a diffusion-generation smoke test and prints prompt/generated/reference triples."""
    rows = _pick_generation_rows(eval_examples, text_examples, tokenizer, prompt_tokens, n_examples)
    results = []
    for i, ex in enumerate(rows, 1):
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        prompt = ids[:prompt_tokens]
        full = pipe.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            max_denoising_steps=denoising_steps,
            entropy_bound=entropy_bound,
            t_min=t_min,
            t_max=t_max,
            seed=seed + i,
        )
        generated = full[prompt_tokens:]
        reference = ids[prompt_tokens : prompt_tokens + max_new_tokens]
        result = {
            "prompt": tokenizer.decode(prompt, skip_special_tokens=True),
            "generated": tokenizer.decode(generated, skip_special_tokens=True),
            "reference": tokenizer.decode(reference, skip_special_tokens=True),
        }
        print(f"--- example {i} ---")
        print("prompt:    ", result["prompt"])
        print("generated: ", result["generated"])
        print("reference: ", result["reference"])
        results.append(result)
    return results


def _pick_generation_rows(eval_examples, text_examples, tokenizer, prompt_tokens: int, n_examples: int) -> list[dict]:
    def _long_enough(ex):
        return ex.get("modality") == "text" and len(tokenizer.encode(ex["text"], add_special_tokens=False)) > (
            prompt_tokens
        )

    rows = [ex for ex in eval_examples if _long_enough(ex)]
    if len(rows) < n_examples:
        rows = [ex for ex in text_examples if _long_enough(ex)]
    return rows[:n_examples]
