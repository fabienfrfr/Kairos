"""Thin orchestration helpers for the marimo pretraining notebook."""

from __future__ import annotations

from .utils import make_progress_callback


def train_with_progress(pipe, force_restart: bool = False, mo=None) -> list[dict]:
    """Runs pipe.train() with resume-aware logging and a live progress display.

    Pass the marimo module as `mo` to get a native marimo progress bar when running
    inside marimo's own runtime; otherwise (or if mo.running_in_notebook() is False,
    e.g. a plain Jupyter/Kaggle kernel), a tqdm bar via make_progress_callback is used.
    """
    resumed = not force_restart and (pipe.ckpt_dir / "last.pt").exists()
    if resumed:
        print(f"found last.pt in {pipe.ckpt_dir} - resuming")
    elif force_restart:
        print("FORCE_RESTART is True - ignoring any existing checkpoint")
    if mo is not None and mo.running_in_notebook():
        logs = _train_with_marimo_bar(pipe, force_restart, mo)
    else:
        cb = make_progress_callback()
        logs = pipe.train(progress_callback=cb, phase_callback=cb.phase, resume=not force_restart)
    _print_training_summary(pipe, logs)
    return logs


def _train_with_marimo_bar(pipe, force_restart: bool, mo) -> list[dict]:
    # on multi-GPU, pipe.train itself spawns a torchrun job (flex + memory gate per rank)
    # and replays its steps into the progress_callback; results come back into this pipe.
    total_steps = pipe.train_config.epochs * len(pipe.loader)
    with mo.status.progress_bar(total=total_steps, title="training") as bar:
        state = {"last_step": 0}

        def _on_step(step, total, loss_val):
            bar.update(increment=step - state["last_step"], subtitle=f"loss={loss_val:.4f}")
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
