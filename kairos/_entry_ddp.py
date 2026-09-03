"""DDP entrypoint: unpickle fresh configs from argv[1], build a pipeline, run the requested action (train/summary)."""

import pickle
import sys
from pathlib import Path

from kairos.pipeline import KairosMultimodalPipeline
from kairos.tokenizer import KairosTokenizer


def _run_train(pipe: KairosMultimodalPipeline, job: dict, job_dir: Path) -> None:
    def on_step(step: int, total: int, loss: float) -> None:
        print(f"step {step}/{total}  loss {loss:.4f}", flush=True)

    logs = pipe.train(progress_callback=on_step, resume=job["resume"])
    if pipe.is_main_process:
        print(f"training complete - steps: {len(logs)}  best avg-epoch loss: {pipe.best_loss:.4f}", flush=True)
        results = {
            "log_rows": pipe.log_rows,
            "eval_log_rows": pipe.eval_log_rows,
            "nan_log": getattr(pipe, "nan_log", []),
            "best_loss": pipe.best_loss,
            "best_eval_loss": pipe.best_eval_loss,
            "skipped_nonfinite_steps": pipe.skipped_nonfinite_steps,
            "global_step": pipe.global_step,
        }
        with (job_dir / "results.pkl").open("wb") as f:
            pickle.dump(results, f)


def _run_summary(pipe: KairosMultimodalPipeline, job: dict, job_dir: Path) -> None:
    summary = pipe.summary(**job["kwargs"])
    if pipe.is_main_process:
        print(str(summary), flush=True)
        with (job_dir / "results.pkl").open("wb") as f:
            pickle.dump({"summary": summary}, f)


_ACTIONS = {"train": _run_train, "summary": _run_summary}


def main() -> None:
    job_dir = Path(sys.argv[1])
    with (job_dir / "configs.pkl").open("rb") as f:
        model_config, data_config, eval_data_config, train_config = pickle.load(f)
    with (job_dir / "job.pkl").open("rb") as f:
        job = pickle.load(f)
    pipe = KairosMultimodalPipeline(
        model_config,
        data_config,
        train_config,
        eval_data_config=eval_data_config,
        tokenizer=KairosTokenizer(),
    )
    pipe.build()
    assert pipe.distributed, "expected DDP env from torchrun"
    _ACTIONS[job["action"]](pipe, job, job_dir)


if __name__ == "__main__":
    main()