"""Compute-cost helpers: parameter counts, memory footprint, step-time estimation."""

from __future__ import annotations

import time
from dataclasses import dataclass


def format_duration(seconds: float | None) -> str:
    """Formats seconds as e.g. "1h 2m 5s"; returns "n/a" for None."""
    if seconds is None:
        return "n/a"
    seconds = round(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def count_parameters(model) -> tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_active_parameters(model, num_experts_per_tok: int | None = None, num_local_experts: int | None = None) -> int:
    """MoE-aware param count actually touched per forward pass: dense params unchanged if not MoE."""
    total = sum(p.numel() for p in model.parameters())
    if not num_experts_per_tok or not num_local_experts:
        return total
    expert_params = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
    ratio = num_experts_per_tok / num_local_experts
    return round(total - expert_params + expert_params * ratio)


def estimate_param_memory_mb(total_params: int, bytes_per_param: int = 4) -> float:
    """Raw weight storage in MB, fp32 by default."""
    return total_params * bytes_per_param / (1024**2)


def estimate_optimizer_memory_mb(trainable_params: int, optimizer_states: int = 2, bytes_per_param: int = 4) -> float:
    """AdamW optimizer state in MB: optimizer_states buffers (m, v) per trainable param."""
    return trainable_params * optimizer_states * bytes_per_param / (1024**2)


def benchmark_step_time(step_fn, n_steps: int = 5, warmup: int = 1) -> float | None:
    """Average seconds/step over n_steps calls to step_fn(), or None if step_fn runs out of batches."""
    try:
        for _ in range(warmup):
            step_fn()
        start = time.perf_counter()
        for _ in range(n_steps):
            step_fn()
        elapsed = time.perf_counter() - start
    except StopIteration:
        return None
    return elapsed / n_steps


def make_progress_callback(desc: str = "training"):
    """Returns a (step, total, loss) -> None callback for pipeline.train(), backed by a tqdm bar."""
    from tqdm.auto import tqdm

    state = {"bar": None}

    def _callback(step: int, total: int, loss_val: float) -> None:
        if state["bar"] is None:
            state["bar"] = tqdm(total=total, desc=desc)
        state["bar"].n = step
        state["bar"].set_postfix(loss=f"{loss_val:.4f}")
        state["bar"].refresh()
        if step >= total:
            state["bar"].close()

    return _callback


@dataclass
class TrainingSummary:
    """Snapshot of a run's compute cost: param counts, memory footprint, and (optional) timing."""

    total_params: int
    trainable_params: int
    active_params: int
    param_memory_mb: float
    optimizer_memory_mb: float
    total_memory_mb: float
    steps_per_epoch: int
    epochs: int
    total_steps: int
    avg_step_time_sec: float | None = None
    estimated_total_time_sec: float | None = None

    def __str__(self) -> str:
        lines = [
            "Kairos training summary",
            "------------------------",
            f"Total params:        {self.total_params / 1e6:.2f}M",
            f"Active params/tok:   {self.active_params / 1e6:.2f}M",
            f"Trainable params:    {self.trainable_params / 1e6:.2f}M",
            f"Est. model memory:   {self.param_memory_mb:.1f} MB",
            f"Est. optimizer mem:  {self.optimizer_memory_mb:.1f} MB",
            f"Est. total memory:   {self.total_memory_mb:.1f} MB",
            f"Steps/epoch:         {self.steps_per_epoch}",
            f"Epochs:              {self.epochs}",
            f"Total steps:         {self.total_steps}",
        ]
        if self.avg_step_time_sec is not None:
            lines.append(f"Avg step time:       {self.avg_step_time_sec * 1000:.1f} ms")
            lines.append(f"Est. total time:     {format_duration(self.estimated_total_time_sec)}")
        else:
            lines.append("Avg step time:       n/a")
        return "\n".join(lines)


def training_summary(
    model,
    loader,
    epochs: int,
    step_fn=None,
    n_bench_steps: int = 5,
    num_experts_per_tok: int | None = None,
    num_local_experts: int | None = None,
) -> TrainingSummary:
    """Builds a TrainingSummary from a model and sized loader; pass step_fn to also time+extrapolate total time."""
    total_params, trainable_params = count_parameters(model)
    active_params = count_active_parameters(model, num_experts_per_tok, num_local_experts)
    param_memory_mb = estimate_param_memory_mb(total_params)
    optimizer_memory_mb = estimate_optimizer_memory_mb(trainable_params)

    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch

    avg_step_time = benchmark_step_time(step_fn, n_steps=n_bench_steps) if step_fn is not None else None
    estimated_total_time = avg_step_time * total_steps if avg_step_time is not None else None

    return TrainingSummary(
        total_params=total_params,
        trainable_params=trainable_params,
        active_params=active_params,
        param_memory_mb=param_memory_mb,
        optimizer_memory_mb=optimizer_memory_mb,
        total_memory_mb=param_memory_mb + optimizer_memory_mb,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        total_steps=total_steps,
        avg_step_time_sec=avg_step_time,
        estimated_total_time_sec=estimated_total_time,
    )
