"""Compute-cost helpers: parameter counts, memory footprint, step-time estimation."""

from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass

import torch


def locate_first_nonfinite_module(model, forward_fn) -> dict | None:
    """Runs forward_fn() with hooks on every submodule and returns diagnostics for the."""
    found: dict | None = None
    handles = []

    def _make_hook(name):
        def _hook(module, inputs, output):
            nonlocal found
            if found is not None:
                return
            tensors = output if isinstance(output, (tuple, list)) else [output]
            for t in tensors:
                if isinstance(t, torch.Tensor) and t.is_floating_point() and not torch.isfinite(t).all():
                    found = {
                        "module": name,
                        "module_type": type(module).__name__,
                        "nan_frac": float(torch.isnan(t).float().mean()),
                        "inf_frac": float(torch.isinf(t).float().mean()),
                    }
                    return

        return _hook

    for name, module in model.named_modules():
        if name:  # skip the root module itself,
            handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        forward_fn()
    finally:
        for h in handles:
            h.remove()

    return found


def real_module_breakdown(model) -> list[dict]:
    """Per top-level submodule param bytes, deduped by storage identity (not double-counting
    shared modules like a tied/shared backbone). Each entry also reports how many attribute
    slots reference that same physical module, so sharing is visible instead of hidden."""
    seen_storage: dict[int, str] = {}  # storage id -> first module name that claimed it
    breakdown = []
    for name, module in model.named_children():
        unique_bytes = 0
        shared_bytes = 0
        n_params = 0
        for p in module.parameters():
            sid = p.untyped_storage().data_ptr()
            nbytes = p.numel() * p.element_size()
            n_params += p.numel()
            if sid in seen_storage:
                shared_bytes += nbytes  # storage already counted under another top-level name
            else:
                seen_storage[sid] = name
                unique_bytes += nbytes
        # detect internal repetition (e.g. self.backbones = ModuleList([shared] * 4)):
        # count how many distinct child slots inside `module` point at the same storage.
        n_slots = sum(1 for _ in module.modules())
        n_unique_instances = len({id(m) for m in module.modules()})
        breakdown.append(
            {
                "name": name,
                "n_params": n_params,
                "unique_bytes": unique_bytes,
                "shared_bytes": shared_bytes,  # counted elsewhere too — informational only
                "n_module_slots": n_slots,
                "n_unique_module_instances": n_unique_instances,
            }
        )
    return breakdown


def _process_rss_mb() -> float:
    """Current process resident memory in MB (Linux/Mac; ru_maxrss is peak, not current, so we
    read /proc/self/status when available for a live number, falling back to peak RSS)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except FileNotFoundError:
        pass
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024 if sys.platform != "darwin" else ru / (1024 * 1024)


def _tensor_bytes(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes(t) for t in x)
    return 0


@dataclass
class DetailedMemoryReport:
    """Real, measured (not estimated) memory footprint of one train step."""

    unique_param_bytes: int
    module_breakdown: list[dict]
    grad_bytes: int
    optimizer_state_bytes: int
    activation_bytes_forward: int  # sum of every module's output tensor(s), one forward pass
    largest_activation: tuple[str, int]  # (module name, bytes) of the single biggest output
    rss_before_mb: float
    rss_after_forward_mb: float
    rss_after_backward_mb: float
    rss_after_optimizer_step_mb: float
    device: str

    def __str__(self) -> str:
        lines = [
            "Kairos detailed memory report (measured, not estimated)",
            "----------------------------------------------------------",
            f"Device:                        {self.device}",
            f"Unique param bytes (deduped):  {self.unique_param_bytes / 1e6:.1f} MB",
            f"Gradient bytes (measured):     {self.grad_bytes / 1e6:.1f} MB",
            f"Optimizer state (measured):    {self.optimizer_state_bytes / 1e6:.1f} MB",
            f"Activation bytes (1 fwd pass): {self.activation_bytes_forward / 1e6:.1f} MB",
            f"  largest single activation:   {self.largest_activation[0]} "
            f"({self.largest_activation[1] / 1e6:.1f} MB)",
            "",
            "Process RSS (real OS memory, /proc/self/status VmRSS):",
            f"  before step:                 {self.rss_before_mb:.1f} MB",
            f"  after forward:                {self.rss_after_forward_mb:.1f} MB "
            f"(+{self.rss_after_forward_mb - self.rss_before_mb:.1f} MB)",
            f"  after backward:               {self.rss_after_backward_mb:.1f} MB "
            f"(+{self.rss_after_backward_mb - self.rss_after_forward_mb:.1f} MB)",
            f"  after optimizer.step():       {self.rss_after_optimizer_step_mb:.1f} MB "
            f"(+{self.rss_after_optimizer_step_mb - self.rss_after_backward_mb:.1f} MB)",
            "",
            "Per top-level module (unique bytes, shared modules counted once):",
        ]
        for row in sorted(self.module_breakdown, key=lambda r: -r["unique_bytes"]):
            if row["unique_bytes"] == 0 and row["shared_bytes"] == 0:
                continue
            tag = ""
            if row["n_unique_module_instances"] < row["n_module_slots"]:
                tag = f"  [SHARED: {row['n_module_slots']} slots -> {row['n_unique_module_instances']} unique instance(s)]"
            lines.append(f"  {row['name']:<16} {row['unique_bytes'] / 1e6:8.1f} MB{tag}")
        accounted = self.unique_param_bytes + self.grad_bytes + self.optimizer_state_bytes + self.activation_bytes_forward
        unaccounted = (self.rss_after_optimizer_step_mb - self.rss_before_mb) * 1e6 - accounted
        lines += [
            "",
            f"Sum of measured components:    {accounted / 1e6:.1f} MB",
            f"Actual RSS growth this step:   {(self.rss_after_optimizer_step_mb - self.rss_before_mb):.1f} MB",
            f"Unaccounted (dataloader/cache/Arrow mmap/fragmentation/etc): {unaccounted / 1e6:.1f} MB",
        ]
        return "\n".join(lines)


def detailed_memory_report(model, optimizer, loss_fn, device, autocast_ctx=None, scaler=None) -> DetailedMemoryReport:
    """Runs exactly one real forward+backward+optimizer-step and measures actual memory at
    each stage — no estimation. loss_fn() must run the forward pass and return the loss
    tensor (e.g. lambda: hf_trainer.compute_loss(model, batch)). Leaves the model/optimizer
    mutated — caller should snapshot and restore state_dict() around this call if it needs
    to keep training from a clean state.
    """
    import gc

    autocast_ctx = autocast_ctx or contextlib.nullcontext

    module_breakdown = real_module_breakdown(model)
    unique_param_bytes = sum(r["unique_bytes"] for r in module_breakdown)

    activation_bytes = 0
    largest = ("(none)", 0)
    handles = []

    def _make_hook(name):
        def _hook(module, inputs, output):
            nonlocal activation_bytes, largest
            n = _tensor_bytes(output)
            activation_bytes += n
            if n > largest[1]:
                largest = (name, n)

        return _hook

    for name, module in model.named_modules():
        if name and sum(1 for _ in module.children()) == 0:  # leaf modules only, avoid double counting
            handles.append(module.register_forward_hook(_make_hook(name)))

    gc.collect()
    rss_before = _process_rss_mb()

    optimizer.zero_grad()
    try:
        with autocast_ctx():
            loss = loss_fn()
    finally:
        for h in handles:
            h.remove()
    rss_after_forward = _process_rss_mb()

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    rss_after_backward = _process_rss_mb()

    grad_bytes = 0
    seen = set()
    for p in model.parameters():
        if p.grad is not None:
            sid = p.grad.untyped_storage().data_ptr()
            if sid not in seen:
                seen.add(sid)
                grad_bytes += p.grad.numel() * p.grad.element_size()

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    rss_after_optimizer_step = _process_rss_mb()

    optimizer_state_bytes = 0
    seen_opt = set()
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                sid = v.untyped_storage().data_ptr()
                if sid not in seen_opt:
                    seen_opt.add(sid)
                    optimizer_state_bytes += v.numel() * v.element_size()

    return DetailedMemoryReport(
        unique_param_bytes=unique_param_bytes,
        module_breakdown=module_breakdown,
        grad_bytes=grad_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        activation_bytes_forward=activation_bytes,
        largest_activation=largest,
        rss_before_mb=rss_before,
        rss_after_forward_mb=rss_after_forward,
        rss_after_backward_mb=rss_after_backward,
        rss_after_optimizer_step_mb=rss_after_optimizer_step,
        device=str(device),
    )


def format_duration(seconds: float | None) -> str:
    """Formats seconds as e.g."""
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
    """MoE-aware param count actually touched per forward pass: dense params unchanged if."""
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
    """Average seconds/step over n_steps calls to step_fn(), or None if step_fn runs."""
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
    """Returns a (step, total, loss) -> None callback for pipeline.train(), backed by."""
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
    """Snapshot of a run's compute cost: param counts, memory footprint, and (optional)."""

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
    measured_memory: bool = False  # True when the memory fields below come from a real
    # forward+backward+step (detailed_memory_report), not from the param-count formulas.

    def __str__(self) -> str:
        mem_label = "Measured" if self.measured_memory else "Est."
        lines = [
            "Kairos training summary",
            "------------------------",
            f"Total params:        {self.total_params / 1e6:.2f}M",
            f"Active params/tok:   {self.active_params / 1e6:.2f}M",
            f"Trainable params:    {self.trainable_params / 1e6:.2f}M",
            f"{mem_label} model memory:".ljust(21) + f"{self.param_memory_mb:.1f} MB",
            f"{mem_label} optimizer mem:".ljust(21) + f"{self.optimizer_memory_mb:.1f} MB",
            f"{mem_label} total memory:".ljust(21) + f"{self.total_memory_mb:.1f} MB",
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
    """Builds a TrainingSummary from a model and sized loader; pass step_fn to."""
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
