"""Thin, declarative wrapper: tokenizer -> dataset -> model -> optimizer/scheduler -> train."""

from __future__ import annotations

import copy
import itertools
import json
import math
import random
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments

from .dataset import KairosPretrainingDataset
from .modeling import KairosConfig, KairosDiffusionFM, KairosMultiCache, gate_memory_bank
from .tokenizer import KairosTokenizer, Modality
from .trainer import KairosDiffusionTrainer, compute_masked_diffusion_losses, make_diffusion_mask
from .utils import (
    DetailedMemoryReport,
    TrainingSummary,
    benchmark_step_time,
    detailed_memory_report,
    locate_first_nonfinite_module,
    training_summary,
)


@dataclass
class DataConfig:
    multimodal_examples: list | None = None
    multimodal_path: str | None = None
    text_examples: list | None = None
    max_len: int = 1024
    stride: int = 3
    batch_size: int = 8
    shuffle: bool = True
    drop_last: bool = True
    pack: bool = False  # concatenate samples before chunking so
    num_workers: int | None = None  # None: 4 if batch_size > 1 else 0. Set explicitly to
    # override — e.g. 0 on a memory-constrained machine, since each DataLoader worker forks
    # (copy-on-write) the whole parent process, and CPython refcounting on touched objects
    # can turn that into real, non-shared memory growth per worker over time.


@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 3
    save_every: int = 200
    last_ckpt_every: int = 20  # how often last.pt (resume point)
    eval_every: int = 0  # run eval on the held-out set every N steps (0 = off)
    eval_batches: int = 2  # eval batches per evaluation, capped; keep small
    grad_clip: float = 1.0
    mask_eps: float = 1e-3  # floor of masked-diffusion rate p; CE/p variance grows sharply as this shrinks
    mask_p_max: float = 1.0  # ceiling of p; cap below 1.0 (e.g. 0.3) for an MAE-style fixed-rate curriculum stage
    mask_reweight: bool = True  # divide CE by p; set False for plain CE (pairs with a capped mask_p_max)
    octet_loss_weight: float = 1.0  # weight of the octet-family loss; family is part of token identity now
    max_consecutive_nan: int = 50  # abort with a diagnosis instead
    run_dir: str = "checkpoints/kairos-multimodal/run_01"
    device: str | None = None  # None -> auto
    report_to: list = field(default_factory=list)
    hub_repo_id: str | None = None  # set to also push each checkpoint
    hub_push_every_ckpt: bool = False  # requires hub_repo_id; pushes checkpoints
    hub_private: bool = False
    hub_subfolder: str | None = None  # push under repo_id/<subfolder>


def _consecutive_run_lengths(ids: torch.Tensor) -> dict[int, int]:
    """Maps each distinct id to the length of its longest run of."""
    if ids.numel() == 0:
        return {}
    values = ids.tolist()
    longest: dict[int, int] = {}
    current_id, current_len = values[0], 1
    for v in values[1:]:
        if v == current_id:
            current_len += 1
        else:
            longest[current_id] = max(longest.get(current_id, 0), current_len)
            current_id, current_len = v, 1
    longest[current_id] = max(longest.get(current_id, 0), current_len)
    return longest


class KairosMultimodalPipeline:
    def __init__(
        self,
        model_config: KairosConfig,
        data_config: DataConfig,
        train_config: TrainConfig,
        eval_data_config: DataConfig | None = None,
        tokenizer: KairosTokenizer | None = None,
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.eval_data_config = eval_data_config
        self.train_config = train_config
        self.tokenizer = tokenizer or KairosTokenizer()

        self.device = train_config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: KairosDiffusionFM | None = None
        self.dataset: KairosPretrainingDataset | None = None
        self.loader: DataLoader | None = None
        self.eval_loader: DataLoader | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.hf_trainer: KairosDiffusionTrainer | None = None
        self.writer: SummaryWriter | None = None

        # AMP: fp16 on pre-Ampere (T4), bf16 on Ampere+; GradScaler only for fp16.
        if torch.cuda.is_available():
            self.amp_device_type = "cuda"
            self.amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16
        elif torch.backends.mps.is_available():
            self.amp_device_type = "mps"
            self.amp_dtype = torch.float16
        else:
            self.amp_device_type = "cpu"
            self.amp_dtype = torch.float16
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.use_amp else "cpu", enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        self.log_rows: list[dict] = []
        self.eval_log_rows: list[dict] = []
        self.best_loss: float = float("inf")
        self.best_eval_loss: float = float("inf")
        self.global_step: int = 0
        self.skipped_nonfinite_steps: int = 0
        self.nan_log: list[dict] = []
        self._last_nonfinite_batch: dict | None = None

    def _autocast(self):
        return torch.autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.use_amp)

    @property
    def _num_workers(self) -> int:
        if self.data_config.num_workers is not None:
            return self.data_config.num_workers
        return 4 if self.data_config.batch_size > 1 else 0

    # ------------------------------------------------------------------ build
    def _build_dataset(self, data_config: DataConfig | None = None) -> KairosPretrainingDataset:
        dc = data_config or self.data_config
        # A second build() reusing the same DataConfig (e.g. resuming training with a fresh
        # KairosMultimodalPipeline instance, or a test that builds twice) used to hit the
        # ValueError below, because the raw examples were freed after the first build. Cache
        # the already-tokenized (arrow-backed, memory-mapped — cheap to hold) dataset on the
        # config instead: a rebuild reuses it directly, which is both correct and faster
        # (skips re-tokenizing) instead of just failing.
        cached = getattr(dc, "_cached_dataset", None)
        if cached is not None:
            return cached

        text_ex = dc.text_examples or []
        multi_ex = dc.multimodal_examples or []
        # chain instead of list+list: avoids holding a second full-length copy of the
        # combined examples in memory just to hand it to KairosPretrainingDataset, which
        # only needs to iterate it once.
        examples = list(itertools.chain(text_ex, multi_ex)) if text_ex or multi_ex else []
        if examples:
            ds = KairosPretrainingDataset(
                multimodal_examples=examples,
                tokenizer=self.tokenizer,
                max_len=dc.max_len,
                stride=dc.stride,
                pack=dc.pack,
            )
        elif dc.multimodal_path:
            ds = KairosPretrainingDataset(
                multimodal_path=dc.multimodal_path,
                tokenizer=self.tokenizer,
                max_len=dc.max_len,
                stride=dc.stride,
                pack=dc.pack,
            )
        else:
            raise ValueError("DataConfig needs multimodal_examples, text_examples, and/or multimodal_path")

        # The dataset above is now fully tokenized and arrow-backed (memory-mapped) — the raw
        # examples that fed it are no longer needed, but dc.text_examples/multimodal_examples
        # would otherwise sit retained on self.data_config for the pipeline's entire lifetime
        # (training, benchmarking, checkpointing...), and get re-inherited by every DataLoader
        # worker process on fork. Free them here, right after they've served their purpose.
        # Stash counts first so run_config_dict() can still report how many examples were used.
        dc._text_examples_count = len(text_ex) or None
        dc._multimodal_examples_count = len(multi_ex) or None
        dc.text_examples = None
        dc.multimodal_examples = None
        dc._cached_dataset = ds
        del examples, text_ex, multi_ex
        return ds

    def build(self) -> KairosMultimodalPipeline:
        """Wires up dataset, model, optimizer, scheduler, and (if resuming later) the checkpoint."""
        dc, tc = self.data_config, self.train_config

        self.dataset = self._build_dataset()
        num_workers = self._num_workers
        self.loader = DataLoader(
            self.dataset,
            batch_size=dc.batch_size,
            shuffle=dc.shuffle,
            drop_last=dc.drop_last,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        if self.eval_data_config is not None:
            self.eval_loader = DataLoader(
                self._build_dataset(self.eval_data_config),
                batch_size=self.eval_data_config.batch_size,
                shuffle=False,
                drop_last=False,
            )

        self.model = KairosDiffusionFM(
            self.model_config, vocab_size=len(self.tokenizer), num_octet_families=self.tokenizer.NUM_OCTET_FAMILIES
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=tc.lr)
        n_steps = max(1, tc.epochs * len(self.loader))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=n_steps)

        run_dir = Path(tc.run_dir)
        self.ckpt_dir = run_dir / "checkpoints"
        self.tb_dir = run_dir / "tensorboard"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "training_config.json").write_text(json.dumps(self.run_config_dict(), indent=2))

        self.hf_trainer = KairosDiffusionTrainer(
            model=self.model, args=TrainingArguments(output_dir=str(run_dir), report_to=tc.report_to)
        )
        self.hf_trainer.mask_eps = tc.mask_eps
        self.hf_trainer.mask_p_max = tc.mask_p_max
        self.hf_trainer.mask_reweight = tc.mask_reweight
        self.hf_trainer.octet_loss_weight = tc.octet_loss_weight
        self.writer = SummaryWriter(str(self.tb_dir))

        if tc.hub_repo_id and tc.hub_push_every_ckpt:
            from huggingface_hub import HfApi

            HfApi().create_repo(tc.hub_repo_id, private=tc.hub_private, exist_ok=True)
        return self

    # -------------------------------------------------------------- summary
    def summary(self, benchmark: bool = True, n_bench_steps: int = 5) -> TrainingSummary:
        """Report params/memory/time. When benchmark=True, memory numbers come from one real
        forward+backward+optimizer.step() (same measurement as memory_report()) instead of the
        param-count formulas, and the remaining n_bench_steps-1 steps are timed for the step-time
        estimate. Model/optimizer state and the loader are restored after a single snapshot —
        this replaces the old pattern of summary() and memory_report() each doing their own
        separate deepcopy (which doubled the transient RAM spike and made it worse each time
        both were called back to back)."""
        self._require_built()

        mem_report = None
        avg_step_time = None
        if benchmark:
            model_state = copy.deepcopy(self.model.state_dict())
            optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            loader_iter = iter(self.loader)

            def step_fn():
                batch = next(loader_iter)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                self.optimizer.zero_grad()
                with self._autocast():
                    loss = self.hf_trainer.compute_loss(self.model, batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            try:
                # one real step, fully measured (params/grads/optimizer/activations/RSS) —
                # this *is* the memory_report() measurement, done once and reused here. Also
                # timed, so it counts as the first of n_bench_steps rather than an extra step
                # outside that budget (previously: n_bench_steps=1 left avg_step_time_sec=None,
                # since "remaining" was 0 and this first step's own time was discarded).
                batch = next(loader_iter)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                def loss_fn():
                    return self.hf_trainer.compute_loss(self.model, batch)

                t0 = time.perf_counter()
                mem_report = detailed_memory_report(
                    self.model, self.optimizer, loss_fn, self.device,
                    autocast_ctx=self._autocast, scaler=self.scaler,
                )
                first_step_time = time.perf_counter() - t0

                # remaining steps just for timing, continuing from the already-stepped model
                remaining = max(0, n_bench_steps - 1)
                rest_avg = benchmark_step_time(step_fn, n_steps=remaining, warmup=0) if remaining else None
                if rest_avg is not None:
                    avg_step_time = (first_step_time + rest_avg * remaining) / (1 + remaining)
                else:
                    avg_step_time = first_step_time
            except StopIteration:
                pass
            finally:
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optimizer_state)
                del model_state, optimizer_state
                self._release_transient_memory()

        ts = training_summary(
            self.model,
            self.loader,
            epochs=self.train_config.epochs,
            step_fn=None,
            n_bench_steps=n_bench_steps,
            num_experts_per_tok=self.model_config.num_experts_per_tok if self.model_config.use_moe else None,
            num_local_experts=self.model_config.num_local_experts if self.model_config.use_moe else None,
        )
        if avg_step_time is not None:
            ts.avg_step_time_sec = avg_step_time
            ts.estimated_total_time_sec = avg_step_time * ts.total_steps
        if mem_report is not None:
            ts.param_memory_mb = mem_report.unique_param_bytes / 1e6
            ts.optimizer_memory_mb = mem_report.optimizer_state_bytes / 1e6
            ts.total_memory_mb = mem_report.rss_after_optimizer_step_mb - mem_report.rss_before_mb
            ts.measured_memory = True
        return ts

    # ------------------------------------------------------------ memory report
    def memory_report(self) -> DetailedMemoryReport:
        """Real measured memory for one train step (params/grads/optimizer/activations/RSS),
        standalone version of the measurement summary(benchmark=True) now folds in directly.
        Runs one real step then restores model/optimizer state."""
        self._require_built()
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        loader_iter = iter(self.loader)
        batch = next(loader_iter)
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        def loss_fn():
            return self.hf_trainer.compute_loss(self.model, batch)

        try:
            return detailed_memory_report(
                self.model,
                self.optimizer,
                loss_fn,
                self.device,
                autocast_ctx=self._autocast,
                scaler=self.scaler,
            )
        finally:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            del model_state, optimizer_state
            self._release_transient_memory()

    @staticmethod
    def _release_transient_memory() -> None:
        """After a deepcopy'd state_dict snapshot is dropped, glibc's malloc doesn't always
        return the freed pages to the OS — RSS stays high and creeps up further on repeated
        calls (the "RAM explodes then plateaus, and grows again next run" symptom). gc.collect()
        clears the Python-level references; malloc_trim(0) asks glibc to actually give the pages
        back. Best-effort: silently no-ops on platforms without glibc (e.g. macOS)."""
        import gc

        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass

    # ---------------------------------------------------------------- eval
    def evaluate(self, step: int | None = None) -> dict | None:
        """Loss on the held-out eval set, capped at ``eval_batches``; logged to tensorboard and ``eval_log_rows``; returns the result dict or ``None`` when no eval set is configured."""
        if self.eval_loader is None:
            return None
        step = self.global_step if step is None else step
        tc = self.train_config
        losses: list[float] = []
        seen = 0
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in self.eval_loader:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    with self._autocast():
                        losses.append(self.hf_trainer.compute_loss(self.model, batch).item())
                    seen += 1
                    if tc.eval_batches and seen >= tc.eval_batches:
                        break
        finally:
            self.model.train()
        if not losses:
            return None

        eval_loss = sum(losses) / len(losses)
        self.best_eval_loss = min(self.best_eval_loss, eval_loss)
        row = {"step": step, "loss": eval_loss, "batches": seen}
        self.eval_log_rows.append(row)
        self.writer.add_scalar("eval/loss", eval_loss, step)
        return row

    # ---------------------------------------------------------- overfit test
    def _optimizer_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, scheduler) -> None:
        """AMP backward + grad clip + optimizer/scheduler step (shared by ``train`` and ``overfit_test``)."""
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_config.grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        scheduler.step()

    def overfit_test(
        self,
        n_examples: int = 64,
        steps: int = 200,
        lr: float = 1e-3,
        seed: int = 0,
        progress_callback=None,
        mask_p_max: float | None = None,
        mask_reweight: bool | None = None,
    ) -> list[dict]:
        """Trains on a tiny subset to verify the model can memorize before a long run (a plateau high means a structural problem; non-destructive, restores model/optimizer/scheduler/loader state on return).

        ``mask_p_max``/``mask_reweight`` temporarily override the trainer's masking curriculum for this call
        only (e.g. MAE-style low, fixed-rate corruption vs. full diffusion), restored afterwards either way.
        """
        self._require_built()
        from torch.utils.data import Subset

        saved_model = copy.deepcopy(self.model.state_dict())
        saved_opt = copy.deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None
        saved_sch = copy.deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None
        saved_loader = self.loader
        saved_step = self.global_step
        saved_logs = self.log_rows
        saved_best = self.best_loss
        saved_eval_logs = self.eval_log_rows
        saved_best_eval = self.best_eval_loss
        saved_mask_p_max = self.hf_trainer.mask_p_max
        saved_mask_reweight = self.hf_trainer.mask_reweight
        if mask_p_max is not None:
            self.hf_trainer.mask_p_max = mask_p_max
        if mask_reweight is not None:
            self.hf_trainer.mask_reweight = mask_reweight

        n = min(n_examples, len(self.dataset))
        if n == 0 or steps <= 0:
            raise ValueError("overfit_test needs a non-empty dataset and steps > 0")
        indices = random.Random(seed).sample(range(len(self.dataset)), n)
        loader = DataLoader(
            Subset(self.dataset, indices),
            batch_size=self.data_config.batch_size,
            shuffle=True,
        )
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
        self.log_rows = []
        self.eval_log_rows = []
        self.best_loss = float("inf")
        self.best_eval_loss = float("inf")

        logs: list[dict] = []
        try:
            self.model.train()
            it = iter(loader)
            for step in range(steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                opt.zero_grad()
                with self._autocast():
                    loss = self.hf_trainer.compute_loss(self.model, batch)
                loss_val = loss.item()
                if not math.isfinite(loss_val):
                    warnings.warn(
                        f"overfit_test hit a non-finite loss at step {step} - the model cannot memorize "
                        f"this data (diagnostics: {getattr(self.hf_trainer, 'last_loss_diagnostics', None)})",
                        stacklevel=2,
                    )
                    break
                self._optimizer_step(loss, opt, sch)

                self.global_step += 1
                logs.append({"step": step, "loss": loss_val})
                if progress_callback is not None:
                    progress_callback(self.global_step, steps, loss_val)

            first, last = logs[0]["loss"], logs[-1]["loss"]
            tail = sum(r["loss"] for r in logs[-max(1, len(logs) // 10) :]) / max(1, len(logs) // 10)
            print(f"overfit_test: {len(logs)} steps on {n} examples - loss {first:.4f} -> {tail:.4f} (last {last:.4f})")
            return logs
        finally:
            self.model.load_state_dict(saved_model)
            if saved_opt is not None:
                self.optimizer.load_state_dict(saved_opt)
            if saved_sch is not None:
                self.scheduler.load_state_dict(saved_sch)
            self.loader = saved_loader
            self.global_step = saved_step
            self.log_rows = saved_logs
            self.best_loss = saved_best
            self.eval_log_rows = saved_eval_logs
            self.best_eval_loss = saved_best_eval
            self.hf_trainer.mask_p_max = saved_mask_p_max
            self.hf_trainer.mask_reweight = saved_mask_reweight

    # ---------------------------------------------------------------- train
    def _require_built(self):
        if self.model is None:
            raise RuntimeError("call .build() before .train()/.evaluate()/.check_per_modality_loss()")

    def train(
        self, progress_callback=None, resume: bool = True, memory_bank: KairosMultiCache | None = None
    ) -> list[dict]:
        """Runs the training loop; resumes from local last.pt or the hub if unavailable."""
        self._require_built()
        tc = self.train_config
        self.model.train()

        last_ckpt = self.ckpt_dir / "last.pt"
        start_epoch = 1
        if resume and last_ckpt.exists():
            start_epoch = self._safe_resume(last_ckpt)
        elif resume and tc.hub_repo_id:
            ckpt = self._try_resume_from_hub(tc.hub_repo_id)
            if ckpt is not None:
                start_epoch = ckpt.get("epoch", 1)

        total_steps = tc.epochs * len(self.loader)
        skipped_nonfinite = 0
        consecutive_nan = 0
        prev_cache = None
        use_memory_gate = getattr(self.model_config, "use_memory_gate", False)

        try:
            for epoch in range(start_epoch, tc.epochs + 1):
                epoch_loss = 0.0
                for batch in self.loader:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                    cache_params = None
                    if use_memory_gate:
                        cur_bsz = batch["input_ids"].size(0)
                        memory_caches = [c for c in (prev_cache, memory_bank) if c is not None]
                        if memory_caches:
                            cache_params = gate_memory_bank(self.model, memory_caches, cur_bsz)
                        else:
                            cache_params = KairosMultiCache(self.model_config)

                    self.optimizer.zero_grad()
                    with self._autocast():
                        loss = self.hf_trainer.compute_loss(self.model, batch, cache_params=cache_params)
                    loss_val = loss.item()
                    if use_memory_gate:
                        for scale_cache in cache_params.caches:
                            for layer_idx, s in enumerate(scale_cache.ssm_caches):
                                if s is not None:
                                    scale_cache.ssm_caches[layer_idx] = s.detach()
                        prev_cache = cache_params
                    if not math.isfinite(loss_val):
                        # a corrupted batch can spike the loss; skip it
                        skipped_nonfinite += 1
                        consecutive_nan += 1
                        self._last_nonfinite_batch = batch
                        diag = getattr(self.hf_trainer, "last_loss_diagnostics", None)
                        self.nan_log.append({"step": self.global_step + 1, "loss": loss_val, **(diag or {})})
                        warnings.warn(
                            f"non-finite loss ({loss_val}) at step {self.global_step + 1}, skipping batch "
                            f"(diagnostics: {diag}); see pipe.nan_log for the full list",
                            stacklevel=2,
                        )
                        if progress_callback is not None:
                            progress_callback(self.global_step, total_steps, loss_val)  # keep progress visible
                        if consecutive_nan >= tc.max_consecutive_nan:
                            source = self.locate_nan_source()
                            raise RuntimeError(
                                f"{consecutive_nan} consecutive non-finite batches at step "
                                f"{self.global_step + 1} — training is not converging, aborting instead of "
                                f"looping silently. Last diagnostics: {diag}. First non-finite module: "
                                f"{source}. Full history in pipe.nan_log."
                            )
                        continue
                    consecutive_nan = 0

                    self._optimizer_step(loss, self.optimizer, self.scheduler)

                    epoch_loss += loss_val
                    self.global_step += 1

                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                    self.log_rows.append({"step": self.global_step, "epoch": epoch, "loss": loss_val})

                    if progress_callback is not None:
                        progress_callback(self.global_step, total_steps, loss_val)

                    if tc.eval_every > 0 and self.global_step % tc.eval_every == 0:
                        eval_row = self.evaluate()
                        if eval_row is not None:
                            print(
                                f"[eval @ step {eval_row['step']}] loss {eval_row['loss']:.4f} "
                                f"(best {self.best_eval_loss:.4f})"
                            )

                    if self.global_step % tc.last_ckpt_every == 0:
                        self._save(last_ckpt, loss_val, epoch)  # periodically overwritten, resumable
                    if self.global_step % tc.save_every == 0:
                        step_ckpt = self.ckpt_dir / f"step_{self.global_step:06d}.pt"
                        self._save(step_ckpt, loss_val, epoch)
                        if tc.hub_repo_id and tc.hub_push_every_ckpt:
                            self._push_checkpoint_to_hub(step_ckpt)
                            self._push_checkpoint_to_hub(last_ckpt)

                self._save(last_ckpt, loss_val, epoch)  # always resumable at epoch boundaries

                avg_loss = epoch_loss / max(1, len(self.loader))
                self.writer.add_scalar("train/epoch_avg_loss", avg_loss, epoch)
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self._save(self.ckpt_dir / "best.pt", avg_loss, epoch)
                    if tc.hub_repo_id and tc.hub_push_every_ckpt:
                        self._push_checkpoint_to_hub(self.ckpt_dir / "best.pt")

            last_ckpt.unlink(missing_ok=True)  # finished cleanly: nothing to resume
            if tc.eval_every > 0:
                self.evaluate()  # final eval on the converged weights
        finally:
            self.skipped_nonfinite_steps = skipped_nonfinite
            self.writer.flush()
            self.writer.close()

        return self.log_rows

    def locate_nan_source(self) -> dict | None:
        """Re-runs the last non-finite batch with hooks to find which module first."""
        if self._last_nonfinite_batch is None:
            return None
        return locate_first_nonfinite_module(
            self.model, lambda: self.hf_trainer.compute_loss(self.model, self._last_nonfinite_batch)
        )

    def inspect_batch(self, n: int = 1, from_loader: bool = True) -> list[dict]:
        """Pulls `n` real post-tokenization batches and reports per row: text, modalities, id."""
        self._require_built()
        vocab_size = len(self.tokenizer)
        num_modalities = self.model_config.num_modalities
        reports = []

        loader = self.loader if from_loader else [next(iter(self.loader))]
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n:
                break
            input_ids = batch["input_ids"]
            modality_ids = batch.get("modality_ids")
            pad_mask = batch.get("mask")
            prompt_len = batch.get("prompt_len")

            for row in range(input_ids.size(0)):
                row_ids = input_ids[row]
                row_modality = modality_ids[row] if modality_ids is not None else None

                oob_token = ((row_ids < 0) | (row_ids >= vocab_size)).nonzero(as_tuple=True)[0]
                oob_modality = (
                    ((row_modality < 0) | (row_modality >= num_modalities)).nonzero(as_tuple=True)[0]
                    if row_modality is not None
                    else torch.empty(0, dtype=torch.long)
                )

                text_mask = (row_modality == int(Modality.TEXT)) if row_modality is not None else None
                text_ids = row_ids[text_mask] if text_mask is not None else row_ids
                try:
                    text_preview = self.tokenizer.decode(text_ids.tolist(), skip_special_tokens=True)[:200]
                except Exception as e:  # noqa: BLE001 - best-effort preview, never blocks inspection
                    text_preview = f"<decode failed: {e}>"

                modality_counts = {}
                if row_modality is not None:
                    values, counts = row_modality.unique(return_counts=True)
                    modality_counts = dict(zip(values.tolist(), counts.tolist()))

                # a long run of the same
                real_len = int(pad_mask[row].sum()) if pad_mask is not None else row_ids.size(0)
                run_lengths = _consecutive_run_lengths(row_ids[:real_len])
                max_run_id, max_run_len = max(run_lengths.items(), key=lambda kv: kv[1]) if run_lengths else (None, 0)
                id_values, id_counts = row_ids[:real_len].unique(return_counts=True)
                top_ids = sorted(zip(id_values.tolist(), id_counts.tolist()), key=lambda kv: -kv[1])[:5]

                reports.append(
                    {
                        "batch": batch_idx,
                        "row": row,
                        "seq_len": row_ids.size(0),
                        "prompt_len": int(prompt_len[row]) if prompt_len is not None else None,
                        "pad_frac": float(1 - pad_mask[row].float().mean()) if pad_mask is not None else None,
                        "modality_counts": modality_counts,
                        "token_id_range": (int(row_ids.min()), int(row_ids.max())),
                        "out_of_bounds": {
                            "token_ids": oob_token.tolist(),  # ids outside [0, vocab)
                            "modality_ids": oob_modality.tolist(),  # ids outside [0, num_modalities)
                        },
                        "text_preview": text_preview,
                        "input_ids": row_ids.tolist(),  # raw ids, as fed to the model
                        "modality_ids": row_modality.tolist() if row_modality is not None else None,
                        "top_token_ids": top_ids,  # most frequent ids
                        "max_repeat_run": {"id": max_run_id, "length": max_run_len},
                    }
                )

        return reports

    def run_config_dict(self) -> dict:
        """model/train/data config as a plain JSON-safe dict — the actual hyperparameters behind."""
        dc = asdict(self.data_config)
        for key, count_attr in (("text_examples", "_text_examples_count"), ("multimodal_examples", "_multimodal_examples_count")):
            if dc.get(key) is not None:
                dc[key] = f"<{len(dc[key])} examples, omitted>"
            elif getattr(self.data_config, count_attr, None):
                # already freed by _build_dataset (see there) — report the count we stashed.
                dc[key] = f"<{getattr(self.data_config, count_attr)} examples, freed after tokenizing>"
        edc = asdict(self.eval_data_config) if self.eval_data_config is not None else None
        if edc is not None:
            for key, count_attr in (
                ("text_examples", "_text_examples_count"),
                ("multimodal_examples", "_multimodal_examples_count"),
            ):
                if edc.get(key) is not None:
                    edc[key] = f"<{len(edc[key])} examples, omitted>"
                elif getattr(self.eval_data_config, count_attr, None):
                    edc[key] = f"<{getattr(self.eval_data_config, count_attr)} examples, freed after tokenizing>"
        return {
            "model_config": self.model_config.to_dict(),
            "train_config": asdict(self.train_config),
            "data_config": dc,
            "eval_data_config": edc,
        }

    def _save(self, path: Path, loss_val: float, epoch: int = 1):
        torch.save(
            {
                "step": self.global_step,
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "loss": loss_val,
                "config": self.model_config.to_dict(),
                "train_config": asdict(self.train_config),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Loads a local .pt checkpoint into the built model/optimizer/scheduler."""
        self._require_built()
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step = ckpt.get("step", self.global_step)
        return ckpt

    def _safe_resume(self, path: Path) -> int:
        # an incompatible checkpoint (different model_config) starts fresh
        try:
            ckpt = self.load_checkpoint(str(path))
            return ckpt.get("epoch", 1)
        except RuntimeError as e:
            warnings.warn(f"{path} is incompatible with the current model_config, starting fresh: {e}", stacklevel=2)
            return 1

    # ------------------------------------------------------------- hf hub
    def generate(self, prompt_ids, max_new_tokens=64, modality=Modality.TEXT, seed=None, **kwargs):
        """Block-diffusion continuation of a token prompt; thin wrapper around ``KairosDiffusionGenerationMixin.generate`` handling device placement, modality ids and AMP; returns prompt + generated ids (decode with ``self.tokenizer.decode(...)``)."""
        self._require_built()
        if seed is not None:
            torch.manual_seed(seed)
        prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        modality_ids = torch.full_like(prompt, int(modality))
        self.model.eval()
        try:
            with self._autocast():
                out = self.model.generate(
                    input_ids=prompt,
                    modality_ids=modality_ids,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
        finally:
            self.model.train()
        sequences = out.sequences if hasattr(out, "sequences") else out
        return sequences[0].tolist()

    def _push_checkpoint_to_hub(self, path: Path):
        from huggingface_hub import HfApi

        prefix = f"{self.train_config.hub_subfolder}/" if self.train_config.hub_subfolder else ""
        HfApi().upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{prefix}checkpoints/{path.name}",
            repo_id=self.train_config.hub_repo_id,
        )

    def load_checkpoint_from_hub(self, repo_id: str, filename: str = "checkpoints/last.pt"):
        """Downloads a checkpoint from a HF hub repo and loads it, same."""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id, filename)
        return self.load_checkpoint(path)

    def _try_resume_from_hub(self, repo_id: str):
        # best-effort: no checkpoint on the repo
        try:
            return self.load_checkpoint_from_hub(repo_id)
        except Exception:  # noqa: BLE001 — no hub checkpoint yet, start fresh
            return None

    def push_to_hub(
        self, repo_id: str, private: bool = False, license: str = "apache-2.0", subfolder: str | None = None
    ):
        """Pushes model, config, checkpoints, logs, and a model card to the hub."""
        from huggingface_hub import HfApi

        self._require_built()
        subfolder = subfolder or self.train_config.hub_subfolder
        prefix = f"{subfolder}/" if subfolder else ""
        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)

        self.model_config.register_for_auto_class()
        self.model.register_for_auto_class("AutoModelForCausalLM")

        # save_pretrained + upload_folder honors the subfolder
        export_dir = Path(self.train_config.run_dir) / "hf_export"
        export_dir.mkdir(parents=True, exist_ok=True)
        # tied SWA/DeltaNet weights break save_pretrained's tied-weight logic
        self.model_config.save_pretrained(str(export_dir))
        torch.save(self.model.state_dict(), export_dir / "pytorch_model.bin")
        api.upload_folder(repo_id=repo_id, folder_path=str(export_dir), path_in_repo=subfolder or ".")

        if self.ckpt_dir.exists():
            api.upload_folder(repo_id=repo_id, folder_path=str(self.ckpt_dir), path_in_repo=f"{prefix}checkpoints")
        if self.tb_dir.exists():
            api.upload_folder(repo_id=repo_id, folder_path=str(self.tb_dir), path_in_repo=f"{prefix}tensorboard")

        api.upload_file(
            path_or_fileobj=json.dumps(self.run_config_dict(), indent=2).encode("utf-8"),
            path_in_repo=f"{prefix}training_config.json",
            repo_id=repo_id,
        )
        api.upload_file(
            path_or_fileobj=self._model_card(repo_id, license).encode("utf-8"),
            path_in_repo=f"{prefix}README.md",
            repo_id=repo_id,
        )
        return repo_id

    def _model_card(self, repo_id: str, license: str) -> str:
        template_path = Path(__file__).parent / "templates" / "model_card.md"
        mc = self.model_config
        best_loss = self.best_loss if self.best_loss != float("inf") else "n/a"
        return template_path.read_text().format(
            license=license,
            name=repo_id.split("/")[-1],
            repo_id=repo_id,
            d_model=getattr(mc, "d_model", "?"),
            n_layers=getattr(mc, "n_layers", "?"),
            n_routed_experts=getattr(mc, "n_routed_experts", "?"),
            n_shared_experts=getattr(mc, "n_shared_experts", "?"),
            num_experts_per_tok=getattr(mc, "num_experts_per_tok", "?"),
            vocab_size=mc.vocab_size,
            best_loss=best_loss,
            global_step=self.global_step,
        )

    # ------------------------------------------------------------- checks
    def check_per_modality_loss(self, n_batches: int = 1) -> dict[str, float]:
        """Diffusion loss averaged per modality, so a router-ignored modality is caught."""
        self._require_built()
        self.model.eval()
        losses_by_modality: dict[str, list[float]] = defaultdict(list)

        seen = 0
        with torch.no_grad():
            for batch in self.loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                for name, loss_val in self._per_modality_loss_for_batch(batch).items():
                    losses_by_modality[name].append(loss_val)
                seen += 1
                if seen >= n_batches:
                    break

        self.model.train()
        return {name: sum(v) / len(v) for name, v in losses_by_modality.items()}

    def _per_modality_loss_for_batch(self, batch: dict) -> dict[str, float]:
        x0 = batch["input_ids"]
        prompt_len = batch["prompt_len"]
        modality_ids = batch.get("modality_ids")
        pad_mask = batch.get("mask")
        if modality_ids is None:
            return {}

        noise_mask, p = make_diffusion_mask(x0, prompt_len, pad_mask)
        if not noise_mask.any():
            return {}

        per_token_loss, _, _, _ = compute_masked_diffusion_losses(self.model, x0, noise_mask, p, modality_ids)

        out = {}
        modality_at_noised = modality_ids[noise_mask]
        for m in modality_at_noised.unique().tolist():
            sel = modality_at_noised == m
            out[Modality(m).name] = per_token_loss[sel].mean().item()
        return out