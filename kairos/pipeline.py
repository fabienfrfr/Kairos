"""Thin, declarative wrapper: tokenizer -> dataset -> model -> optimizer/scheduler -> train."""

from __future__ import annotations

import copy
import datetime
import itertools
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments

from .attentions import ATTN_IMPL as _ATTN_IMPL
from .attentions import CAUSAL_CONV1D_BACKEND as _CAUSAL_CONV1D_BACKEND
from .attentions import DELTA_RULE_BACKEND as _DELTA_RULE_BACKEND
from .dataset import (
    KairosPretrainingDataset,
    diagnose_built_dataset,
    diagnose_control_alternation,
    diagnose_multimodal_examples,
    find_rows_with_modality,
    plot_tokenized_row,
    preview_tokenized_examples,
)
from .modeling import KairosConfig, KairosDiffusionFM, KairosMultiCache, gate_memory_bank
from .tokenizer import KairosTokenizer, Modality
from .trainer import (
    KairosDiffusionTrainer,
    compute_masked_diffusion_losses,
    make_diffusion_mask,
    stage_mask_schedule,
)
from .utils import (
    DetailedMemoryReport,
    ModuleTimeReport,
    TrainingSummary,
    benchmark_step_time,
    detailed_memory_report,
    locate_first_nonfinite_module,
    profile_module_time,
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
    pack: bool = False  # concatenate examples before windowing (see KairosPretrainingDataset)
    # None: min(4, os.cpu_count()-1) if batch_size > 1 else 0; override (e.g. 0) if memory-tight.
    num_workers: int | None = None
    # per-modality-key encode_* scale_factor override; unset keys use tokenizer defaults
    modality_scale_factors: dict | None = None


@dataclass
class TrainConfig:
    lr: float = 3e-4
    # epochs (deprecated): if set, means diffusion_epochs=epochs, mae/transition=0.
    epochs: int | None = None
    # single-pipeline MAE -> transition -> diffusion; see trainer.stage_mask_schedule
    mae_epochs: int = 1  # stage 1: fixed-rate corruption at mask_mae_p_max, no reweighting
    transition_epochs: int = 1  # stage 2: linear ramp from the MAE values to the diffusion targets
    diffusion_epochs: int = 1  # stage 3: full masked-diffusion at mask_p_max/mask_reweight
    save_every: int = 200
    last_ckpt_every: int = 20  # how often last.pt (resume point)
    eval_every: int = 0  # explicit step-based cadence; 0 = derive from eval_every_epochs instead
    eval_every_epochs: float | None = 0.5  # cadence in epochs; ignored if eval_every > 0
    eval_at_start: bool = True  # also evaluate once before the first training step
    eval_batches: int = 2  # eval batches per evaluation, capped; keep small
    grad_clip: float = 1.0
    mask_eps: float = 1e-3  # floor of masked-diffusion rate p; CE/p variance grows as this shrinks
    mask_p_max: float = 1.0  # diffusion-stage (target) ceiling of p
    mask_reweight: bool = True  # diffusion-stage (target): divide CE by p
    mask_mae_p_max: float = 0.3  # MAE-stage fixed-ish corruption ceiling (cheap/stable to optimize)
    mask_mae_reweight: bool = False  # MAE-stage: plain CE, no 1/p variance blowup
    octet_loss_weight: float = 1.0  # weight of the octet-family loss
    # train-time self-conditioning rate; 0.0 disables it (generate() then sees OOD input).
    self_conditioning_prob: float = 0.5
    max_consecutive_nan: int = 50  # abort with a diagnosis instead
    run_dir: str = "checkpoints/kairos-multimodal/run_01"
    device: str | None = None  # None -> auto
    report_to: list = field(default_factory=list)
    hub_repo_id: str | None = None  # set to also push each checkpoint
    hub_push_every_ckpt: bool = False  # requires hub_repo_id; pushes checkpoints
    hub_private: bool = False
    hub_subfolder: str | None = None  # push under repo_id/<subfolder>
    compile_model: bool = True  # torch.compile(model_forward); memory measurement bypasses it
    amp_dtype: str | None = None  # None=auto (bf16 whenever supported); "bf16"/"fp16" forces it

    def __post_init__(self):
        if self.epochs is not None:
            self.mae_epochs = 0
            self.transition_epochs = 0
            self.diffusion_epochs = self.epochs
        for name in ("mae_epochs", "transition_epochs", "diffusion_epochs"):
            if getattr(self, name) < 0:
                raise ValueError(f"TrainConfig.{name} must be >= 0, got {getattr(self, name)}")
        self.epochs = self.mae_epochs + self.transition_epochs + self.diffusion_epochs
        if self.epochs <= 0:
            raise ValueError(
                "TrainConfig needs at least one epoch: mae_epochs + transition_epochs + "
                f"diffusion_epochs must be > 0, got {self.epochs}"
            )
        if self.amp_dtype is not None and self.amp_dtype not in ("bf16", "fp16"):
            raise ValueError(f"TrainConfig.amp_dtype must be None, 'bf16', or 'fp16', got {self.amp_dtype!r}")


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


def _resolve_amp_dtype(amp_dtype_override: str | None, bf16_supported: bool) -> torch.dtype:
    """Picks the CUDA autocast dtype: override wins; else bf16 if supported, fp16 as the fallback."""
    if amp_dtype_override == "bf16":
        return torch.bfloat16
    if amp_dtype_override == "fp16":
        return torch.float16
    return torch.bfloat16 if bf16_supported else torch.float16


def init_distributed() -> bool:
    """Initializes the torch.distributed process group from torchrun env vars; True if DDP."""
    if dist.is_available() and not dist.is_initialized() and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=30))
        return True
    return False


def launch_ddp(
    model_config,
    data_config,
    eval_data_config,
    train_config,
    n_proc=None,
    wait: bool = True,
    resume: bool = True,
    action: str = "train",
    action_kwargs: dict | None = None,
):
    """Spawns a fresh pipeline via torchrun; one GPU per rank (flex + memory gate work)."""
    # Configs are pickled to run_dir/ddp_job; pass fresh copies (post-build DataConfig is emptied).
    run_dir = Path(train_config.run_dir)
    job_dir = run_dir / "ddp_job"
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "configs.pkl").open("wb") as f:
        pickle.dump((model_config, data_config, eval_data_config, train_config), f)
    with (job_dir / "job.pkl").open("wb") as f:
        pickle.dump({"action": action, "resume": resume, "kwargs": action_kwargs or {}}, f)
    n_proc = n_proc or (torch.cuda.device_count() or 1)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={n_proc}",
        "--rdzv-backend=c10d",
        str(Path(__file__).parent / "_entry_ddp.py"),
        str(job_dir),
    ]
    log_path = run_dir / f"{action}_ddp.log"
    with log_path.open("ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        if wait and proc.wait() != 0:
            raise RuntimeError(f"DDP {action} failed (see {log_path})")
    return proc


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

        # build() empties data_config in place, so snapshot pre-build configs for a DDP-launched job.
        self._ddp_snapshot = (
            copy.deepcopy(model_config),
            copy.deepcopy(data_config),
            copy.deepcopy(eval_data_config),
            copy.deepcopy(train_config),
        )

        # distributed (DDP) state; init_distributed()/build() fill these from torchrun env.
        self.distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.compiled = False  # build() sets this once torch.compile is (or isn't) applied

        self.device = train_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"train_config.device={self.device!r} but no CUDA device is available")

        self.model: KairosDiffusionFM | None = None
        self.dataset: KairosPretrainingDataset | None = None
        self.loader: DataLoader | None = None
        self.eval_loader: DataLoader | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.hf_trainer: KairosDiffusionTrainer | None = None
        self.writer: SummaryWriter | None = None

        # AMP: bf16 only on real tensor-core hardware (Ampere+); T4 "supports" bf16 unaccelerated.
        if torch.cuda.is_available():
            self.amp_device_type = "cuda"
            bf16_hw = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability() >= (8, 0)
            self.amp_dtype = _resolve_amp_dtype(train_config.amp_dtype, bf16_hw)
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

        # background disk writes for the frequent resumable checkpoint; see _save()/_flush_checkpoint_writes()
        self._ckpt_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_ckpt_futures: list[Future] = []

    @property
    def is_main_process(self) -> bool:
        """True on rank 0 (or single process): logging / checkpoint / hub work is gated on this."""
        return not self.distributed or self.rank == 0

    @property
    def _forward_model(self):
        """Rank-0 diagnostic forward uses the raw model; DDP wrapper needs all ranks in sync."""
        return self.model if self.distributed else self.model_forward

    def _autocast(self):
        return torch.autocast(device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.use_amp)

    @property
    def _num_workers(self) -> int:
        if self.data_config.num_workers is not None:
            return self.data_config.num_workers
        if self.data_config.batch_size <= 1:
            return 0
        # cap at available CPUs; a flat "8" can stall/hang on a 1-2 core machine.
        cpu_count = os.cpu_count() or 1
        return max(0, min(8, cpu_count - 1))

    # ------------------------------------------------------------------ build
    def _build_dataset(self, data_config: DataConfig | None = None) -> KairosPretrainingDataset:
        dc = data_config or self.data_config
        # cache the tokenized dataset on dc: a rebuild reuses it (correct + faster)
        cached = getattr(dc, "_cached_dataset", None)
        if cached is not None:
            return cached

        text_ex = dc.text_examples or []
        multi_ex = dc.multimodal_examples or []
        # chain instead of list+list: avoids a second full-length copy just to iterate it once
        examples = list(itertools.chain(text_ex, multi_ex)) if text_ex or multi_ex else []
        if examples:
            ds = KairosPretrainingDataset(
                multimodal_examples=examples,
                tokenizer=self.tokenizer,
                max_len=dc.max_len,
                stride=dc.stride,
                pack=dc.pack,
                modality_scale_factors=dc.modality_scale_factors,
            )
        elif dc.multimodal_path:
            ds = KairosPretrainingDataset(
                multimodal_path=dc.multimodal_path,
                tokenizer=self.tokenizer,
                max_len=dc.max_len,
                stride=dc.stride,
                pack=dc.pack,
                modality_scale_factors=dc.modality_scale_factors,
            )
        else:
            raise ValueError("DataConfig needs multimodal_examples, text_examples, and/or multimodal_path")

        # free the raw examples now that the dataset is tokenized; stash counts for reporting
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

        self.distributed = init_distributed()
        if torch.cuda.is_available():
            # TF32 matmul: free precision/perf trade on Ampere+ for the fp32 ops autocast leaves untouched.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        if self.distributed:
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.rank = int(os.environ.get("RANK", "0"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = f"cuda:{self.local_rank}"
            if not dist.is_initialized():
                raise RuntimeError("distributed env present but process group failed to initialize")

        self.dataset = self._build_dataset()
        num_workers = self._num_workers
        train_sampler = None
        if self.distributed:
            train_sampler = DistributedSampler(
                self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=dc.shuffle
            )
        self._train_sampler = train_sampler
        self.loader = DataLoader(
            self.dataset,
            batch_size=dc.batch_size,
            shuffle=False if self.distributed else dc.shuffle,
            sampler=train_sampler,
            drop_last=dc.drop_last,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        if self.eval_data_config is not None:
            eval_dataset = self._build_dataset(self.eval_data_config)
            eval_sampler = None
            if self.distributed:
                eval_sampler = DistributedSampler(eval_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.eval_data_config.batch_size,
                shuffle=False,
                sampler=eval_sampler,
                drop_last=False,
            )

        self.model = KairosDiffusionFM(
            self.model_config, vocab_size=len(self.tokenizer), num_octet_families=self.tokenizer.NUM_OCTET_FAMILIES
        ).to(self.device)
        # state_dict/generate keep using self.model; self.model_forward is the parallel wrapper.
        should_compile = tc.compile_model and torch.cuda.is_available()
        self.compiled = should_compile
        if self.distributed:
            # Conditional forward (unused params) -> DDP needs find_unused_parameters.
            forward_module = torch.compile(self.model) if should_compile else self.model
            self.model_forward = DDP(
                forward_module,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,
            )
        else:
            if torch.cuda.device_count() > 1:
                warnings.warn(
                    "Multiple GPUs visible but not launched via torchrun: model_forward will only "
                    "use cuda:0. train() still auto-launches DDP across all visible GPUs; launch "
                    "the whole script via torchrun to use every GPU for summary/generate/evaluate "
                    "too.",
                    stacklevel=2,
                )
            self.model_forward = torch.compile(self.model) if should_compile else self.model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=tc.lr, fused=torch.cuda.is_available())
        n_steps = max(1, tc.epochs * len(self.loader))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=n_steps)

        run_dir = Path(tc.run_dir)
        self.ckpt_dir = run_dir / "checkpoints"
        self.tb_dir = run_dir / "tensorboard"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main_process:
            (run_dir / "training_config.json").write_text(json.dumps(self.run_config_dict(), indent=2))

        self.hf_trainer = KairosDiffusionTrainer(
            model=self.model, args=TrainingArguments(output_dir=str(run_dir), report_to=tc.report_to)
        )
        self.hf_trainer.mask_eps = tc.mask_eps
        self.hf_trainer.mask_p_max = tc.mask_p_max
        self.hf_trainer.mask_reweight = tc.mask_reweight
        self.hf_trainer.octet_loss_weight = tc.octet_loss_weight
        self.hf_trainer.self_conditioning_prob = tc.self_conditioning_prob
        self.writer = SummaryWriter(str(self.tb_dir)) if self.is_main_process else None

        if tc.hub_repo_id and tc.hub_push_every_ckpt and self.is_main_process:
            from huggingface_hub import HfApi

            HfApi().create_repo(tc.hub_repo_id, private=tc.hub_private, exist_ok=True)
        return self

    # -------------------------------------------------------------- summary
    def summary(self, benchmark: bool = True, n_bench_steps: int = 5, ddp_benchmark: bool = False) -> TrainingSummary:
        """Report params/memory/time; benchmark=True uses one real step, not param formulas."""
        self._require_built()
        if ddp_benchmark and benchmark and not self.distributed and torch.cuda.device_count() > 1:
            kwargs = {"benchmark": True, "n_bench_steps": n_bench_steps}
            results = self._run_via_ddp("summary", resume=False, action_kwargs=kwargs)
            return results["summary"]
        if benchmark and not self.distributed and torch.cuda.device_count() > 1:
            warnings.warn(
                "Multiple GPUs visible: benchmark will only use cuda:0 (fast, single-process). "
                "Pass summary(ddp_benchmark=True) for a real multi-GPU torchrun benchmark -- slower "
                "to start (full pipeline rebuild per rank) and can hang in constrained notebook "
                "environments (Kaggle/Colab rendezvous issues).",
                stacklevel=2,
            )

        mem_report = None
        avg_step_time = None
        if benchmark and self.is_main_process:
            model_state = copy.deepcopy(self.model.state_dict())
            optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            loader_iter = iter(self.loader)

            def step_fn():
                batch = next(loader_iter)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                self.optimizer.zero_grad()
                with self._autocast():
                    loss = self.hf_trainer.compute_loss(self._forward_model, batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            try:
                # memory: uncompiled self.model only, isolated from any compiled timing below.
                batch = next(loader_iter)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                def loss_fn():
                    return self.hf_trainer.compute_loss(self.model, batch)

                mem_report = detailed_memory_report(
                    self.model,
                    self.optimizer,
                    loss_fn,
                    self.device,
                    autocast_ctx=self._autocast,
                    scaler=self.scaler,
                )

                # timing: self._forward_model, warmup absorbs the one-time compile cost if any.
                warmup = 1 if self.compiled else 0
                avg_step_time = benchmark_step_time(step_fn, n_steps=n_bench_steps, warmup=warmup)
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
        # here only for the benchmark=False static estimate; benchmark=True routes via DDP above.
        n_gpus = torch.cuda.device_count() if not self.distributed and torch.cuda.device_count() > 1 else 1
        ts.n_gpus = n_gpus
        ts.attn_impl = _ATTN_IMPL
        ts.delta_rule_backend = _DELTA_RULE_BACKEND
        ts.causal_conv1d_backend = _CAUSAL_CONV1D_BACKEND
        if n_gpus > 1:
            ts.steps_per_epoch = math.ceil(ts.steps_per_epoch / n_gpus)
            ts.total_steps = ts.epochs * ts.steps_per_epoch
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
        """Real measured memory for one train step; runs it then restores model/optimizer state."""
        self._require_built()
        if not self.is_main_process:
            return None
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        loader_iter = iter(self.loader)
        batch = next(loader_iter)
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        def loss_fn():
            # hooks mutate a closure var every layer; torch.compile would recompile on each change.
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

    # ------------------------------------------------------------ time profile
    def profile(self, n_steps: int = 3) -> ModuleTimeReport:
        """Per-module wall-clock time (fwd+bwd), avg over n_steps; restores state after."""
        self._require_built()
        if not self.is_main_process:
            return None
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        loader_iter = iter(self.loader)
        batch = next(loader_iter)
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        def step_fn():
            self.optimizer.zero_grad()
            with self._autocast():
                loss = self.hf_trainer.compute_loss(self._forward_model, batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        try:
            return profile_module_time(self.model, step_fn, self.device, n_steps=n_steps)
        finally:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            del model_state, optimizer_state
            self._release_transient_memory()

    def data_report(self, sample_size: int = 200, split: str = "train"):
        """Raw-vs-tokenized per-modality stats; falls back to the built dataset if freed."""
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        dc = self.data_config if split == "train" else self.eval_data_config
        loader = self.loader if split == "train" else self.eval_loader
        if dc is None:
            raise RuntimeError(f"data_report(split={split!r}) needs an eval_data_config on this pipeline")

        examples = dc.multimodal_examples
        if examples is None and dc.multimodal_path:
            examples = torch.load(dc.multimodal_path, weights_only=False)
        try:
            if examples:
                return diagnose_multimodal_examples(
                    examples,
                    tokenizer=self.tokenizer,
                    modality_scale_factors=dc.modality_scale_factors,
                    max_len=dc.max_len,
                    sample_size=sample_size,
                )
            if loader is not None and getattr(loader.dataset, "ds", None) is not None:
                return diagnose_built_dataset(loader.dataset.ds, sample_size=sample_size)
            raise RuntimeError(
                f"data_report(split={split!r}) needs multimodal_examples/multimodal_path, or a built pipeline"
            )
        finally:
            self._release_transient_memory()

    def control_alternation_report(self, sample_size: int = 200, split: str = "train"):
        """Decodes each sampled row's CONTROL segments; surfaces rows truncated mid-byte by a window cut (see dataset module)."""
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        loader = self.loader if split == "train" else self.eval_loader
        if loader is None or getattr(loader.dataset, "ds", None) is None:
            raise RuntimeError(f"control_alternation_report(split={split!r}) needs a built pipeline")
        return diagnose_control_alternation(loader.dataset.ds, self.tokenizer, sample_size=sample_size)

    def plot_row(self, row: int = 0, split: str = "train", max_segments: int | None = None) -> None:
        """Reconstructs and plots row `row` of the built (tokenized) dataset in document order."""
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        loader = self.loader if split == "train" else self.eval_loader
        if loader is None or getattr(loader.dataset, "ds", None) is None:
            raise RuntimeError(f"plot_row(split={split!r}) needs a built pipeline")
        input_ids = loader.dataset.ds.with_format("torch")[row]["input_ids"]
        plot_tokenized_row(self.tokenizer, input_ids, max_segments=max_segments)

    def show(
        self,
        n: int = 3,
        modality: str | None = None,
        split: str = "train",
        seed: int = 0,
        max_segments: int | None = None,
    ) -> None:
        """Plots n real tokenized rows; modality=None: n random, else only rows containing it."""
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        loader = self.loader if split == "train" else self.eval_loader
        if loader is None or getattr(loader.dataset, "ds", None) is None:
            raise RuntimeError(f"show(split={split!r}) needs a built pipeline")
        built = loader.dataset.ds

        if modality is None:
            total = len(built)
            rows = random.Random(seed).sample(range(total), min(n, total)) if total else []
        else:
            rows = find_rows_with_modality(built, modality, n=n, seed=seed)
            if not rows:
                print(f"no rows with modality={modality!r} found in this sample")
                return

        for row in rows:
            print(f"--- row {row} ---")
            self.plot_row(row=row, split=split, max_segments=max_segments)

    def preview_tokenized(
        self, n: int = 1, modality: str | None = None, split: str = "train", sample_size: int = 200, seed: int = 0
    ) -> None:
        """Tokenized pipe.show(): overlays CONTROL state/action; n rows/modality by default."""
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        loader = self.loader if split == "train" else self.eval_loader
        if loader is None or getattr(loader.dataset, "ds", None) is None:
            raise RuntimeError(f"preview_tokenized(split={split!r}) needs a built pipeline")
        preview_tokenized_examples(
            self.tokenizer, loader.dataset.ds, n=n, modality=modality, sample_size=sample_size, seed=seed
        )

    def inspect_batch_df(self, n: int = 1, from_loader: bool = True):
        """inspect_batch(), pre-flattened into a pandas DataFrame ready to print."""
        import pandas as pd

        reports = self.inspect_batch(n=n, from_loader=from_loader)
        return pd.DataFrame(
            [
                {
                    "row": r["row"],
                    "modality_counts": r["modality_counts"],
                    "token_id_range": r["token_id_range"],
                    "top_token_ids": r["top_token_ids"],
                    "max_repeat_run": r["max_repeat_run"],
                    "out_of_bounds_tokens": len(r["out_of_bounds"]["token_ids"]),
                    "out_of_bounds_modality": len(r["out_of_bounds"]["modality_ids"]),
                    "pad_frac": round(r["pad_frac"], 3) if r["pad_frac"] is not None else None,
                }
                for r in reports
            ]
        )

    @staticmethod
    def _release_transient_memory() -> None:
        """gc.collect() + malloc_trim(0): return freed deepcopy pages to the OS; no-op off glibc."""
        import gc

        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass

    # ---------------------------------------------------------------- eval
    def evaluate(self, step: int | None = None) -> dict | None:
        """Loss on the held-out eval set, capped at eval_batches; logs to tensorboard."""
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
                        losses.append(self.hf_trainer.compute_loss(self.model_forward, batch).item())
                    seen += 1
                    if tc.eval_batches and seen >= tc.eval_batches:
                        break
        finally:
            self.model.train()
        if not losses:
            return None

        local_sum = sum(losses)
        local_count = len(losses)
        if self.distributed:
            stats = torch.tensor([local_sum, local_count], device=self.device, dtype=torch.float)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            local_sum, local_count = float(stats[0]), int(stats[1])
        eval_loss = local_sum / max(1, local_count)

        if not self.is_main_process:
            return None
        self.best_eval_loss = min(self.best_eval_loss, eval_loss)
        row = {"step": step, "loss": eval_loss, "batches": local_count}
        self.eval_log_rows.append(row)
        self.writer.add_scalar("eval/loss", eval_loss, step)
        return row

    # ---------------------------------------------------------- overfit test
    def _optimizer_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, scheduler) -> None:
        """AMP backward + grad clip + optimizer/scheduler step, shared by train and overfit_test."""
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
        """Trains on a tiny subset to check memorization; walks the active curriculum stages."""
        self._require_built()
        if not self.is_main_process:
            return []
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

        tc = self.train_config
        if mask_p_max is not None or mask_reweight is not None:
            # single fixed regime for the whole call, explicitly requested
            fixed_p_max = mask_p_max if mask_p_max is not None else saved_mask_p_max
            fixed_reweight = mask_reweight if mask_reweight is not None else saved_mask_reweight

            def stage_at(_step: int) -> tuple[float, float]:
                return fixed_p_max, float(fixed_reweight)
        else:
            # same curriculum as train(), proportionally compressed into `steps` total steps
            mae_steps = round(steps * tc.mae_epochs / tc.epochs)
            transition_steps = round(steps * tc.transition_epochs / tc.epochs)

            def stage_at(_step: int) -> tuple[float, float]:
                return stage_mask_schedule(
                    _step,
                    mae_steps,
                    transition_steps,
                    tc.mask_mae_p_max,
                    tc.mask_mae_reweight,
                    tc.mask_p_max,
                    tc.mask_reweight,
                )

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
                self.hf_trainer.mask_p_max, self.hf_trainer.mask_reweight = stage_at(step)
                with self._autocast():
                    loss = self.hf_trainer.compute_loss(self._forward_model, batch)
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
        self,
        progress_callback=None,
        resume: bool = True,
        memory_bank: KairosMultiCache | None = None,
        ddp_launch: bool | None = None,
        n_proc: int | None = None,
    ) -> list[dict]:
        """Runs the training loop; resumes from local last.pt or the hub if unavailable."""
        self._require_built()
        # single process, several GPUs visible -> spawn a torchrun job (one GPU per rank).
        auto_launch = ddp_launch is None and not self.distributed and torch.cuda.device_count() > 1
        if auto_launch or ddp_launch:
            results = self._run_via_ddp("train", n_proc=n_proc, progress_callback=progress_callback, resume=resume)
            for key, value in results.items():
                setattr(self, key, value)
            return self.log_rows
        tc = self.train_config
        self.model.train()

        last_ckpt = self.ckpt_dir / "last.pt"
        start_epoch = 1
        if resume:
            if self.distributed:
                # rank0 materializes last.pt (local or hub); all ranks then load the same file.
                if self.is_main_process and not last_ckpt.exists() and tc.hub_repo_id:
                    ckpt = self._try_resume_from_hub(tc.hub_repo_id)
                    if ckpt is not None:
                        torch.save(ckpt, last_ckpt)
                dist.barrier()
                if last_ckpt.exists():
                    start_epoch = self._safe_resume(last_ckpt)
            else:
                if last_ckpt.exists():
                    start_epoch = self._safe_resume(last_ckpt)
                elif tc.hub_repo_id:
                    ckpt = self._try_resume_from_hub(tc.hub_repo_id)
                    if ckpt is not None:
                        start_epoch = ckpt.get("epoch", 1)

        total_steps = tc.epochs * len(self.loader)
        mae_steps = tc.mae_epochs * len(self.loader)
        transition_steps = tc.transition_epochs * len(self.loader)
        # eval_every (steps) wins if set explicitly; else derive from eval_every_epochs
        if tc.eval_every > 0:
            eval_every_steps = tc.eval_every
        elif tc.eval_every_epochs:
            eval_every_steps = max(1, round(tc.eval_every_epochs * len(self.loader)))
        else:
            eval_every_steps = 0
        skipped_nonfinite = 0
        consecutive_nan = 0
        prev_cache = None
        use_memory_gate = getattr(self.model_config, "use_memory_gate", False)

        if tc.eval_at_start and self.eval_loader is not None:
            self.evaluate(step=self.global_step)

        try:
            for epoch in range(start_epoch, tc.epochs + 1):
                if self._train_sampler is not None:
                    self._train_sampler.set_epoch(epoch)  # reshuffle per epoch identically on all ranks
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
                    # pure function of global_step: resuming mid-curriculum picks the right stage
                    self.hf_trainer.mask_p_max, self.hf_trainer.mask_reweight = stage_mask_schedule(
                        self.global_step,
                        mae_steps,
                        transition_steps,
                        tc.mask_mae_p_max,
                        tc.mask_mae_reweight,
                        tc.mask_p_max,
                        tc.mask_reweight,
                    )
                    with self._autocast():
                        loss = self.hf_trainer.compute_loss(self.model_forward, batch, cache_params=cache_params)
                    loss_val = loss.item()
                    if self.distributed:
                        # identical loss on every rank -> identical skip/abort decisions and logs
                        loss_reduced = loss.detach().clone()
                        dist.all_reduce(loss_reduced, op=dist.ReduceOp.SUM)
                        loss_val = float(loss_reduced / self.world_size)
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
                            progress_callback(self.global_step, total_steps, loss_val)
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

                    if self.is_main_process:
                        self.writer.add_scalar("train/loss", loss_val, self.global_step)
                        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                        self.writer.add_scalar("train/mask_p_max", self.hf_trainer.mask_p_max, self.global_step)
                        self.writer.add_scalar(
                            "train/mask_reweight", self.hf_trainer.mask_reweight, self.global_step
                        )
                        self.log_rows.append({"step": self.global_step, "epoch": epoch, "loss": loss_val})

                        if progress_callback is not None:
                            progress_callback(self.global_step, total_steps, loss_val)

                    if eval_every_steps > 0 and self.global_step % eval_every_steps == 0:
                        eval_row = self.evaluate()  # all ranks join the all_reduce inside
                        if eval_row is not None:
                            print(
                                f"[eval @ step {eval_row['step']}] loss {eval_row['loss']:.4f} "
                                f"(best {self.best_eval_loss:.4f})"
                            )

                    if self.is_main_process:
                        if self.global_step % tc.last_ckpt_every == 0:
                            self._save(last_ckpt, loss_val, epoch, wait=False)  # overwritten, resumable
                        if self.global_step % tc.save_every == 0:
                            step_ckpt = self.ckpt_dir / f"step_{self.global_step:06d}.pt"
                            self._save(step_ckpt, loss_val, epoch)
                            if tc.hub_repo_id and tc.hub_push_every_ckpt:
                                self._flush_checkpoint_writes()  # last_ckpt may still be in-flight (async save)
                                self._push_checkpoint_to_hub(step_ckpt)
                                self._push_checkpoint_to_hub(last_ckpt)

                if self.is_main_process:
                    self._save(last_ckpt, loss_val, epoch)  # always resumable at epoch boundaries

                    avg_loss = epoch_loss / max(1, len(self.loader))
                    self.writer.add_scalar("train/epoch_avg_loss", avg_loss, epoch)
                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss
                        self._save(self.ckpt_dir / "best.pt", avg_loss, epoch)
                        if tc.hub_repo_id and tc.hub_push_every_ckpt:
                            self._push_checkpoint_to_hub(self.ckpt_dir / "best.pt")

            if self.is_main_process:
                self._flush_checkpoint_writes()  # last_ckpt's last async write must land before unlink
                last_ckpt.unlink(missing_ok=True)  # finished cleanly: nothing to resume
            # final eval on the converged weights (skip if the last step already evaluated)
            if eval_every_steps > 0 and self.global_step % eval_every_steps != 0:
                self.evaluate()  # all ranks join the all_reduce inside
        finally:
            self._flush_checkpoint_writes()
            self.skipped_nonfinite_steps = skipped_nonfinite
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

        return self.log_rows

    def _run_via_ddp(
        self, action: str, n_proc=None, progress_callback=None, resume: bool = True, action_kwargs: dict | None = None
    ) -> dict:
        """Spawns a torchrun job (one GPU per rank), shared by train() and summary()."""
        tc = self.train_config
        log_path = Path(tc.run_dir) / f"{action}_ddp.log"
        log_path.unlink(missing_ok=True)  # fresh log so step replay below never sees stale lines
        proc = launch_ddp(
            *self._ddp_snapshot, n_proc=n_proc, resume=resume, wait=False, action=action, action_kwargs=action_kwargs
        )
        index = 0
        while proc.poll() is None:
            index = self._replay_ddp_log(log_path, index, progress_callback)
            time.sleep(0.25)
        index = self._replay_ddp_log(log_path, index, progress_callback)
        if proc.returncode != 0:
            raise RuntimeError(f"DDP {action} failed (see {log_path})")
        return self._load_ddp_results(Path(tc.run_dir))

    @staticmethod
    def _replay_ddp_log(log_path: Path, index: int, progress_callback) -> int:
        """Feeds progress_callback with step lines freshly written by the torchrun rank-0 process."""
        if progress_callback is None or not log_path.exists():
            return index
        lines = log_path.read_text().splitlines()
        for line in lines[index:]:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "step":
                step_total = parts[1].split("/", 1)
                if len(step_total) == 2:
                    try:
                        progress_callback(int(step_total[0]), int(step_total[1]), float(parts[3]))
                    except ValueError:
                        pass
            index += 1
        return index

    @staticmethod
    def _load_ddp_results(run_dir: Path) -> dict:
        res_path = run_dir / "ddp_job" / "results.pkl"
        if not res_path.exists():
            raise RuntimeError(f"DDP job finished but no {res_path} was written")
        with res_path.open("rb") as f:
            return pickle.load(f)

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
                except Exception as e:  # noqa: BLE001 - best-effort preview
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
                            "modality_ids": oob_modality.tolist(),  # ids outside valid range
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
        """model/train/data config as a plain JSON-safe dict; the actual hyperparameters behind."""
        dc = asdict(self.data_config)
        for key, count_attr in (
            ("text_examples", "_text_examples_count"),
            ("multimodal_examples", "_multimodal_examples_count"),
        ):
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

    def _flush_checkpoint_writes(self) -> None:
        """Waits for pending async checkpoint writes; re-raises the first failure instead of losing it."""
        while self._pending_ckpt_futures:
            self._pending_ckpt_futures.pop(0).result()

    def _save(self, path: Path, loss_val: float, epoch: int = 1, wait: bool = True):
        """Clones state (fast) then writes off the main thread; wait=True blocks until it lands."""
        payload = {
            "step": self.global_step,
            "epoch": epoch,
            "model_state": copy.deepcopy(self.model.state_dict()),
            "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler_state": copy.deepcopy(self.scheduler.state_dict()),
            "loss": loss_val,
            "config": self.model_config.to_dict(),
            "train_config": asdict(self.train_config),
        }
        self._pending_ckpt_futures.append(self._ckpt_executor.submit(torch.save, payload, path))
        if wait:
            self._flush_checkpoint_writes()

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
        """Block-diffusion continuation of a prompt; wraps generate() with device/AMP handling."""
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
