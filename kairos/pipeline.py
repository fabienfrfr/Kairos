"""Thin, declarative wrapper: tokenizer -> dataset -> model -> optimizer/scheduler -> train loop -> per-modality check."""

from __future__ import annotations

import copy
import json
import math
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments

from .dataset import KairosPretrainingDataset
from .modeling import (
    KairosConfig,
    KairosDiffusionLLM,
    all_rows_carry_plan,
    build_carried_cache,
    build_memory_cache,
    random_state_carry_plan,
)
from .tokenizer import KairosTokenizer, Modality
from .trainer import KairosDiffusionTrainer
from .utils import TrainingSummary, locate_first_nonfinite_module, training_summary


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
    pack: bool = False  # concatenate samples before chunking so only the last chunk is padded


@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 3
    save_every: int = 200
    last_ckpt_every: int = 20  # how often last.pt (resume point) is overwritten; was every step, too slow
    grad_clip: float = 1.0
    max_consecutive_nan: int = 50  # abort with a diagnosis instead of silently looping through a dead run
    run_dir: str = "checkpoints/kairos-multimodal/run_01"
    device: str | None = None  # None -> auto
    report_to: list = field(default_factory=list)
    hub_repo_id: str | None = None  # set to also push each periodic checkpoint to this HF repo
    hub_push_every_ckpt: bool = False  # requires hub_repo_id; pushes step_*.pt/last.pt/best.pt as they're saved
    hub_private: bool = False
    hub_subfolder: str | None = None  # push checkpoints/model under repo_id/<subfolder> instead of repo root
    state_carry: bool = False  # carry DeltaNet cache across batches (memory/robustness regularizer)
    state_carry_mode: str = "all"  # "all": every row gets agg(all prev rows); "random": per-row random recipe
    state_carry_agg: str = "mean"  # "mean" or "sum", used by both modes
    state_carry_max_group: int = 3  # only used when state_carry_mode == "random"


def _consecutive_run_lengths(ids: torch.Tensor) -> dict[int, int]:
    """Maps each distinct id to the length of its longest run of consecutive occurrences. A single id repeating for hundreds of positions in a row is a classic sign of a corrupted or degenerate example (e.g. a modality encoder producing a constant/clipped output)."""
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
        tokenizer: KairosTokenizer | None = None,
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.tokenizer = tokenizer or KairosTokenizer()

        self.device = train_config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: KairosDiffusionLLM | None = None
        self.dataset: KairosPretrainingDataset | None = None
        self.loader: DataLoader | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.hf_trainer: KairosDiffusionTrainer | None = None
        self.writer: SummaryWriter | None = None

        self.log_rows: list[dict] = []
        self.best_loss: float = float("inf")
        self.global_step: int = 0
        self.skipped_nonfinite_steps: int = 0
        self.nan_log: list[dict] = []
        self._last_nonfinite_batch: dict | None = None

    # ------------------------------------------------------------------ build
    def _build_dataset(self) -> KairosPretrainingDataset:
        dc = self.data_config
        examples = list(dc.text_examples or []) + list(dc.multimodal_examples or [])
        if examples:
            return KairosPretrainingDataset(
                multimodal_examples=examples,
                tokenizer=self.tokenizer,
                max_len=dc.max_len,
                stride=dc.stride,
                pack=dc.pack,
            )
        if dc.multimodal_path:
            return KairosPretrainingDataset(
                multimodal_path=dc.multimodal_path,
                tokenizer=self.tokenizer,
                max_len=dc.max_len,
                stride=dc.stride,
                pack=dc.pack,
            )
        raise ValueError("DataConfig needs multimodal_examples, text_examples, and/or multimodal_path")

    def build(self) -> KairosMultimodalPipeline:
        """Wires up dataset, model, optimizer, scheduler, and (if resuming later) the checkpoint dirs."""
        dc, tc = self.data_config, self.train_config

        self.dataset = self._build_dataset()
        self.loader = DataLoader(self.dataset, batch_size=dc.batch_size, shuffle=dc.shuffle, drop_last=dc.drop_last)

        self.model = KairosDiffusionLLM(self.model_config, vocab_size=len(self.tokenizer)).to(self.device)
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
        self.writer = SummaryWriter(str(self.tb_dir))

        if tc.hub_repo_id and tc.hub_push_every_ckpt:
            from huggingface_hub import HfApi

            HfApi().create_repo(tc.hub_repo_id, private=tc.hub_private, exist_ok=True)
        return self

    # -------------------------------------------------------------- summary
    def summary(self, benchmark: bool = True, n_bench_steps: int = 5) -> TrainingSummary:
        """Params/memory/estimated-time report; benchmark steps are timed then reverted, no side effects."""
        self._require_built()

        step_fn = None
        model_state = optimizer_state = None
        if benchmark:
            model_state = copy.deepcopy(self.model.state_dict())
            optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            loader_iter = iter(self.loader)

            def step_fn():
                batch = next(loader_iter)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                loss = self.hf_trainer.compute_loss(self.model, batch)
                loss.backward()
                self.optimizer.step()

        try:
            return training_summary(
                self.model,
                self.loader,
                epochs=self.train_config.epochs,
                step_fn=step_fn,
                n_bench_steps=n_bench_steps,
                num_experts_per_tok=self.model_config.num_experts_per_tok if self.model_config.use_moe else None,
                num_local_experts=self.model_config.num_local_experts if self.model_config.use_moe else None,
            )
        finally:
            if benchmark:
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optimizer_state)

    # ---------------------------------------------------------------- train
    def _require_built(self):
        if self.model is None:
            raise RuntimeError("call .build() before .train()/.check_per_modality_loss()")

    def train(self, progress_callback=None, resume: bool = True) -> list[dict]:
        """Runs the training loop; resumes from local last.pt or, failing that, the hub repo, if resume=True."""
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
        prev_bsz = None
        running_memory = {}
        use_memory_bank = getattr(self.model_config, "use_memory_bank", False)

        try:
            for epoch in range(start_epoch, tc.epochs + 1):
                epoch_loss = 0.0
                for batch in self.loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    cache_params = None
                    if use_memory_bank:
                        cur_bsz = batch["input_ids"].size(0)
                        cache_params, running_memory = build_memory_cache(
                            self.model, prev_cache, running_memory, cur_bsz
                        )
                    elif tc.state_carry:
                        cur_bsz = batch["input_ids"].size(0)
                        if prev_cache is not None and prev_bsz != cur_bsz:
                            prev_cache = None  # batch size changed; start fresh
                        plan = (
                            random_state_carry_plan(cur_bsz, tc.state_carry_max_group)
                            if tc.state_carry_mode == "random"
                            else all_rows_carry_plan(cur_bsz)
                        )
                        cache_params = build_carried_cache(self.model_config, prev_cache, plan, agg=tc.state_carry_agg)
                        prev_bsz = cur_bsz

                    self.optimizer.zero_grad()
                    loss = self.hf_trainer.compute_loss(self.model, batch, cache_params=cache_params)
                    loss_val = loss.item()
                    if use_memory_bank:
                        prev_cache = cache_params
                        running_memory = {k: v.detach() for k, v in running_memory.items()}
                    elif tc.state_carry:
                        prev_cache = cache_params  # forward wrote this step's final states into it
                    if not math.isfinite(loss_val):
                        # a corrupted batch can spike the loss to inf/nan; skip it rather than step on garbage
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

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=tc.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()

                    epoch_loss += loss_val
                    self.global_step += 1

                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                    self.log_rows.append({"step": self.global_step, "epoch": epoch, "loss": loss_val})

                    if progress_callback is not None:
                        progress_callback(self.global_step, total_steps, loss_val)

                    if self.global_step % tc.last_ckpt_every == 0:
                        self._save(last_ckpt, loss_val, epoch)  # overwritten periodically: resumable, not every step
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

            last_ckpt.unlink(missing_ok=True)  # finished cleanly: nothing to resume from anymore
        finally:
            self.skipped_nonfinite_steps = skipped_nonfinite
            self.writer.flush()
            self.writer.close()

        return self.log_rows

    def locate_nan_source(self) -> dict | None:
        """Re-runs the last non-finite batch with hooks to find which module first outputs NaN/Inf."""
        if self._last_nonfinite_batch is None:
            return None
        return locate_first_nonfinite_module(
            self.model, lambda: self.hf_trainer.compute_loss(self.model, self._last_nonfinite_batch)
        )

    def inspect_batch(self, n: int = 1, from_loader: bool = True) -> list[dict]:
        """Pulls `n` real post-tokenization batches and reports per row: text, modalities, id bounds."""
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
                except Exception as e:  # noqa: BLE001 - best-effort preview, never block the inspection itself
                    text_preview = f"<decode failed: {e}>"

                modality_counts = {}
                if row_modality is not None:
                    values, counts = row_modality.unique(return_counts=True)
                    modality_counts = dict(zip(values.tolist(), counts.tolist()))

                # numeric anomaly signals: a long run of the exact same token id (degenerate/
                # corrupted example) is a stronger red flag than any single-value stat — but
                # exclude the padding tail, where a long run of pad_token_id is normal, not a bug
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
                            "token_ids": oob_token.tolist(),  # positions with id outside [0, vocab_size)
                            "modality_ids": oob_modality.tolist(),  # positions with id outside [0, num_modalities)
                        },
                        "text_preview": text_preview,
                        "input_ids": row_ids.tolist(),  # raw ids, exactly what the embedding layer indexes with
                        "modality_ids": row_modality.tolist() if row_modality is not None else None,
                        "top_token_ids": top_ids,  # [(id, count), ...] most frequent ids in this row
                        "max_repeat_run": {"id": max_run_id, "length": max_run_len},
                    }
                )

        return reports

    def run_config_dict(self) -> dict:
        """model/train/data config as a plain JSON-safe dict — the actual hyperparameters behind a run, since model_config alone (saved in checkpoints) doesn't capture lr/epochs/pack/state_carry/etc."""
        dc = asdict(self.data_config)
        for key in ("text_examples", "multimodal_examples"):
            if dc.get(key) is not None:
                dc[key] = f"<{len(dc[key])} examples, omitted>"
        return {"model_config": self.model_config.to_dict(), "train_config": asdict(self.train_config), "data_config": dc}

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
        # an incompatible checkpoint (different model_config) starts fresh instead of crashing
        try:
            ckpt = self.load_checkpoint(str(path))
            return ckpt.get("epoch", 1)
        except RuntimeError as e:
            warnings.warn(f"{path} is incompatible with the current model_config, starting fresh: {e}", stacklevel=2)
            return 1

    # ------------------------------------------------------------- hf hub
    def _push_checkpoint_to_hub(self, path: Path):
        from huggingface_hub import HfApi

        prefix = f"{self.train_config.hub_subfolder}/" if self.train_config.hub_subfolder else ""
        HfApi().upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{prefix}checkpoints/{path.name}",
            repo_id=self.train_config.hub_repo_id,
        )

    def load_checkpoint_from_hub(self, repo_id: str, filename: str = "checkpoints/last.pt"):
        """Downloads a checkpoint from a HF hub repo and loads it, same as load_checkpoint."""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id, filename)
        return self.load_checkpoint(path)

    def _try_resume_from_hub(self, repo_id: str):
        # best-effort: no checkpoint on the repo yet (fresh run) is not an error
        try:
            return self.load_checkpoint_from_hub(repo_id)
        except Exception:  # noqa: BLE001 — no checkpoint on the hub yet is not an error, just start fresh
            return None

    def push_to_hub(self, repo_id: str, private: bool = False, license: str = "apache-2.0", subfolder: str | None = None):
        """Pushes model, config, checkpoints, tensorboard logs, training config, and a model card to a HF hub repo — under repo_id/<subfolder> if given, so multiple runs/configs can share one repo."""
        from huggingface_hub import HfApi

        self._require_built()
        subfolder = subfolder or self.train_config.hub_subfolder
        prefix = f"{subfolder}/" if subfolder else ""
        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)

        self.model_config.register_for_auto_class()
        self.model.register_for_auto_class("AutoModelForCausalLM")

        # save_pretrained + upload_folder honors the subfolder prefix consistently across versions
        export_dir = Path(self.train_config.run_dir) / "hf_export"
        export_dir.mkdir(parents=True, exist_ok=True)
        # tied SWA/DeltaNet weights break save_pretrained's tied-weight check; save state_dict directly instead
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
        """Diffusion loss averaged per modality over n_batches, so a router-ignored modality can't hide in the global mean."""
        self._require_built()
        self.model.eval()
        losses_by_modality: dict[str, list[float]] = defaultdict(list)

        seen = 0
        with torch.no_grad():
            for batch in self.loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                for name, loss_val in self._per_modality_loss_for_batch(batch).items():
                    losses_by_modality[name].append(loss_val)
                seen += 1
                if seen >= n_batches:
                    break

        self.model.train()
        return {name: sum(v) / len(v) for name, v in losses_by_modality.items()}

    def _per_modality_loss_for_batch(self, batch: dict) -> dict[str, float]:
        import torch.nn.functional as F

        x0 = batch["input_ids"]
        prompt_len = batch["prompt_len"]
        modality_ids = batch.get("modality_ids")
        pad_mask = batch.get("mask")
        if modality_ids is None:
            return {}

        eps = 1e-3
        t = torch.rand(x0.size(0), device=x0.device)
        p = (1 - eps) * t + eps
        p = p[:, None].expand_as(x0)

        noise_mask = torch.rand(x0.shape, device=x0.device) < p
        for i in range(x0.size(0)):
            noise_mask[i, : prompt_len[i]] = False
        if pad_mask is not None:
            noise_mask &= pad_mask.bool()

        if not noise_mask.any():
            return {}

        xt = x0.clone()
        noise = torch.randint_like(x0, self.model.lm_head.vocab_size)
        xt[noise_mask] = noise[noise_mask]

        logits = self.model(decoder_input_ids=xt, modality_ids=modality_ids).logits
        per_token_loss = F.cross_entropy(logits[noise_mask], x0[noise_mask], reduction="none") / p[noise_mask]

        out = {}
        modality_at_noised = modality_ids[noise_mask]
        for m in modality_at_noised.unique().tolist():
            sel = modality_at_noised == m
            out[Modality(m).name] = per_token_loss[sel].mean().item()
        return out
