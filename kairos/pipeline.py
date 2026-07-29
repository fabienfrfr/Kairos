"""
kairos/pipeline.py — a thin, declarative wrapper around the pieces already in
kairos/{modeling,tokenizer,dataset,trainer}.py: build tokenizer -> dataset ->
model -> optimizer/scheduler -> train loop -> per-modality sanity check.

Not "AutoML" in the hyperparameter-search sense — no search, no NAS. It's a
single entry point (`KairosMultimodalPipeline`) driven entirely by three small
dataclasses (`DataConfig`, `TrainConfig`, + the existing `KairosConfig`), so a
notebook only has to declare config and call `.build()` / `.train()`. All the
plumbing that used to be copy-pasted notebook cells (dataset construction,
checkpointing, TensorBoard, the training loop itself) lives here once, tested
once, and is reused by every notebook/script instead of re-typed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments

from .dataset import KairosPretrainingDataset
from .modeling import KairosConfig, KairosDiffusionLLM
from .tokenizer import KairosTokenizer, Modality
from .trainer import KairosDiffusionTrainer


@dataclass
class DataConfig:
    """Everything needed to build the training dataset. Pass either
    `multimodal_examples` (already-built list[dict]) or `multimodal_path`
    (a .pt file, see scripts/pretrain/build_keep_it_simple_multimodal.py).
    `text_examples` (list[dict] with kind="text") is merged in alongside,
    since multimodal builds intentionally don't include their own text."""

    multimodal_examples: list | None = None
    multimodal_path: str | None = None
    text_examples: list | None = None
    max_len: int = 1024
    stride: int = 3
    batch_size: int = 8
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 3
    save_every: int = 200
    grad_clip: float = 1.0
    run_dir: str = "checkpoints/kairos-multimodal/run_01"
    device: str | None = None  # None -> auto (cuda if available else cpu)
    report_to: list = field(default_factory=list)


class KairosMultimodalPipeline:
    """Usage:
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()
    logs = pipe.train()
    pipe.check_per_modality_loss()
    """

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
            )
        if dc.multimodal_path:
            return KairosPretrainingDataset(
                multimodal_path=dc.multimodal_path,
                tokenizer=self.tokenizer,
                max_len=dc.max_len,
                stride=dc.stride,
            )
        raise ValueError("DataConfig needs multimodal_examples, text_examples, and/or multimodal_path")

    def build(self) -> KairosMultimodalPipeline:
        dc, tc = self.data_config, self.train_config

        self.dataset = self._build_dataset()
        self.loader = DataLoader(
            self.dataset,
            batch_size=dc.batch_size,
            shuffle=dc.shuffle,
            drop_last=dc.drop_last,
        )

        self.model = KairosDiffusionLLM(self.model_config, vocab_size=len(self.tokenizer)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=tc.lr)
        n_steps = max(1, tc.epochs * len(self.loader))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=n_steps)

        run_dir = Path(tc.run_dir)
        self.ckpt_dir = run_dir / "checkpoints"
        self.tb_dir = run_dir / "tensorboard"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        self.hf_trainer = KairosDiffusionTrainer(
            model=self.model,
            args=TrainingArguments(output_dir=str(run_dir), report_to=tc.report_to),
        )
        self.writer = SummaryWriter(str(self.tb_dir))
        return self

    # ---------------------------------------------------------------- train
    def _require_built(self):
        if self.model is None:
            raise RuntimeError("call .build() before .train()/.check_per_modality_loss()")

    def train(self) -> list[dict]:
        self._require_built()
        tc = self.train_config
        self.model.train()

        for epoch in range(1, tc.epochs + 1):
            epoch_loss = 0.0
            for batch in self.loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                self.optimizer.zero_grad()
                loss = self.hf_trainer.compute_loss(self.model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=tc.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                self.global_step += 1

                self.writer.add_scalar("train/loss", loss_val, self.global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                self.log_rows.append({"step": self.global_step, "epoch": epoch, "loss": loss_val})

                if self.global_step % tc.save_every == 0:
                    self._save(self.ckpt_dir / f"step_{self.global_step:06d}.pt", loss_val)

            avg_loss = epoch_loss / max(1, len(self.loader))
            self.writer.add_scalar("train/epoch_avg_loss", avg_loss, epoch)
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save(self.ckpt_dir / "best.pt", avg_loss)

        self.writer.flush()
        self.writer.close()
        return self.log_rows

    def _save(self, path: Path, loss_val: float):
        torch.save(
            {
                "step": self.global_step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "loss": loss_val,
                "config": self.model_config.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        self._require_built()
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step = ckpt.get("step", self.global_step)
        return ckpt

    # ------------------------------------------------------------- checks
    def check_per_modality_loss(self, n_batches: int = 1) -> dict[str, float]:
        """Masks the diffusion loss to each modality present in the batch so a
        modality being silently ignored by the router doesn't hide behind a
        healthy-looking global average. Returns {modality_name: mean_loss}."""
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
        """Same masked-diffusion recipe as KairosDiffusionTrainer.compute_loss,
        but the cross-entropy is aggregated separately per modality id instead
        of averaged over the whole batch."""
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
