"""Checks that the pretraining model/pipeline can train on data with the schema and sizes of the project's dataset.

The Hub datasets (ffurfaro/keep-it-simple, ffurfaro/keep-it-simple-multimodal) need network access; this script builds
synthetic rows with the exact row schema and array shapes written by scripts/pretrain/build_keep_it_simple_multimodal.py
(structured signals, so a loss decrease is possible) and runs them through the real dataset, model and trainer of
configs/pretrain.yaml. It cannot say anything about the real data content.

Usage: python scripts/check_pretraining_readiness.py data|train|bench [--source synthetic|hf] [--batch-size B]
  data   tokens per modality and windows needed per example
  train  three-stage run of configs/pretrain.yaml, plain CE per modality before/after, checkpoint round trip, generation
  bench  seconds per step, tokens per second and peak GPU memory for several batch sizes (use it on the GPU)
--source hf reads the real datasets named in the `hf_data` section of --spec (repo ids or local folders).
"""

from __future__ import annotations

import argparse
import itertools
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from kairos.dataset import KairosPretrainingDataset, pack_multimodal_data
from kairos.tokenizer import KairosTokenizer

MODALITIES = ("text", "image_caption", "audio_caption", "video_caption", "lidar", "imu", "control")
COLORS = {"red": (220, 30, 30), "green": (30, 200, 60), "blue": (40, 60, 220)}
TEXTS = (
    "the cat sat on the mat",
    "a small robot learns to walk",
    "water boils at one hundred degrees",
    "kairos reads bytes",
)


def _row(modality: str, caption: str, arrays: dict, meta: dict | None = None) -> dict:
    return {
        "modality": modality,
        "caption": caption,
        "source": "synthetic",
        "data": pack_multimodal_data(arrays),
        "meta": json.dumps(meta) if meta else None,
    }


def synthetic_rows(per_modality: int, seed: int = 0) -> dict[str, list[dict]]:
    """Rows in the builder's schema; shapes: image 32x32x3, audio 8000 @ 8 kHz, video 6x16x16x3, lidar 300x4 [x, y, z, intensity], control 16000."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[dict]] = {m: [] for m in MODALITIES}
    for i in range(per_modality):
        rows["text"].append({"modality": "text", "text": TEXTS[i % len(TEXTS)] + f" number {i % 10}"})
        name = list(COLORS)[i % 3]
        left = i % 2 == 0
        img = np.tile(np.linspace(0, 80, 32, dtype=np.uint8)[None, :, None], (32, 1, 3))
        img[10:22, 4:16] = COLORS[name] if left else img[10:22, 4:16]
        img[10:22, 16:28] = img[10:22, 16:28] if left else COLORS[name]
        rows["image_caption"].append(
            _row("image_caption", f"a {name} square on the {'left' if left else 'right'}", {"image": img})
        )
        freq = (220.0, 440.0, 880.0)[i % 3]
        audio = (0.5 * np.sin(2 * np.pi * freq * np.arange(8000) / 8000)).astype(np.float32)
        rows["audio_caption"].append(
            _row("audio_caption", f"a tone of {int(freq)} hertz", {"audio": audio}, {"sample_rate": 8000})
        )
        video = np.zeros((6, 16, 16, 3), dtype=np.uint8)
        for t in range(6):
            x = t * 2 if i % 2 == 0 else 10 - t * 2
            video[t, 6:10, x : x + 4] = 255
        rows["video_caption"].append(
            _row("video_caption", f"a square moving {'right' if i % 2 == 0 else 'left'}", {"video": video})
        )
        radius = 5.0 + (i % 3) * 5.0
        angle = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        intensity = np.full(300, 0.2 * (1 + i % 3))
        points = np.stack(
            [radius * np.cos(angle), radius * np.sin(angle), 0.1 * rng.standard_normal(300), intensity], axis=1
        )
        rows["lidar"].append(_row("lidar", "", {"points": points.astype(np.float32)}))
        t = np.arange(128) / 32.0
        signal = np.stack([np.sin(2 * np.pi * (1 + i % 3) * t + k) for k in range(3)], axis=1)
        rows["imu"].append(_row("imu", "", {"signal": (0.8 * signal).astype(np.float32)}))
        u = np.linspace(0, 1, 16000, dtype=np.float32)
        state = np.sin(2 * np.pi * (2 + i % 3) * u).astype(np.float32)
        rows["control"].append(_row("control", json.dumps({"k": 1 + i % 3}), {"state": state, "action": -0.5 * state}))
    return rows


def token_stats(
    rows: dict[str, list[dict]], max_len: int = 1024, stride: int = 3, scale_factors: dict | None = None
) -> dict[str, dict]:
    """Per modality: tokens per example, share of the window, windows needed (unpacked), non-pad fraction."""
    tok = KairosTokenizer()
    out = {}
    for name, examples in rows.items():
        ds = KairosPretrainingDataset(
            multimodal_examples=examples,
            tokenizer=tok,
            max_len=max_len,
            stride=stride,
            pack=False,
            modality_scale_factors=scale_factors,
        )
        lens = [int(ds[i]["mask"].sum()) for i in range(len(ds))]
        out[name] = {
            "examples": len(examples),
            "windows": len(ds),
            "tokens_total": sum(lens),
            "tokens_per_example": sum(lens) / len(examples),
            "pad_fraction": 1 - sum(lens) / (len(ds) * ((max_len // stride) * stride)),
        }
    return out


def group_by_modality(examples: list) -> dict[str, list]:
    """Groups example dicts by their `modality` key (text rows carry modality "text")."""
    groups: dict[str, list] = {}
    for row in examples:
        groups.setdefault(row["modality"], []).append(row)
    return groups


def limit_per_modality(examples: list, n: int) -> list:
    """Keeps the first n rows of every modality, in the original order of modalities."""
    return [r for rows in group_by_modality(examples).values() for r in rows[:n]]


def load_real_rows(spec: dict, max_per_modality: int) -> tuple[list, list]:
    """(train, eval) rows from the datasets of an `hf_data` spec; an empty eval split is carved out of the train rows."""
    from kairos.hf_data import load_hf_examples

    train, held = load_hf_examples(spec)
    train, held = limit_per_modality(train, max_per_modality), limit_per_modality(held, max_per_modality)
    if len(held) < 2:
        cut = max(2, len(train) // 10)
        train, held = train[:-cut], train[-cut:]
    return train, held


def load_rows(args) -> tuple[list, list]:
    """(train, eval) rows from --source: synthetic rows (eval uses another seed) or the real datasets."""
    if args.source == "hf":
        import yaml

        spec = dict(yaml.safe_load(Path(args.spec).read_text())["hf_data"])
        spec.update({k: v for k, v in (("text_repo", args.text_repo), ("multimodal_repo", args.multimodal_repo)) if v})
        return load_real_rows(spec, args.max_per_modality)
    flat = lambda rows: [r for name in MODALITIES for r in rows[name]]
    return flat(synthetic_rows(args.per_modality, 0)), flat(synthetic_rows(max(2, args.per_modality // 5), 1))


DEFAULT_SCALE_FACTORS = {"image_caption": 2, "video_caption": 4, "lidar": 4, "audio_caption": 16, "control": 32}


def build_pipeline_from_config(
    train_rows: list, eval_rows: list, batch_size: int, scale_factors: dict, run_dir: str, lr=None, compile_model=None
):
    """The model, curriculum and optimizer of configs/pretrain.yaml on the given rows."""
    import yaml

    from kairos.cli import _model_kwargs
    from kairos.modeling import KairosConfig
    from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

    cfg = yaml.safe_load((ROOT / "configs" / "pretrain.yaml").read_text())
    data = {**cfg["data"], "batch_size": batch_size, "modality_scale_factors": scale_factors, "num_workers": 0}
    train_cfg = {**cfg["train"], "run_dir": run_dir, "hub_repo_id": None, "save_every": 10**6, "last_ckpt_every": 10}
    train_cfg.update(eval_every=0, eval_every_epochs=1.0, eval_batches=2)
    train_cfg["lr"] = lr if lr is not None else train_cfg["lr"]
    if compile_model is not None:
        train_cfg["compile_model"] = compile_model
    held = {**data, "pack": False, "shuffle": False, "drop_last": False}
    return KairosMultimodalPipeline.from_configs(
        KairosConfig(**_model_kwargs(cfg["model"])),
        DataConfig(multimodal_examples=train_rows, text_examples=[], **data),
        TrainConfig(**train_cfg),
        DataConfig(multimodal_examples=eval_rows, text_examples=[], **held),
    )


def parameter_counts(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    experts = sum(p.numel() for n, p in model.named_parameters() if "experts" in n)
    return {"total": total, "routed_experts": experts}


def plain_ce_by_modality(pipe, loader, max_batches: int = 20, seed: int = 0) -> dict:
    """Plain CE on noised tokens (linear path, p ~ U(0, 1), no 1/p weight), overall and per modality, on seeded masks."""
    import torch

    from kairos.tokenizer import Modality
    from kairos.trainer import compute_masked_diffusion_losses, make_diffusion_mask

    torch.manual_seed(seed)
    losses, mods = [], []
    pipe.model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(pipe.device) for k, v in batch.items()}
            x0, modality = batch["input_ids"], batch["modality_ids"]
            noise_mask, p = make_diffusion_mask(x0, batch["prompt_len"], batch.get("mask"))
            per_token, _, _, _ = compute_masked_diffusion_losses(
                pipe.model, x0, noise_mask, p, modality, reweight=False
            )
            losses.append(per_token.float().flatten())
            mods.append(modality[noise_mask].flatten())
    pipe.model.train()
    loss, mod = torch.cat(losses), torch.cat(mods)
    by = {Modality(m).name: loss[mod == m].mean().item() for m in mod.unique().tolist() if (mod == m).any()}
    return {"all": loss.mean().item(), "by_modality": by}


def modality_token_share(dataset) -> dict:
    """Share of the non-pad tokens of the training windows carried by each modality."""

    from kairos.tokenizer import Modality

    counts: dict[str, int] = {}
    for i in range(len(dataset)):
        row = dataset[i]
        keep = row["mask"].bool()
        for m in row["modality_ids"][keep].unique().tolist():
            counts[Modality(m).name] = counts.get(Modality(m).name, 0) + int((row["modality_ids"][keep] == m).sum())
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())}


def run_train(args) -> dict:
    import torch

    train_rows, eval_rows = load_rows(args)
    pipe = build_pipeline_from_config(
        train_rows,
        eval_rows,
        args.batch_size,
        json.loads(args.scale_factors),
        args.run_dir,
        args.lr,
        False if args.no_compile else None,
    )
    pipe.build()
    summary = {
        "source": args.source,
        "device": device_info(pipe),
        "params": parameter_counts(pipe.model),
        "windows": len(pipe.dataset),
        "steps_per_epoch": len(pipe.loader),
    }
    summary["modality_token_share"] = modality_token_share(pipe.dataset)
    summary["ce_before"] = {
        "train": plain_ce_by_modality(pipe, pipe.loader),
        "heldout": plain_ce_by_modality(pipe, pipe.eval_loader),
    }
    start = time.time()
    rows = pipe.train(resume=True)
    summary["train_seconds_this_call"] = time.time() - start
    summary["ce_after"] = {
        "train": plain_ce_by_modality(pipe, pipe.loader),
        "heldout": plain_ce_by_modality(pipe, pipe.eval_loader),
    }
    losses = [r["loss"] for r in rows if "loss" in r]
    summary.update(steps=len(losses), skipped_nonfinite=pipe.skipped_nonfinite_steps)
    epochs = sorted({r["epoch"] for r in rows if "epoch" in r})
    # epochs 1, 2, 3 are the MAE, transition and diffusion stages (mean training loss; the loss definition changes by stage)
    summary["loss_by_epoch"] = {
        e: sum(r["loss"] for r in rows if r.get("epoch") == e) / sum(r.get("epoch") == e for r in rows) for e in epochs
    }
    summary["eval_loss_rows"] = pipe.eval_log_rows
    if pipe.device.startswith("cuda"):
        summary["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 2**20
    summary["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # last.pt is removed after a clean run, so save a fresh checkpoint, perturb the weights and reload it
    path = pipe.ckpt_dir / "roundtrip.pt"
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    pipe._save(path, losses[-1], 3)
    next(iter(pipe.model.parameters())).data.add_(1.0)
    pipe.load_checkpoint(str(path))
    summary["checkpoint_reload_matches"] = all(torch.equal(before[k], v) for k, v in pipe.model.state_dict().items())
    out = pipe.generate(list(b"the cat"), max_new_tokens=8, seed=0)
    summary["generate_returns_tokens"] = len(out) > 0
    return summary


def device_info(pipe) -> dict:
    import torch

    info = {
        "device": pipe.device,
        "amp_dtype": str(pipe.amp_dtype),
        "compiled": bool(pipe.compiled),
        "torch": torch.__version__,
    }
    if pipe.device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["total_mb"] = torch.cuda.get_device_properties(0).total_memory / 2**20
    return info


def time_steps(pipe, batch_size: int, steps: int, warmup: int = 2) -> dict:
    """Seconds per optimizer step, tokens per second (padded window tokens) and peak GPU memory at one batch size."""
    import torch
    from torch.utils.data import DataLoader

    from kairos.trainer import make_optimizer

    loader = DataLoader(pipe.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = make_optimizer(pipe.model, "adamw", 1e-3)
    on_gpu = pipe.device.startswith("cuda")
    batches, start = itertools.cycle(loader), 0.0
    pipe.model.train()
    for i in range(warmup + steps):
        if i == warmup:
            if on_gpu:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
        batch = {k: v.to(pipe.device) for k, v in next(batches).items()}
        with pipe._autocast():
            loss = pipe.hf_trainer.compute_loss(pipe._forward_model, batch)
        pipe.scaler.scale(loss).backward()
        pipe.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(pipe.model.parameters(), 1.0)
        pipe.scaler.step(opt)
        pipe.scaler.update()
        opt.zero_grad(set_to_none=True)
    if on_gpu:
        torch.cuda.synchronize()
    seconds = (time.time() - start) / steps
    tokens = batch["input_ids"].numel()
    return {
        "batch_size": batch_size,
        "sec_per_step": seconds,
        "tokens_per_sec": tokens / seconds,
        "peak_cuda_mb": torch.cuda.max_memory_allocated() / 2**20 if on_gpu else None,
        "finite": bool(torch.isfinite(loss).item()),
    }


def run_bench(args) -> dict:
    import torch

    train_rows, eval_rows = load_rows(args)
    pipe = build_pipeline_from_config(
        train_rows,
        eval_rows,
        1,
        json.loads(args.scale_factors),
        args.run_dir,
        args.lr,
        False if args.no_compile else None,
    )
    pipe.build()
    results = []
    for size in (int(x) for x in args.batch_sizes.split(",")):
        try:
            results.append(time_steps(pipe, size, args.steps))
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": size, "oom": True})
            torch.cuda.empty_cache()
    return {
        "device": device_info(pipe),
        "params": parameter_counts(pipe.model),
        "windows": len(pipe.dataset),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["data", "train", "bench"])
    parser.add_argument("--source", choices=["synthetic", "hf"], default="synthetic")
    parser.add_argument(
        "--spec", default=str(ROOT / "configs" / "pretrain_sample.yaml"), help="config holding `hf_data`"
    )
    parser.add_argument("--text-repo", default=None, help="override hf_data.text_repo (repo id or local folder)")
    parser.add_argument("--multimodal-repo", default=None, help="override hf_data.multimodal_repo")
    parser.add_argument("--max-per-modality", type=int, default=100, help="rows kept per modality with --source hf")
    parser.add_argument("--batch-sizes", default="4,8,16,32", help="bench: comma-separated batch sizes")
    parser.add_argument("--steps", type=int, default=10, help="bench: timed steps per batch size")
    parser.add_argument(
        "--no-compile", action="store_true", help="disable torch.compile (needed without Triton, e.g. native Windows)"
    )
    parser.add_argument("--per-modality", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None, help="override train.lr of configs/pretrain.yaml")
    parser.add_argument("--run-dir", default="/tmp/readiness")
    parser.add_argument("--out", default=None, help="write the train summary as JSON")
    parser.add_argument(
        "--scale-factors", default=json.dumps(DEFAULT_SCALE_FACTORS), help='JSON, e.g. {"image_caption": 2}'
    )
    args = parser.parse_args()
    if args.command in ("train", "bench"):
        summary = run_train(args) if args.command == "train" else run_bench(args)
        print(json.dumps(summary, indent=1))
        if args.out:
            Path(args.out).write_text(json.dumps(summary, indent=1))
        return
    train_rows, _ = load_rows(args)
    stats = token_stats(group_by_modality(train_rows), args.max_len, scale_factors=json.loads(args.scale_factors))
    for name, s in stats.items():
        print(
            f"{name:14s} examples={s['examples']:3d} windows={s['windows']:3d} tokens/example={s['tokens_per_example']:8.1f} pad={s['pad_fraction']:.2f}"
        )


if __name__ == "__main__":
    main()
