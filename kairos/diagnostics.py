"""Diagnostic checks for the pipeline: currently a MoE-bias A/B regime check; add more here as needed."""

from __future__ import annotations

import math

import torch

from .modeling import KairosConfig, KairosTopkRouter
from .pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

DEFAULT_TEXTS = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog " * 10}] * 16

REGIMES = {
    "mae": {"mask_p_max": 0.3, "mask_reweight": False},  # matches notebook TRAIN_MAE_P_MAX/TRAIN_MAE_REWEIGHT
    "diffusion": {"mask_p_max": 1.0, "mask_reweight": True},  # matches notebook TRAIN_MASK_P_MAX/TRAIN_MASK_REWEIGHT
    "diffusion_noreweight": {"mask_p_max": 1.0, "mask_reweight": False},  # full masking range, no 1/p weight
    "diffusion_eps01": {"mask_p_max": 1.0, "mask_reweight": True},  # same but with a higher mask_eps floor
    "curriculum": {},  # default mae_epochs=1/transition_epochs=1/diffusion_epochs=1, same as the notebook
}


def run(
    regime: str,
    bias_update_rate: float,
    mask_eps: float = 1e-3,
    mask_reweight_clip: float | None = 10.0,
    steps: int = 200,
    seed: int = 0,
    num_local_experts: int = 7,
) -> tuple[list[float], list[float]]:
    """Runs a tiny MoE overfit_test under `regime` and returns (per-step losses, per-expert usage counts)."""
    torch.manual_seed(seed)
    model_config = KairosConfig(
        d_model=64, n_heads=4, n_layers=4, use_moe=True, num_local_experts=num_local_experts, num_experts_per_tok=1
    )
    data_config = DataConfig(text_examples=DEFAULT_TEXTS, max_len=128, batch_size=4)
    train_config = TrainConfig(
        save_every=10_000,
        run_dir="/tmp/ab_run",
        moe_bias_update_rate=bias_update_rate,
        mask_eps=mask_eps,
        mask_reweight_clip=mask_reweight_clip,
    )
    pipe = KairosMultimodalPipeline.from_configs(model_config, data_config, train_config)

    usage_totals = torch.zeros(num_local_experts)
    routers = [m for m in pipe.model.modules() if isinstance(m, KairosTopkRouter)]

    def _record(_module, _inputs, output):
        _, _, topk_indices = output
        usage_totals[: _module.num_experts] += torch.bincount(
            topk_indices.reshape(-1), minlength=_module.num_experts
        ).float()

    for r in routers:
        r.register_forward_hook(_record)

    losses: list[float] = []
    pipe.overfit_test(
        n_examples=16,
        steps=steps,
        seed=seed,
        progress_callback=lambda s, t, loss_val: losses.append(loss_val),
        **REGIMES[regime],
    )
    return losses, usage_totals.tolist()


def format_report(losses: list[float], usage: list[float], n_chunks: int = 8) -> str:
    """Renders the same windowed-average report the original script printed to stdout."""
    chunk = max(1, len(losses) // n_chunks)
    lines = []
    for start in range(0, len(losses), chunk):
        window = losses[start : start + chunk]
        lines.append(f"steps {start}-{start + len(window)}: moyenne={sum(window) / len(window):.4f}")
    lines.append(f"loss finale: {losses[-1]:.4f}")
    lines.append(f"tous finis: {all(math.isfinite(x) for x in losses)}")
    lines.append(f"usage par expert (cumulé, tous layers): {usage}")
    return "\n".join(lines)
