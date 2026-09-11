"""A/B check per regime: does update_moe_bias help a tiny 7-expert top-1 MoE overfit-test converge?"""

import math
import sys

import torch

from kairos.modeling import KairosConfig, KairosTopkRouter
from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog " * 10}] * 16

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
    mask_reweight_clip=10.0,
    steps: int = 200,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    torch.manual_seed(seed)
    model_config = KairosConfig(
        d_model=64, n_heads=4, n_layers=4, use_moe=True, num_local_experts=7, num_experts_per_tok=1
    )
    data_config = DataConfig(text_examples=texts, max_len=128, batch_size=4)
    train_config = TrainConfig(
        save_every=10_000,
        run_dir="/tmp/ab_run",
        moe_bias_update_rate=bias_update_rate,
        mask_eps=mask_eps,
        mask_reweight_clip=mask_reweight_clip,
    )
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    usage_totals = torch.zeros(7)
    routers = [m for m in pipe.model.modules() if isinstance(m, KairosTopkRouter)]

    def _record(_module, _inputs, output):
        _, _, topk_indices = output
        usage_totals[: _module.num_experts] += torch.bincount(
            topk_indices.reshape(-1), minlength=_module.num_experts
        ).float()

    for r in routers:
        r.register_forward_hook(_record)

    losses = []
    pipe.overfit_test(
        n_examples=16,
        steps=steps,
        seed=seed,
        progress_callback=lambda s, t, loss_val: losses.append(loss_val),
        **REGIMES[regime],
    )
    return losses, usage_totals.tolist()


if __name__ == "__main__":
    regime = sys.argv[1]
    rate = float(sys.argv[2])
    eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    clip = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    n_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    losses, usage = run(regime, rate, mask_eps=eps, mask_reweight_clip=clip, steps=n_steps)
    chunk = max(1, n_steps // 8)
    for start in range(0, n_steps, chunk):
        window = losses[start : start + chunk]
        print(f"steps {start}-{start + len(window)}: moyenne={sum(window) / len(window):.4f}")
    print(f"loss finale: {losses[-1]:.4f}")
    print(f"tous finis: {all(math.isfinite(x) for x in losses)}")
    print(f"usage par expert (cumulé, tous layers): {usage}")
