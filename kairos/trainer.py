import random

import torch
import torch.nn.functional as F
from transformers import Trainer


def make_diffusion_mask(x0, prompt_len, pad_mask=None, eps=1e-3, p_max=1.0):
    """Random per-token mask + per-row rate p for the masked-diffusion objective.

    p ~ U(eps, p_max). Full diffusion uses p_max=1.0 (rows can be up to 100% noised).
    A capped p_max (e.g. 0.3) gives a fixed-ish-rate MAE-style corruption instead.
    """
    p = (p_max - eps) * torch.rand(x0.size(0), device=x0.device) + eps
    p = p[:, None].expand_as(x0)
    noise_mask = torch.rand(x0.shape, device=x0.device) < p
    for i in range(x0.size(0)):
        noise_mask[i, : prompt_len[i]] = False
    if pad_mask is not None:
        noise_mask &= pad_mask.bool()  # never noise/score padding
    return noise_mask, p


def compute_masked_diffusion_losses(
    model,
    x0,
    noise_mask,
    p,
    modality_ids=None,
    family_ids=None,
    cache_params=None,
    reweight=True,
    self_conditioning_prob=0.0,
):
    """Noises ``x0`` on ``noise_mask``. With self_conditioning_prob>0, warms up with a no-grad
    pass and feeds its detached logits back in, matching generate()'s inference-time usage."""
    xt = x0.clone()
    noise = torch.randint_like(x0, model.lm_head.vocab_size)
    xt[noise_mask] = noise[noise_mask]

    xt_family, octet_targets = None, None
    family_embed = model.embedding.family_embed
    if family_ids is not None and family_embed is not None:
        xt_family = family_ids.clone()
        noise_family = torch.randint_like(family_ids, family_embed.num_embeddings)
        xt_family[noise_mask] = noise_family[noise_mask]
        octet_targets = family_ids[noise_mask]

    self_cond = None
    if self_conditioning_prob > 0 and random.random() < self_conditioning_prob:
        with torch.no_grad():
            warm_out = model(
                decoder_input_ids=xt,
                modality_ids=modality_ids,
                family_ids=xt_family,
                logits_mask=noise_mask,
            )
        vocab_size = model.lm_head.vocab_size
        self_cond = torch.zeros(*x0.shape, vocab_size, device=x0.device, dtype=warm_out.logits.dtype)
        self_cond[noise_mask] = warm_out.logits.detach()

    # logits_mask: only project noised positions - lm_head cost scales with vocab_size
    out = model(
        decoder_input_ids=xt,
        modality_ids=modality_ids,
        family_ids=xt_family,
        self_conditioning_logits=self_cond,
        cache_params=cache_params,
        logits_mask=noise_mask,
    )
    per_token_loss = F.cross_entropy(out.logits, x0[noise_mask], reduction="none")
    # reweight is a continuous blend factor in [0, 1], not just a bool: alpha=1.0 -> full 1/p
    # reweight (original behaviour of reweight=True), alpha=0.0 -> plain CE (reweight=False).
    # bool True/False still work unchanged since Python bools are ints (True==1.0, False==0.0);
    # intermediate floats interpolate smoothly, which is what lets the MAE->diffusion transition
    # ramp the loss weighting instead of switching it instantly. See anneal_mask_schedule().
    alpha = float(reweight)
    if alpha:
        weight = 1.0 + alpha * (1.0 / p[noise_mask] - 1.0)
        per_token_loss = per_token_loss * weight
    return per_token_loss, out.logits, out.octet_logits, octet_targets


def anneal_mask_schedule(step: int, anneal_steps: int, start: float, end: float) -> float:
    """Linear ramp from `start` at step 0 to `end` at step >= anneal_steps. anneal_steps <= 0
    means no ramp: returns `end` immediately (the original hard-switch behaviour)."""
    if anneal_steps <= 0:
        return end
    t = min(1.0, max(0.0, step / anneal_steps))
    return start + t * (end - start)


def stage_mask_schedule(
    global_step: int,
    mae_steps: int,
    transition_steps: int,
    mae_p_max: float,
    mae_reweight: float,
    target_p_max: float,
    target_reweight: float,
) -> tuple[float, float]:
    """(mask_p_max, mask_reweight) for the unified MAE -> transition -> diffusion curriculum, as
    a pure function of `global_step` — this is what makes resuming from a checkpoint "just work"
    without any special-casing: the regime only ever depends on the (persisted) global_step, not
    on which stage/pipeline object happened to run it.

    - step < mae_steps: flat at (mae_p_max, mae_reweight) — MAE stage.
    - mae_steps <= step < mae_steps + transition_steps: linear ramp to (target_p_max,
      target_reweight) — transition stage. transition_steps<=0 skips straight to the target
      right after the MAE stage (no ramp).
    - step >= mae_steps + transition_steps: flat at (target_p_max, target_reweight) — diffusion
      stage.
    """
    if global_step < mae_steps:
        return mae_p_max, float(mae_reweight)
    t_step = global_step - mae_steps
    p_max = anneal_mask_schedule(t_step, transition_steps, mae_p_max, target_p_max)
    reweight = anneal_mask_schedule(t_step, transition_steps, float(mae_reweight), float(target_reweight))
    return p_max, reweight


class KairosDiffusionTrainer(Trainer):
    """Masked-diffusion loss: mask a random fraction of non-prompt tokens with noise and."""

    last_loss_diagnostics: dict | None = None
    mask_eps: float = 1e-3  # floor of p ~ U(eps, p_max); CE/p makes rare low-p rows dominate
    mask_p_max: float = 1.0  # ceiling of p; 1.0 = full diffusion, cap for MAE-style corruption
    mask_reweight: bool = True  # divide CE by p; False for plain CE
    octet_loss_weight: float = 1.0  # weight of the octet-family loss
    # train-time self-conditioning rate; keep >0 so generate()'s usage isn't out-of-distribution.
    self_conditioning_prob: float = 0.5

    def compute_loss(self, model, inputs, return_outputs=False, cache_params=None):
        x0 = inputs["input_ids"]
        prompt_len = inputs["prompt_len"]
        modality_ids = inputs.get("modality_ids")
        family_ids = inputs.get("octet_family_ids")
        pad_mask = inputs.get("mask")

        noise_mask, p = make_diffusion_mask(x0, prompt_len, pad_mask, eps=self.mask_eps, p_max=self.mask_p_max)
        if not noise_mask.any():
            # exceedingly rare: short sequence + low mask ratio
            eligible = pad_mask.bool() if pad_mask is not None else torch.ones_like(noise_mask)
            for i in range(x0.size(0)):
                row_idx = eligible[i].nonzero(as_tuple=True)[0]
                if len(row_idx) > 0:
                    noise_mask[i, row_idx[0]] = True

        per_token_loss, logits, octet_logits, octet_targets = compute_masked_diffusion_losses(
            model,
            x0,
            noise_mask,
            p,
            modality_ids,
            family_ids,
            cache_params,
            reweight=self.mask_reweight,
            self_conditioning_prob=self.self_conditioning_prob,
        )
        loss = per_token_loss.mean()
        if octet_logits is not None and octet_targets is not None:
            loss = loss + self.octet_loss_weight * F.cross_entropy(octet_logits, octet_targets)

        if not torch.isfinite(loss):
            # capture context here (access to logits/inputs)
            self.last_loss_diagnostics = self._build_nan_diagnostics(x0, modality_ids, pad_mask, prompt_len, logits)

        return loss

    @staticmethod
    def _build_nan_diagnostics(x0, modality_ids, pad_mask, prompt_len, logits) -> dict:
        diag = {
            "batch_size": x0.size(0),
            "seq_len": x0.size(1),
            "prompt_len_min": int(prompt_len.min()),
            "prompt_len_max": int(prompt_len.max()),
            "logits_nan_frac": float(torch.isnan(logits).float().mean()),
            "logits_inf_frac": float(torch.isinf(logits).float().mean()),
            "logits_absmax": float(logits.detach().abs().amax()) if torch.isfinite(logits).any() else float("nan"),
        }
        if modality_ids is not None:
            values, counts = modality_ids.unique(return_counts=True)
            diag["modality_histogram"] = dict(zip(values.tolist(), counts.tolist()))
        if pad_mask is not None:
            diag["pad_frac"] = float(1 - pad_mask.float().mean())
        return diag