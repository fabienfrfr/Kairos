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


def compute_masked_diffusion_losses(model, x0, noise_mask, p, modality_ids=None, cache_params=None, reweight=True):
    """Noises ``x0`` on ``noise_mask`` and returns per-token loss (``CE / p`` if ``reweight``, else plain CE)."""
    xt = x0.clone()
    noise = torch.randint_like(x0, model.lm_head.vocab_size)
    xt[noise_mask] = noise[noise_mask]

    logits = model(decoder_input_ids=xt, modality_ids=modality_ids, cache_params=cache_params).logits
    per_token_loss = F.cross_entropy(logits[noise_mask], x0[noise_mask], reduction="none")
    if reweight:
        per_token_loss = per_token_loss / p[noise_mask]
    return per_token_loss, logits


class KairosDiffusionTrainer(Trainer):
    """Masked-diffusion loss: mask a random fraction of non-prompt tokens with noise and."""

    last_loss_diagnostics: dict | None = None
    mask_eps: float = 1e-3  # floor of the per-row masking rate p ~ U(eps, p_max); CE is divided by p, so a
    # small eps makes rare low-p rows dominate the loss with high variance. Expose/tune via TrainConfig.mask_eps.
    mask_p_max: float = 1.0  # ceiling of p. 1.0 = full diffusion (rows can be up to 100% noised); cap it
    # (e.g. 0.3) for an MAE-style fixed-rate corruption curriculum stage. Expose via TrainConfig.mask_p_max.
    mask_reweight: bool = True  # divide CE by p (standard masked-diffusion ELBO weighting). Set False for
    # plain CE (standard MAE loss, no variance blow-up at low p) — pairs naturally with a capped mask_p_max.

    def compute_loss(self, model, inputs, return_outputs=False, cache_params=None):
        x0 = inputs["input_ids"]
        prompt_len = inputs["prompt_len"]
        modality_ids = inputs.get("modality_ids")
        pad_mask = inputs.get("mask")

        noise_mask, p = make_diffusion_mask(x0, prompt_len, pad_mask, eps=self.mask_eps, p_max=self.mask_p_max)
        if not noise_mask.any():
            # exceedingly rare: short sequence + low mask ratio
            eligible = pad_mask.bool() if pad_mask is not None else torch.ones_like(noise_mask)
            for i in range(x0.size(0)):
                row_idx = eligible[i].nonzero(as_tuple=True)[0]
                if len(row_idx) > 0:
                    noise_mask[i, row_idx[0]] = True

        per_token_loss, logits = compute_masked_diffusion_losses(
            model, x0, noise_mask, p, modality_ids, cache_params, reweight=self.mask_reweight
        )
        loss = per_token_loss.mean()

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
