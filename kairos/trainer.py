import torch
import torch.nn.functional as F
from transformers import Trainer


class KairosDiffusionTrainer(Trainer):
    """Masked-diffusion loss: mask a random fraction of non-prompt tokens with noise and predict the originals back."""

    last_loss_diagnostics: dict | None = None

    def compute_loss(self, model, inputs, return_outputs=False):
        x0 = inputs["input_ids"]
        prompt_len = inputs["prompt_len"]
        modality_ids = inputs.get("modality_ids")
        pad_mask = inputs.get("mask")

        eps = 1e-3
        t = torch.rand(x0.size(0), device=x0.device)
        p = (1 - eps) * t + eps
        p = p[:, None].expand_as(x0)

        noise_mask = torch.rand(x0.shape, device=x0.device) < p
        for i in range(x0.size(0)):
            noise_mask[i, : prompt_len[i]] = False
        if pad_mask is not None:
            noise_mask &= pad_mask.bool()  # never noise/score padding

        if not noise_mask.any():
            # exceedingly rare (short sequence + low sampled p): force one
            # position per row so cross_entropy never sees an empty tensor
            eligible = pad_mask.bool() if pad_mask is not None else torch.ones_like(noise_mask)
            for i in range(x0.size(0)):
                row_idx = eligible[i].nonzero(as_tuple=True)[0]
                if len(row_idx) > 0:
                    noise_mask[i, row_idx[0]] = True

        xt = x0.clone()
        noise = torch.randint_like(x0, model.lm_head.vocab_size)
        xt[noise_mask] = noise[noise_mask]

        logits = model(decoder_input_ids=xt, modality_ids=modality_ids).logits

        loss = F.cross_entropy(logits[noise_mask], x0[noise_mask], reduction="none")
        loss = (loss / p[noise_mask]).mean()

        if not torch.isfinite(loss):
            # capture enough context here (with access to logits/inputs) for the caller to
            # print an actionable report instead of a bare "loss=nan"
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
