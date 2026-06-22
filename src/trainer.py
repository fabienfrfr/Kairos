
import torch
import torch.nn.functional as F

from transformers import Trainer


# =========================
# Trainer (standard HF-like)
# =========================
class KairosDiffusionTrainer(Trainer):
    """ https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/DiffusionGemma_(26B-A4B)-Sudoku.ipynb"""
    def compute_loss(self, model, inputs, return_outputs=False):
        x0 = inputs["input_ids"]
        prompt_len = inputs["prompt_len"]

        eps = 1e-3
        t = torch.rand(x0.size(0), device=x0.device)
        
        p = (1 - eps) * t + eps
        p = p[:, None].expand_as(x0)


        mask = torch.rand(x0.shape, device=x0.device) < p

        for i in range(x0.size(0)):
            mask[i, :prompt_len[i]] = False

        xt = x0.clone()
        noise = torch.randint_like(x0, model.lm_head.vocab_size)
        # The noise level (timestep) is directly observable in xt through the explicit masking, so no timestep embedding is required.
        xt[mask] = noise[mask]

        logits = model(decoder_input_ids=xt).logits

        loss = F.cross_entropy(
            logits[mask],
            x0[mask],
            reduction='none'
        )

        loss = (loss / p[mask]).mean()

        return loss