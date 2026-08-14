"""Block-diffusion generation for the Kairos single-tower architecture.

Reuses the DiffusionGemma generation machinery shipped with transformers
(EntropyBoundSampler, LinearTemperatureScheduleLogitsProcessor and
StableAndConfidentStoppingCriteria) but replaces the encoder/decoder + KV-cache
loop with a single forward pass over ``[prompt, canvas]`` at every denoising
step, which is the natural fit for KairosDiffusionLLM.

``KairosDiffusionLLM`` inherits this mixin (see ``kairos/modeling.py``); the base
``DiffusionGemmaGenerationMixin.generate`` is *not* usable as-is because it is
hard-wired to the DiffusionGemma architecture (separate encoder/decoder towers,
``config.canvas_length`` / ``config.text_config``, KV caches, diffusion decoder
attention masks).
"""

from __future__ import annotations

import copy
import math

import torch
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig,
    DiffusionGemmaGenerationOutput,
    EntropyBoundSampler,
    EntropyBoundSamplerConfig,
    LinearTemperatureScheduleLogitsProcessor,
    StableAndConfidentStoppingCriteria,
)

DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 256,
    "max_denoising_steps": 48,
    "sampler_config": EntropyBoundSamplerConfig(entropy_bound=0.1),
    "t_min": 0.4,
    "t_max": 0.8,
    "stability_threshold": 1,
    "confidence_threshold": 0.005,
}

_CFG_FIELDS = (
    "max_new_tokens",
    "max_length",
    "max_denoising_steps",
    "t_min",
    "t_max",
    "stability_threshold",
    "confidence_threshold",
    "return_dict_in_generate",
)


class KairosDiffusionGenerationMixin:
    """Single-tower block-diffusion generation, adapted from DiffusionGemma."""

    @torch.no_grad()
    def generate(self, input_ids=None, modality_ids=None, generation_config=None, **kwargs):
        """Denoises ``max_new_tokens`` continuation tokens after a fixed prompt.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, prompt_len)`):
                Fixed prompt tokens (kept unnoised).
            modality_ids (`torch.LongTensor` of shape `(batch_size, prompt_len)`, *optional*):
                Modality of each prompt token; defaults to the text modality.
            generation_config (`DiffusionGemmaGenerationConfig`, *optional*):
                Base configuration; kwargs below override it.
            max_new_tokens (`int`, *optional*):
                Number of continuation tokens to generate.
            max_denoising_steps (`int`, *optional*):
                Diffusion iterations per canvas.
            entropy_bound (`float`, *optional*):
                Cumulative-entropy acceptance bound (higher accepts more tokens/step).
            t_min/t_max (`float`, *optional*):
                Final/initial temperature of the linear schedule.
            stability_threshold/confidence_threshold (`int`/`float`, *optional*):
                Adaptive stopping criteria (disabled when both are unset).
            **kwargs: forwarded to the model ``forward`` call.

        Returns:
            `DiffusionGemmaGenerationOutput` with ``sequences`` of shape
            `(batch_size, prompt_len + max_new_tokens)`, or the raw tensor when
            ``return_dict_in_generate=False``.
        """
        cfg = copy.deepcopy(generation_config) if generation_config is not None else DiffusionGemmaGenerationConfig()
        for key, val in DEFAULT_GENERATION_PARAMS.items():
            if getattr(cfg, key, None) is None:
                setattr(cfg, key, val)
        if "sampler_config" in kwargs:
            cfg.sampler_config = kwargs.pop("sampler_config")
        if "entropy_bound" in kwargs:
            cfg.sampler_config = EntropyBoundSamplerConfig(kwargs.pop("entropy_bound"))
        for key in _CFG_FIELDS:
            if key in kwargs:
                setattr(cfg, key, kwargs.pop(key))
        cfg.validate()
        model_kwargs = kwargs

        config = self.config
        device = input_ids.device
        batch_size, prompt_len = input_ids.shape
        canvas_length = config.canvas_length
        max_new_tokens = cfg.max_new_tokens
        max_denoising_steps = cfg.max_denoising_steps
        text_mod = int(config.text_modality_id)
        vocab_size = config.vocab_size

        if modality_ids is None:
            modality_ids = torch.full_like(input_ids, text_mod)

        sampler = EntropyBoundSampler(cfg.sampler_config, canvas_length, vocab_size, max_denoising_steps)
        temp_processor = (
            LinearTemperatureScheduleLogitsProcessor(cfg.t_min, cfg.t_max, max_denoising_steps)
            if cfg.t_min is not None and cfg.t_max is not None
            else None
        )
        stopping = (
            StableAndConfidentStoppingCriteria(cfg.stability_threshold, cfg.confidence_threshold)
            if cfg.stability_threshold is not None and cfg.confidence_threshold is not None
            else None
        )

        final_len = prompt_len + max_new_tokens
        n_canvases = max(1, math.ceil(max_new_tokens / canvas_length))

        for _ in range(n_canvases):
            current_canvas = sampler.initialize_canvas(batch_size, device)
            argmax_canvas = current_canvas.clone()
            self_cond = None
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if stopping is not None:
                stopping.reset()

            for step in range(max_denoising_steps, 0, -1):
                full_ids = torch.cat([input_ids, current_canvas], dim=-1)
                full_mod = torch.cat([modality_ids, torch.full_like(current_canvas, text_mod)], dim=-1)
                cond = None
                if self_cond is not None:
                    cond = torch.zeros(batch_size, full_ids.shape[1], vocab_size, device=device, dtype=torch.float32)
                    cond[:, prompt_len:] = self_cond
                outputs = self.forward(
                    decoder_input_ids=full_ids,
                    modality_ids=full_mod,
                    self_conditioning_logits=cond,
                    **model_kwargs,
                )
                logits = outputs.logits[:, prompt_len:]
                if temp_processor is not None:
                    logits = temp_processor(full_ids, logits, cur_step=step)
                logits = logits.float()

                probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                denoiser_canvas = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).squeeze(-1)
                denoiser_canvas = denoiser_canvas.view(batch_size, canvas_length)
                new_argmax = logits.argmax(dim=-1)

                prev_canvas = current_canvas
                accepted = sampler.accept_canvas(current_canvas, denoiser_canvas, logits, step)
                current_canvas = sampler.renoise_canvas(accepted, step)

                if stopping is not None:
                    if finished.any():
                        new_argmax = torch.where(finished[:, None], argmax_canvas, new_argmax)
                        current_canvas = torch.where(finished[:, None], prev_canvas, current_canvas)
                        logits = torch.where(finished[:, None, None], self_cond, logits)
                    argmax_canvas = new_argmax
                    finished = finished | stopping(argmax_canvas, logits)
                    if finished.all():
                        break
                else:
                    argmax_canvas = new_argmax

                self_cond = logits.detach()

            input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)
            modality_ids = torch.cat([modality_ids, torch.full_like(argmax_canvas, text_mod)], dim=-1)
            prompt_len = input_ids.shape[1]

        input_ids = input_ids[:, :final_len]
        if cfg.return_dict_in_generate:
            return DiffusionGemmaGenerationOutput(sequences=input_ids)
        return input_ids
