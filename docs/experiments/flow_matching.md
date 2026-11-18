# Flow matching for Kairos (analysis, session 11)

**Answer: yes for the discrete form, with no change to the loss; continuous forms need an embedding space.**

## Why the discrete form already fits
- Kairos corrupts each position with prob. `p` by a uniform token. This is the DFM mixture path
  (uniform source, linear scheduler `kappa_t = 1 - p`), Gat et al. 2024.
- With `mask_reweight=False` (alpha = 0) the loss is the DFM posterior cross-entropy. The `1/p` weight is the
  MDLM-style ELBO weight; it is a choice, not a requirement of the path.
- Missing piece: a sampler matching the path. Added `dfm_euler_step`/`dfm_sample` and a `mask_path` option (linear, cubic) in `trainer.py`: each
  position jumps to a posterior sample at rate `h / (1 - t)`. Tested on an oracle denoiser only.
- Not added: corrector sampling, time conditioning, kinetic-optimal schedulers (all untested here).

## Continuous options (not implemented)
- ELF (embedding flow, x-prediction, 105M-652M), LangFlow, FLM (one-step maps): need a continuous token
  embedding space. Kairos is byte-level, so a frozen byte embedding or a codec latent would be required.
- Benefit: classifier-free guidance and distillation for few steps. Risk: ELF reports x-prediction is needed.

## Models trained this way (verified by search, Nov 2026)
- DFM (Meta): 1.7B, code; HumanEval pass@1 6.7%, MBPP 6.7% (reported by the authors).
- ELF 105M/342M/652M, LangFlow and FLM: OpenWebText, ~0.1-0.7B scale.
- No flow-matching LLM at frontier scale was found. Masked diffusion (LLaDA, Dream) dominates.
- Correlation caveat: ELF's gain over MDLM/Duo mixes the objective with data, encoder and optimizer.

## Overfit comparison (ablations log §11)
Configs `flow_mdlm_control`, `flow_dfm_linear`, `flow_dfm_cubic`; common reference CE, 3 seeds, 600 steps:
0.548 ± 0.170, 0.795 ± 0.341, 0.777 ± 0.157. The `1/p` control is lowest on every seed; no DFM gain observed.

## Next experiment
Compare `dfm_sample` against the current entropy-bound sampler at equal steps on a trained control and DFM model, 3 seeds.
