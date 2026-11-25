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
- ELF (embedding flow, x-prediction, 105M-652M), LangFlow (130M, Bregman/CE flow matching), FLM (one-step maps): need a continuous token
  embedding space. Kairos is byte-level, so a frozen byte embedding or a codec latent would be required.
- Benefit: classifier-free guidance and distillation for few steps. Risk: ELF reports x-prediction is needed.

## Models trained this way (verified by search, Nov 2026)
- DFM (Meta): 1.7B, code; HumanEval pass@1 6.7%, MBPP 6.7% (reported by the authors).
- ELF 105M/342M/652M, LangFlow 130M (LM1B and OpenWebText, 1M steps), FLM (evaluated on OpenWebText; its size was not checked).
- No flow-matching LLM at frontier scale was found. Masked diffusion (LLaDA, Dream) dominates.
- Correlation caveat: ELF's gain over MDLM/Duo mixes the objective with data, encoder and optimizer.

## Overfit comparison (ablations log §11)
Configs `flow_mdlm_control`, `flow_dfm_linear`, `flow_dfm_cubic` (memorization) and `flow_heldout_*` (unseen passages);
seeds are reproducible, 4 paired seeds per arm.
- Memorization (1800 steps): reference CE 0.140 / 0.132 / 0.145 (control / linear / cubic), no detectable difference.
- Unseen passages (1000 steps): held-out CE 3.494 / 3.526 / 3.571; both DFM arms worse in 4/4 seeds (+0.032, +0.077).
- Sampling: `dfm_sample` below the entropy-bound `generate()` (paired gap +0.12 at 8 and 16 steps in 11a).
- At the tuned rate 8e-3 (§11c): memorization 0.082 / 0.073 / 0.034 (no detectable difference); held-out CE 3.591 / 3.675 /
  3.805, both DFM arms worse on 4/4 seeds (+0.084, +0.214) with a lower train CE: same conclusion as at 1e-3.
- Not controlled: the `1/p` weight also changes the effective gradient scale, so the cause is not established.

## Next experiment
Repeat 11b with a larger model and a real corpus, 5+ seeds (the rate sweep is already done: §11c); add a corrector and time conditioning to `dfm_sample`; match the effective gradient scale across arms.
