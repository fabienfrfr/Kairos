# Kairos — Compute Budget Analysis
> $N_\text{total} = 200\text{M}$, $N_\text{act} = 4\text{M}$, $E = 32$ experts ($k=4$ active), $D = 30\text{B}$ tokens.

---
## 1. Total FLOPS (Kaplan 2020, Hoffmann 2022)
Training cost (forward + backward) for a transformer:
$$C \approx 6 \cdot N \cdot D$$
For MoE, compute scales with $N_\text{act}$, not $N_\text{total}$ — only $k/E$ experts fire per token (Ludziejewski et al., arXiv 2502.05172, Eq. 5):
$$\boxed{C_\text{Kairos} = 6 \times 4\text{M} \times 30\text{B} = 7.2 \times 10^{17} \text{ FLOPS}}$$
This is **720× cheaper** than an estimated Gemma-3 270M dense run (${\sim}5 \times 10^{20}$ FLOPS).

---
## 2. Roofline Model (Williams et al., 2009)
$$\text{Throughput} = \min\!\left(\pi,\; \mathcal{I} \cdot b_s\right)$$
where $\pi$ = peak FLOP/s, $b_s$ = HBM bandwidth, $\mathcal{I}$ = arithmetic intensity [FLOP/byte].
The **ridge point** $\mathcal{I}^* = \pi / b_s$ separates the two regimes:
| GPU | $\pi$ | $b_s$ | Ridge point $\mathcal{I}^*$ |
|---|---|---|---|
| A100 | 312 TFLOPS | 2.0 TB/s | 156 FLOP/byte |
| H100 SXM5 | 989 TFLOPS | 3.35 TB/s | 295 FLOP/byte |
Arithmetic intensity of Kairos training (fwd+bwd, no Adam):
$$\mathcal{I}_\text{train} = \frac{6 \cdot N_\text{act}}{4 \cdot N_\text{act}} = \frac{6}{4} = 1.5 \text{ FLOP/byte}$$

**$1.5 \ll 156$ → deeply memory-bound on every GPU, including H100** (Chronicals, arXiv 2601.02609, §S25.1).

---
## 3. MoE Bandwidth Bottleneck
The key question: which $N$ drives memory bandwidth?
| Pass | Bytes/token | Why |
|---|---|---|
| Forward | $4 \cdot N_\text{act} = 16\text{ MB}$ | only active experts + attention |
| **Backward** | $4 \cdot N_\text{total} = 800\text{ MB}$ | gradients flow through all experts via router |
| Adam | $8 \cdot N_\text{total} / B$ | amortized over batch $B$, negligible at $B \geq 32$ |
The backward pass reads $N_\text{total}$, not $N_\text{act}$ — this is the binding constraint (Abnar et al., arXiv 2501.12370).
$$B_\text{eff} = 4 \cdot N_\text{total} + \frac{8 \cdot N_\text{total}}{B} \;\xrightarrow{B \to \infty}\; 4 \cdot N_\text{total}$$

---
## 4. Estimated Throughput
$$\text{tok/s} = \frac{b_s}{B_\text{eff}} = \frac{b_s}{4 \cdot N_\text{total}}$$
| GPU | $b_s$ | tok/s (theory) | tok/s (MFU 25%) | Days for 30B |
|---|---|---|---|---|
| T4 | 300 GB/s | 375 | ~94 | ~3 700 ❌ |
| RTX 5060Ti | 600 GB/s | 750 | ~187 | ~1 850 ❌ |
| A100 40G | 2 000 GB/s | 2 500 | ~625 | ~555 ⚠️ |
| H100 SXM5 | 3 350 GB/s | 4 188 | ~1 047 | ~332 ⚠️ |
$$\text{days} = \frac{D}{\text{tok/s} \times 86\,400}$$
These are **theoretical upper bounds**. Real MFU on a custom model without optimized kernels is typically 15–25%. The only reliable number comes from profiling: `torch.cuda.Event` over 100 steps.

---
## 5. Scaling Laws for MoE
**Chinchilla** (Hoffmann 2022):
$$\mathcal{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$
with $E=1.69,\ A=406.4,\ B=410.7,\ \alpha=0.34,\ \beta=0.28$.
⚠️ Fitted on 70M–16B **dense** models. Extrapolation to 4M active params is out of range.
**Joint MoE scaling law** (Ludziejewski et al., arXiv 2502.05172):
$$\mathcal{L}(N_\text{act}, D, E) = m(E)\cdot N_\text{act}^{\mu(E)} + n(E)\cdot D^{\nu(E)} + c$$
Key result: with $E=32$ experts, the marginal benefit of more tokens **grows** with $E$. More tokens is better justified for a MoE than for a dense model of equal $N_\text{act}$.
**Token ratio** for Kairos: $D / N_\text{act} = 30\text{B} / 4\text{M} = 7{,}500\times$ — aggressive over-training, consistent with edge-deployment strategy (SmolLM, MobileLLM).

---
## 6. Training Inspirations at Similar Scale

The following models are the closest publicly documented precedents for training in the sub-50M dense / low-active-param regime. They inform realistic expectations for data volume, compute, and achievable loss at Kairos scale.

| Model | Params (total) | Params (active) | Tokens seen | Dataset | Batch size | Steps | Architecture | Key takeaway for Kairos |
|---|---|---|---|---|---|---|---|---|
| **Pythia-14M** | 14M | 14M (dense) | ~300B | The Pile (825 GB, 22 sources) | 2M tokens | 143 000 | GPT-NeoX, 6L × d128 × 4h, Flash Attention | 300B tokens on 14M params = $D/N \approx 21{,}000\times$ — far beyond Chinchilla; proves aggressive over-training is standard at this scale. 154 public checkpoints enable loss-curve comparison. https://github.com/EleutherAI/pythia/blob/main/models/14M/pythia-14m.yml |
| **Pythia-31M** | 31M | 31M (dense) | ~300B | The Pile | 2M tokens | 143 000 | GPT-NeoX, 6L × d256 × 8h | Same regime as 14M. Non-embedding params only 4.7M — direct architectural comp to $N_\text{act}=4\text{M}$. Confirms depth-6 is sufficient at this width. |
| **TinyStories-1M** | ~1M | ~1M (dense) | ~3B (synthetic) | TinyStories (GPT-4 generated children's stories) | — | — | Transformer, ~8L × d64 | Demonstrates a 1M-param model *can* produce coherent English on a narrow domain with 3B curated tokens. Sets a lower bound: coherent text generation is possible well below 4M active params if data is high-quality and domain-scoped. |
| **TinyStories-33M** | ~33M | ~33M (dense) | ~3B (synthetic) | TinyStories | — | — | Transformer, larger variant | Upper bound of the TinyStories family. At 33M dense on 3B tokens ($D/N \approx 90\times$), perplexity is low within the domain but generalisation is zero. Shows that domain-scoped training with little data leaves the model brittle. |

**Reading these for Kairos:**
- Pythia-14/31M are the best compute analogues: same $N_\text{act}$ order of magnitude, same over-training philosophy, publicly reproducible.
- TinyStories shows the floor: 1M params + curated data → coherent narrow output. Kairos at 4M active with 30B general tokens should clear this floor by a wide margin.
- None of these is MoE. The $N_\text{total}=200\text{M}$ backward bottleneck has no precedent in this table — it is a Kairos-specific constraint not present in any of these dense baselines.

---
## 7. Verdict
| Question | Answer |
|---|---|
| Is 30B tokens justified? | Yes — MoE benefits more from tokens than dense; over-training is correct for edge |
| Is 30B tokens feasible on a T4? | No — ~3 700 days due to $N_\text{total}$ backward bottleneck |
| What's the real bottleneck? | HBM bandwidth on $N_\text{total}$ (200M), not compute on $N_\text{act}$ (4M) |
| Fastest fix? | Reduce $N_\text{total}$ to ~50M (keep 4M active) → 4× faster backward |
| Before any of this? | Run 100 steps, measure real tok/s with `torch.cuda.Event` |

---
## References
| | |
|---|---|
| Kaplan 2020 | *Scaling Laws for Neural Language Models*, arXiv:2001.08361 |
| Hoffmann 2022 | *Training Compute-Optimal LLMs (Chinchilla)*, arXiv:2203.15556 |
| Williams 2009 | *Roofline: An Insightful Visual Performance Model*, CACM 2009 |
| Ludziejewski 2025 | *Joint MoE Scaling Laws*, arXiv:2502.05172 |
| Abnar 2025 | *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity*, arXiv:2501.12370 |
| Chronicals 2025 | *High-Performance Framework for LLM Fine-Tuning*, arXiv:2601.02609 |
| Biderman 2023 | *Pythia: A Suite for Analyzing LLMs Across Training and Scaling*, arXiv:2304.01373 |
| Eldan & Li 2023 | *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, arXiv:2305.07759 |