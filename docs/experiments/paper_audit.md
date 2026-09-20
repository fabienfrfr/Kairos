# Paper audit (`docs/paper/kairos_paper.tex`)

Scope: read-only review of the draft against the code and configs. No code or paper change made.
Checks: `latexmk -pdf` (builds, 14 pages, 1 overfull hbox of 31 pt), regex cross-check of `\cite`/`\ref`
against the `.bib` and labels (no missing key, no dangling ref), manual arithmetic on tables,
manual comparison with `kairos/attentions.py` and `configs/ablations/baseline.yaml`.

## Verified inconsistencies

| # | Where | Finding | Evidence |
|---|-------|---------|----------|
| 1 | Abstract, Table 1 vs §compute | Target is "25M active" but every compute number uses 4M | Eq. 1 with E=32, k=4 gives at most 25M; 4M needs k/E ~ 1/50. At 25M: C = 4.5e18 (~110x, not 720x), D/N = 1 200x (not 7 500x), forward bytes 100 MB (not 16 MB) |
| 2 | Alg. 3, lines 3-5 | Paper: `delta_f + delta_b combination`. Code: concat, then `merge_norm`, `out_left_right`, `out_proj` | `attentions.py` L580-588; reverse pass also runs with `cache_params=None` (memory gate not applied), undocumented |
| 3 | Table 2 caption | "under half patch's cost" is false: 2.50M / 4.93M = 50.7% | "2.7x saving" holds vs vanilla (1.81M), 2.5x vs liz2 (1.98M) |
| 4 | Table 1 vs Table 2 | Table 2 row labelled "PoC's actual setting" is 1 layer, MoE off, 2.50M; Table 1 PoC is 4 layers, MoE, ~14-15M | `baseline.yaml` (`n_layers: 1`, `use_moe: false`) |
| 5 | Table 4, Table 1 | `TBD` / "not yet logged" for PoC N_act, against the paper's own writing rule | Computable with `count_active_parameters` |
| 6 | Table 3 caption | Cites `liz2_shared_vs_separate_followup_raw.jsonl`, absent from the project snapshot | Also cited in `docs/experiments/*.md` |
| 7 | Labels | `sec:codec`, `sec:memgate`, `tab:bandwidth`, `tab:throughput` never referenced | cosmetic |

## Open concern (not verified, needs a decision)

The bandwidth argument charges `4 N` bytes of weight traffic **per token** and amortizes only Adam over the
batch. In a standard training step, weights are read once per micro-batch and reused by all its tokens, so
matmul arithmetic intensity grows with tokens per batch; 1.5 FLOP/byte is a one-token regime. With the
paper's own byte count, intensity is 6 N_act / 4 N_total = 0.03, not 1.5, so Fig. 2 and Table 5 use
different intensities. Expert-level reuse depends on batch tokens x k/E. This is the abstract's central
claim, so it should be checked (or restated as a bound) before other edits.

## Causal wording (correlation is not causation)

§QKV sharing: "effective higher early learning rate" is stated as the mechanism. The freeze control is
consistent with it but does not isolate it (e.g. no LR-scaled separate variant). Table 3 gives single
values with no seed count. Suggest "consistent with" and one direct control.

## Analysis of the paper's argument

Three papers in one: architecture spec, compute sizing, ablation methodology. No single testable hypothesis.

| Claim | Evidence in the paper | Verdict |
|-------|-----------------------|---------|
| Backward is bandwidth-gated by N_total | Analytical only, no measurement; see open concern above | Unsupported as stated |
| Multimodal (text, image, audio, video, lidar, control) | Only text memorization reported | Untested |
| Codec / memory gate have "no direct precedent" | `.bib` has no byte-level, hierarchical or memory-transformer work (checked by grep) | Absence of search, not absence of precedent |
| Masking curriculum fixes instability | "Observed", no curve or number | Undocumented |
| liz2 / codec / depth / MoE effects | 9 of 11 ablation configs train on ONE unique string x16 (`yaml` check); metric is min loss over 600 steps | Convergence speed on a trivial task; the paper itself shows liz2's gap vanishes |
| QKV-sharing case study | 3 controls, one mechanism-consistent | The one robust result (methodological) |

Title says "Compute-Efficient", yet Table 5 puts the target at 332 to 3 700 days for 30B tokens.

Cheapest falsifiable test of the central claim (planned experiment 12): vary N_total at fixed N_act
(e.g. 200M vs 50M) on one GPU and measure tokens/s at several batch sizes.

## Resolution: rewrite of `docs/paper/kairos_paper.tex`

Every finding above is addressed in the rewrite. Checks: `tests/test_paper.py` (citations, labels,
placeholders, seed statistics, generated table) and `tests/test_roofline.py` (sizing model).

| Change | Reason |
|--------|--------|
| Bandwidth thesis replaced by a threshold model: `T_min = (N_total / N_act) x ridge` | Weights are read once per microbatch, not per token; the old 94-375 tok/s figures are the T=1 regime |
| Active parameters fixed at 25M everywhere (Eq. 1 with E=32, k=4); 4M dropped | Internal inconsistency (finding 1); 25M is a lower bound, dense parameters raise it |
| Algorithms 1, 2, 4 rewritten from `modeling.py` / `attentions.py` | Codec fuses scales in decode; LiZ2 concatenates then norm and linear; gate keys include the query, `round` not `ceil`, forward pass only |
| Corruption described as uniform token replacement (D3PM-like), clip `c` added | Code fills masked positions with random tokens and gives no mask indicator; MDLM uses an absorbing state |
| Ablation setup stated exactly | One 440-byte string x16, about 2.9 copies per 128 tokens, window 128 is global, min loss is an extreme value of per-step losses, stride 5 in ablations vs 3 in the PoC |
| Removed: Gemma FLOP comparison, TinyStories rows, "39 runs", "no precedent" claims, RTX 5060 Ti row | Unverifiable or wrong (5060 Ti bandwidth is 448 GB/s, not 600) |
| Added: per-stage table (item 13), clip sweep (section 4), memory-gate numbers, 10 references | Measured data present in the log but absent from the draft |

To confirm by the author: Table 3 parameter counts other than the baseline (1.81, 2.50, 8.10, 4.93M),
PoC `N_total` (~14-15M) and `N_act`, dense `I=2048` in the ablation baseline, MFU 25% (assumed),
and arXiv identifiers of the 10 added references (BLT and the Kimi report were checked online).
