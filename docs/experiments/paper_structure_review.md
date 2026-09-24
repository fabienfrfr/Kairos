# Paper structure review: TPTT (`main.tex`) vs Kairos (`kairos_paper.tex`)

Scope: read-only comparison of narrative and style, plus one fix. Observations, not causal claims.

## TPTT pattern (reference)

| Element | TPTT |
|---------|------|
| Abstract | Structured (Background / Methods / Results / Conclusions), ~240 words |
| After abstract | Overview figure, cited in the Introduction |
| Introduction | Context -> prior work -> gap -> "In this paper" -> 3 contributions |
| Related work | Own section, one paragraph per family, closing "Our Positioning" |
| Body | One object (LiZAttention) built up in Methodology, then Training, then Experiments |
| Experiments | Setup first, then results, then one paragraph per variant |
| Conclusion | Summary of findings (bullets), Practical implications, Limitations, Future |
| Tone | Conditional wording ("may", "suggests"), "preliminary" stated repeatedly |

Thread: one method, one claim, one experiment family.

## Kairos: where the sequence of ideas breaks

Measured on `kairos_paper.tex` (862 lines, 5 algorithms, 6 tables, 3 figures).

1. **No single thread.** Architecture spec, compute sizing and ablation methodology are three
   contributions of equal weight (`paper_audit.md` reaches the same reading).
2. **Headline differs by section.** The abstract leads with the roofline threshold; the conclusion says
   the main result is methodological (ablation artifact).
3. **Question answered late.** The research question is at line 124 (compute-bound microbatch); the
   sizing model that answers it is at line 509, after ~280 lines of architecture.
4. **Methods weight exceeds results weight.** 3 of 5 algorithms (AttnRes, memory gate, denoising
   objective) have at most a single-run sweep in Results (AttnRes block size gives identical loss);
   the Limitations state no supporting result for the multimodal, curriculum and memory-gate claims.
5. **Table 1 comes before its vocabulary.** The three-scales table uses codec stride, AttnRes block
   size and shared backbone before they are defined.
6. **Duplicate figure.** Figure 1 (SVG) and the TikZ Figure 2 show the same data flow.
7. **Protocol far from results.** The ablation protocol (line 554) is ~90 lines before the ablation
   results (line 646). TPTT puts setup at the start of Experiments.
8. **Results opener concedes the split.** "Two unrelated kinds of result follow".
9. **Figure 1 was never cited.** `test_every_label_is_referenced` failed on `fig:architecture_overview`.
   Fixed: cited in Contribution 1. All 30 tests in `test_paper.py` / `test_roofline.py` pass.

## Kairos figure (`kairos_architecture.svg`)

Strengths: left-to-right flow, masked/denoised byte example, curriculum strip, memory gate visibly
apart from the main path.

Issues:

- **Legibility.** viewBox 1280 px wide, text 7-10 px. At 17 cm text width (A4, 2 cm margins) one px is
  ~0.38 pt, so text is ~2.6-3.8 pt. Print-readable text needs >= 6 pt (~16 px at this width).
- **Aspect ratio ~3:1** forces the scale-down; two rows and fewer secondary notes would fit the column.
- **Image/stride example** (lower left) is the densest element; keep it (multimodal training is planned in the notebook), but give it more room.
- **No evidence coding.** Memory gate is dashed; codec, curriculum and multimodal are not, although
  none has a supporting result either.
- **Constants differ across sources.** The figure shows stride 3 (PoC); the ablation baseline uses
  stride 5.
- **No PDF in the snapshot.** The paper includes `../architecture/kairos_architecture.pdf`; the
  Makefile has no target that builds it from the SVG.

## Proposed outline (needs the author's choice of headline)

Thread A (compute question first): Abstract (structured) -> Figure -> Introduction (gap + question +
contributions) -> Related work (own section, "Our Positioning") -> Design (architecture, ordered as the
figure) -> Sizing model -> Experiments (setup, sizing results, ablations, QKV case) -> Discussion.

Thread B (methodology first): same order, but the QKV correlation/causation case is the headline and
the sizing model moves to a shorter design-rationale section.

Common to both: drop TikZ Figure 2, move Table 1 after the architecture, move the ablation protocol
into Experiments, move memory-gate and curriculum algorithms to an appendix.

## Author's message (session 2) and consequences

Message: the PoC (~14-15M, the size trainable locally) converges (overfit tests) and its performance
is equivalent or superior to Pythia-14M, depending on what can be measured. Multimodal training is
in `notebook/kairos_pretraining.py` (text + `keep-it-simple-multimodal`, MAE -> transition -> diffusion).

Facts checked:

- Pythia-14M: 6 layers, d=128, 14.1M parameters of which 1.2M non-embedding (untied 50k-vocab
  embeddings), autoregressive, ~300B tokens of the Pile.
- Kairos PoC: byte vocabulary (259), so nearly all parameters are backbone; ~2.5M inactive (top-1 MoE).
- Data (Hub, checked): `keep-it-simple` 640,646 rows (513 MB, bidirectional prompt<->text pairs, BabyLM-inspired);
  `keep-it-simple-multimodal` 6,336 rows (33 MB, 6 sources), i.e. ~1% of rows. The dataset card in
  `scripts/pretrain/readme_multimodal.md` still says n<1K: stale, to update.

Missing elements (each one is a confound if absent):

1. **Metric.** Perplexity is not comparable (BPE vs bytes, autoregressive vs denoising). Use
   bits-per-byte on one held-out text; state whether the denoising loss is a bound or a proxy
   (corruption is uniform replacement, not an absorbing mask).
2. **Size accounting.** Report total, non-embedding, active parameters and FLOPs per byte.
3. **Matched baseline.** Pythia has 300B Pile tokens; Kairos far fewer. Train a Pythia-14M-shaped
   model on the same data and budget, otherwise data, architecture and objective are confounded.
4. **Variance.** One run, 2 eval batches: add seeds and a larger fixed eval set.
5. **Convergence evidence.** Train and held-out curves across the three curriculum stages
   (the audit flagged "observed, no curve").
6. **Multimodal.** A deliberately small pilot: report per-modality held-out loss against trivial
   baselines, and text BPB with vs without multimodal data (interference); no Pythia comparison is possible.
   Report the multimodal share in bytes as well as in rows.
8. **In-domain advantage.** Held-out `keep-it-simple` favors Kairos (Pythia never saw it); add a neutral
   corpus neither model trained on, and the matched baseline of item 3.
7. **Run record.** Hardware, tokens seen, wall-clock, config path.

Consequence for the outline: the 200M/25M roofline becomes a short forward-looking section; the QKV
case is the caution on reading overfit tests (memorization speed), which motivates held-out results.

## Outline v2 (thread: converges -> generalizes -> compares with Pythia-14M -> multimodal pilot)

1. Abstract (structured, as TPTT), then Figure 1, cited in the Introduction.
2. Introduction: context, gap, question, "In this paper", 3 contributions (architecture, protocol,
   findings incl. the QKV lesson). The roofline question leaves the Introduction.
3. Related Work as its own section, closing with "Our Positioning"; scaling-law paragraph moves out.
4. Model, ordered as Figure 1 (codec, LiZAttention2, AttnRes, MoE, memory gate, objective and
   curriculum); the three-scales table closes the section; TikZ Figure 2 removed.
5. Experimental protocol (TPTT "Experimental Setup"): data, size accounting, training, metric (BPB),
   baselines (pretrained Pythia-14M, Pythia-shaped model on the same data), seeds, overfit protocol.
6. Results, one paragraph per question: convergence (overfit + QKV case), held-out text vs Pythia,
   multimodal pilot. Only measured numbers (`test_paper.py` rejects placeholders).
7. Discussion and Conclusion: findings (bullets), practical implications, limitations once, future work.
8. Appendix: compute sizing of the 200M/25M target (roofline table and figure), the next-step design.

## Session 3: v1 rewritten with measured results only

Applied by moving existing blocks; only the abstract, Introduction, Results opener and Discussion were
rewritten. Order: structured abstract, Figure 1, Introduction, Related Work (own section, ends with
"Our Positioning"), Model, Experimental Protocol (three scales, overfitting protocol), Results
(convergence, QKV case, curriculum and memory gate), Discussion, Conclusion, Appendix A (compute sizing).

- Removed: TikZ Figure 2 (duplicate of Figure 1); roofline as a contribution and as a research question.
- Held-out, multimodal and Pythia-14M results are not reported; they appear once, in Future work.
- Checks: `test_paper.py` and `test_roofline.py` pass (30); a `latexmk` build (Figure 1 PDF generated
  from the SVG for the check only) gives 15 pages, no undefined reference or citation.
- Still open: Figure 1 text is ~3 pt at page width (see figure section above); the QKV table caption cites
  `liz2_shared_vs_separate_followup_raw.jsonl`, absent from the snapshot.

## Session 4: appendix removed, tokenizer added

- Appendix A (sizing model, table, roofline figure) removed. Its sources (Williams roofline, Kaplan, Chinchilla,
  two MoE scaling-law papers, Pythia, three NVIDIA datasheets) are cited in two Discussion paragraphs with one
  formula (`T_min = (N_total / N_act) * I*`); headline numbers (4.5e18 FLOPs, D/N_act = 1,200) kept, so
  `test_headline_numbers_follow_from_the_model` still holds. `scripts/roofline.py`, `tables/compute.tex` and
  their tests are unchanged; the table is no longer included in the paper.
- New subsection "Multimodal byte tokenizer" (Table + Eq. for place-value quantization), derived from
  `kairos/tokenizer.py`; the codec subsection keeps only embedding and head.
- Removed unused `tikz`/`pgfplots` packages, added `graphicx`. Build: 14 pages, no undefined reference.

Code observations (tokenizer not executed here: `torch`/`transformers` absent; to verify):

1. `len(KairosTokenizer())` is 290 in `tests/test_tokenizer.py`, but `KairosConfig.vocab_size` defaults to 291
   and a comment in `tokenizer.py` says 291. The paper uses 290.
2. `decode_audio` returns `len(waveform) / AUDIO_SAMPLE_RATE`, while `encode_audio` decimates by
   `PCM_SCALE_FACTOR = 4` by default: the returned duration would be 4x too short. The `<TICK>` comment
   ("one per second") holds only at factor 1; the paper says "every 16,000 samples".

## Session 5: tokenizer in Figure 1

- SVG: the input box is now "Kairos Tokenizer" ("bytes + modality id"); the image example caption reads
  "2D image, flattened by tokenizer". The toy strip omits the `<ENDLINE>` row markers the real tokenizer
  inserts; the Figure 1 caption says so. No element leaves its box (checked on a 3x render).
- The output "byte stream" box is unchanged (it is the tokenizer's decode side).
- Text in the figure is still ~3 pt at page width; the two-row redesign is not done.
