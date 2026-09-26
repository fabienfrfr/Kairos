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

## Session 6: two-row redesign of Figure 1, PDFs generated

- Rebuilt `kairos_architecture.svg` from scratch as a fixed 880x644 canvas, two rows (input row: tokenizer,
  embedding, codec, image example, masked-input box; output row: backbone, scatter-back, head, output
  stream, denoised-output box, streaming cache, memory gate), same content and colors as the one-row
  version, plus two small purple weight-sharing badges (Embedding<->Output Head, Codec encode<->decode)
  since the return arrow that used to show this (top dashed loop) no longer fits the layout; the caption
  now explains the badges. Original one-row file kept as `kairos_architecture_wide.svg` (not referenced
  by the paper).
- At the paper's actual print width (measured on the compiled PDF, 300dpi), the smallest labels
  ("row wrap (4)" etc.) render close to 7pt, legible; previously ~3pt. Figure changed from `[H]` to `[t]`,
  full `\linewidth` (was 0.95).
- Generated `docs/architecture/kairos_architecture.pdf` (cairosvg) and `docs/paper/kairos_paper.pdf`
  (latexmk) and put both in the zip. `tests/test_paper.py`/`test_roofline.py`: 30 passed. Build: 15 pages,
  no undefined reference/citation.
- Not redone: the SVG is now plain rects/paths (no `<style>` classes to hand-tune later); porting further
  paper-text tweaks means re-running `make_svg.py`, not hand-editing the SVG.

## Session 7: single-direction figure, on page 1

Complaint: the two-row figure still had crossing/zigzag arrows and did not fit page 1. Rebuilt again:

- **Flow direction, strictly one-way.** Main pipeline is now one row, left to right, seven boxes
  (Tokenizer -> Embedding -> Codec -> Backbone -> Scatter back -> Output Head -> byte stream), arrows
  only rightward. Every annotation (input/output example, image example, streaming cache, memory gate)
  hangs directly below its box on straight vertical arrows, never back up or sideways. Curriculum strip
  at the bottom, left to right. No arrow crosses another module's box.
- **Sizing.** Column widths are computed once (max of a box's own width and its child annotation's
  width) so nothing overlaps; verified by rendering at 3x and by the compiled PDF. Overall canvas
  1220x288 (aspect ~4.2:1), shorter than the two-row version (was 880x644, ~1.4:1).
- **Page 1.** `\maketitle` spacing was tightened (`titling` package, no code/text removed) to recover
  the whitespace LaTeX puts around the title block; the figure environment changed from `[t]` (which
  only ever lands at the top of *a* page) to `[H]` (float package, "here", placed inline at that point
  in the text). Combined, the figure, its caption and the first paragraph of the Introduction now fit
  on page 1 with the compiled PDF (checked: no Overfull vbox, no undefined reference, 14 pages).
- The one-row/no-zigzag constraint means several previous nested details (AttnRes dashed box inside
  Backbone, the per-scale codec grid) are now one-line captions on the main-row box instead of a nested
  diagram; the full detail is still in the Model section text and Algorithms 1-3.
- `kairos_architecture_wide.svg` (session 6, two-row) is superseded and can be deleted; kept for now.

## Session 8: fixed overlap bug, restored internal blocks

Bug found: session 7's box widths for the annotation row (image example, streaming/memory) were
hardcoded pixel values that no longer matched the column widths after a manual rescale-to-1220 pass,
so those boxes bled into the neighboring column (e.g. image-example box: x in [270.7, 565.7] vs the
Streaming-cache box starting at x=506.3 -- a 59-unit overlap, confirmed on a 3x render).

Rebuilt without the rescale step: the canvas is now sized to its natural content width (no forced
total), so every annotation box is drawn at exactly its own column's width and position -- overlap is
impossible by construction (checked: printed column x-ranges are strictly increasing and non-touching).
LaTeX's `\includegraphics[width=\linewidth]` still scales the whole PDF down proportionally, so this
adds no manual-scale bug surface.

Internal blocks restored (asked for): the Pyramidal Codec box now shows its 2x2 per-scale grid; the
KairosBackbone box shows the dashed AttnRes strip and the KairosBlock (LiZAttention2 + FFN/MoE) rows;
Scatter back shows the Codec-decode sub-box. This only grew the figure's height (canvas 1632x466,
aspect ~3.5:1 vs ~4.2:1 before), not its width, so the single left-to-right row is unchanged.

Checked: 30 tests pass; compiled PDF is 14 pages, 0 undefined references, 0 Overfull warnings; Figure 1
with all internals still fits on page 1 (\maketitle tightened in session 7, [H] placement).
`kairos_architecture_wide.svg` (superseded two-row draft) removed from the repo.

## Session 9: legibility check, title, "proof of concept" reframed, missing citations

- **Legibility, verified.** At 300dpi on the compiled PDF (true print size), the figure's smallest
  labels are clearly readable, comparable to the caption text size -- the session-7/8 fix holds; the
  earlier "looks like session 1 again" impression was the overlap bug (session 8), now fixed, not a
  size regression.
- **Title.** Changed from a flat two-line description to a hook + question:
  "Kairos: One Byte Space for Every Modality" / "A Hybrid Delta/Window-Attention Mixture-of-Experts
  Denoising Network -- Does It Converge? Sanity Checks at Tiny Scale."
- **"Proof of concept" removed.** The abstract and conclusion no longer brand Kairos as a PoC (an
  engineering-deliverable framing); both now pose it as a research question ("this report asks whether
  the combination optimizes at all"). The unrelated technical shorthand "PoC" used throughout Methods/
  Results/Table 2 to name one of the three model scales (the notebook's default config, as opposed to
  the ablation baseline and the sizing target) is renamed "notebook config" everywhere, for the same
  reason, with no change in meaning.
- **Missing citations added:**
  - `furfaro2024pixelbytes` (arXiv:2410.01820, PixelBytes): cited in Related Work as the two-modality
    precursor this report substantially revises, and in the tokenizer table caption as the source of
    the control-stream action/state encoding.
  - `keepitsimple2026` / `keepitsimplemm2026`: both HF datasets now cited (not just named) where first
    introduced, in Future Work.
  - All three added to `kairos_references.bib` as `@misc` entries with direct `\url{}` links (visible
    in the reference list, not the margin -- this class has no margin-note mechanism).
- Checked: 30 tests pass; build is 15 pages (bibliography gained 3 entries, spilling one extra line to
  a mostly-blank final page); 0 undefined citation/reference, 0 Overfull; Figure 1 still fits page 1
  with room to spare under the new 3-line title.

## Session 10: real overflow bug fixed, numbered references, new title, foundation-model aim

- **Overflow, confirmed and fixed.** The user's screenshot was right: the Pyramidal Codec's caption
  ("reshape per scale, not pooling; each modality uses only the scales it needs", 76 chars) was a
  single unwrapped `<text>` spanning past both edges of its 300-unit-wide box into the neighboring
  KairosBackbone box. A systematic check (every other caption's estimated width against its containing
  box) found this was the only real overflow; one other flagged case was a false positive from the
  check script matching a small grid-cell rect instead of the actual container. Fixed by wrapping the
  caption onto two lines within the box. Regenerated PDF/PNG; re-verified visually at 3x.
- **References renumbered.** `\usepackage{natbib}` -> `\usepackage[numbers,sort&compress]{natbib}`;
  bibliography stays alphabetical (`plainnat`) but both in-text citations and the final list are now
  numbered ([1], [2], ... [46]), which is shorter given how citation-dense the Related Work paragraph is.
- **arXiv readiness, read through fully section by section:** no undefined references/citations, no
  Overfull warnings, no leftover TODOs or dangling cross-references from the multi-session edits. Two
  non-blocking notes: (i) page size is Letter, not A4 -- both are accepted by arXiv, purely a choice;
  (ii) arXiv auto-runs bibtex from the .bib when no .bbl is supplied, which works here, but including a
  precompiled `.bbl` alongside the submission avoids relying on arXiv's toolchain matching this local
  build exactly. Nothing else found that would not compile or display correctly on arXiv.
- **Title, changed again** per a follow-up request to foreground the more distinctive design choices:
  "Kairos: A Pyramidal Codec and Diffusion over One Byte Space" / "Bidirectional Delta/Window-Attention
  Mixture-of-Experts -- Does It Converge? Sanity Checks at Tiny Scale." Fits on one line at print width;
  Figure 1 still fits page 1.
- **Foundation-model aim, added once** (per the paper's own no-restatement rule), at the start of
  Future Work: "The long-term aim behind Kairos is a small multimodal foundation model that a single
  local machine can pretrain end to end; this report only asks whether the substrate optimizes."
- Checked: 30 tests pass; build is 14 pages, 0 undefined reference/citation, 0 Overfull.

## Session 11: "denoising" restored in title; full critical re-read for v1

Confirmed the user's observation: the previous title change had dropped "denoising" in favor of
"diffusion." Both are real and distinct in this paper (denoising is the umbrella training objective;
diffusion is only its final curriculum stage), so both now appear:
"Kairos: A Pyramidal Codec and Denoising Diffusion over One Byte Space" (manual line break added so
the title doesn't leave an orphan word on its own line). Still fits page 1 with Figure 1 intact.

Full section-by-section critical read of the paper text, cross-checking every number against its own
table and against the underlying `docs/experiments/*` logs. Four real issues found and fixed:

1. **Capitalization parallelism.** The three model scales are introduced as bolded terms in one
   sentence ("ablation networks", "notebook configuration", "target"); only the middle one was
   capitalized ("Notebook configuration"). Lowercased for consistency.
2. **A factual inaccuracy, now fixed.** Both mentions of "the seed spread" gave the range
   "$\pm0.03$ to $\pm0.19$", but Table 2's own sd column runs from $0.002$ (patch codec) to $0.193$
   (baseline) -- the stated range silently dropped the two tightest configurations (0.002, 0.007).
   Corrected to "$\pm0.002$ to $\pm0.193$" in both places (after Table 2, and in Discussion's
   Practical implications).
3. **A reproducibility gap.** Table 3's caption cited a per-step-curve file,
   `docs/experiments/liz2_shared_vs_separate_followup_raw.jsonl`, that is not in the repository
   snapshot (checked: absent). Removed the dangling reference from the caption; the remaining
   citation (`ablations_codec_attention.md`, \S7) does exist and matches. **Not fixed, flagged for the
   author:** the Results paragraph right after Table 3 still reports a number derived from that same
   missing file ("+9 to +14% steps for separate projections", from "the saved curves"). This is a
   live quantitative claim with no traceable artifact in what was provided -- either the raw JSONL
   needs to be added back to the repo before submission, or the claim should be softened/removed.
4. Checked and found consistent (no fix needed): the active-parameter formula
   ($N_\text{act}=N_\text{total}/8+\tfrac78 N_\text{dense}$) against Eq.~\ref{eq:active}; the
   200M/25M target's 8x ratio against the roofline paragraph's $N_\text{total}/N_\text{act}=8$; every
   per-seed number in Results/Discussion against Table 2 and Table 3; both cited log section numbers
   (\S7, item 13) against the actual files; US spelling throughout (no -ise/-ize mixing).

Not touched (judgment calls, not defects): the repeated "does it optimize at all" framing across
Abstract/Introduction/Conclusion -- standard IMRaD practice for a paper's throughline, distinct from
restating a *finding* three times, which the paper avoids elsewhere. "Kairos'" (bare apostrophe
possessive) used consistently in four places; defensible style, left as is.

## Session 12: title restructured -- value proposition up front, stack in the subtitle

Feedback: the previous main title ("A Pyramidal Codec and Denoising Diffusion over One Byte Space")
led with jargon and didn't communicate the point to a reader who doesn't already know the components.
Restructured so the hook states the actual value proposition and the subtitle carries the technical
stack:
- Title: "Kairos: One Byte Space for Every Modality" -- the thing a reader should remember: one shared
  representation across modalities.
- Subtitle: "A Pyramidal Codec and Denoising Diffusion over Bidirectional Delta/Window-Attention
  Mixture-of-Experts -- Does It Converge? Sanity Checks at Tiny Scale" -- every component (codec,
  denoising diffusion, bidirectional hybrid attention, MoE) plus the humble question framing.
Checked: 30 tests pass, 14 pages, 0 undefined reference/citation, 0 Overfull; Figure 1 still fits
page 1 under the four-line title block.

## Session 13: paper folder made self-contained for submission

The user's instinct was right: a paper's `\includegraphics` should not reach outside its own
directory. `docs/paper/kairos_paper.tex` referenced `../architecture/kairos_architecture.pdf` --
harmless when compiling from this repo (the relative path resolves), but not how a submission bundle
should be organized, and fragile if anyone ever zips just the `paper` folder. Fixed:

- Copied `kairos_architecture.pdf` into `docs/paper/` itself; the figure is now
  `\includegraphics{kairos_architecture.pdf}`, no `../`. `docs/architecture/` remains the design
  source (svg, puml, and its own pdf copy for the diagram tooling) -- that separation is fine, the
  paper just no longer depends on reaching into it.
- **Verified with a real self-contained build**: copied only `kairos_paper.tex`,
  `kairos_references.bib` and `kairos_architecture.pdf` into an empty directory and ran
  `pdflatex` -> `bibtex` -> `pdflatex` x2 from there, exactly the arXiv toolchain. Clean: 0 undefined
  citations/references, 0 Overfull, 14 pages, and a `.bbl` generated with no bibtex warnings.
- **Found one more orphan**: `docs/paper/tables/compute.tex` is not `\input` anywhere in the current
  paper (the old Appendix that used to include it was replaced by inline Discussion prose in session 9)
  but is still read directly by `tests/test_paper.py::test_compute_table_is_generated_by_the_script`,
  which checks it stays in sync with `scripts/roofline.py`'s output. Harmless for the PDF (arXiv won't
  see or need it) but stale relative to the current paper structure -- flagged for the author to decide
  whether to keep it as a repo-hygiene fixture or retire it and the test with it.
- Produced `kairos_arxiv_submission.zip`: exactly the four files arXiv needs
  (`kairos_paper.tex`, `kairos_references.bib`, `kairos_paper.bbl`, `kairos_architecture.pdf`), flat,
  no subdirectories, matching arXiv's own upload model (subdirectories only survive if uploaded as a
  single archive; flat is the safer default). Including the precompiled `.bbl` means arXiv does not
  need to run bibtex itself.

## Session 14: title -- "Toward a Foundation Model", not "Foundation Model"

The author's first instinct was "Kairos Foundation Model: One Byte Space for Every Modality" -- too
strong a claim for a paper whose only evidence is convergence sanity checks at 1-15M parameters (no
held-out results, no multimodal training run yet). Landed on a phrasing that keeps the destination
visible without asserting arrival:
"Kairos: One Byte Space for Every Modality / Toward a Foundation Model" (both at title size), with the
technical-stack subtitle unchanged below. "Toward" carries the same function as the paper's own
"this report only asks whether the substrate optimizes" (Future Work) -- an aim, not a result.
Checked: 30 tests pass, 14 pages, 0 undefined reference/citation, 0 Overfull; still 4 title/subtitle
lines total, Figure 1 still fits page 1.

## Session 15: title collapsed to one line

Two stacked title lines ("One Byte Space for Every Modality" / "Toward a Foundation Model") read as
two separate name-like fragments rather than one phrase. Merged into a single sentence:
"Kairos: Toward a Byte-Level Foundation Model for Every Modality" -- "toward" keeps the humility,
"byte-level" carries what "one byte space" meant, "for every modality" keeps the multimodal hook, and
it reads as one line rather than a stacked pair. Technical-stack subtitle unchanged below.
Checked: 30 tests pass, 14 pages, 0 undefined reference/citation, 0 Overfull; the title block is now
3 lines total (was 4), leaving Figure 1 even more comfortably on page 1.

## Session 16: every bibliography entry checked for a valid DOI

Went through all 46 entries in `kairos_references.bib` individually and added a real, verified DOI
(or explained why none exists):

- **30 arXiv-only preprints**: given the arXiv-issued DOI (`10.48550/arXiv.<id>`). Verified this is
  valid for arXiv's entire corpus, including pre-2022 postings (arXiv/DataCite completed registering
  DOIs for all existing articles by Feb 2022, confirmed via arXiv's own blog announcement).
- **9 entries upgraded to their actual published version** (were mislabeled as arXiv preprints; now
  reflect the peer-reviewed venue, with its DOI): BERT (NAACL 2019, ACL Anthology DOI), RoFormer
  (Neurocomputing 2024, Elsevier DOI), GQA (EMNLP 2023, ACL Anthology DOI), ResNet and DenseNet (CVPR
  2016/2017, IEEE DOIs), Hourglass Transformers (Findings of NAACL 2022, ACL Anthology DOI),
  Transformer-XL (ACL 2019, ACL Anthology DOI), ByT5 (already had the right venue fields, only the DOI
  was missing). Each keeps the original arXiv id in a `note` field.
- **2 entries checked and given the arXiv DOI with an explanatory note**: Switch Transformers (JMLR
  2022) and Pythia (ICML/PMLR 2023) are published at venues that do not register a real Crossref DOI
  (JMLR's own site and PMLR proceedings have none; the `10.5555/...` sometimes seen on ACM's Digital
  Library is a placeholder, not a resolvable DOI) -- confirmed via dblp, which explicitly flags Pythia
  as "does not have a DOI." The arXiv DOI is the only valid one available. MobileLLM is the same
  situation (ICML/PMLR) and was handled the same way.
- **1 already-correct entry re-verified**: `williams2009roofline`'s existing DOI
  (10.1145/1498765.1498785, CACM) confirmed correct.
- **3 NVIDIA datasheets and 2 Hugging Face datasets**: no DOI exists for these (product pages and
  user-uploaded HF datasets are not the kind of object Crossref/DataCite issue academic DOIs for by
  default); kept as direct `\url{}` links, which is what the paper already did.
- **One real inconsistency fixed in passing**: `behrouz2025titans`'s bibkey said "2025" but its `year`
  field said 2024; the arXiv ID (2501.00663) and its own paper page date it 2025 (submitted 31 Dec
  2024, posted as a January-2025 arXiv id) -- `year` corrected to 2025 to match the key and the ID.
- **Two entries specifically checked for existence** given how recent/unusual they looked:
  `kimiteam2026attnres` (arXiv:2603.15031, March 2026) is real, confirmed via multiple independent
  citations and the paper itself; not a hallucination.

Checked: all 46 entries still parse (brace-balanced); a real `pdflatex` -> `bibtex` -> `pdflatex` x2
cycle produced no bibtex warnings and no undefined citations; 30 project tests still pass; the
compiled paper grew to 15 pages (bibliography only, each upgraded/DOI'd entry takes 1-2 more lines);
page 1 (title, abstract, Figure 1) is unaffected.
