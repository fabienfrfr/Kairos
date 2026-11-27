# Pretraining readiness: can the model train on the project's datasets?

Question: can the model of `configs/pretrain.yaml` be trained on the multimodal dataset built for it
(`ffurfaro/keep-it-simple` text, `ffurfaro/keep-it-simple-multimodal`), and with which settings does it converge fastest?

## What could and could not be checked

- **The two datasets could not be installed here.** The sandbox cannot resolve `huggingface.co` (DNS failure; the network
  allow-list has no Hub domain), so nothing below says anything about the real data content. `python scripts/fetch_datasets.py`
  (`make fetch-datasets`) downloads both repos once network access exists; `configs/pretrain_sample.yaml` then trains from the
  local folders (`make sample-pretrain`). Set `HF_TOKEN` if the repos are private or gated.
- **What was run instead**: synthetic rows with the exact row schema and array shapes written by
  `scripts/pretrain/build_keep_it_simple_multimodal.py` (image 32x32x3, audio 8000 samples at 8 kHz, video 6x16x16x3, lidar
  300x4, IMU 128x3, control state/action of 16000 samples), generated as structured signals so that a loss decrease is
  possible, through the real `KairosPretrainingDataset`, model, curriculum and trainer of `configs/pretrain.yaml`.
- Offline smoke test shipped with the package: `kairos pretrain minimal --no-push` trained 400 steps over the three stages in
  about 30 s of training (70 s in total) with no skipped non-finite batch and a best epoch-average loss of 0.0408.

## Findings

1. **The pipeline trains end to end on every modality.** Text, image, audio, video, lidar, IMU and control rows tokenize,
   pack, train for three curriculum stages, evaluate, round-trip a checkpoint and generate, with no non-finite batch in any run.
2. **At the configured `max_len: 1024`, almost no multimodal example fits in one window.** With the default scale factors an
   image (32x32x3) takes about 3 100 tokens, an audio second 4 000, a video clip 4 700, a lidar scan 2 400 and a control clip
   16 000 (text and IMU stay under 200). Each example is cut into 3 to 16 windows that do not see each other. The
   `data.modality_scale_factors` setting fixes this at the cost of resolution:

   | Modality | tokens/example, default | windows | tokens/example, `{image 2, video 4, lidar 4, audio 16, control 32}` | windows |
   |---|---:|---:|---:|---:|
   | image + caption | 3 133 | 4 | 813 | 1 |
   | audio + caption | 4 024 | 4 | 1 024 | 2 |
   | video + caption | 4 735 | 5 | 343 | 1 |
   | lidar | 2 412 | 3 | 605 | 1 |
   | control | 16 012 | 16 | 2 012 | 2 |

   Audio stays at the limit and control above it. A larger `max_len` is the alternative; its cost was not measured.
3. **The token budget is dominated by control and audio**, not by the number of examples: with 14 examples per modality the
   training windows carry 44% control (IMU is folded into it), 20% audio, 16% image,
   12% lidar, 6% video and 2% text. Mixture weights should be set in tokens.
4. **The shipped learning rate is the slowest of those tried.** After 51 steps the held-out plain CE (initial value `ln 290` =
   5.67) is:

   | lr | train CE | held-out CE | non-finite batches |
   |---|---|---|---:|
   | 0.0003 | 5.687 -> 5.471 | 5.697 -> 5.479 | 0 |
   | 0.003 | 5.687 -> 4.975 | 5.697 -> 4.977 | 0 |
   | 0.008 | 5.687 -> 4.851 | 5.697 -> 4.861 | 0 |
   | 0.016 | 5.687 -> 4.835 | 5.697 -> 4.856 | 0 |

   `lr = 8e-3` is the best compromise (1.6e-2 is equal overall but worse on text, image and audio); it is also the rate
   selected by the tiny-scale sweep of `training_algorithms.md`. `configs/pretrain_sample.yaml` uses it.
5. **Held-out plain CE per modality after 51 steps** (modality share of the training tokens in the first column):

   | Modality | token share | before | 3e-4 | 3e-3 | 8e-3 | 1.6e-2 |
   |---|---:|---:|---:|---:|---:|---:|
   | TEXT | 2.3% | 5.67 | 5.46 | 5.17 | 4.99 | 5.37 |
   | IMAGE | 15.6% | 5.75 | 5.29 | 4.23 | 3.94 | 4.15 |
   | VIDEO | 6.4% | 5.56 | 5.09 | 3.65 | 3.02 | 3.06 |
   | AUDIO | 19.9% | 5.68 | 5.41 | 4.70 | 4.52 | 4.65 |
   | LIDAR | 12.0% | 5.73 | 5.46 | 5.02 | 4.98 | 4.41 |
   | CONTROL | 43.7% | 5.69 | 5.63 | 5.51 | 5.51 | 5.51 |

   Every modality improves; control (the largest share) improves least and video most.
6. **Cost on one CPU core** (batch 4 x 1024 tokens, the model of `pretrain.yaml`): 3 818 397 parameters (3 342 336 in the
   routed experts), about 3.8 s per step including evaluations, 1.7 GB peak resident memory. The configured batch of 32 was not
   measured.
7. **`pipe.check_per_modality_loss` is not a stable diagnostic.** It calls `compute_masked_diffusion_losses` with the `1/p`
   weight and no clipping, so a modality read 1 946 before training (audio) and 8.6 after in the first run of this study (not stored). This script measures plain CE on
   seeded masks instead; the method itself was not changed.

## What the dataset cards add (not measured here)

From the Hub dataset cards and viewers (pasted by the author; not downloaded): `keep-it-simple` has 640 646 rows (513 MB, MIT) with
columns `prompt`, `text`, `seed_data`; `keep-it-simple-multimodal` has 6 336 rows (33.2 MB, MIT) with `modality`, `caption`,
`source`, `data` (an `.npz` with named arrays, e.g. `image.npy`), `meta`. Two differences from the synthetic rows used above:

- the `image_caption` rows shown come from `detection-datasets-coco` and their caption is a list of boxes
  (`cat=45 box=(1, 188, 613, 474); ...`), not a sentence, so token counts per example depend on the number of boxes;
- the viewers show very long text fields (the pasted headers are garbled; text up to roughly 1.2 M characters and captions up to
  roughly 42 k characters), which would span many 1024-token windows. `check_pretraining_readiness.py data --source hf`
  reports this on the real rows.

## Limits

Synthetic rows with the dataset's schema (the held-out rows share the same classes, so held-out CE is not generalization to new
content), 51 steps, one seed, one CPU core, no real data, no GPU. Convergence on the real datasets and the stability of
`lr = 8e-3` over a long run are not established; a short warmup (`train.warmup_frac`) is a precaution that was not tested
at this scale.

## Reproduce

```bash
python scripts/check_pretraining_readiness.py data --per-modality 6 --scale-factors '{}'   # tokens per modality, default factors
python scripts/check_pretraining_readiness.py train --per-modality 14 --lr 0.008 \
  --run-dir /tmp/readiness --out docs/experiments/results/pretraining_readiness_lr0.008.json   # about 200 s
```

Raw results: `results/pretraining_readiness_lr*.json` (rates 3e-4, 3e-3, 8e-3, 1.6e-2). Tests:
`tests/test_pretraining_readiness_script.py`. Environment: `docs/experiments/README.md`.

## Next steps with network access and a GPU

See `rtx_quickstart.md`: `make fetch-datasets`, then `check_pretraining_readiness.py data|bench|train --source hf` on the real
rows (the real-data reader is tested on local parquet files in the Hub layout, not on the Hub datasets themselves), then
`make sample-pretrain`.
