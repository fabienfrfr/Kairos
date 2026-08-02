---
license: mit
task_categories:
- other
tags:
- multimodal
- image
- audio
- video
- lidar
- imu
- robotics
pretty_name: keep-it-simple-multimodal
size_categories:
- n<1K
---

# keep-it-simple-multimodal

A mini, standalone multimodal dataset: image+caption, audio+caption, video+caption, lidar, IMU, and
optimal-control state/action pairs. Companion to [keep-it-simple](https://huggingface.co/datasets/ffurfaro/keep-it-simple)
(text), built to feed `KairosPretrainingDataset` in [kairos](https://github.com/ffurfaro/kairos).

## Structure

One generic schema for every row — no per-modality columns, no fixed shape/dtype assumptions:

| Column | Type | Description |
| --- | --- | --- |
| `modality` | string | `image_caption` \| `audio_caption` \| `video_caption` \| `lidar` \| `imu` \| `control` |
| `caption` | string | text label (or control's parsed transfer-function params, as JSON) |
| `source` | string | which upstream dataset this row came from (flickr8k, audiocaps, ...) |
| `data` | binary | one or more named numpy arrays packed with `numpy.savez` — self-describing (shape/dtype travel with the data) |
| `meta` | string | small extra scalars as JSON (e.g. `sample_rate`), or null |

`data` unpacks with `numpy.load` on a `BytesIO` wrapper (or `kairos.dataset.unpack_multimodal_data`),
giving back a dict of named arrays, e.g. `{"image": (H,W,3) uint8}` for `image_caption` or
`{"action": (T,) float32, "state": (T,) float32}` for `control`. Shapes are whatever the source
example actually had — nothing here assumes e.g. every image is 32x32.

## Sources

Flickr8k, AudioCaps, Molmo2-VideoCapQA (allenai), nuScenes-mini, MotionSense,
PixelBytes-OptimalControl — small slices only, aggressively downsized (see the build script for
exact sizes). Video clips are downloaded from AI2's public GCS mirror at build time, not re-hosted here.
