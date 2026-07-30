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

One `kind` per row, dispatched by `KairosPretrainingDataset._segments_for`:

| `kind` | Fields |
| --- | --- |
| `image_caption` | `image` (H,W,3 uint8), `caption` |
| `audio_caption` | `audio` (float32 [-1,1]), `sample_rate`, `caption` |
| `video_caption` | `video` (T,H,W,3 uint8), `caption` |
| `lidar` | `points` (N,4 float32) |
| `imu` | `signal` (T,6 float32) — acc xyz + gyro xyz |
| `control` | `action`, `state` (float32), `sample_rate`, `context` |

Arrays are stored as nested lists (Arrow has no native ndarray type); cast back to numpy on load, e.g.
via `scripts/pretrain/build_keep_it_simple_multimodal.py::pull_from_hub`.

## Sources

Flickr8k, AudioCaps, MSR-VTT, nuScenes-mini, MotionSense, PixelBytes-OptimalControl — small
slices only, aggressively downsized (see the build script for exact sizes).
