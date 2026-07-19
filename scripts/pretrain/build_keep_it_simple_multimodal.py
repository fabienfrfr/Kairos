"""
scripts/build_keep_it_simple_multimodal.py

Builds keep-it-simple-multimodal: a mini, standalone multimodal dataset
(image+caption, audio+caption, video+caption, lidar, IMU, optimal control
state/action) — NOT including keep-it-simple's text itself, and at least
10x smaller than keep-it-simple (513MB / 641k rows -> target well under 51MB).

Sources (small slices only, downsized aggressively):
  - Flickr8k (ariG23498/flickr8k)               -> image_caption
  - AudioCaps (d0rj/audiocaps)                  -> audio_caption
  - MSR-VTT (friedrichor/MSR-VTT)                -> video_caption
  - nuScenes mini (KevinNotSmile/nuscenes-qa-mini) -> lidar
  - MotionSense (best-effort HF id, may not exist) -> imu
  - ffurfaro/PixelBytes-OptimalControl           -> control

Each source is wrapped in try/except: if a dataset id, column name, or
network call fails, that source is skipped with a warning and the rest of
the script keeps going. Run it, it either builds what it can or tells you
clearly what failed.

Output schema is a list[dict], one "kind" per source, fields kept minimal
and consistent so KairosPretrainingDataset._segments_for can dispatch on
`kind` alone:
  image_caption: {kind, image (H,W,3 u8), caption (str)}
  audio_caption: {kind, audio (float32 [-1,1]), sample_rate, caption}
  video_caption: {kind, video (T,H,W,3 u8), caption}
  lidar:         {kind, points (N,4 float32)}
  imu:           {kind, signal (T,6 float32)}  # acc xyz + gyro xyz
  control:       {kind, action, state (float32), sample_rate, context}
"""

import numpy as np


N_PER_SOURCE = 60          # ultra-minimal: 6 sources x 60 ~ 360 examples total
IMAGE_SIZE = 32
VIDEO_FRAMES = 6
VIDEO_SIZE = 16
AUDIO_SECONDS = 1.0
AUDIO_SAMPLE_RATE = 8000
LIDAR_POINTS = 300
IMU_TIMESTEPS = 200


def build_image_caption():
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset("ariG23498/flickr8k", split="train", streaming=True)
    out = []
    for row in ds.take(N_PER_SOURCE):
        img = row.get("image")
        caption = row.get("caption") or row.get("captions")
        if img is None or not caption:
            continue
        if isinstance(caption, list):
            caption = caption[0]
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        out.append({"kind": "image_caption", "image": np.array(img, dtype=np.uint8), "caption": str(caption)})
    return out


def build_audio_caption():
    from datasets import load_dataset

    ds = load_dataset("d0rj/audiocaps", split="train", streaming=True)
    out = []
    for row in ds.take(N_PER_SOURCE):
        audio = row.get("audio")
        caption = row.get("caption")
        if audio is None or "array" not in audio or not caption:
            continue
        arr = np.asarray(audio["array"], dtype=np.float32)
        max_samples = int(AUDIO_SECONDS * AUDIO_SAMPLE_RATE)
        arr = arr[:max_samples]
        arr = np.clip(arr, -1.0, 1.0)
        out.append({
            "kind": "audio_caption", "audio": arr,
            "sample_rate": AUDIO_SAMPLE_RATE, "caption": str(caption),
        })
    return out


def build_video_caption():
    from datasets import load_dataset
    import av

    ds = load_dataset("friedrichor/MSR-VTT", split="train", streaming=True)
    out = []
    for row in ds.take(N_PER_SOURCE):
        video = row.get("video")
        caption = row.get("caption") or row.get("sentence")
        if video is None or not caption:
            continue
        if isinstance(caption, list):
            caption = caption[0]
        path = video if isinstance(video, str) else video.get("path")
        container = av.open(path)
        frames = []
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")
            h, w = arr.shape[:2]
            step_h, step_w = max(h // VIDEO_SIZE, 1), max(w // VIDEO_SIZE, 1)
            small = arr[::step_h, ::step_w][:VIDEO_SIZE, :VIDEO_SIZE]
            frames.append(small)
            if len(frames) >= VIDEO_FRAMES:
                break
        container.close()
        if len(frames) < VIDEO_FRAMES:
            continue
        video_arr = np.stack(frames, axis=0).astype(np.uint8)
        out.append({"kind": "video_caption", "video": video_arr, "caption": str(caption)})
    return out


def build_lidar():
    from datasets import load_dataset

    ds = load_dataset("KevinNotSmile/nuscenes-qa-mini", split="train", streaming=True)
    out = []
    for row in ds.take(N_PER_SOURCE):
        points = row.get("lidar") or row.get("points") or row.get("point_cloud")
        if points is None:
            continue
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[-1] < 4:
            continue
        arr = arr[:LIDAR_POINTS, :4]
        out.append({"kind": "lidar", "points": arr})
    return out


def build_imu():
    from datasets import load_dataset

    ds = load_dataset("MotionSense/motionsense", split="train", streaming=True)
    out = []
    for row in ds.take(N_PER_SOURCE):
        cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        if not all(c in row for c in cols):
            continue
        signal = np.stack([np.asarray(row[c], dtype=np.float32)[:IMU_TIMESTEPS] for c in cols], axis=-1)
        out.append({"kind": "imu", "signal": signal})
    return out


def build_control():
    from datasets import load_dataset

    ds = load_dataset("ffurfaro/PixelBytes-OptimalControl", split="train", streaming=True)
    out = []
    for row in ds.take(N_PER_SOURCE):
        audio = row.get("audio")
        if audio is None:
            continue
        arr = np.asarray(audio["array"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        out.append({
            "kind": "control",
            "action": np.clip(arr[:, 0], -1.0, 1.0).astype(np.float32),
            "state": np.clip(arr[:, 1], -1.0, 1.0).astype(np.float32),
            "sample_rate": audio["sampling_rate"],
            "context": str(row.get("text", "")),
        })
    return out


if __name__ == "__main__":
    import os
    import torch

    OUT_PATH = "data/keep-it-simple-multimodal.pt"

    builders = {
        "image_caption": build_image_caption,
        "audio_caption": build_audio_caption,
        "video_caption": build_video_caption,
        "lidar": build_lidar,
        "imu": build_imu,
        "control": build_control,
    }

    examples = []
    for name, builder in builders.items():
        try:
            rows = builder()
            print(f"[{name}] got {len(rows)} examples")
            examples += rows
        except Exception as e:
            print(f"[{name}] SKIPPED ({e})")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(examples, OUT_PATH)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\nSaved {len(examples)} examples to {OUT_PATH} ({size_mb:.2f} MB)")
    print("keep-it-simple is 513 MB -> target < 51 MB (10x smaller); this lands far below that.")
