"""
scripts/build_keep_it_simple_multimodal.py

Builds keep-it-simple-multimodal: a mini, standalone multimodal dataset
(image+caption, audio+caption, video+caption, lidar, IMU, optimal control
state/action) — NOT including keep-it-simple's text itself, target ~51MB.

Sources (small slices only, downsized aggressively):
  - Flickr8k (ariG23498/flickr8k)                    -> image_caption
  - AudioCaps (OpenSound/AudioCaps, has real audio)   -> audio_caption
  - MSR-VTT (friedrichor/MSR-VTT)                     -> video_caption
  - nuScenes mini (KevinNotSmile/nuscenes-qa-mini)    -> lidar
  - MotionSense (github.com/mmalekzadeh/motion-sense) -> imu
  - ffurfaro/PixelBytes-OptimalControl                -> control

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

# per-source example count, sized so each modality lands around ~8.5MB (~51MB total)
N_PER_SOURCE = {
    "image_caption": 2700,
    "audio_caption": 270,
    "video_caption": 1900,
    "lidar": 1800,
    "imu": 1800,
    "control": 1100,
}
IMAGE_SIZE = 32
VIDEO_FRAMES = 6
VIDEO_SIZE = 16
AUDIO_SECONDS = 1.0
AUDIO_SAMPLE_RATE = 8000
LIDAR_POINTS = 300
IMU_TIMESTEPS = 200
MOTIONSENSE_ZIP_URL = "https://codeload.github.com/mmalekzadeh/motion-sense/zip/refs/heads/master"
HF_REPO_ID = "ffurfaro/keep-it-simple-multimodal"


def _decode_audio_bytes(raw_bytes: bytes, layout: str = "mono", rate: int | None = None) -> tuple[np.ndarray, int]:
    """Decode encoded audio bytes with PyAV (bundles its own FFmpeg, no system libs needed).
    Returns (channels, samples) float32 in [-1, 1] at `rate` (source rate if None)."""
    import io

    import av

    container = av.open(io.BytesIO(raw_bytes))
    stream = container.streams.audio[0]
    out_rate = rate or stream.rate
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=out_rate)
    chunks = [rframe.to_ndarray() for frame in container.decode(stream) for rframe in resampler.resample(frame)]
    container.close()
    return np.concatenate(chunks, axis=1).astype(np.float32), out_rate


def build_image_caption():
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset("ariG23498/flickr8k", split="train", streaming=True)
    out = []
    for row in ds.take(N_PER_SOURCE["image_caption"]):
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
    from datasets import Audio, load_dataset

    # d0rj/audiocaps only has {audiocap_id, youtube_id, start_time, caption} — no audio at all.
    # OpenSound/AudioCaps has real audio; decode=False + av avoids torchcodec's system FFmpeg requirement.
    ds = load_dataset("OpenSound/AudioCaps", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for row in ds.take(N_PER_SOURCE["audio_caption"]):
        audio = row.get("audio")
        caption = row.get("caption")
        if audio is None or not audio.get("bytes") or not caption:
            continue
        try:
            arr, _ = _decode_audio_bytes(audio["bytes"], layout="mono", rate=AUDIO_SAMPLE_RATE)
        except Exception:  # noqa: BLE001, S112 — a handful of malformed/unsupported clips is expected, just skip them
            continue
        max_samples = int(AUDIO_SECONDS * AUDIO_SAMPLE_RATE)
        arr = np.clip(arr[0, :max_samples], -1.0, 1.0)
        out.append(
            {
                "kind": "audio_caption",
                "audio": arr,
                "sample_rate": AUDIO_SAMPLE_RATE,
                "caption": str(caption),
            }
        )
    return out


def build_video_caption():
    import io

    import av
    from datasets import Video, load_dataset

    ds = load_dataset("friedrichor/MSR-VTT", "train_9k", split="train", streaming=True)
    # decode=False: streaming gives a source-relative path ("video0.mp4") that isn't a real local
    # file, only raw bytes are usable — decode ourselves with av instead.
    ds = ds.cast_column("video", Video(decode=False))
    out = []
    for row in ds.take(N_PER_SOURCE["video_caption"]):
        video = row.get("video")
        caption = row.get("caption") or row.get("sentence")
        if video is None or not video.get("bytes") or not caption:
            continue
        if isinstance(caption, list):
            caption = caption[0]
        try:
            container = av.open(io.BytesIO(video["bytes"]))
        except Exception:  # noqa: BLE001, S112 — skip a handful of malformed/unsupported clips
            continue
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

    # this dataset requires an explicit config name ("day" or "night")
    ds = load_dataset("KevinNotSmile/nuscenes-qa-mini", "day", split="train", streaming=True)
    out, warned = [], False
    for row in ds.take(N_PER_SOURCE["lidar"]):
        raw = row.get("LIDAR_TOP")
        if raw is None:
            if not warned:
                print(f"[lidar] no LIDAR_TOP key found; row keys are {list(row.keys())}")
                warned = True
            continue
        if isinstance(raw, dict):
            raw = raw.get("bytes") or raw.get("array")
        # nuScenes stores lidar sweeps as flat (x, y, z, intensity, ring); ring is dropped below.
        if isinstance(raw, (bytes, bytearray)):
            arr = np.frombuffer(raw, dtype=np.float32)
            arr = arr[: (len(arr) // 5) * 5].reshape(-1, 5)[:, :4]
        elif isinstance(raw, list):
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[: (len(arr) // 5) * 5].reshape(-1, 5)[:, :4]
            elif arr.ndim == 2 and arr.shape[-1] >= 4:
                arr = arr[:, :4]
            else:
                continue
        else:
            continue
        if arr.shape[0] == 0:
            continue
        out.append({"kind": "lidar", "points": arr[:LIDAR_POINTS].astype(np.float32)})
    return out


def build_imu():
    """MotionSense isn't on the HF Hub as a clean dataset; pull the real CSVs from GitHub directly."""
    import io
    import random
    import zipfile

    import requests

    resp = requests.get(MOTIONSENSE_ZIP_URL, timeout=60)
    resp.raise_for_status()
    outer = zipfile.ZipFile(io.BytesIO(resp.content))
    inner_bytes = outer.read("motion-sense-master/data/A_DeviceMotion_data.zip")
    inner = zipfile.ZipFile(io.BytesIO(inner_bytes))

    csv_names = [
        n for n in inner.namelist() if n.endswith(".csv") and "__MACOSX" not in n and "/._" not in n
    ]
    random.shuffle(csv_names)

    out = []
    cols = ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z", "rotationRate.x", "rotationRate.y", "rotationRate.z"]
    for name in csv_names:
        if len(out) >= N_PER_SOURCE["imu"]:
            break
        try:
            with inner.open(name) as f:
                lines = f.read().decode("utf-8").splitlines()
            header = lines[0].split(",")
            idx = [header.index(c) for c in cols]
            rows = [line.split(",") for line in lines[1 : IMU_TIMESTEPS + 1]]
            if len(rows) < IMU_TIMESTEPS:
                continue
            signal = np.array([[float(r[i]) for i in idx] for r in rows], dtype=np.float32)
        except (UnicodeDecodeError, ValueError, IndexError):
            continue
        out.append({"kind": "imu", "signal": signal})
    return out


def build_control():
    from datasets import Audio, load_dataset

    ds = load_dataset("ffurfaro/PixelBytes-OptimalControl", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for row in ds.take(N_PER_SOURCE["control"]):
        audio = row.get("audio")
        if audio is None or not audio.get("bytes"):
            continue
        try:
            arr, sample_rate = _decode_audio_bytes(audio["bytes"], layout="stereo")
        except Exception:  # noqa: BLE001, S112 — skip a handful of malformed/unsupported clips
            continue
        if arr.shape[0] != 2 or arr.shape[1] < 2:
            continue
        out.append(
            {
                "kind": "control",
                "action": np.clip(arr[0], -1.0, 1.0).astype(np.float32),
                "state": np.clip(arr[1], -1.0, 1.0).astype(np.float32),
                "sample_rate": sample_rate,
                "context": str(row.get("text", "")),
            }
        )
    return out


def _to_arrow_row(example: dict) -> dict:
    """numpy arrays/scalars aren't Arrow-safe; convert to native Python for push_to_hub."""
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in example.items()}


def _from_arrow_row(row: dict) -> dict:
    """Inverse of _to_arrow_row: restore numpy arrays for the fields KairosTokenizer expects."""
    array_fields = {
        "image": np.uint8,
        "video": np.uint8,
        "points": np.float32,
        "signal": np.float32,
        "audio": np.float32,
        "action": np.float32,
        "state": np.float32,
    }
    out = {k: v for k, v in row.items() if v is not None}
    for field, dtype in array_fields.items():
        if field in out:
            out[field] = np.array(out[field], dtype=dtype)
    return out


def push_to_hub(examples: list[dict], repo_id: str = HF_REPO_ID):
    """Push the built examples as a HF dataset, with the README alongside it."""
    from datasets import Dataset, Features, Sequence, Value
    from huggingface_hub import HfApi

    rows = [_to_arrow_row(ex) for ex in examples]
    all_keys = {k for row in rows for k in row}  # Dataset.from_list only infers columns from row 0
    rows = [{k: row.get(k) for k in all_keys} for row in rows]

    # Dataset.from_list would otherwise infer float64 from plain Python lists, ~doubling storage
    # vs. the float32 arrays we actually built — pin the real dtype/nesting per field explicitly.
    field_shapes = {
        "image": (Value("uint8"), 3),  # H, W, 3
        "video": (Value("uint8"), 4),  # T, H, W, 3
        "points": (Value("float32"), 2),  # N, 4
        "signal": (Value("float32"), 2),  # T, 6
        "audio": (Value("float32"), 1),
        "action": (Value("float32"), 1),
        "state": (Value("float32"), 1),
    }
    features = {}
    for key in all_keys:
        if key in field_shapes:
            base, depth = field_shapes[key]
            feat = base
            for _ in range(depth):
                feat = Sequence(feat)
            features[key] = feat
        elif key == "sample_rate":
            features[key] = Value("int64")
        else:
            features[key] = Value("string")

    dataset = Dataset.from_list(rows, features=Features(features))
    print(f"Arrow dataset built: {len(dataset)} rows, {dataset.data.nbytes / 1e6:.2f} MB in memory")
    dataset.push_to_hub(repo_id)
    HfApi().upload_file(
        path_or_fileobj="scripts/pretrain/readme_multimodal.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed {len(examples)} examples and README to {repo_id}")


def pull_from_hub(repo_id: str = HF_REPO_ID) -> list[dict]:
    """Load a pushed dataset back into the list[dict] shape KairosPretrainingDataset expects."""
    from datasets import load_dataset

    return [_from_arrow_row(row) for row in load_dataset(repo_id, split="train")]


if __name__ == "__main__":
    import argparse
    import os

    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/keep-it-simple-multimodal.pt", help="Local .pt output path.")
    parser.add_argument("--repo", default=HF_REPO_ID, help="HF dataset repo id to push to.")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing to the HF Hub.")
    args = parser.parse_args()

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

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(examples, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nSaved {len(examples)} examples to {args.out} ({size_mb:.2f} MB)")

    if not args.no_push:
        push_to_hub(examples, args.repo)
