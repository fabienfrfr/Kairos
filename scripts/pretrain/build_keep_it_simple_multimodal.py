"""

Builds keep-it-simple-multimodal: a mini, standalone multimodal dataset
(image+caption, audio+caption, video+caption, lidar, IMU, optimal control
state/action) — NOT including keep-it-simple's text itself, target ~51MB.

Sources (small slices only, downsized aggressively):
  - Flickr8k (ariG23498/flickr8k)                    -> image_caption
  - AudioCaps (OpenSound/AudioCaps, has real audio)   -> audio_caption
  - Molmo2-VideoCapQA (allenai) + GCS video mapping   -> video_caption
  - nuScenes mini (KevinNotSmile/nuscenes-qa-mini)    -> lidar
  - MotionSense (github.com/mmalekzadeh/motion-sense) -> imu
  - ffurfaro/PixelBytes-OptimalControl                -> control
"""

import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from kairos.dataset import pack_multimodal_data

# a stalled read otherwise hangs forever (seen on nuscenes-qa-mini); bound it so a slow/dead
# connection surfaces as a normal exception instead of an unkillable-looking freeze.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

CACHE_DIR = "data/cache"
CACHE_SCHEMA_VERSION = 2  # bump this whenever the row schema (make_row's keys) changes


def make_row(modality, source, caption=None, **fields):
    """Build a generic {modality, caption, source, data, meta} row: numpy arrays go into `data`
    (self-describing via .npz, no fixed shape/dtype assumed here), everything else into `meta`."""
    arrays = {k: v for k, v in fields.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in fields.items() if not isinstance(v, np.ndarray)}
    return {
        "modality": modality,
        "caption": caption,
        "source": source,
        "data": pack_multimodal_data(arrays) if arrays else None,
        "meta": json.dumps(meta) if meta else None,
    }


# per-source example count, sized so each modality lands around ~8.5MB (~51MB total across 5 sources)
N_PER_SOURCE = {
    "image_caption": 2700,
    "audio_caption": 270,
    "video_caption": 250,
    "lidar": 100,
    "imu": 1800,
    "control": 1100,
}
IMAGE_SIZE = 32
AUDIO_SECONDS = 1.0
AUDIO_SAMPLE_RATE = 8000
VIDEO_FRAMES = 6
VIDEO_SIZE = 16
VIDEO_MAX_BYTES = 20_000_000  # skip absurdly large source clips rather than downloading them fully
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
    n = N_PER_SOURCE["image_caption"]
    for row in tqdm(ds.take(n), total=n, desc="image_caption"):
        img = row.get("image")
        caption = row.get("caption") or row.get("captions")
        if img is None or not caption:
            continue
        if isinstance(caption, list):
            caption = caption[0]
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        out.append(make_row("image_caption", "flickr8k", caption=str(caption), image=np.array(img, dtype=np.uint8)))
    return out


def build_audio_caption():
    from datasets import Audio, load_dataset

    # d0rj/audiocaps only has {audiocap_id, youtube_id, start_time, caption} — no audio at all.
    # OpenSound/AudioCaps has real audio; decode=False + av avoids torchcodec's system FFmpeg requirement.
    ds = load_dataset("OpenSound/AudioCaps", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    n = N_PER_SOURCE["audio_caption"]
    for row in tqdm(ds.take(n), total=n, desc="audio_caption"):
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
            make_row("audio_caption", "audiocaps", caption=str(caption), audio=arr, sample_rate=AUDIO_SAMPLE_RATE)
        )
    return out


def _decode_video_bytes(raw_bytes: bytes) -> list | None:
    """Evenly sample VIDEO_FRAMES frames from an encoded video via av (self-contained, no system ffmpeg)."""
    import io

    import av

    container = av.open(io.BytesIO(raw_bytes))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        h, w = arr.shape[:2]
        step_h, step_w = max(h // VIDEO_SIZE, 1), max(w // VIDEO_SIZE, 1)
        frames.append(arr[::step_h, ::step_w][:VIDEO_SIZE, :VIDEO_SIZE])
        if len(frames) >= VIDEO_FRAMES:
            break
    container.close()
    return frames if len(frames) >= VIDEO_FRAMES else None


def build_video_caption():
    """allenai/Molmo2-VideoCapQA only ships video_ids; AI2 separately maps them to public GCS mp4
    URLs (plain https, no gsutil/auth needed — see allenai/Molmo2-8B's own usage example)."""
    import datasets
    import requests
    from huggingface_hub import hf_hub_download

    ds = datasets.load_dataset("allenai/Molmo2-VideoCapQA", split="CapQA")

    try:
        mapping_path = hf_hub_download(
            repo_id="allenai/Molmo2-VideoCapQA", filename="youtube_id_to_urls_mapping.json", repo_type="dataset"
        )
    except Exception as e:  # noqa: BLE001 — mapping location isn't guaranteed; fail soft with a clear pointer
        print(f"[video_caption] couldn't fetch youtube_id_to_urls_mapping.json from the dataset repo ({e})")
        print("[video_caption] set VIDEO_MAPPING_PATH to a local copy of that file to unblock this source")
        mapping_path = os.environ.get("VIDEO_MAPPING_PATH")
        if not mapping_path:
            return []
    with open(mapping_path) as f:
        mapping = json.load(f)

    caption_col = next((c for c in ("caption", "merged_caption", "video_caption") if c in ds.column_names), None)
    if caption_col is None:
        print(f"[video_caption] no known caption column found; columns are {ds.column_names}")
        return []

    out, warned = [], False
    n = min(N_PER_SOURCE["video_caption"], len(ds))
    for row in tqdm(ds.select(range(n)), total=n, desc="video_caption"):
        video_id, caption = row.get("video_id"), row.get(caption_col)
        entry = mapping.get(video_id) if video_id else None
        if entry is None or not caption:
            continue
        url = entry.get("gcp_url") if isinstance(entry, dict) else entry
        try:
            resp = requests.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            raw = resp.raw.read(VIDEO_MAX_BYTES + 1)
            if len(raw) > VIDEO_MAX_BYTES:
                continue
            frames = _decode_video_bytes(raw)
        except Exception as e:  # noqa: BLE001 — a handful of dead links/unsupported clips is expected
            if not warned:
                print(f"[video_caption] a video failed to download/decode ({e}); skipping rows like this")
                warned = True
            continue
        if not frames:
            continue
        video_arr = np.stack(frames, axis=0).astype(np.uint8)
        out.append(make_row("video_caption", "molmo2-videocapqa", caption=str(caption), video=video_arr))
    return out


def build_lidar():
    from datasets import load_dataset

    # this dataset requires an explicit config name ("day" or "night")
    ds = load_dataset("KevinNotSmile/nuscenes-qa-mini", "day", split="train", streaming=True)
    # each row also carries 6 full camera images we don't use; dropping them cuts streaming
    # bandwidth/latency a lot (this dataset is slow to iterate otherwise).
    cam_cols = [c for c in ds.column_names if c.startswith("CAM_")]
    if cam_cols:
        ds = ds.remove_columns(cam_cols)
    out, warned = [], False
    n = N_PER_SOURCE["lidar"]

    def warn_once(msg):
        nonlocal warned
        if not warned:
            print(f"[lidar] {msg}")
            warned = True

    try:
        for row in tqdm(ds.take(n), total=n, desc="lidar"):
            raw = row.get("LIDAR_TOP")
            if raw is None:
                warn_once(f"no LIDAR_TOP key found; row keys are {list(row.keys())}")
                continue
            original_type = type(raw)
            if isinstance(raw, dict):
                resolved = raw.get("bytes") or raw.get("array")
                if resolved is None:
                    warn_once(f"LIDAR_TOP is a dict but has neither 'bytes' nor 'array'; keys are {list(raw.keys())}")
                    continue
                raw = resolved
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
                    warn_once(f"LIDAR_TOP list has unexpected shape {arr.shape}")
                    continue
            else:
                warn_once(f"LIDAR_TOP has unhandled type {original_type}")
                continue
            if arr.shape[0] == 0:
                warn_once("decoded points array is empty after reshaping")
                continue
            out.append(make_row("lidar", "nuscenes-qa-mini", points=arr[:LIDAR_POINTS].astype(np.float32)))
    except Exception as e:  # noqa: BLE001 — a network hiccup shouldn't discard rows already fetched
        print(f"[lidar] network error after {len(out)} rows, keeping what we have ({e})")
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
    for name in tqdm(csv_names, desc="imu"):
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
        out.append(make_row("imu", "motionsense", signal=signal))
    return out


def _parse_control_params(text: str) -> dict:
    """PixelBytes-OptimalControl's `text` field is a tiny 2-row CSV, not free text:
    header 'numerator,denominator,method,controller_params,u,y' + one quoted-list-valued row."""
    import ast
    import csv
    import io

    try:
        header, values = list(csv.reader(io.StringIO(text)))
    except (ValueError, csv.Error):
        return {"raw": text}
    params = {}
    for key, value in zip(header, values):
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value
    return params


def build_control():
    from datasets import Audio, load_dataset

    ds = load_dataset("ffurfaro/PixelBytes-OptimalControl", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    n = N_PER_SOURCE["control"]
    for row in tqdm(ds.take(n), total=n, desc="control"):
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
            make_row(
                "control",
                "pixelbytes-optimalcontrol",
                caption=json.dumps(_parse_control_params(str(row.get("text", "")))),
                action=np.clip(arr[0], -1.0, 1.0).astype(np.float32),
                state=np.clip(arr[1], -1.0, 1.0).astype(np.float32),
                sample_rate=sample_rate,
            )
        )
    return out


def push_to_hub(examples: list[dict], repo_id: str = HF_REPO_ID):
    """Push the built examples as a HF dataset, with the README alongside it."""
    from datasets import Dataset, Features, Value
    from huggingface_hub import HfApi

    features = Features(
        {
            "modality": Value("string"),
            "caption": Value("string"),
            "source": Value("string"),
            "data": Value("binary"),
            "meta": Value("string"),
        }
    )
    dataset = Dataset.from_list(examples, features=features)
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

    return list(load_dataset(repo_id, split="train"))


if __name__ == "__main__":
    import argparse

    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/keep-it-simple-multimodal.pt", help="Local .pt output path.")
    parser.add_argument("--repo", default=HF_REPO_ID, help="HF dataset repo id to push to.")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing to the HF Hub.")
    parser.add_argument(
        "--force-rebuild",
        nargs="*",
        default=None,
        metavar="SOURCE",
        help="Re-download these sources even if cached (no names = rebuild everything).",
    )
    args = parser.parse_args()

    builders = {
        "image_caption": build_image_caption,
        "audio_caption": build_audio_caption,
        "video_caption": build_video_caption,
        "lidar": build_lidar,
        "imu": build_imu,
        "control": build_control,
    }
    force = set(builders) if args.force_rebuild == [] else set(args.force_rebuild or [])

    os.makedirs(CACHE_DIR, exist_ok=True)
    examples = []
    for name, builder in builders.items():
        cache_path = os.path.join(CACHE_DIR, f"{name}_v{CACHE_SCHEMA_VERSION}.pkl")
        if name not in force and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                rows = pickle.load(f)
            print(f"[{name}] loaded {len(rows)} examples from cache ({cache_path})")
        else:
            try:
                rows = builder()
                print(f"[{name}] got {len(rows)} examples")
                if rows:  # don't cache empty results — a source that got 0 rows should retry, not stay stuck
                    with open(cache_path, "wb") as f:
                        pickle.dump(rows, f)
            except Exception as e:  # noqa: BLE001 — one broken source shouldn't abort the whole build
                print(f"[{name}] SKIPPED ({e})")
                rows = []
        examples += rows

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(examples, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nSaved {len(examples)} examples to {args.out} ({size_mb:.2f} MB)")

    if not args.no_push:
        push_to_hub(examples, args.repo)
