"""
Builds keep-it-simple-multimodal: mini multimodal dataset (image+caption, audio+caption,
video+caption, lidar, control state/action), target ~51MB.

Datasets used:
  - detection-datasets/coco            -> image_caption (bbox serialized as text) (32, 32, 3) uint8
  - laion/relaion-coco                  -> image_caption (URL download, punsafe-filtered)
  - OpenSound/AudioCaps                  -> audio_caption
  - HuggingFaceFV/finevideo (gated)      -> video_caption - video (6, 16, 16, 3) uint8
  - nvidia/Cosmos-Transfer-LidarGen-Example (gated) -> lidar (300, 4) float32
  - ffurfaro/PixelBytes-OptimalControl   -> control - stereo

Gated sources: accept terms on the HF page, then `huggingface-cli login` or export HF_TOKEN.
"""

import io
import json
import os
import pickle
import tarfile

import numpy as np
from tqdm import tqdm

from kairos.dataset import pack_multimodal_data

# a stalled read otherwise hangs forever; bound it so a slow/dead connection
# surfaces as a normal exception instead of an unkillable-looking freeze.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

CACHE_DIR = "data/cache"
CACHE_SCHEMA_VERSION = 5  # bumped: caches now store RAW source rows; processing is re-run every build
CHECKPOINT_EVERY = 10  # rows between checkpoint saves for streaming sources

# per-source example count, sized so each modality lands around ~10MB (~51MB total across 5 sources)
N_PER_SOURCE = {
    "image_bbox": 2700,
    "image_caption": 1800,
    "audio_caption": 270,
    "video_caption": 250,
    "lidar": 2200,
    "control": 1100,
}
IMAGE_SIZE = 32
AUDIO_SECONDS = 1.0
AUDIO_SAMPLE_RATE = 8000
VIDEO_FRAMES = 6
VIDEO_SIZE = 16
VIDEO_MAX_DURATION_SEC = 1.0  # only sample within this window so frames capture short-term dynamics
LIDAR_POINTS = 300
LIDAR_REPO_ID = "nvidia/Cosmos-Transfer-LidarGen-Example"
LIDAR_TAR_FILENAME = None  # None = auto-pick the first .tar under lidar/ in the repo
HF_REPO_ID = "ffurfaro/keep-it-simple-multimodal"


def make_row(modality, source, caption=None, **fields):
    """Generic {modality, caption, source, data, meta} row: arrays -> data, rest -> meta."""
    arrays = {k: v for k, v in fields.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in fields.items() if not isinstance(v, np.ndarray)}
    return {
        "modality": modality,
        "caption": caption,
        "source": source,
        "data": pack_multimodal_data(arrays) if arrays else None,
        "meta": json.dumps(meta) if meta else None,
    }


def _iterate_resumable(ds, cache_path: str, process_row, n: int, desc: str) -> list[dict]:
    """Streams `ds`, but caches the RAW source rows — not the processed results — so every
    build re-runs `process_row` and picks up edits to the processing code. `--force-rebuild`
    clears the cache to re-fetch from the network. Returns the first `n` processed rows."""
    raw_rows, consumed = [], 0
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            state = pickle.load(f)
        raw_rows, consumed = state["raw_rows"], state["consumed"]
        print(f"[{desc}] resuming: {len(raw_rows)} raw rows cached, {consumed} source rows consumed")

    # always recompute from the raw rows, so processing changes take effect
    results = [r for r in (process_row(r) for r in raw_rows) if r is not None]

    def save_checkpoint():
        with open(cache_path, "wb") as f:
            pickle.dump({"raw_rows": raw_rows, "consumed": consumed}, f)

    with tqdm(total=n, initial=len(results), desc=desc) as pbar:
        if len(results) < n:
            for row in ds.skip(consumed):
                consumed += 1
                raw_rows.append(row)
                result = process_row(row)
                if result is not None:
                    results.append(result)
                    pbar.update(1)
                if len(results) >= n:
                    break
                if consumed % CHECKPOINT_EVERY == 0:
                    save_checkpoint()

    save_checkpoint()
    return results[:n]


def _peak_normalize(arr: np.ndarray, target_peak: float = 0.95) -> tuple[np.ndarray, float]:
    """Rescales `arr` so its max absolute value is `target_peak`, instead of hard-clipping."""
    peak = float(np.abs(arr).max())
    if peak < 1e-8:
        return arr.astype(np.float32), 1.0
    scale = target_peak / peak
    return (arr * scale).astype(np.float32), peak


def _decode_audio_bytes(raw_bytes: bytes, layout: str = "mono", rate: int | None = None) -> tuple[np.ndarray, int]:
    """Decode audio bytes via PyAV. Returns (channels, samples) float32 in [-1, 1]."""
    import av

    container = av.open(io.BytesIO(raw_bytes))
    stream = container.streams.audio[0]
    out_rate = rate or stream.rate
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=out_rate)
    chunks = [rframe.to_ndarray() for frame in container.decode(stream) for rframe in resampler.resample(frame)]
    container.close()
    return np.concatenate(chunks, axis=1).astype(np.float32), out_rate


def _decode_video_bytes(raw_bytes: bytes) -> tuple[list, dict] | None:
    """Samples VIDEO_FRAMES frames evenly spaced over min(duration, VIDEO_MAX_DURATION_SEC)"""
    import av
    from PIL import Image

    container = av.open(io.BytesIO(raw_bytes))
    stream = container.streams.video[0]
    orig_w, orig_h = stream.width, stream.height
    fps = float(stream.average_rate) if stream.average_rate else None
    duration_sec = float(stream.duration * stream.time_base) if stream.duration else None

    window = min(duration_sec or VIDEO_MAX_DURATION_SEC, VIDEO_MAX_DURATION_SEC)
    buffer = []  # (time, resized_array)
    for frame in container.decode(stream):
        t = float(frame.time or 0.0)
        arr = frame.to_ndarray(format="rgb24")
        # proper resize (not a strided crop) so the whole frame is represented, not just its
        # top-left corner after a coarse stride
        resized = np.array(Image.fromarray(arr).resize((VIDEO_SIZE, VIDEO_SIZE), Image.BILINEAR))
        buffer.append((t, resized))
        if t >= window:
            break
    container.close()
    if not buffer:
        return None

    targets = np.linspace(0.0, window, VIDEO_FRAMES)
    times = np.array([t for t, _ in buffer])
    frames = [buffer[np.abs(times - target).argmin()][1] for target in targets]
    meta = {"original_width": orig_w, "original_height": orig_h, "fps": fps, "duration_sec": duration_sec}
    return frames, meta


def build_image_bbox():
    """Image + bbox-as-text. Dataset: detection-datasets/coco. Bbox stored as plain text."""
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset("detection-datasets/coco", split="train", streaming=True)

    def process(row):
        img, objects = row.get("image"), row.get("objects") or {}
        bbox, category = objects.get("bbox"), objects.get("category")
        if img is None or not bbox:
            return None
        caption = "; ".join(f"cat={c} box={tuple(round(v) for v in b)}" for c, b in zip(category or [], bbox))
        orig_w, orig_h = img.size
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        return make_row(
            "image_caption",
            "detection-datasets-coco",
            caption=caption,
            image=np.array(img, dtype=np.uint8),
            original_width=orig_w,
            original_height=orig_h,
        )

    cache_path = os.path.join(CACHE_DIR, f"image_bbox_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, cache_path, process, N_PER_SOURCE["image_bbox"], desc="image_bbox")


def build_image_caption():
    """Image + caption. Dataset: laion/relaion-coco."""
    import requests
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset("laion/relaion-coco", split="train", streaming=True)
    PUNSAFE_MAX = 0.1

    def process(row):
        url, caption, punsafe = row.get("URL"), row.get("top_caption"), row.get("punsafe")
        if not url or not caption or (punsafe is not None and punsafe > PUNSAFE_MAX):
            return None
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            img_full = Image.open(io.BytesIO(resp.content)).convert("RGB")
            orig_w, orig_h = img_full.size
            img = img_full.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        except Exception:  # noqa: BLE001 — dead links/unsupported images are expected at scale
            return None
        return make_row(
            "image_caption",
            "laion-relaion-coco",
            caption=str(caption),
            image=np.array(img, dtype=np.uint8),
            original_width=orig_w,
            original_height=orig_h,
        )

    cache_path = os.path.join(CACHE_DIR, f"image_caption_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, cache_path, process, N_PER_SOURCE["image_caption"], desc="image_caption")


def _time_stretch_to_fixed_length(signal: np.ndarray, out_samples: int) -> np.ndarray:
    """Resamples the whole clip (not just its start) to exactly `out_samples` via linear interpolation over time"""
    if signal.shape[0] == out_samples:
        return signal
    x_old = np.linspace(0.0, 1.0, signal.shape[0])
    x_new = np.linspace(0.0, 1.0, out_samples)
    return np.interp(x_new, x_old, signal)


def build_audio_caption():
    """Audio + caption. Dataset: OpenSound/AudioCaps (real audio bytes embedded)."""
    from datasets import Audio, load_dataset

    ds = load_dataset("OpenSound/AudioCaps", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out_samples = int(AUDIO_SECONDS * AUDIO_SAMPLE_RATE)

    def process(row):
        audio, caption = row.get("audio"), row.get("caption")
        if audio is None or not audio.get("bytes") or not caption:
            return None
        try:
            arr, _ = _decode_audio_bytes(audio["bytes"], layout="mono", rate=AUDIO_SAMPLE_RATE)
        except Exception:  # noqa: BLE001, S112 — a handful of malformed/unsupported clips is expected
            return None
        original_samples = arr.shape[1]
        original_duration_sec = original_samples / AUDIO_SAMPLE_RATE
        stretched = _time_stretch_to_fixed_length(arr[0], out_samples)
        arr, peak = _peak_normalize(stretched)
        return make_row(
            "audio_caption",
            "audiocaps",
            caption=str(caption),
            audio=arr,
            stretch_factor=original_samples / out_samples,  # divide output's time axis by this to undo
            sample_rate=AUDIO_SAMPLE_RATE,
            original_duration_sec=original_duration_sec,
            peak_scale=peak,  # multiply by this / 0.95 to approximately undo the normalization
        )

    cache_path = os.path.join(CACHE_DIR, f"audio_caption_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, cache_path, process, N_PER_SOURCE["audio_caption"], desc="audio_caption")


def build_video_caption():
    """Video + caption. Dataset: HuggingFaceFV/finevideo (mp4 bytes + JSON metadata embedded, gated)."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)

    def process(row):
        raw, meta = row.get("mp4"), row.get("json") or {}
        caption = (meta.get("content_metadata") or {}).get("description") or meta.get("youtube_title")
        if raw is None or not caption:
            return None
        raw_bytes = raw if isinstance(raw, (bytes, bytearray)) else raw.get("bytes")
        try:
            decoded = _decode_video_bytes(raw_bytes)
        except Exception:  # noqa: BLE001 — a handful of malformed clips is expected
            return None
        if not decoded:
            return None
        frames, video_meta = decoded
        return make_row(
            "video_caption",
            "finevideo",
            caption=str(caption),
            video=np.stack(frames, axis=0).astype(np.uint8),
            **video_meta,
        )

    cache_path = os.path.join(CACHE_DIR, f"video_caption_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, cache_path, process, N_PER_SOURCE["video_caption"], desc="video_caption")


def _find_lidar_tar(repo_id: str) -> str:
    """Picks one lidar .tar file from the repo (10 available, one per clip)."""
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(repo_id, repo_type="dataset") if f.startswith("lidar_dataset_release/lidar/") and f.endswith(".tar")]
    if not files:
        raise RuntimeError(f"No lidar .tar files found in {repo_id}")
    return sorted(files)[0]


def _load_npz_arrays(raw: bytes) -> dict[str, np.ndarray]:
    """Loads ALL arrays out of an .npz file's bytes, keyed by name — a single component file."""
    with np.load(io.BytesIO(raw)) as npz:
        return {k: npz[k] for k in npz.files if npz[k].size > 0}


def _stack_component_columns(arrays: dict[str, np.ndarray]) -> np.ndarray | None:
    """Turns a component's {key: array} into a single (N, D) array"""
    if not arrays:
        return None
    two_d = [a for a in arrays.values() if a.ndim == 2]
    if len(two_d) == 1:
        return two_d[0].astype(np.float32)
    one_d = {k: a for k, a in arrays.items() if a.ndim == 1}
    if not one_d:
        return None
    n = min(a.shape[0] for a in one_d.values())
    cols = [a[:n] for _, a in sorted(one_d.items())]
    return np.stack(cols, axis=1).astype(np.float32)


def _group_lidar_members(members: list) -> dict[str, dict[str, "tarfile.TarInfo"]]:
    """Groups tar members by frame stem """
    groups: dict[str, dict[str, object]] = {}
    for m in members:
        if ".lidar_" not in m.name or not m.name.endswith(".npz"):
            continue
        stem, _, rest = m.name.partition(".lidar_")
        component = rest.removesuffix(".npz")
        groups.setdefault(stem, {})[component] = m
    return groups


def _merge_lidar_frame(tar, components: dict) -> np.ndarray | None:
    """Merges a frame's separate .npz components into a single array."""
    per_component = {}
    for name, member in components.items():
        f = tar.extractfile(member)
        if f is None:
            continue
        try:
            arrays = _load_npz_arrays(f.read())
            stacked = _stack_component_columns(arrays)
        except Exception:  # noqa: BLE001 — a handful of corrupt npz entries is expected
            continue
        if stacked is not None:
            per_component[name] = stacked

    if not per_component:
        return None

    geometry = next((a for k, a in per_component.items() if "col" in k or "xyz" in k or "point" in k), None)
    if geometry is None:
        geometry = max(per_component.values(), key=lambda a: a.shape[1] * a.shape[0])

    others = [a for k, a in per_component.items() if a is not geometry]
    n = geometry.shape[0]
    others = [a[:n] for a in others if a.shape[0] == n]
    points = np.concatenate([geometry] + others, axis=1) if others else geometry

    if points.shape[1] < 4:
        return None  # not enough columns to form (x, y, z, intensity) — caller logs & skips
    return points if np.isfinite(points).all() and points.shape[0] > 0 else None


def _subsample_lidar_azimuth(points: np.ndarray, n: int) -> np.ndarray:
    """Subsamples to `n` points uniformly spread across the full 360° azimuth """
    if points.shape[0] <= n:
        return points
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    order = np.argsort(azimuth)
    idx = np.linspace(0, len(order) - 1, n).astype(int)
    return points[order[idx]]


def build_lidar():
    """Lidar points. Dataset: nvidia/Cosmos-Transfer-LidarGen-Example (gated, one .tar clip).

    Caches the RAW merged frames (`_merge_lidar_frame` output, before subsampling) so edits to
    `_subsample_lidar_azimuth` / `make_row` are picked up on every build. `--force-rebuild`
    clears the cache to re-merge straight from the tar."""
    from huggingface_hub import hf_hub_download

    cache_path = os.path.join(CACHE_DIR, f"lidar_v{CACHE_SCHEMA_VERSION}.pkl")
    n = N_PER_SOURCE["lidar"]

    frames = []
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            frames = pickle.load(f)
        print(f"[lidar] loaded {len(frames)} raw frames from cache ({cache_path})")

    if len(frames) < n:
        tar_filename = LIDAR_TAR_FILENAME or _find_lidar_tar(LIDAR_REPO_ID)
        local_path = hf_hub_download(repo_id=LIDAR_REPO_ID, repo_type="dataset", filename=tar_filename)
        warned = False

        def warn_once(msg):
            nonlocal warned
            if not warned:
                print(f"[lidar] {msg}")
                warned = True

        with tarfile.open(local_path) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            groups = _group_lidar_members(members)
            if not groups:
                print(f"[lidar] no '<stem>.lidar_<component>.npz' members found; example names: {[m.name for m in members[:5]]}")
                return []

            for stem, components in tqdm(list(groups.items()), total=min(n, len(groups)), desc="lidar"):
                if len(frames) >= n:
                    break
                points = _merge_lidar_frame(tar, components)
                if points is None:
                    if not warned:
                        # dump npz internals once so a real failure is debuggable, not just "0 examples"
                        for name, member in components.items():
                            f = tar.extractfile(member)
                            try:
                                with np.load(io.BytesIO(f.read())) as npz:
                                    print(f"[lidar] debug {name}: keys={npz.files} shapes={[npz[k].shape for k in npz.files]}")
                            except Exception as e:  # noqa: BLE001
                                print(f"[lidar] debug {name}: failed to inspect ({e})")
                        warn_once(f"couldn't merge frame '{stem}' from components {list(components)}; skipping")
                    continue
                frames.append({"points": points, "components": list(components)})

        with open(cache_path, "wb") as f:
            pickle.dump(frames, f)

    out = []
    for frame in frames[:n]:
        points = frame["points"]
        out.append(
            make_row(
                "lidar",
                "cosmos-transfer-lidargen",
                points=_subsample_lidar_azimuth(points, LIDAR_POINTS).astype(np.float32),
                n_points_original=int(points.shape[0]),
                components=frame["components"],
            )
        )
    return out


def _parse_control_params(text: str) -> dict:
    """Parses PixelBytes-OptimalControl's `text` field (tiny 2-row CSV, not free text)."""
    import ast
    import csv

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
    """Control state + action. Dataset: ffurfaro/PixelBytes-OptimalControl (channel 1 = state, 0 = action)."""
    from datasets import Audio, load_dataset

    ds = load_dataset("ffurfaro/PixelBytes-OptimalControl", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    def process(row):
        audio = row.get("audio")
        if audio is None or not audio.get("bytes"):
            return None
        try:
            arr, sample_rate = _decode_audio_bytes(audio["bytes"], layout="stereo")
        except Exception:  # noqa: BLE001, S112 — skip a handful of malformed/unsupported clips
            return None
        if arr.shape[0] != 2 or arr.shape[1] < 2:
            return None
        state, state_peak = _peak_normalize(arr[1])
        action, action_peak = _peak_normalize(arr[0])
        # stereo layout: channel 0 (left) = state, channel 1 (right) = action
        return make_row(
            "control",
            "pixelbytes-optimalcontrol",
            caption=json.dumps(_parse_control_params(str(row.get("text", "")))),
            state=state,
            action=action,
            sample_rate=sample_rate,
            state_peak_scale=state_peak,
            action_peak_scale=action_peak,
        )

    cache_path = os.path.join(CACHE_DIR, f"control_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, cache_path, process, N_PER_SOURCE["control"], desc="control")


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
        help="Re-fetch RAW source rows for these sources (clears their raw cache; no names = all). "
        "Processing is always re-run anyway.",
    )
    args = parser.parse_args()

    builders = {
        "image_bbox": build_image_bbox,
        "image_caption": build_image_caption,
        "audio_caption": build_audio_caption,
        "video_caption": build_video_caption,
        "lidar": build_lidar,
        "control": build_control,
    }
    force = set(builders) if args.force_rebuild == [] else set(args.force_rebuild or [])

    os.makedirs(CACHE_DIR, exist_ok=True)
    examples = []
    for name, builder in builders.items():
        cache_path = os.path.join(CACHE_DIR, f"{name}_v{CACHE_SCHEMA_VERSION}.pkl")
        if name in force and os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"[{name}] force-rebuild: cleared raw cache ({cache_path})")
        try:
            rows = builder()
            print(f"[{name}] got {len(rows)} examples")
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