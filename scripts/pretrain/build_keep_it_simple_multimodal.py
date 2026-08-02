"""
Builds keep-it-simple-multimodal: mini multimodal dataset (image+caption, audio+caption,
video+caption, lidar, control state/action), target ~51MB.

Datasets used:
  - detection-datasets/coco            -> image_caption (bbox serialized as text)
  - laion/relaion-coco                  -> image_caption (URL download, punsafe-filtered)
  - OpenSound/AudioCaps                  -> audio_caption
  - HuggingFaceFV/finevideo (gated)      -> video_caption
  - nvidia/Cosmos-Transfer-LidarGen-Example (gated) -> lidar
  - ffurfaro/PixelBytes-OptimalControl   -> control

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
CACHE_SCHEMA_VERSION = 4  # bumped: added resumable checkpointing for streaming builders
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


def _iterate_resumable(ds, checkpoint_path: str, process_row, n: int, desc: str) -> list[dict]:
    """Streams `ds` up to `n` kept rows, checkpointing every CHECKPOINT_EVERY rows so a
    Ctrl-C/crash resumes via ds.skip(consumed) instead of restarting from scratch."""
    rows, consumed = [], 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        rows, consumed = state["rows"], state["consumed"]
        print(f"[{desc}] resuming: {len(rows)} rows kept, {consumed} source rows already consumed")
        if len(rows) >= n:
            return rows[:n]
        ds = ds.skip(consumed)

    def save_checkpoint():
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"rows": rows, "consumed": consumed}, f)

    with tqdm(total=n, initial=len(rows), desc=desc) as pbar:
        for row in ds:
            consumed += 1
            result = process_row(row)
            if result is not None:
                rows.append(result)
                pbar.update(1)
            if consumed % CHECKPOINT_EVERY == 0:
                save_checkpoint()
            if len(rows) >= n:
                break

    save_checkpoint()
    return rows


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


def _decode_video_bytes(raw_bytes: bytes) -> list | None:
    """Sample VIDEO_FRAMES evenly-spaced frames from encoded video via PyAV."""
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


def build_image_bbox():
    """Image + bbox-as-text. Dataset: detection-datasets/coco. Bbox stored as plain text
    (same modality as image_caption: it's just text describing the image)."""
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset("detection-datasets/coco", split="train", streaming=True)

    def process(row):
        img, objects = row.get("image"), row.get("objects") or {}
        bbox, category = objects.get("bbox"), objects.get("category")
        if img is None or not bbox:
            return None
        caption = "; ".join(f"cat={c} box={tuple(round(v) for v in b)}" for c, b in zip(category or [], bbox))
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        return make_row("image_caption", "detection-datasets-coco", caption=caption, image=np.array(img, dtype=np.uint8))

    checkpoint_path = os.path.join(CACHE_DIR, f"image_bbox_partial_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, checkpoint_path, process, N_PER_SOURCE["image_bbox"], desc="image_bbox")


def build_image_caption():
    """Image + caption. Dataset: laion/relaion-coco (image bytes not embedded, downloaded by URL;
    filtered on `punsafe` since LAION-scale scrapes can contain unsafe content)."""
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
            img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        except Exception:  # noqa: BLE001 — dead links/unsupported images are expected at scale
            return None
        return make_row("image_caption", "laion-relaion-coco", caption=str(caption), image=np.array(img, dtype=np.uint8))

    checkpoint_path = os.path.join(CACHE_DIR, f"image_caption_partial_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, checkpoint_path, process, N_PER_SOURCE["image_caption"], desc="image_caption")


def build_audio_caption():
    """Audio + caption. Dataset: OpenSound/AudioCaps (real audio bytes embedded)."""
    from datasets import Audio, load_dataset

    ds = load_dataset("OpenSound/AudioCaps", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    max_samples = int(AUDIO_SECONDS * AUDIO_SAMPLE_RATE)

    def process(row):
        audio, caption = row.get("audio"), row.get("caption")
        if audio is None or not audio.get("bytes") or not caption:
            return None
        try:
            arr, _ = _decode_audio_bytes(audio["bytes"], layout="mono", rate=AUDIO_SAMPLE_RATE)
        except Exception:  # noqa: BLE001, S112 — a handful of malformed/unsupported clips is expected
            return None
        arr = np.clip(arr[0, :max_samples], -1.0, 1.0)
        return make_row("audio_caption", "audiocaps", caption=str(caption), audio=arr, sample_rate=AUDIO_SAMPLE_RATE)

    checkpoint_path = os.path.join(CACHE_DIR, f"audio_caption_partial_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, checkpoint_path, process, N_PER_SOURCE["audio_caption"], desc="audio_caption")


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
            frames = _decode_video_bytes(raw_bytes)
        except Exception:  # noqa: BLE001 — a handful of malformed clips is expected
            return None
        if not frames:
            return None
        return make_row(
            "video_caption", "finevideo", caption=str(caption), video=np.stack(frames, axis=0).astype(np.uint8)
        )

    checkpoint_path = os.path.join(CACHE_DIR, f"video_caption_partial_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, checkpoint_path, process, N_PER_SOURCE["video_caption"], desc="video_caption")


def _find_lidar_tar(repo_id: str) -> str:
    """Picks one lidar .tar file from the repo (10 available, one per clip)."""
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(repo_id, repo_type="dataset") if f.startswith("lidar_dataset_release/lidar/") and f.endswith(".tar")]
    if not files:
        raise RuntimeError(f"No lidar .tar files found in {repo_id}")
    return sorted(files)[0]


def _parse_lidar_frame(raw: bytes) -> np.ndarray | None:
    """Parses one lidar sweep as flat float32 (x, y, z, [extra]); stride inferred (4 or 5)."""
    n_floats = len(raw) // 4
    for stride in (4, 5):
        if n_floats % stride == 0:
            arr = np.frombuffer(raw, dtype=np.float32, count=(n_floats // stride) * stride)
            return arr.reshape(-1, stride)[:, :4]
    return None


def build_lidar():
    """Lidar points. Dataset: nvidia/Cosmos-Transfer-LidarGen-Example (gated, one .tar clip)."""
    from huggingface_hub import hf_hub_download

    tar_filename = LIDAR_TAR_FILENAME or _find_lidar_tar(LIDAR_REPO_ID)
    local_path = hf_hub_download(repo_id=LIDAR_REPO_ID, repo_type="dataset", filename=tar_filename)

    out, warned = [], False
    n = N_PER_SOURCE["lidar"]

    def warn_once(msg):
        nonlocal warned
        if not warned:
            print(f"[lidar] {msg}")
            warned = True

    with tarfile.open(local_path) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for member in tqdm(members, total=min(n, len(members)), desc="lidar"):
            if len(out) >= n:
                break
            f = tar.extractfile(member)
            if f is None:
                continue
            raw = f.read()
            arr = _parse_lidar_frame(raw)
            if arr is None:
                warn_once(f"couldn't infer point stride for {member.name} ({len(raw)} bytes); skipping")
                continue
            if arr.shape[0] == 0:
                continue
            out.append(make_row("lidar", "cosmos-transfer-lidargen", points=arr[:LIDAR_POINTS].astype(np.float32)))
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
    """Control state + action. Dataset: ffurfaro/PixelBytes-OptimalControl (channel 0 = state, 1 = action)."""
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
        return make_row(
            "control",
            "pixelbytes-optimalcontrol",
            caption=json.dumps(_parse_control_params(str(row.get("text", "")))),
            state=np.clip(arr[0], -1.0, 1.0).astype(np.float32),
            action=np.clip(arr[1], -1.0, 1.0).astype(np.float32),
            sample_rate=sample_rate,
        )

    checkpoint_path = os.path.join(CACHE_DIR, f"control_partial_v{CACHE_SCHEMA_VERSION}.pkl")
    return _iterate_resumable(ds, checkpoint_path, process, N_PER_SOURCE["control"], desc="control")


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
