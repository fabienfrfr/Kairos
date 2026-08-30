import array
import io
import json
import random
import uuid
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import Features, Sequence, Value, concatenate_datasets, get_dataset_config_names, load_dataset
from torch.utils.data import Dataset

from .tokenizer import KairosTokenizer, Modality, MultimodalSegment

MAX_LEN = 3 * 2048

# explicit schema for _build_multimodal: stable column order, works even for an empty dataset.
_MULTIMODAL_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int64")),
        "modality_ids": Sequence(Value("int64")),
        "octet_family_ids": Sequence(Value("int64")),
        "mask": Sequence(Value("int64")),
        "prompt_len": Value("int64"),
    }
)


class NonFiniteDataError(ValueError):
    """Raised when a multimodal example contains NaN/Inf — a data-quality issue to."""


def pack_multimodal_data(arrays: dict) -> bytes:
    """Serialize named numpy arrays into one self-describing blob — shape/dtype travel with."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def unpack_multimodal_data(data: bytes) -> dict:
    """Inverse of pack_multimodal_data."""
    with np.load(io.BytesIO(data)) as npz:
        return {k: npz[k] for k in npz.files}


_KNOWN_MULTIMODAL_MODALITIES = frozenset({"image_caption", "audio_caption", "video_caption", "lidar", "imu", "control"})


def _segments_for(ex, modality_scale_factors: dict | None = None) -> list[MultimodalSegment]:
    """Dispatch by `modality` (see build_keep_it_simple_multimodal.py)."""
    modality_scale_factors = modality_scale_factors or {}
    modality = ex["modality"]

    if modality == "text":
        return [MultimodalSegment(Modality.TEXT, ex["text"].encode("utf-8"))]

    if modality not in _KNOWN_MULTIMODAL_MODALITIES:
        raise ValueError(f"unknown example modality: {modality!r}")

    arrays = unpack_multimodal_data(ex["data"])
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise NonFiniteDataError(f"non-finite values in {modality!r} field {name!r}")
    meta = json.loads(ex["meta"]) if ex.get("meta") else {}
    caption = ex.get("caption") or ""

    if modality == "image_caption":
        sf = modality_scale_factors.get("image_caption")
        img_markers = KairosTokenizer.encode_image(arrays["image"], scale_factor=sf)
        return [
            MultimodalSegment(Modality.TEXT, caption.encode("utf-8")),
            MultimodalSegment(Modality.IMAGE, img_markers),
        ]

    if modality == "audio_caption":
        sr = meta.get("sample_rate", KairosTokenizer.AUDIO_SAMPLE_RATE)
        sf = modality_scale_factors.get("audio_caption")
        audio_markers = KairosTokenizer.encode_audio(arrays["audio"], tick_samples=sr, scale_factor=sf)
        return [
            MultimodalSegment(Modality.TEXT, caption.encode("utf-8")),
            MultimodalSegment(Modality.AUDIO, audio_markers),
        ]

    if modality == "video_caption":
        sf = modality_scale_factors.get("video_caption")
        video_markers = KairosTokenizer.encode_video(arrays["video"], scale_factor=sf)
        return [
            MultimodalSegment(Modality.TEXT, caption.encode("utf-8")),
            MultimodalSegment(Modality.VIDEO, video_markers),
        ]

    if modality == "lidar":
        sf = modality_scale_factors.get("lidar")
        return [MultimodalSegment(Modality.LIDAR, KairosTokenizer.encode_lidar(arrays["points"], scale_factor=sf))]

    if modality == "imu":
        # observation-only CONTROL (no action): folded into the same fused modality, <OBS>-tagged.
        sf = modality_scale_factors.get("imu")
        flat = np.clip(arrays["signal"].flatten(), -1.0, 1.0).astype(np.float32)
        return [MultimodalSegment(Modality.CONTROL, KairosTokenizer.encode_control(flat, action=None, scale_factor=sf))]

    if modality == "control":
        sf = modality_scale_factors.get("control")
        if len(arrays["state"]) != len(arrays["action"]):
            warnings.warn(
                f"control example: state ({len(arrays['state'])} samples) vs action "
                f"({len(arrays['action'])} samples) length mismatch - check arrays['state']/['action']",
                stacklevel=2,
            )
        segments = []
        if caption:
            segments.append(MultimodalSegment(Modality.TEXT, caption.encode("utf-8")))
        # single CONTROL segment via encode_control: one tag pair per clip, not per sample
        segments.append(
            MultimodalSegment(
                Modality.CONTROL, KairosTokenizer.encode_control(arrays["state"], arrays["action"], scale_factor=sf)
            )
        )
        return segments
    return None


def _pad_and_gen_mask(ids, prompt_len, max_len, pad_token_id):
    """Pad `ids` to max_len and build a gen_mask that's 0 on the."""
    pad_len = max_len - len(ids)
    gen_len = len(ids) - prompt_len
    ids = ids + [pad_token_id] * pad_len
    gen_mask = [0] * prompt_len + [1] * gen_len + [0] * pad_len
    return ids, gen_mask


@dataclass
class ModalityDataStats:
    """Raw-count and tokenized-length stats for one modality key, sampled from a corpus."""

    modality: str
    total_examples: int
    sampled: int
    tokens_mean: float
    tokens_min: int
    tokens_max: int
    chunks_mean: float
    chunks_total_estimate: int


@dataclass
class DataDiagnosticReport:
    """Per-modality raw-vs-tokenized breakdown; see diagnose_multimodal_examples."""

    rows: list[ModalityDataStats]
    max_len: int
    total_examples: int

    def __str__(self) -> str:
        lines = [
            "Kairos data diagnostic (raw examples -> tokenized chunk estimate)",
            "----------------------------------------------------------------",
            f"Total examples:  {self.total_examples}",
            f"max_len (chunk): {self.max_len}",
            "",
            f"{'modality':<16} {'count':>8} {'sampled':>8} {'tok mean':>10} {'tok min':>8} {'tok max':>8} {'chunks/ex':>10} {'chunks (est)':>13}",
        ]
        for r in sorted(self.rows, key=lambda r: -r.chunks_total_estimate):
            lines.append(
                f"{r.modality:<16} {r.total_examples:>8} {r.sampled:>8} {r.tokens_mean:>10.0f} "
                f"{r.tokens_min:>8} {r.tokens_max:>8} {r.chunks_mean:>10.2f} {r.chunks_total_estimate:>13}"
            )
        lines.append("")
        lines.append(f"Total estimated chunks (all modalities): {sum(r.chunks_total_estimate for r in self.rows)}")
        return "\n".join(lines)


def diagnose_multimodal_examples(
    examples, tokenizer=None, modality_scale_factors=None, max_len=1024, sample_size=200, seed=0
) -> DataDiagnosticReport:
    """Per-modality raw example count + tokenized-length stats, sampled for speed on big corpora."""
    tokenizer = tokenizer or KairosTokenizer()
    modality_scale_factors = modality_scale_factors or {}
    rng = random.Random(seed)

    by_modality: dict[str, list] = {}
    for ex in examples:
        by_modality.setdefault(ex["modality"], []).append(ex)

    rows = []
    for modality, exs in by_modality.items():
        sample = exs if len(exs) <= sample_size else rng.sample(exs, sample_size)
        lengths = []
        for ex in sample:
            segments = _segments_for(ex, modality_scale_factors)
            encoded = tokenizer.encode_multimodal(segments)
            lengths.append(len(encoded["input_ids"]))
        mean_tokens = sum(lengths) / len(lengths) if lengths else 0.0
        mean_chunks = mean_tokens / max_len if max_len else 0.0
        rows.append(
            ModalityDataStats(
                modality=modality,
                total_examples=len(exs),
                sampled=len(sample),
                tokens_mean=mean_tokens,
                tokens_min=min(lengths) if lengths else 0,
                tokens_max=max(lengths) if lengths else 0,
                chunks_mean=mean_chunks,
                chunks_total_estimate=round(mean_chunks * len(exs)),
            )
        )
    return DataDiagnosticReport(rows=rows, max_len=max_len, total_examples=len(examples))


@dataclass
class BuiltDatasetReport:
    """Token-level per-modality composition of an already-tokenized (built) dataset."""

    modality_tokens: dict[str, int]  # modality name -> token count in the sample
    sampled_rows: int
    total_rows: int

    def __str__(self) -> str:
        total = sum(self.modality_tokens.values()) or 1
        lines = [
            "Kairos built-dataset diagnostic (token-level, sampled from already-tokenized rows)",
            "-------------------------------------------------------------------------------",
            f"Total rows:  {self.total_rows}  (sampled: {self.sampled_rows})",
            "",
            f"{'modality':<10} {'tokens':>12} {'%':>6}",
        ]
        for name, count in sorted(self.modality_tokens.items(), key=lambda kv: -kv[1]):
            lines.append(f"{name:<10} {count:>12} {100 * count / total:>5.1f}%")
        return "\n".join(lines)


def diagnose_built_dataset(built_dataset, sample_size: int = 2000, seed: int = 0) -> BuiltDatasetReport:
    """Token-level modality composition of a built KairosPretrainingDataset (post pipe.build())."""
    total_rows = len(built_dataset)
    sample_n = min(sample_size, total_rows)
    idx = random.Random(seed).sample(range(total_rows), sample_n) if total_rows else []
    plain = built_dataset.with_format(None)
    batch = plain[idx] if idx else {"modality_ids": []}

    id_to_name = {m.value: m.name for m in Modality}
    counts: dict[str, int] = {}
    for mod_ids in batch["modality_ids"]:
        for m in mod_ids:
            name = id_to_name.get(m, str(m))
            counts[name] = counts.get(name, 0) + 1
    return BuiltDatasetReport(modality_tokens=counts, sampled_rows=sample_n, total_rows=total_rows)


def modality_counts(examples) -> dict[str, int]:
    """Raw example count per modality, no tokenization — cheap, safe to run on a full corpus."""
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex["modality"]] = counts.get(ex["modality"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def split_examples(
    examples, eval_pct: float = 10, seed: int = 0, contiguous_modalities: frozenset = frozenset({"control"})
) -> tuple[list, list]:
    """Shuffles then splits into (train, eval); contiguous_modalities keep relative order as one block."""
    if not contiguous_modalities:
        shuffled = list(examples)
        random.Random(seed).shuffle(shuffled)
    else:
        runs, current, current_mod = [], [], None
        for ex in examples:
            mod = ex.get("modality")
            if current and mod in contiguous_modalities and mod == current_mod:
                current.append(ex)
            else:
                if current:
                    runs.append(current)
                # non-grouped modalities get a unique sentinel so each stays its own run
                current, current_mod = [ex], mod if mod in contiguous_modalities else object()
        if current:
            runs.append(current)
        random.Random(seed).shuffle(runs)
        shuffled = [ex for run in runs for ex in run]
    n_eval = int(len(shuffled) * eval_pct / 100)
    return shuffled[n_eval:], shuffled[:n_eval]


@dataclass
class RawControlBalanceReport:
    """Sums raw state/action array lengths across "control" examples, before tokenization/windowing."""

    n_control_examples: int
    total_state_samples: int
    total_action_samples: int
    mismatched_examples: list[dict]  # [{"index": i, "state_samples": ..., "action_samples": ...}]

    def __str__(self) -> str:
        lines = [
            "Kairos raw control state/action balance (pre-tokenization)",
            "-------------------------------------------------------------",
            f"control examples: {self.n_control_examples}",
            f"total state samples: {self.total_state_samples}  total action samples: {self.total_action_samples}",
            f"examples with a raw state/action length mismatch: {len(self.mismatched_examples)}",
        ]
        if self.mismatched_examples:
            lines.append("First few mismatches:")
            for m in self.mismatched_examples[:5]:
                lines.append(f"  example {m['index']}: state={m['state_samples']} action={m['action_samples']}")
        return "\n".join(lines)


def diagnose_raw_control_balance(examples) -> RawControlBalanceReport:
    """Checks every "control" example's raw state/action arrays for equal length, before tokenization."""
    n = 0
    total_state = total_action = 0
    mismatched: list[dict] = []
    for i, ex in enumerate(examples):
        if ex.get("modality") != "control":
            continue
        arrays = unpack_multimodal_data(ex["data"])
        state_n, action_n = len(arrays["state"]), len(arrays["action"])
        n += 1
        total_state += state_n
        total_action += action_n
        if state_n != action_n:
            mismatched.append({"index": i, "state_samples": state_n, "action_samples": action_n})
    return RawControlBalanceReport(
        n_control_examples=n,
        total_state_samples=total_state,
        total_action_samples=total_action,
        mismatched_examples=mismatched,
    )


def preview_multimodal_examples(examples, n: int = 3, seed: int = 0) -> None:
    """Prints caption/meta and plots a small sample of each multimodal modality (matplotlib)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    by_modality: dict[str, list] = {}
    for ex in examples:
        if ex["modality"] != "text":
            by_modality.setdefault(ex["modality"], []).append(ex)

    for modality, rows in by_modality.items():
        print(f"--- {modality} ({len(rows)} examples) ---")
        sample_idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
        for i in sample_idx:
            row = rows[i]
            arrays = unpack_multimodal_data(row["data"])
            caption = (row.get("caption") or "")[:80]
            meta = json.loads(row["meta"]) if row.get("meta") else {}
            print(f"  caption: {caption!r}  meta: {meta}")
            _plot_multimodal_row(plt, modality, arrays)


def _plot_multimodal_row(plt, modality: str, arrays: dict) -> None:
    """One matplotlib panel for a single preview_multimodal_examples row."""
    if modality == "image_caption":
        plt.figure(figsize=(2, 2))
        plt.imshow(arrays["image"])
        plt.axis("off")
        plt.show()
    elif modality == "audio_caption":
        plt.figure(figsize=(3, 1))
        plt.plot(arrays["audio"], linewidth=0.5)
        plt.axis("off")
        plt.show()
    elif modality == "video_caption":
        video = arrays["video"]
        n_frames = min(4, video.shape[0])
        _fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 1.2, 1.2))
        for j, ax in enumerate(axes if n_frames > 1 else [axes]):
            ax.imshow(video[j])
            ax.axis("off")
        plt.show()
    elif modality == "lidar":
        points = np.asarray(arrays["points"], dtype=np.float32)
        fig = plt.figure(figsize=(2.5, 2.5))
        ax = fig.add_subplot(projection="3d") if points.shape[1] >= 3 else fig.add_subplot()
        if points.shape[1] >= 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        else:
            ax.scatter(points[:, 0], points[:, 1], s=1)
        plt.show()
    elif modality == "control":
        state_n, action_n = len(arrays["state"]), len(arrays["action"])
        plt.figure(figsize=(3, 1.2))
        plt.plot(arrays["state"], label=f"state ({state_n})")
        plt.plot(arrays["action"], label=f"action ({action_n})")
        plt.legend(fontsize=6)
        title = f"state={state_n} action={action_n}"
        if state_n != action_n:
            title += "  /!\\ LENGTH MISMATCH"
        plt.title(title, fontsize=6, color="red" if state_n != action_n else "black")
        plt.show()


# modality kinds preview_tokenized_examples covers when modality=None (see find_rows_with_modality).
_ALL_PREVIEW_MODALITIES = ("text", "image", "audio", "video", "lidar", "control")


def preview_tokenized_examples(
    tokenizer, built_dataset, n: int = 1, modality: str | None = None, sample_size: int = 200, seed: int = 0
) -> None:
    """Tokenized preview_multimodal_examples; modality=None: n rows/modality; gap<=2=truncation."""
    import matplotlib.pyplot as plt

    if modality is not None:
        rows = find_rows_with_modality(built_dataset, modality, n=n, sample_size=sample_size, seed=seed)
        if not rows:
            print(f"no rows with {modality!r} content found in this sample")
            return
    else:
        rows, seen = [], set()
        for kind in _ALL_PREVIEW_MODALITIES:
            for row_i in find_rows_with_modality(built_dataset, kind, n=n, sample_size=sample_size, seed=seed):
                if row_i not in seen:
                    seen.add(row_i)
                    rows.append(row_i)
        if not rows:
            print("no rows found in this sample")
            return
    plain = built_dataset.with_format(None)

    for row_i in rows:
        input_ids = plain[row_i]["input_ids"]
        segments = tokenizer.reconstruct_segments(input_ids)
        print(f"--- row {row_i}: " + " -> ".join(f"{s['modality']}({s['n_tokens']})" for s in segments) + " ---")

        # CONTROL is paired (state+action) or observation-only (<OBS>); keep the two totals apart.
        paired_state, paired_action, obs_state, undecodable = [], [], [], []
        for i, seg in enumerate(segments):
            if seg["modality"] != "CONTROL":
                _render_tokenized_segment(seg, i)
                continue
            if seg.get("error"):
                undecodable.append(f"  [truncated] CONTROL: undecodable tail ({seg['error']})")
                continue
            if seg["decoded"]["action"] is None:
                obs_state.append(seg["decoded"]["state"])
            else:
                paired_state.append(seg["decoded"]["state"])
                paired_action.append(seg["decoded"]["action"])

        if paired_state or paired_action:
            state_arr = np.concatenate(paired_state) if paired_state else np.array([], dtype=np.float32)
            action_arr = np.concatenate(paired_action) if paired_action else np.array([], dtype=np.float32)
            state_n, action_n = len(state_arr), len(action_arr)
            plt.figure(figsize=(3, 1.2))
            plt.plot(state_arr, label=f"state ({state_n})")
            plt.plot(action_arr, label=f"action ({action_n})")
            plt.legend(fontsize=6)
            gap = abs(state_n - action_n)
            title = f"row {row_i}: state={state_n} action={action_n}"
            # a packed row has 2 truncation edges max (start+end cut), each +/-1: gap<=2 is legit.
            if gap > 2:
                title += "  /!\\ LENGTH MISMATCH"
            elif gap > 0:
                title += "  (window-edge truncation)"
            plt.title(title, fontsize=6, color="red" if gap > 2 else "black")
            plt.show()
        if obs_state:
            obs_arr = np.concatenate(obs_state)
            plt.figure(figsize=(3, 1.2))
            plt.plot(obs_arr)
            plt.title(f"row {row_i}: CONTROL observation-only ({len(obs_arr)} samples)", fontsize=6)
            plt.show()
        for msg in undecodable:
            print(msg)


def _render_tokenized_segment(seg: dict, i: int) -> None:
    """Shared per-segment renderer for TEXT/IMAGE/VIDEO/AUDIO/LIDAR, used by both plot helpers."""
    import matplotlib.pyplot as plt

    modality, decoded = seg["modality"], seg["decoded"]
    if seg.get("error"):
        print(f"  [{i}] {modality}: decode failed ({seg['error']})")
    elif modality == "TEXT":
        print(f"  [{i}] TEXT: {decoded[:120]!r}")
    elif modality == "IMAGE":
        plt.figure(figsize=(1.5, 1.5))
        plt.title(f"[{i}] IMAGE", fontsize=6)
        plt.imshow(decoded)
        plt.axis("off")
        plt.show()
    elif modality == "VIDEO":
        n_frames = min(4, decoded.shape[0])
        _fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 1.0, 1.0))
        for j, ax in enumerate(axes if n_frames > 1 else [axes]):
            ax.imshow(decoded[j])
            ax.axis("off")
        _fig.suptitle(f"[{i}] VIDEO", fontsize=6)
        plt.show()
    elif modality == "AUDIO":
        plt.figure(figsize=(3, 1))
        plt.title(f"[{i}] AUDIO ({seg.get('duration_s', 0):.3f}s)", fontsize=6)
        plt.plot(decoded, linewidth=0.5)
        plt.axis("off")
        plt.show()
    elif modality == "LIDAR":
        points = decoded
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(projection="3d") if points.shape[1] >= 3 else fig.add_subplot()
        if points.shape[1] >= 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        else:
            ax.scatter(points[:, 0], points[:, 1], s=1)
        ax.set_title(f"[{i}] LIDAR", fontsize=6)
        plt.show()


def plot_tokenized_row(tokenizer, input_ids, max_segments: int | None = None) -> None:
    """Decodes a real tokenized row and plots each segment in order; spots a truncated segment."""
    import matplotlib.pyplot as plt

    segments = tokenizer.reconstruct_segments(input_ids)
    if max_segments is not None:
        segments = segments[:max_segments]
    print(" -> ".join(f"{s['modality']}({s['n_tokens']})" for s in segments))

    for i, seg in enumerate(segments):
        modality, decoded = seg["modality"], seg["decoded"]
        if modality == "CONTROL" and not seg.get("error"):
            plt.figure(figsize=(3, 1))
            if decoded["action"] is None:
                plt.plot(decoded["state"], linewidth=0.7)
                plt.title(f"[{i}] CONTROL observation-only ({len(decoded['state'])} samples)", fontsize=6)
            else:
                plt.plot(decoded["state"], linewidth=0.7, label=f"state ({len(decoded['state'])})")
                plt.plot(decoded["action"], linewidth=0.7, label=f"action ({len(decoded['action'])})")
                plt.legend(fontsize=6)
                plt.title(f"[{i}] CONTROL", fontsize=6)
            plt.axis("off")
            plt.show()
        else:
            _render_tokenized_segment(seg, i)


@dataclass
class ControlAlternationReport:
    """Per-row CONTROL stats: rows with control content, total decoded samples, and rows with a segment truncated mid-byte by a window cut (state/action counts are always equal for any segment that decodes at all - encode_control enforces it)."""

    n_rows_with_control: int
    n_rows_sampled: int
    total_control_samples: int
    mismatched_rows: list[dict]  # rows with a CONTROL segment that failed to decode (window-truncated)

    def __str__(self) -> str:
        lines = [
            "Kairos control diagnostic",
            "--------------------------",
            f"Rows sampled: {self.n_rows_sampled}  (with control content: {self.n_rows_with_control})",
            f"Total decoded control samples: {self.total_control_samples}",
            f"Rows with a truncated CONTROL segment: {len(self.mismatched_rows)}",
        ]
        if self.mismatched_rows:
            lines.append("First few:")
            for m in self.mismatched_rows[:5]:
                lines.append(f"  row {m['row']}: {m['error']}")
        return "\n".join(lines)


def find_rows_with_modality(
    built_dataset, modality: str, n: int = 3, sample_size: int = 200, seed: int = 0
) -> list[int]:
    """Row indices with modality_ids containing `modality` ("state"/"action" both mean CONTROL)."""
    key = modality.lower()
    if key in ("control", "action", "state", "imu"):
        wanted = {int(Modality.CONTROL)}
    else:
        try:
            wanted = {int(Modality[key.upper()])}
        except KeyError:
            valid = ", ".join(m.name.lower() for m in Modality) + ", control"
            raise ValueError(f"unknown modality {modality!r}; expected one of: {valid}") from None

    total_rows = len(built_dataset)
    sample_n = min(sample_size, total_rows)
    idx = random.Random(seed).sample(range(total_rows), sample_n) if total_rows else []
    plain = built_dataset.with_format(None)
    batch = plain[idx] if idx else {"modality_ids": []}

    found = []
    for row_i, mods in zip(idx, batch["modality_ids"]):
        if any(m in wanted for m in mods):
            found.append(row_i)
            if len(found) >= n:
                break
    return found


def diagnose_control_alternation(
    built_dataset, tokenizer, sample_size: int = 200, seed: int = 0
) -> ControlAlternationReport:
    """Decodes each sampled row's CONTROL segments and counts samples, surfacing any segment truncated mid-byte by a window cut."""
    total_rows = len(built_dataset)
    sample_n = min(sample_size, total_rows)
    idx = random.Random(seed).sample(range(total_rows), sample_n) if total_rows else []
    plain = built_dataset.with_format(None)

    n_with_control = 0
    total_samples = 0
    mismatched: list[dict] = []

    for row_i in idx:
        segments = tokenizer.reconstruct_segments(plain[row_i]["input_ids"])
        control_segs = [s for s in segments if s["modality"] == "CONTROL"]
        if not control_segs:
            continue
        n_with_control += 1
        for seg in control_segs:
            if seg.get("error"):
                mismatched.append({"row": row_i, "error": seg["error"]})
            else:
                total_samples += len(seg["decoded"]["state"])

    return ControlAlternationReport(
        n_rows_with_control=n_with_control,
        n_rows_sampled=sample_n,
        total_control_samples=total_samples,
        mismatched_rows=mismatched,
    )


class KairosPretrainingDataset(Dataset):
    """Full diffusion pretraining dataset: text or multimodal, chunked to token ids."""

    def __init__(
        self,
        texts=None,
        tokenizer=None,
        max_len=MAX_LEN,
        stride=3,
        multimodal_examples=None,
        multimodal_path=None,
        pack=False,
        modality_scale_factors=None,
    ):
        self.tokenizer = tokenizer
        self.stride = stride
        self.target_len = max_len
        self.max_len = (max_len // stride) * stride
        self.pack = pack
        # per-modality-key override for encode_*'s scale_factor; unset keys use tokenizer defaults
        self.modality_scale_factors = modality_scale_factors or {}

        if multimodal_examples is not None or multimodal_path is not None:
            if multimodal_examples is None:
                multimodal_examples = torch.load(multimodal_path, weights_only=False)
            self._build_multimodal(multimodal_examples)
            return

        if texts is None:
            configs = get_dataset_config_names("HuggingFaceTB/cosmopedia")
            parts = [load_dataset("HuggingFaceTB/cosmopedia", c, split="train[:98.00%]") for c in configs]
            self.ds = concatenate_datasets(parts)
        else:
            self.ds = HFDataset.from_dict({"text": texts})

        self.ds = self.ds.map(self.preprocess, batched=True, remove_columns=self.ds.column_names)
        self.ds.set_format("torch")

    def _chunk(self, token_ids, modality_ids, family_ids):
        """Fixed-length windowing for text/multimodal, padded to target_len; returns plain lists."""
        for i in range(0, len(token_ids), self.max_len):
            ids_chunk = list(token_ids[i : i + self.max_len])
            mod_chunk = list(modality_ids[i : i + self.max_len])
            fam_chunk = list(family_ids[i : i + self.max_len])
            pad_len = self.target_len - len(ids_chunk)
            ids_chunk += [self.tokenizer.pad_token_id] * pad_len
            mod_chunk += [int(Modality.TEXT)] * pad_len
            fam_chunk += [0] * pad_len
            mask = [1] * (len(ids_chunk) - pad_len) + [0] * pad_len
            yield ids_chunk, mod_chunk, fam_chunk, mask

    def _collect_chunks(self, chunk_sources):
        """Runs each (ids, modality_ids, family_ids) through _chunk, flattened; yields lazily."""
        if self.pack:
            # "H"/"B" (uint16/uint8) are far cheaper per element than a plain Python-list int.
            packed_ids = array.array("H")
            packed_mods = array.array("B")
            packed_fams = array.array("B")
            for ids, mods, fams in chunk_sources:
                packed_ids.extend(ids)
                packed_mods.extend(mods)
                packed_fams.extend(fams)
            chunk_sources = [(packed_ids, packed_mods, packed_fams)]
        for ids, mods, fams in chunk_sources:
            yield from self._chunk(ids, mods, fams)

    def preprocess(self, examples):
        prompts = examples.get("prompt", [""] * len(examples["text"]))
        texts = examples.get("text", [""] * len(examples["text"]))

        def sources():
            for prompt, text in zip(prompts, texts):
                # anti-Reversal Curse: randomize prompt/text order
                merged = " ".join([prompt, text] if random.random() < 0.5 else [text, prompt]).strip()
                if not merged:
                    continue  # empty example: nothing to learn from
                tokens = self.tokenizer.encode(merged, add_special_tokens=False)
                if not tokens:
                    continue
                yield tokens, [int(Modality.TEXT)] * len(tokens), [0] * len(tokens)

        # one batch's worth at a time (inside ds.map(batched=True)); arrow chunks across batches.
        all_input_ids, all_modality_ids, all_family_ids, all_masks = [], [], [], []
        for ids_chunk, mod_chunk, fam_chunk, mask in self._collect_chunks(sources()):
            all_input_ids.append(ids_chunk)
            all_modality_ids.append(mod_chunk)
            all_family_ids.append(fam_chunk)
            all_masks.append(mask)
        return {
            "input_ids": all_input_ids,
            "modality_ids": all_modality_ids,
            "octet_family_ids": all_family_ids,
            "mask": all_masks,
            "prompt_len": [0] * len(all_input_ids),
        }

    def _segments_for(self, ex):
        return _segments_for(ex, self.modality_scale_factors)

    def _build_multimodal(self, examples):
        if self.tokenizer is None:
            self.tokenizer = KairosTokenizer()

        skip_messages = []  # populated during iteration, warned about after (see below)

        def sources():
            for ex in examples:
                try:
                    segments = self._segments_for(ex)
                except NonFiniteDataError as e:
                    skip_messages.append(str(e))
                    continue
                encoded = self.tokenizer.encode_multimodal(segments)
                yield (
                    encoded["input_ids"].tolist(),
                    encoded["modality_ids"].tolist(),
                    encoded["octet_family_ids"].tolist(),
                )

        def rows():
            # one dict per row, streamed straight to the arrow writer; nothing sits in RAM at once.
            for ids_chunk, mod_chunk, fam_chunk, mask in self._collect_chunks(sources()):
                yield {
                    "input_ids": ids_chunk,
                    "modality_ids": mod_chunk,
                    "octet_family_ids": fam_chunk,
                    "mask": mask,
                    "prompt_len": 0,
                }

        # writer_batch_size caps rows buffered before an arrow write; turn down if RAM is tight.
        # An explicit schema (features=) lets from_generator succeed even with zero rows.
        try:
            # fresh fingerprint each call (avoid stale cache reuse); never keep_in_memory=True.
            self.ds = HFDataset.from_generator(
                rows, features=_MULTIMODAL_FEATURES, writer_batch_size=256, fingerprint=uuid.uuid4().hex
            )
        except ValueError as e:
            # can still fail on a truly empty stream; build the empty dataset directly in that case.
            if "corresponds to no data" in str(e):
                self.ds = HFDataset.from_dict(
                    {"input_ids": [], "modality_ids": [], "octet_family_ids": [], "mask": [], "prompt_len": []},
                    features=_MULTIMODAL_FEATURES,
                )
            else:
                raise
        except Exception as e:
            # surface the original exception (e.g. ValueError), not DatasetGenerationError's wrapper.
            cause = e.__cause__ or e.__context__
            if isinstance(cause, ValueError):
                raise cause from None
            raise
        self.ds.set_format("torch")

        # warn here, not in sources(): warnings from inside the consumed generator may not propagate.
        for i, msg in enumerate(skip_messages, 1):
            warnings.warn(f"skipping corrupt example ({msg}); {i} skipped so far", stacklevel=2)

    def __getitem__(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)


class KairosSFTDataset(Dataset):
    """SFT dataset: flattens a conversation to tags and diffuses only the last."""

    def __init__(self, tokenizer, max_len=512, examples=None, source="toolace"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._build(examples, source)

    def _build(self, examples, source):
        if examples is not None:
            return [self._process(ex) for ex in examples]
        if source == "toolace":
            ds = load_dataset("Team-ACE/ToolACE", split="train")
            return [self._process(ex) for ex in ds]
        if source == "alpaca":
            ds = load_dataset("yahma/alpaca-cleaned", split="train")
            return [self._process_alpaca(ex) for ex in ds]
        raise ValueError(f"Unknown source: {source}")

    def _process(self, ex):
        system = ex.get("system", "")
        turns = ex.get("conversations", [])
        parts = [f"<system>\n{system}\n</system>\n"] if system else []
        last_assistant_start = None

        for turn in turns:
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", ""))
            if role in ("user", "human"):
                parts.append(f"<user>\n{value}\n</user>\n")
            elif role in ("assistant", "gpt"):
                prefix_ids = self.tokenizer.encode("".join(parts), add_special_tokens=False)
                last_assistant_start = len(prefix_ids)
                parts.append(f"<assistant>\n{value}\n</assistant>\n")
            elif role == "tool":
                parts.append(f"<tool_result>\n{value}\n</tool_result>\n")

        all_ids = self.tokenizer.encode("".join(parts), add_special_tokens=False)
        prompt_len = last_assistant_start if last_assistant_start is not None else len(all_ids)

        all_ids = all_ids[: self.max_len]
        prompt_len = min(prompt_len, len(all_ids))
        all_ids, gen_mask = _pad_and_gen_mask(all_ids, prompt_len, self.max_len, self.tokenizer.pad_token_id)

        return {"input_ids": all_ids, "gen_mask": gen_mask, "prompt_len": prompt_len}

    def _process_alpaca(self, ex):
        user = ex["instruction"]
        if ex.get("input", "").strip():
            user += f"\n\n{ex['input']}"
        conversations = [
            {"from": "user", "value": user},
            {"from": "assistant", "value": ex["output"]},
        ]
        return self._process({"system": "", "conversations": conversations})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "gen_mask": torch.tensor(s["gen_mask"], dtype=torch.long),
            "prompt_len": torch.tensor(s["prompt_len"], dtype=torch.long),
        }


class KairosDPODataset(Dataset):
    """DPO dataset: fixed prompt plus separately-tokenized chosen/rejected responses."""

    def __init__(self, tokenizer, max_len=512, examples=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._build(examples)

    def _build(self, examples):
        if examples is not None:
            return [self._process(ex) for ex in examples]
        ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
        return [self._process(ex) for ex in ds]

    def _render_messages(self, messages):
        return "".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>\n" for m in messages)

    def _encode_pair(self, prompt_text, response_text):
        # byte-level tokenizer: encode(A) + encode(B) == encode(A + B)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        response_ids = response_ids[: self.max_len - len(prompt_ids)]
        prompt_ids = prompt_ids[: self.max_len]
        all_ids = prompt_ids + response_ids
        all_ids, gen_mask = _pad_and_gen_mask(all_ids, len(prompt_ids), self.max_len, self.tokenizer.pad_token_id)
        return all_ids, gen_mask, len(prompt_ids)

    def _process(self, ex):
        prompt_text = f"<user>\n{ex['prompt']}\n</user>\n<assistant>\n"
        chosen_text = self._render_messages(ex.get("chosen", [])[-1:])
        rejected_text = self._render_messages(ex.get("rejected", [])[-1:])
        chosen_ids, chosen_mask, plen = self._encode_pair(prompt_text, chosen_text)
        rejected_ids, rejected_mask, _ = self._encode_pair(prompt_text, rejected_text)
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
            "prompt_len": plen,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "chosen_ids": torch.tensor(s["chosen_ids"], dtype=torch.long),
            "chosen_mask": torch.tensor(s["chosen_mask"], dtype=torch.long),
            "rejected_ids": torch.tensor(s["rejected_ids"], dtype=torch.long),
            "rejected_mask": torch.tensor(s["rejected_mask"], dtype=torch.long),
            "prompt_len": torch.tensor(s["prompt_len"], dtype=torch.long),
        }


class KairosRLDataset(Dataset):
    """RL dataset for reasoning via masked diffusion: prompt tokens are never noised."""

    def __init__(self, tokenizer, max_len=2048, split="train", max_samples=None, examples=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._build(examples, split, max_samples)

    def _build(self, examples, split, max_samples):
        if examples is not None:
            return [self._process(ex) for ex in examples]
        configs = get_dataset_config_names("ffurfaro/bigbench")
        raw, count = [], 0
        for name in configs:
            for ex in load_dataset("ffurfaro/bigbench", name, split=split, streaming=True):
                if "multiple_choice_targets" not in ex:
                    continue
                raw.append(self._process(ex))
                count += 1
                if max_samples and count >= max_samples:
                    return raw
        return raw

    def _process(self, ex):
        question = ex["inputs"]
        choices = list(ex["multiple_choice_targets"])
        scores = list(ex.get("multiple_choice_scores", [0] * len(choices)))
        reasoning = ex.get("reasoning", "")

        choices.append("not sure / I don't know")
        scores.append(0.1)

        paired = list(zip(choices, scores))
        random.shuffle(paired)  # anti-position bias
        choices, scores = zip(*paired)
        best = choices[int(torch.tensor(list(scores)[:-1]).argmax())]

        level = random.choice(["low", "medium", "flex"])
        mask_ratio = {"low": 0.25, "medium": 0.5, "flex": random.uniform(0.1, 0.9)}[level]

        choice_lines = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))
        prompt = f"<inputs>\n{question}\n<choices>\n{choice_lines}\n"

        gen_blocks = [f"<reasoning={level}>\n{reasoning}", f"<answer>\n{best}"]
        random.shuffle(gen_blocks)  # anti-Reversal Curse
        generation = "\n".join(gen_blocks) + "\n"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        gen_ids = self.tokenizer.encode(generation, add_special_tokens=False)
        gen_ids = gen_ids[: self.max_len - len(prompt_ids)]

        ids = prompt_ids + gen_ids
        ids, gen_mask = _pad_and_gen_mask(ids, len(prompt_ids), self.max_len, self.tokenizer.pad_token_id)

        return {
            "input_ids": ids,
            "gen_mask": gen_mask,
            "prompt_len": len(prompt_ids),
            "mask_ratio": mask_ratio,
            "choices": list(choices),
            "scores": list(scores),
            "level": level,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "gen_mask": torch.tensor(s["gen_mask"], dtype=torch.long),
            "prompt_len": torch.tensor(s["prompt_len"], dtype=torch.long),
            "mask_ratio": torch.tensor(s["mask_ratio"], dtype=torch.float),
            "choices": s["choices"],
            "scores": s["scores"],
            "level": s["level"],
        }
