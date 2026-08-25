"""Byte-level multimodal tokenizer built on ByT5Tokenizer with periodic LOCAL markers."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import torch
from transformers.models.byt5.tokenization_byt5 import ByT5Tokenizer


class Modality(enum.IntEnum):
    """Must match KairosConfig.text_modality_id / modality_scales ordering."""

    TEXT = 0
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3
    LIDAR = 4
    STATE = 5
    ACTION = 6
    META = 7


_MODALITY_TAGS: dict[Modality, tuple[str, str]] = {
    Modality.TEXT: ("<TEXT>", "</TEXT>"),
    Modality.IMAGE: ("<IMG>", "</IMG>"),
    Modality.VIDEO: ("<VIDEO>", "</VIDEO>"),
    Modality.AUDIO: ("<AUDIO>", "</AUDIO>"),
    Modality.LIDAR: ("<LIDAR>", "</LIDAR>"),
    Modality.STATE: ("<STATE>", "</STATE>"),
    Modality.ACTION: ("<ACTION>", "</ACTION>"),
    Modality.META: ("<META>", "</META>"),
}

# sub-channel tags, e.g. RGB planes or channel-stacked audio
_CHANNEL_TAGS: dict[str, tuple[str, str]] = {
    "R": ("<R>", "</R>"),
    "G": ("<G>", "</G>"),
    "B": ("<B>", "</B>"),
    "LEFT": ("<LEFT>", "</LEFT>"),
    "RIGHT": ("<RIGHT>", "</RIGHT>"),
}

_STRUCTURAL_TOKENS = ["<ENDLINE>", "<ENDFRAME>", "<TICK>", "<PTSEP>"]

ALL_SPECIAL_TOKENS = (
    [t for pair in _MODALITY_TAGS.values() for t in pair]
    + [t for pair in _CHANNEL_TAGS.values() for t in pair]
    + _STRUCTURAL_TOKENS
    + ["<SEP>", "<MASK>"]
)

# Octet-family ids: WHICH byte-plane/channel a position's *value* belongs to, carried as a small
# parallel stream (octet_family_ids) instead of inflating the main token vocab. Byte values 0-255
# are shared by every modality (a value id never encodes "which family" by itself); the family
# tells the model whether that value is e.g. an image R channel byte, an audio hi byte, etc.
# Family 0 is the catch-all for text and structural/marker tokens.
_FAMILY_NAMES = (
    [f"IMG{c}" for c in range(3)]  # R, G, B (shared by video)
    + [f"AUD{p}" for p in range(2)]  # hi, lo byte of a 16-bit audio sample
    + [f"LID{i}" for i in range(8)]  # 4 point channels x hi/lo byte
    + [f"ACT{p}" for p in range(2)]  # hi, lo byte of a 16-bit action sample
    + [f"STA{p}" for p in range(2)]  # hi, lo byte of a 16-bit state sample
)
_FAMILY_ID = {name: idx + 1 for idx, name in enumerate(_FAMILY_NAMES)}  # 0 reserved for text/other
NUM_OCTET_FAMILIES = len(_FAMILY_NAMES) + 1


def _quantize_planes(x: np.ndarray, lo, hi, n_bytes: int) -> np.ndarray:
    """Float in [lo, hi] -> n_bytes big-endian byte planes (place-value digits, like RGB)."""
    levels = 256**n_bytes - 1
    q = np.clip((np.asarray(x) - lo) / (hi - lo) * levels, 0, levels).astype(np.int64)
    shifts = [(n_bytes - 1 - k) * 8 for k in range(n_bytes)]
    return np.stack([(q >> s) & 0xFF for s in shifts], axis=-1).astype(np.uint8)


def _dequantize_planes(planes: np.ndarray, lo, hi, n_bytes: int) -> np.ndarray:
    """Inverse of `_quantize_planes`."""
    levels = 256**n_bytes - 1
    shifts = [(n_bytes - 1 - k) * 8 for k in range(n_bytes)]
    q = sum(planes[..., k].astype(np.int64) << s for k, s in enumerate(shifts))
    return q / levels * (hi - lo) + lo


@dataclass
class MultimodalSegment:
    """One typed chunk of a sequence: UTF-8 bytes for TEXT, or an."""

    modality: Modality
    data: object
    channel: str | None = None


class KairosTokenizer(ByT5Tokenizer):
    """Byte-level multimodal tokenizer, ByT5-compatible for text; use encode_multimodal."""

    # pipeline-level constants — not stored per-instance
    IMAGE_CHANNELS = 3
    VIDEO_CHANNELS = 3
    AUDIO_SAMPLE_RATE = 16_000
    AUDIO_TICK_SAMPLES = 16_000  # one <TICK> per second of
    LIDAR_POINTS_PER_GROUP = 32  # one <PTSEP> every N points
    LIDAR_XYZ_RANGE = (-100.0, 100.0)
    LIDAR_INTENSITY_RANGE = (0.0, 1.0)
    SIGNAL_VALUE_RANGE = (-1.0, 1.0)  # state/action/imu channels are expected pre-clipped to this
    IMAGE_SCALE_FACTOR = 1  # default spatial block-mean factor for encode_image/encode_video
    LIDAR_SCALE_FACTOR = 1  # default point-stride factor for encode_lidar
    PCM_SCALE_FACTOR = 4  # default block-mean factor for encode_audio/encode_signal
    NUM_OCTET_FAMILIES = NUM_OCTET_FAMILIES

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("extra_ids", 0)
        super().__init__(*args, **kwargs)
        self.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})

        # byte -> id offset, shared by every modality (value only, no family)
        self._byte_offset = self.convert_tokens_to_ids(chr(0))
        assert self.convert_tokens_to_ids(chr(255)) == self._byte_offset + 255, (
            "byte->id mapping is not contiguous — review _byte_offset calibration"
        )
        self._endline_id = self.convert_tokens_to_ids("<ENDLINE>")
        self._endframe_id = self.convert_tokens_to_ids("<ENDFRAME>")
        self._tick_id = self.convert_tokens_to_ids("<TICK>")
        self._ptsep_id = self.convert_tokens_to_ids("<PTSEP>")

    def _bytes_to_ids(self, raw: bytes) -> list[int]:
        return [b + self._byte_offset for b in raw]

    def _ids_to_bytes(self, ids: list[int]) -> bytes:
        return bytes([max(0, i - self._byte_offset) & 0xFF for i in ids])

    @staticmethod
    def _family_ids_for(n: int, family_cycle: tuple | None) -> list[int]:
        """n values -> parallel octet-family ids, cycling family_cycle (or all-0 if None)."""
        if not family_cycle:
            return [0] * n
        cycle = [_FAMILY_ID[name] for name in family_cycle]
        return [cycle[i % len(cycle)] for i in range(n)]

    # ---------------- IMAGE: row-delimited, no header ----------------
    @staticmethod
    def _scale_hw(frame: np.ndarray, scale_factor: int) -> np.ndarray:
        """Block-mean downsamples the H,W dims of a (H,W,C) uint8 array by an integer factor."""
        if scale_factor <= 1:
            return frame
        h, w, c = frame.shape
        pad_h, pad_w = (-h) % scale_factor, (-w) % scale_factor
        if pad_h or pad_w:
            frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        h2, w2 = frame.shape[0] // scale_factor, frame.shape[1] // scale_factor
        blocks = frame.reshape(h2, scale_factor, w2, scale_factor, c)
        return blocks.mean(axis=(1, 3)).round().astype(np.uint8)

    @classmethod
    def encode_image(cls, image: np.ndarray, scale_factor: int | None = None) -> list:
        """(H, W, C) uint8 -> marker-list, one <ENDLINE> per row. scale_factor downsamples H,W."""
        if image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError("encode_image expects a (H, W, C) uint8 array")
        image = cls._scale_hw(image, scale_factor if scale_factor is not None else cls.IMAGE_SCALE_FACTOR)
        return cls._encode_frame_rows(image)

    @staticmethod
    def _encode_frame_rows(frame: np.ndarray) -> list:
        channels = frame.shape[-1]
        family_cycle = tuple(f"IMG{c}" for c in range(channels))  # R, G, B, ...
        out = []
        for row in frame:
            out.append(("bytes", row.tobytes(), family_cycle))
            out.append(("marker", "<ENDLINE>"))
        return out

    def _resolve_markers(self, marker_seq: list) -> tuple[list[int], list[int]]:
        """Returns (ids, octet_family_ids), both the same length."""
        ids: list[int] = []
        families: list[int] = []
        for entry in marker_seq:
            kind, payload = entry[0], entry[1]
            if kind == "bytes":
                family_cycle = entry[2] if len(entry) > 2 else None
                ids.extend(self._bytes_to_ids(payload))
                families.extend(self._family_ids_for(len(payload), family_cycle))
            else:
                ids.append(self.convert_tokens_to_ids(payload))
                families.append(0)  # structural markers are family 0 (text/other)
        return ids, families

    def decode_image(self, ids: list[int], channels: int | None = None) -> np.ndarray:
        """Dimensions recovered by counting <ENDLINE> markers; raises on bad row lengths."""
        channels = channels or self.IMAGE_CHANNELS
        rows, current = [], []
        for i in ids:
            if i == self._endline_id:
                rows.append(current)
                current = []
            else:
                current.append(i)
        if current:
            rows.append(current)
        if not rows:
            raise ValueError("no <ENDLINE> markers found — not a valid image stream")
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            raise ValueError(f"inconsistent row lengths {sorted(widths)} — malformed/truncated generation")
        w_bytes = widths.pop()
        if w_bytes % channels != 0:
            raise ValueError(f"row byte length {w_bytes} not divisible by channels={channels}")
        w, h = w_bytes // channels, len(rows)
        raw = self._ids_to_bytes([i for row in rows for i in row])
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, channels)

    # ---------------- VIDEO: rows + <ENDFRAME> ----------------
    @classmethod
    def encode_video(cls, frames: np.ndarray, stride: int = 1, scale_factor: int | None = None) -> list:
        """stride subsamples frames (temporal); scale_factor downsamples each frame's H,W."""
        if frames.dtype != np.uint8 or frames.ndim != 4:
            raise ValueError("encode_video expects a (T, H, W, C) uint8 array")
        factor = scale_factor if scale_factor is not None else cls.IMAGE_SCALE_FACTOR
        out = []
        for frame in frames[::stride]:
            out.extend(cls._encode_frame_rows(cls._scale_hw(frame, factor)))
            out.append(("marker", "<ENDFRAME>"))
        return out

    def decode_video(self, ids: list[int], channels: int | None = None, fps: float = 1.0):
        """Returns (frames, duration_seconds); fps is supplied at decode time, not stored in."""
        channels = channels or self.VIDEO_CHANNELS
        frame_lists, current = [], []
        for i in ids:
            if i == self._endframe_id:
                frame_lists.append(current)
                current = []
            else:
                current.append(i)
        if current:
            frame_lists.append(current)
        if not frame_lists:
            raise ValueError("no <ENDFRAME> markers found — not a valid video stream")
        frames = [self.decode_image(f, channels=channels) for f in frame_lists]
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            raise ValueError(f"inconsistent frame shapes across video: {shapes}")
        stacked = np.stack(frames, axis=0)
        duration = stacked.shape[0] / fps if fps > 0 else float("nan")
        return stacked, duration

    # ---------------- shared PCM codec: N-byte place-value quantization ----------------
    @staticmethod
    def _decimate_1d(values: np.ndarray, scale_factor: int) -> np.ndarray:
        """Block-means every scale_factor consecutive raw samples into 1 (no-op if factor <= 1)."""
        if scale_factor <= 1:
            return values
        n = len(values)
        pad = (-n) % scale_factor
        if pad:
            values = np.concatenate([values, np.full(pad, values[-1], dtype=values.dtype)])
        return values.reshape(-1, scale_factor).mean(axis=1).astype(values.dtype)

    @classmethod
    def _encode_pcm_ticks(
        cls, values: np.ndarray, lo: float, hi: float, tick_samples: int, family_cycle: tuple, n_bytes: int, scale_factor: int
    ) -> list[list]:
        """Same as _encode_pcm but returns one marker-list per tick, for interleaving."""
        values = cls._decimate_1d(values, scale_factor)
        planes = _quantize_planes(values, lo, hi, n_bytes)  # (N, n_bytes), one family id per byte-plane
        ticks = []
        for start in range(0, len(planes), tick_samples):
            ticks.append([("bytes", planes[start : start + tick_samples].tobytes(), family_cycle), ("marker", "<TICK>")])
        return ticks

    @classmethod
    def _encode_pcm(
        cls, values: np.ndarray, lo: float, hi: float, tick_samples: int, family_cycle: tuple, n_bytes: int, scale_factor: int
    ) -> list:
        out = []
        for tick in cls._encode_pcm_ticks(values, lo, hi, tick_samples, family_cycle, n_bytes, scale_factor):
            out.extend(tick)
        return out

    def _decode_pcm(self, ids: list[int], lo: float, hi: float, n_bytes: int) -> np.ndarray:
        samples, current = [], []
        for i in ids:
            if i == self._tick_id:
                samples.extend(current)
                current = []
            else:
                current.append(i)
        if current:
            samples.extend(current)
        raw = self._ids_to_bytes(samples)
        if len(raw) % n_bytes != 0:
            raise ValueError(f"payload ({len(raw)} bytes) not a multiple of n_bytes={n_bytes}")
        planes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, n_bytes)
        return _dequantize_planes(planes, lo, hi, n_bytes).astype(np.float32)

    # ---------------- AUDIO: flat PCM + periodic <TICK>, 16-bit by default ----------------
    @classmethod
    def encode_audio(
        cls, waveform: np.ndarray, tick_samples: int | None = None, n_bytes: int = 2, scale_factor: int | None = None
    ) -> list:
        """scale_factor downsamples raw samples before quantizing (default PCM_SCALE_FACTOR)."""
        if waveform.dtype != np.float32:
            raise ValueError("encode_audio expects a float32 waveform in [-1, 1]")
        family_cycle = tuple(f"AUD{p}" for p in range(n_bytes))  # hi, lo, ...
        factor = scale_factor if scale_factor is not None else cls.PCM_SCALE_FACTOR
        return cls._encode_pcm(waveform, -1.0, 1.0, tick_samples or cls.AUDIO_TICK_SAMPLES, family_cycle, n_bytes, factor)

    def decode_audio(self, ids: list[int], tick_samples: int | None = None, n_bytes: int = 2):
        """Returns (waveform, duration_seconds); duration = len(waveform)/AUDIO_SAMPLE_RATE."""
        waveform = self._decode_pcm(ids, -1.0, 1.0, n_bytes)
        return waveform, len(waveform) / self.AUDIO_SAMPLE_RATE

    # ------------- SIGNAL: generic control channel (state/action/imu) -------------
    @classmethod
    def encode_signal(
        cls, values: np.ndarray, family: str, tick_samples: int | None = None, n_bytes: int = 2, scale_factor: int | None = None
    ) -> list:
        """Same place-value PCM scheme as audio, tagged with its own family."""
        if values.dtype != np.float32:
            raise ValueError("encode_signal expects a float32 array in [-1, 1]")
        family_cycle = tuple(f"{family}{p}" for p in range(n_bytes))
        factor = scale_factor if scale_factor is not None else cls.PCM_SCALE_FACTOR
        return cls._encode_pcm(
            values, *cls.SIGNAL_VALUE_RANGE, tick_samples or cls.AUDIO_TICK_SAMPLES, family_cycle, n_bytes, factor
        )

    def decode_signal(self, ids: list[int], n_bytes: int = 2) -> np.ndarray:
        return self._decode_pcm(ids, *self.SIGNAL_VALUE_RANGE, n_bytes)

    @classmethod
    def encode_signal_ticks(
        cls, values: np.ndarray, family: str, tick_samples: int | None = None, n_bytes: int = 2, scale_factor: int | None = None
    ) -> list[list]:
        """Same as encode_signal but returns one marker-list per tick, for interleaving."""
        if values.dtype != np.float32:
            raise ValueError("encode_signal expects a float32 array in [-1, 1]")
        family_cycle = tuple(f"{family}{p}" for p in range(n_bytes))
        factor = scale_factor if scale_factor is not None else cls.PCM_SCALE_FACTOR
        return cls._encode_pcm_ticks(
            values, *cls.SIGNAL_VALUE_RANGE, tick_samples or cls.AUDIO_TICK_SAMPLES, family_cycle, n_bytes, factor
        )

    # ---------------- LIDAR: point groups + fixed quantization bounds ----------------
    @classmethod
    def encode_lidar(
        cls, points: np.ndarray, points_per_group: int | None = None, n_bytes: int = 2, scale_factor: int | None = None
    ) -> list:
        """scale_factor keeps every Nth point (uniform stride subsample) before quantizing."""
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[-1] != 4:
            raise ValueError("encode_lidar expects a (N, 4) float32 array [x,y,z,intensity]")
        factor = scale_factor if scale_factor is not None else cls.LIDAR_SCALE_FACTOR
        if factor > 1:
            points = points[::factor]
        points_per_group = points_per_group or cls.LIDAR_POINTS_PER_GROUP
        xyz_lo, xyz_hi = cls.LIDAR_XYZ_RANGE
        int_lo, int_hi = cls.LIDAR_INTENSITY_RANGE
        lo = np.array([xyz_lo, xyz_lo, xyz_lo, int_lo], dtype=np.float32)
        hi = np.array([xyz_hi, xyz_hi, xyz_hi, int_hi], dtype=np.float32)
        planes = _quantize_planes(points, lo, hi, n_bytes)  # (N, 4, n_bytes)
        family_cycle = tuple(f"LID{c * n_bytes + p}" for c in range(4) for p in range(n_bytes))
        out = []
        for start in range(0, len(planes), points_per_group):
            out.append(("bytes", planes[start : start + points_per_group].tobytes(), family_cycle))
            out.append(("marker", "<PTSEP>"))
        return out

    def decode_lidar(self, ids: list[int], n_bytes: int = 2) -> np.ndarray:
        """Dequantized via fixed LIDAR_XYZ_RANGE/LIDAR_INTENSITY_RANGE (lossy but header-free)."""
        point_bytes, current = [], []
        for i in ids:
            if i == self._ptsep_id:
                point_bytes.extend(current)
                current = []
            else:
                current.append(i)
        if current:
            point_bytes.extend(current)
        raw = self._ids_to_bytes(point_bytes)
        stride = 4 * n_bytes
        if len(raw) % stride != 0:
            raise ValueError(f"lidar payload ({len(raw)} bytes) not a multiple of {stride}")
        planes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4, n_bytes)
        xyz_lo, xyz_hi = self.LIDAR_XYZ_RANGE
        int_lo, int_hi = self.LIDAR_INTENSITY_RANGE
        lo = np.array([xyz_lo, xyz_lo, xyz_lo, int_lo], dtype=np.float32)
        hi = np.array([xyz_hi, xyz_hi, xyz_hi, int_hi], dtype=np.float32)
        return _dequantize_planes(planes, lo, hi, n_bytes)

    # ---------------- multimodal sequence assembly ----------------
    def encode_multimodal(self, segments: list[MultimodalSegment], max_len: int | None = None) -> dict:
        """Returns aligned {input_ids, modality_ids, octet_family_ids}; pads with Modality.TEXT."""
        all_ids: list[int] = []
        all_modality: list[int] = []
        all_family: list[int] = []

        for seg in segments:
            open_tag, close_tag = _MODALITY_TAGS[seg.modality]
            open_id = self.convert_tokens_to_ids(open_tag)
            close_id = self.convert_tokens_to_ids(close_tag)

            if seg.modality is Modality.TEXT:
                body_ids = self.encode(seg.data.decode("utf-8"), add_special_tokens=False)
                body_families = [0] * len(body_ids)
            else:
                body_ids, body_families = self._resolve_markers(seg.data)

            if seg.channel is not None:
                c_open, c_close = _CHANNEL_TAGS[seg.channel]
                body_ids = [self.convert_tokens_to_ids(c_open)] + body_ids + [self.convert_tokens_to_ids(c_close)]
                body_families = [0] + body_families + [0]

            seg_ids = [open_id] + body_ids + [close_id]
            seg_families = [0] + body_families + [0]
            all_ids.extend(seg_ids)
            all_modality.extend([int(seg.modality)] * len(seg_ids))
            all_family.extend(seg_families)

        if max_len is not None:
            all_ids = all_ids[:max_len]
            all_modality = all_modality[:max_len]
            all_family = all_family[:max_len]
            pad_len = max_len - len(all_ids)
            if pad_len > 0:
                all_ids += [self.pad_token_id] * pad_len
                all_modality += [int(Modality.TEXT)] * pad_len
                all_family += [0] * pad_len

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "modality_ids": torch.tensor(all_modality, dtype=torch.long),
            "octet_family_ids": torch.tensor(all_family, dtype=torch.long),
        }

    def decode_multimodal(self, input_ids: torch.Tensor) -> list[MultimodalSegment]:
        """Splits on modality delimiters; non-text `.data` is the raw id list, feed."""
        ids = input_ids.tolist()
        tokens = self.convert_ids_to_tokens(ids)
        open_to_modality = {tags[0]: m for m, tags in _MODALITY_TAGS.items()}

        segments: list[MultimodalSegment] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in open_to_modality:
                modality = open_to_modality[tok]
                _, close_tag = _MODALITY_TAGS[modality]
                j = i + 1
                body_ids: list[int] = []
                while j < len(tokens) and tokens[j] != close_tag:
                    body_ids.append(ids[j])
                    j += 1
                if modality is Modality.TEXT:
                    data = self.decode(body_ids, skip_special_tokens=False).encode("utf-8")
                else:
                    data = body_ids
                segments.append(MultimodalSegment(modality=modality, data=data))
                i = j + 1
            else:
                i += 1
        return segments


# len(KairosTokenizer()) == 259 base bytes/specials + 30 modality/channel/structural tags == 289.
# The (up to) 17 "which byte/channel" families live in a separate small octet_family_ids stream
# (see NUM_OCTET_FAMILIES), not in the main vocab - that's what keeps this vocab small.