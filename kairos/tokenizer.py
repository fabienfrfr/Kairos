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

# Raw-byte id blocks, one per (modality family, position-in-group), so distinct positions never
# share a token id: 3 for IMG (R,G,B channels, shared by video), 2 for AUD (hi/lo byte of each
# 16-bit sample), 8 for LID (4 point channels x hi/lo byte), 2 each for ACT/STA (hi/lo byte).
_BLOCK_GROUPS = {"IMG": 3, "AUD": 2, "LID": 8, "ACT": 2, "STA": 2}
_SUBBLOCKS = [f"{family}{sub}" for family, n in _BLOCK_GROUPS.items() for sub in range(n)]
_BLOCK_TOKENS = [f"<{sb}_{i}>" for sb in _SUBBLOCKS for i in range(256)]

ALL_SPECIAL_TOKENS = (
    [t for pair in _MODALITY_TAGS.values() for t in pair]
    + [t for pair in _CHANNEL_TAGS.values() for t in pair]
    + _STRUCTURAL_TOKENS
    + ["<SEP>", "<MASK>"]
    + _BLOCK_TOKENS
)


def _quantize_planes(x: np.ndarray, lo, hi, n_bytes: int) -> np.ndarray:
    """Float in [lo, hi] -> n_bytes big-endian byte planes (place-value digits, like RGB channels), no single-byte rounding."""
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

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("extra_ids", 0)
        super().__init__(*args, **kwargs)
        self.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})

        # byte -> id offset, calibrated empirically
        self._byte_offset = self.convert_tokens_to_ids(chr(0))
        assert self.convert_tokens_to_ids(chr(255)) == self._byte_offset + 255, (
            "byte->id mapping is not contiguous — review _byte_offset calibration"
        )
        self._block_offset = {}
        for sb in _SUBBLOCKS:
            offset = self.convert_tokens_to_ids(f"<{sb}_0>")
            assert self.convert_tokens_to_ids(f"<{sb}_255>") == offset + 255, f"{sb} byte block not contiguous"
            self._block_offset[sb] = offset
        self._endline_id = self.convert_tokens_to_ids("<ENDLINE>")
        self._endframe_id = self.convert_tokens_to_ids("<ENDFRAME>")
        self._tick_id = self.convert_tokens_to_ids("<TICK>")
        self._ptsep_id = self.convert_tokens_to_ids("<PTSEP>")

    def _offsets(self, block) -> list[int]:
        """block: "text", a single sub-block name, or a tuple/list of names cycled across bytes."""
        names = [block] if isinstance(block, str) else block
        return [self._byte_offset if b == "text" else self._block_offset[b] for b in names]

    def _bytes_to_ids(self, raw: bytes, block: str = "text") -> list[int]:
        offsets = self._offsets(block)
        return [b + offsets[i % len(offsets)] for i, b in enumerate(raw)]

    def _ids_to_bytes(self, ids: list[int], block: str = "text") -> bytes:
        offsets = self._offsets(block)
        return bytes([max(0, i - offsets[k % len(offsets)]) & 0xFF for k, i in enumerate(ids)])

    # ---------------- IMAGE: row-delimited, no header ----------------
    @classmethod
    def encode_image(cls, image: np.ndarray) -> list:
        """(H, W, C) uint8 -> marker-list, one <ENDLINE> per row."""
        if image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError("encode_image expects a (H, W, C) uint8 array")
        return cls._encode_frame_rows(image)

    @staticmethod
    def _encode_frame_rows(frame: np.ndarray) -> list:
        channels = frame.shape[-1]
        block_cycle = tuple(f"IMG{c}" for c in range(channels))  # one token block per channel (R,G,B,...)
        out = []
        for row in frame:
            out.append(("bytes", row.tobytes(), block_cycle))
            out.append(("marker", "<ENDLINE>"))
        return out

    def _resolve_markers(self, marker_seq: list) -> list[int]:
        ids: list[int] = []
        for entry in marker_seq:
            kind, payload = entry[0], entry[1]
            if kind == "bytes":
                block = entry[2] if len(entry) > 2 else "text"
                ids.extend(self._bytes_to_ids(payload, block))
            else:
                ids.append(self.convert_tokens_to_ids(payload))
        return ids

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
        block_cycle = tuple(f"IMG{c}" for c in range(channels))
        raw = self._ids_to_bytes([i for row in rows for i in row], block_cycle)
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, channels)

    # ---------------- VIDEO: rows + <ENDFRAME> ----------------
    @classmethod
    def encode_video(cls, frames: np.ndarray, stride: int = 1) -> list:
        if frames.dtype != np.uint8 or frames.ndim != 4:
            raise ValueError("encode_video expects a (T, H, W, C) uint8 array")
        out = []
        for frame in frames[::stride]:
            out.extend(cls._encode_frame_rows(frame))
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

    # ---------------- shared PCM codec: N-byte place-value quantization for 1D signals ----------------
    @classmethod
    def _encode_pcm(cls, values: np.ndarray, lo: float, hi: float, tick_samples: int, block_cycle: tuple, n_bytes: int) -> list:
        planes = _quantize_planes(values, lo, hi, n_bytes)  # (N, n_bytes), one distinct block per byte-plane
        out = []
        for start in range(0, len(planes), tick_samples):
            out.append(("bytes", planes[start : start + tick_samples].tobytes(), block_cycle))
            out.append(("marker", "<TICK>"))
        return out

    def _decode_pcm(self, ids: list[int], lo: float, hi: float, block_cycle: tuple, n_bytes: int) -> np.ndarray:
        samples, current = [], []
        for i in ids:
            if i == self._tick_id:
                samples.extend(current)
                current = []
            else:
                current.append(i)
        if current:
            samples.extend(current)
        raw = self._ids_to_bytes(samples, block_cycle)
        if len(raw) % n_bytes != 0:
            raise ValueError(f"payload ({len(raw)} bytes) not a multiple of n_bytes={n_bytes}")
        planes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, n_bytes)
        return _dequantize_planes(planes, lo, hi, n_bytes).astype(np.float32)

    # ---------------- AUDIO: flat PCM + periodic <TICK>, n_bytes=2 (16-bit) by default ----------------
    @classmethod
    def encode_audio(cls, waveform: np.ndarray, tick_samples: int | None = None, n_bytes: int = 2) -> list:
        if waveform.dtype != np.float32:
            raise ValueError("encode_audio expects a float32 waveform in [-1, 1]")
        block_cycle = tuple(f"AUD{p}" for p in range(n_bytes))  # one token block per byte-plane (hi, lo, ...)
        return cls._encode_pcm(waveform, -1.0, 1.0, tick_samples or cls.AUDIO_TICK_SAMPLES, block_cycle, n_bytes)

    def decode_audio(self, ids: list[int], tick_samples: int | None = None, n_bytes: int = 2):
        """Returns (waveform, duration_seconds); duration = len(waveform)/AUDIO_SAMPLE_RATE."""
        block_cycle = tuple(f"AUD{p}" for p in range(n_bytes))
        waveform = self._decode_pcm(ids, -1.0, 1.0, block_cycle, n_bytes)
        return waveform, len(waveform) / self.AUDIO_SAMPLE_RATE

    # ------------- SIGNAL: generic control channel (state/action/imu), family picks its own token block -------------
    @classmethod
    def encode_signal(cls, values: np.ndarray, family: str, tick_samples: int | None = None, n_bytes: int = 2) -> list:
        """Same place-value PCM scheme as audio, but its own block per family (e.g. "ACT" vs "STA") so channels never share ids."""
        if values.dtype != np.float32:
            raise ValueError("encode_signal expects a float32 array in [-1, 1]")
        block_cycle = tuple(f"{family}{p}" for p in range(n_bytes))
        return cls._encode_pcm(values, *cls.SIGNAL_VALUE_RANGE, tick_samples or cls.AUDIO_TICK_SAMPLES, block_cycle, n_bytes)

    def decode_signal(self, ids: list[int], family: str, n_bytes: int = 2) -> np.ndarray:
        block_cycle = tuple(f"{family}{p}" for p in range(n_bytes))
        return self._decode_pcm(ids, *self.SIGNAL_VALUE_RANGE, block_cycle, n_bytes)

    # ---------------- LIDAR: point groups + fixed quantization bounds, n_bytes=2 per channel ----------------
    @classmethod
    def encode_lidar(cls, points: np.ndarray, points_per_group: int | None = None, n_bytes: int = 2) -> list:
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[-1] != 4:
            raise ValueError("encode_lidar expects a (N, 4) float32 array [x,y,z,intensity]")
        points_per_group = points_per_group or cls.LIDAR_POINTS_PER_GROUP
        xyz_lo, xyz_hi = cls.LIDAR_XYZ_RANGE
        int_lo, int_hi = cls.LIDAR_INTENSITY_RANGE
        lo = np.array([xyz_lo, xyz_lo, xyz_lo, int_lo], dtype=np.float32)
        hi = np.array([xyz_hi, xyz_hi, xyz_hi, int_hi], dtype=np.float32)
        planes = _quantize_planes(points, lo, hi, n_bytes)  # (N, 4, n_bytes)
        block_cycle = tuple(f"LID{c * n_bytes + p}" for c in range(4) for p in range(n_bytes))  # one block per (channel, byte-plane)
        out = []
        for start in range(0, len(planes), points_per_group):
            out.append(("bytes", planes[start : start + points_per_group].tobytes(), block_cycle))
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
        block_cycle = tuple(f"LID{c * n_bytes + p}" for c in range(4) for p in range(n_bytes))
        raw = self._ids_to_bytes(point_bytes, block_cycle)
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
        """Returns aligned {"input_ids", "modality_ids"} tensors; padding uses Modality.TEXT."""
        all_ids: list[int] = []
        all_modality: list[int] = []

        for seg in segments:
            open_tag, close_tag = _MODALITY_TAGS[seg.modality]
            open_id = self.convert_tokens_to_ids(open_tag)
            close_id = self.convert_tokens_to_ids(close_tag)

            if seg.modality is Modality.TEXT:
                body_ids = self.encode(seg.data.decode("utf-8"), add_special_tokens=False)
            else:
                body_ids = self._resolve_markers(seg.data)

            if seg.channel is not None:
                c_open, c_close = _CHANNEL_TAGS[seg.channel]
                body_ids = [self.convert_tokens_to_ids(c_open)] + body_ids + [self.convert_tokens_to_ids(c_close)]

            seg_ids = [open_id] + body_ids + [close_id]
            all_ids.extend(seg_ids)
            all_modality.extend([int(seg.modality)] * len(seg_ids))

        if max_len is not None:
            all_ids = all_ids[:max_len]
            all_modality = all_modality[:max_len]
            pad_len = max_len - len(all_ids)
            if pad_len > 0:
                all_ids += [self.pad_token_id] * pad_len
                all_modality += [int(Modality.TEXT)] * pad_len

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "modality_ids": torch.tensor(all_modality, dtype=torch.long),
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


# len(KairosTokenizer()) == 259 base bytes/specials + 32 modality/channel/structural tags + 4352
# per-position raw-byte block tokens (17 sub-blocks x 256: 3 IMG + 2 AUD + 8 LID + 2 ACT + 2 STA) == 4643
