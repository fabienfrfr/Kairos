"""Byte-level multimodal tokenizer built on ByT5Tokenizer, using periodic LOCAL markers (PixelBytes, arxiv.org/html/2410.01820v2)."""

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

# sub-channel tags, e.g. RGB planes or stereo audio
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


@dataclass
class MultimodalSegment:
    """One typed chunk of a sequence: UTF-8 bytes for TEXT, or an encode_*'s marker-list otherwise."""

    modality: Modality
    data: object
    channel: str | None = None


class KairosTokenizer(ByT5Tokenizer):
    """Byte-level multimodal tokenizer, ByT5-compatible for text; `encode_multimodal` is the entry point for mixed sequences."""

    # pipeline-level constants — not stored per-instance in the stream
    IMAGE_CHANNELS = 3
    VIDEO_CHANNELS = 3
    AUDIO_SAMPLE_RATE = 16_000
    AUDIO_TICK_SAMPLES = 16_000  # one <TICK> per second of audio
    LIDAR_POINTS_PER_GROUP = 32  # one <PTSEP> every N points
    LIDAR_XYZ_RANGE = (-100.0, 100.0)
    LIDAR_INTENSITY_RANGE = (0.0, 1.0)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("extra_ids", 0)
        super().__init__(*args, **kwargs)
        self.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})

        # byte -> id offset, calibrated empirically (id = byte + offset)
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

    # ---------------- IMAGE: row-delimited, no header ----------------
    @classmethod
    def encode_image(cls, image: np.ndarray) -> list:
        """(H, W, C) uint8 -> marker-list, one <ENDLINE> per row."""
        if image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError("encode_image expects a (H, W, C) uint8 array")
        return cls._encode_frame_rows(image)

    @staticmethod
    def _encode_frame_rows(frame: np.ndarray) -> list:
        out = []
        for row in frame:
            out.append(("bytes", row.tobytes()))
            out.append(("marker", "<ENDLINE>"))
        return out

    def _resolve_markers(self, marker_seq: list) -> list[int]:
        ids: list[int] = []
        for kind, payload in marker_seq:
            if kind == "bytes":
                ids.extend(self._bytes_to_ids(payload))
            else:
                ids.append(self.convert_tokens_to_ids(payload))
        return ids

    def decode_image(self, ids: list[int], channels: int | None = None) -> np.ndarray:
        """Dimensions recovered by counting <ENDLINE> markers; raises on inconsistent row lengths."""
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
    def encode_video(cls, frames: np.ndarray, stride: int = 1) -> list:
        if frames.dtype != np.uint8 or frames.ndim != 4:
            raise ValueError("encode_video expects a (T, H, W, C) uint8 array")
        out = []
        for frame in frames[::stride]:
            out.extend(cls._encode_frame_rows(frame))
            out.append(("marker", "<ENDFRAME>"))
        return out

    def decode_video(self, ids: list[int], channels: int | None = None, fps: float = 1.0):
        """Returns (frames, duration_seconds); fps is supplied at decode time, not stored in the stream."""
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

    # ---------------- AUDIO: flat PCM + periodic <TICK> ----------------
    @classmethod
    def encode_audio(cls, waveform: np.ndarray, tick_samples: int | None = None) -> list:
        if waveform.dtype != np.float32:
            raise ValueError("encode_audio expects a float32 waveform in [-1, 1]")
        tick_samples = tick_samples or cls.AUDIO_TICK_SAMPLES
        pcm = np.clip(waveform * 127.5 + 127.5, 0, 255).astype(np.uint8)
        out = []
        for start in range(0, len(pcm), tick_samples):
            out.append(("bytes", pcm[start : start + tick_samples].tobytes()))
            out.append(("marker", "<TICK>"))
        return out

    def decode_audio(self, ids: list[int], tick_samples: int | None = None):
        """Returns (waveform, duration_seconds); duration = len(waveform)/AUDIO_SAMPLE_RATE."""
        samples, current = [], []
        for i in ids:
            if i == self._tick_id:
                samples.extend(current)
                current = []
            else:
                current.append(i)
        if current:
            samples.extend(current)
        pcm = np.frombuffer(self._ids_to_bytes(samples), dtype=np.uint8)
        waveform = (pcm.astype(np.float32) - 127.5) / 127.5
        return waveform, len(waveform) / self.AUDIO_SAMPLE_RATE

    # ---------------- LIDAR: point groups + fixed quant bounds ----------------
    @classmethod
    def encode_lidar(cls, points: np.ndarray, points_per_group: int | None = None) -> list:
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[-1] != 4:
            raise ValueError("encode_lidar expects a (N, 4) float32 array [x,y,z,intensity]")
        points_per_group = points_per_group or cls.LIDAR_POINTS_PER_GROUP
        xyz_lo, xyz_hi = cls.LIDAR_XYZ_RANGE
        int_lo, int_hi = cls.LIDAR_INTENSITY_RANGE
        lo = np.array([xyz_lo, xyz_lo, xyz_lo, int_lo], dtype=np.float32)
        hi = np.array([xyz_hi, xyz_hi, xyz_hi, int_hi], dtype=np.float32)
        q = ((np.clip(points, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
        out = []
        for start in range(0, len(q), points_per_group):
            out.append(("bytes", q[start : start + points_per_group].tobytes()))
            out.append(("marker", "<PTSEP>"))
        return out

    def decode_lidar(self, ids: list[int]) -> np.ndarray:
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
        if len(raw) % 4 != 0:
            raise ValueError(f"lidar payload ({len(raw)} bytes) not a multiple of 4")
        q = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4).astype(np.float32)
        xyz_lo, xyz_hi = self.LIDAR_XYZ_RANGE
        int_lo, int_hi = self.LIDAR_INTENSITY_RANGE
        lo = np.array([xyz_lo, xyz_lo, xyz_lo, int_lo], dtype=np.float32)
        hi = np.array([xyz_hi, xyz_hi, xyz_hi, int_hi], dtype=np.float32)
        return q / 255.0 * (hi - lo) + lo

    # ---------------- multimodal sequence assembly ----------------
    def encode_multimodal(self, segments: list[MultimodalSegment], max_len: int | None = None) -> dict:
        """Returns {"input_ids", "modality_ids"} aligned tensors; padding uses Modality.TEXT (cheapest routing scale)."""
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
        """Splits on modality delimiters; non-text `.data` is the raw id list, feed it to the matching decode_* method."""
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


# len(KairosTokenizer()) == 291  (259 base bytes/pad/eos/unk + 32 special tokens:
# 8 modality pairs, 5 channel pairs, 4 structural markers, <SEP>, <MASK>)
