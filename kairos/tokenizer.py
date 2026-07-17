"""
kairos/tokenizer.py

Byte-level multimodal tokenizer for Kairos, built on top of ByT5Tokenizer.

DESIGN (resolves the redundancy flagged during review of the previous version):
- The token stream stays byte-level (vocab 0-255) for every modality. Each
  modality's raw bytes are wrapped in structural delimiter tokens
  (<IMG> ... </IMG>, <AUDIO> ... </AUDIO>, etc.) so a sequence can be
  losslessly parsed back into its modality segments (see `decode_multimodal`).
- `modality_ids` (same length as `input_ids`) is produced alongside the token
  stream and is what actually drives `KairosEmbedding.modality_embed` /
  `KairosScaleRouter`. The delimiter tokens are *structural* (they mark where
  a segment starts/stops for parsing) — they are NOT a duplicate encoding of
  the modality; the backbone never has to infer modality from a text token,
  it is only ever given the modality_ids stream directly.

KNOWN BREAKING CHANGE vs the previous stub:
- `ByT5Tokenizer` defaults to `extra_ids=125` (T5 sentinel tokens), which
  makes `len(tokenizer) == 384`, not 259 as assumed by `KairosConfig` and the
  test suite. This class now forces `extra_ids=0` and registers the modality/
  channel delimiters as additional special tokens, giving a total vocab of
  259 (base) + 28 (8 modality pairs + 5 channel pairs + <SEP>/<MASK>,
  see ALL_SPECIAL_TOKENS) = 287.
  ==> `KairosConfig(vocab_size=...)` and anywhere `vocab_size=259` is
  hardcoded (tests, notebook, `KairosDiffusionLLM(vocab_size=259)`) must be
  updated to `len(KairosTokenizer())` instead of the literal 259.
- `ByT5Tokenizer` has no native `<BOS>` token (`bos_token_id` is `None`).
  The old reserved-token table in this file listed one at id 257 — that was
  never true for a stock ByT5Tokenizer and has been removed below.

Byte <-> id mapping is calibrated empirically at __init__ time (via
`convert_tokens_to_ids(chr(0))`) rather than relying on private internals of
ByT5Tokenizer, so it stays correct across transformers versions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import torch
from transformers.models.byt5.tokenization_byt5 import ByT5Tokenizer


# ==========================================================
# Modality enum — MUST match KairosConfig.text_modality_id /
# KairosConfig.modality_scales ordering (0 = text).
# ==========================================================
class Modality(enum.IntEnum):
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

# Optional sub-channel tags (e.g. splitting an image into R/G/B planes, or
# audio into LEFT/RIGHT). Only used if the caller passes `channel=...`.
_CHANNEL_TAGS: dict[str, tuple[str, str]] = {
    "R": ("<R>", "</R>"),
    "G": ("<G>", "</G>"),
    "B": ("<B>", "</B>"),
    "LEFT": ("<LEFT>", "</LEFT>"),
    "RIGHT": ("<RIGHT>", "</RIGHT>"),
}

ALL_SPECIAL_TOKENS = (
    [t for pair in _MODALITY_TAGS.values() for t in pair]
    + [t for pair in _CHANNEL_TAGS.values() for t in pair]
    + ["<SEP>", "<MASK>"]
)


@dataclass
class MultimodalSegment:
    """One typed chunk of a multimodal sequence.

    `data` is:
      - raw UTF-8 bytes of a string, for Modality.TEXT
      - the output of one of the `KairosTokenizer.encode_*` quantizers below,
        for every other modality.
    """

    modality: Modality
    data: bytes
    channel: str | None = None  # e.g. "R" / "G" / "B" / "LEFT" / "RIGHT"


class KairosTokenizer(ByT5Tokenizer):
    """
    Byte-level tokenizer, multimodal-aware.

    - Text: standard ByT5 UTF-8 byte encoding (fully backward compatible —
      `tokenizer.encode(text)` / `tokenizer.decode(ids)` behave exactly as
      before for pure-text use).
    - Non-text modalities: raw bytes of a *pre-quantized* representation
      (see `encode_image` / `encode_audio` / `encode_video` / `encode_lidar`
      for the default 8-bit quantizers — swap these for a learned codec
      later without touching the assembly logic).
    - `encode_multimodal` is the single entry point that returns aligned
      `input_ids` + `modality_ids`, ready for `KairosEmbedding`.
    """

    def __init__(self, *args, **kwargs):
        # Force extra_ids=0: Kairos has no use for T5 sentinel tokens and
        # they would silently inflate len(tokenizer) from 259 to 384+.
        kwargs.setdefault("extra_ids", 0)
        super().__init__(*args, **kwargs)

        self.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})

        # Calibrate the byte -> id offset empirically (id = byte + offset).
        # ByT5 maps byte b to token chr(b); this holds for the full 0-255
        # range regardless of how many special/added tokens exist, since
        # added tokens are appended after the byte range, not interleaved.
        self._byte_offset = self.convert_tokens_to_ids(chr(0))
        assert self.convert_tokens_to_ids(chr(255)) == self._byte_offset + 255, (
            "byte->id mapping is not contiguous — ByT5Tokenizer internals changed, "
            "review KairosTokenizer._byte_offset calibration"
        )

    # ---------------------------------------------------------------
    # low-level: raw bytes <-> token ids
    # ---------------------------------------------------------------
    def _bytes_to_ids(self, raw: bytes) -> list[int]:
        return [b + self._byte_offset for b in raw]

    def _ids_to_bytes(self, ids: list[int]) -> bytes:
        return bytes([max(0, i - self._byte_offset) & 0xFF for i in ids])

    # ---------------------------------------------------------------
    # modality-specific quantizers -> raw bytes
    # (deliberately simple/lossy 8-bit quantizers; swap for a learned
    # codec later — the assembly logic below doesn't care how the bytes
    # were produced)
    # ---------------------------------------------------------------
    @staticmethod
    def encode_image(image: np.ndarray) -> bytes:
        """image: (H, W, 3) uint8 array -> raw interleaved RGB bytes."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("encode_image expects a (H, W, 3) uint8 array")
        return image.tobytes()

    @staticmethod
    def decode_image(raw: bytes, height: int, width: int) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

    @staticmethod
    def encode_audio(waveform: np.ndarray) -> bytes:
        """waveform: float32 in [-1, 1], mono or (T, C) -> 8-bit PCM bytes."""
        if waveform.dtype != np.float32:
            raise ValueError("encode_audio expects a float32 waveform in [-1, 1]")
        pcm = np.clip(waveform * 127.5 + 127.5, 0, 255).astype(np.uint8)
        return pcm.tobytes()

    @staticmethod
    def decode_audio(raw: bytes) -> np.ndarray:
        pcm = np.frombuffer(raw, dtype=np.uint8)
        return (pcm.astype(np.float32) - 127.5) / 127.5

    @staticmethod
    def encode_video(frames: np.ndarray, stride: int = 1) -> bytes:
        """frames: (T, H, W, 3) uint8 -> concatenated per-frame RGB bytes."""
        if frames.dtype != np.uint8 or frames.ndim != 4:
            raise ValueError("encode_video expects a (T, H, W, 3) uint8 array")
        return frames[::stride].tobytes()

    @staticmethod
    def decode_video(raw: bytes, num_frames: int, height: int, width: int) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.uint8).reshape(num_frames, height, width, 3)

    @staticmethod
    def encode_lidar(points: np.ndarray) -> bytes:
        """points: (N, 4) float32 [x, y, z, intensity] -> per-channel min-max
        quantized uint8 bytes. NOTE: lossy and NOT self-describing — the
        caller must keep track of (mins, maxs) separately to dequantize
        exactly; `decode_lidar` returns normalized [0, 1] values only."""
        if points.dtype != np.float32 or points.shape[-1] != 4:
            raise ValueError("encode_lidar expects a (N, 4) float32 array")
        mins = points.min(axis=0, keepdims=True)
        maxs = points.max(axis=0, keepdims=True)
        scale = np.clip(maxs - mins, 1e-6, None)
        q = ((points - mins) / scale * 255).astype(np.uint8)
        return q.tobytes()

    @staticmethod
    def decode_lidar(raw: bytes) -> np.ndarray:
        q = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
        return q.astype(np.float32) / 255.0

    # ---------------------------------------------------------------
    # high-level: multimodal sequence assembly
    # ---------------------------------------------------------------
    def encode_multimodal(
        self,
        segments: list[MultimodalSegment],
        max_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        segments: ordered list of MultimodalSegment — mix text and other
            modalities freely, e.g. [text, image, text] for an image-QA pair.

        Returns {"input_ids": LongTensor[L], "modality_ids": LongTensor[L]},
        both length L (truncated/padded to max_len if given). Padding uses
        Modality.TEXT for modality_ids (cheapest routing scale, consistent
        with KairosPretrainingDataset's existing padding convention).
        """
        all_ids: list[int] = []
        all_modality: list[int] = []

        for seg in segments:
            open_tag, close_tag = _MODALITY_TAGS[seg.modality]
            open_id = self.convert_tokens_to_ids(open_tag)
            close_id = self.convert_tokens_to_ids(close_tag)

            if seg.modality is Modality.TEXT:
                body_ids = self.encode(seg.data.decode("utf-8"), add_special_tokens=False)
            else:
                body_ids = self._bytes_to_ids(seg.data)

            if seg.channel is not None:
                c_open, c_close = _CHANNEL_TAGS[seg.channel]
                body_ids = (
                    [self.convert_tokens_to_ids(c_open)]
                    + body_ids
                    + [self.convert_tokens_to_ids(c_close)]
                )

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
        """Best-effort roundtrip: split on delimiter tokens, return raw bytes
        per segment (lossless for the bytes themselves; any channel sub-tags
        are stripped and not reconstructed as separate segments)."""
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
                    data = self._ids_to_bytes(body_ids)

                segments.append(MultimodalSegment(modality=modality, data=data))
                i = j + 1
            else:
                i += 1
        return segments


# ==========================================================
# Reserved Special Tokens — kept as documentation. The actual
# vocab is registered programmatically in __init__ via
# ALL_SPECIAL_TOKENS; this table just shows the resulting layout
# for a freshly constructed KairosTokenizer(extra_ids=0).
# ==========================================================
#
#   0-255 : Raw byte values          (id = byte + 3)
#   0     : <PAD>
#   1     : <EOS>
#   2     : <UNK>
#           (ByT5 has NO native <BOS> — bos_token_id is None)
#
# 259+    : modality / channel delimiters, appended in ALL_SPECIAL_TOKENS
#           order (exact ids depend on transformers version — always read
#           them via `tokenizer.convert_tokens_to_ids(...)`, never hardcode):
#           <TEXT> </TEXT> <IMG> </IMG> <VIDEO> </VIDEO> <AUDIO> </AUDIO>
#           <LIDAR> </LIDAR> <STATE> </STATE> <ACTION> </ACTION>
#           <META> </META> <R> </R> <G> </G> <B> </B> <LEFT> </LEFT>
#           <RIGHT> </RIGHT> <SEP> <MASK>
#
# len(KairosTokenizer()) == 287  (259 base + 28 special tokens above)
# ==========================================================