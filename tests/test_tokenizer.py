import numpy as np
import pytest
import torch

from kairos.tokenizer import KairosTokenizer, Modality, MultimodalSegment


# ========================= Fixtures =========================
@pytest.fixture(scope="module")
def tokenizer():
    return KairosTokenizer()


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (5, 7, 3), dtype=np.uint8)


@pytest.fixture
def sample_video():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (6, 3, 4, 3), dtype=np.uint8)


@pytest.fixture
def sample_audio():
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, 40_000).astype(np.float32)


@pytest.fixture
def sample_lidar():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-10, 10, (50, 4)).astype(np.float32)
    pts[:, 3] = rng.uniform(0, 1, 50)  # intensity in [0, 1]
    return pts


# ========================= Vocab / backward compatibility =========================
def test_vocab_size_is_4643(tokenizer):
    """Regression: 259 base bytes/specials + 32 tags + 17 sub-blocks x 256 (3 IMG + 2 AUD + 8 LID + 2 ACT + 2 STA)."""
    assert len(tokenizer) == 4643


def test_no_native_bos(tokenizer):
    assert tokenizer.bos_token_id is None


def test_byte_offset_contiguous(tokenizer):
    assert tokenizer.convert_tokens_to_ids(chr(0)) == tokenizer._byte_offset
    assert tokenizer.convert_tokens_to_ids(chr(255)) == tokenizer._byte_offset + 255


def test_image_and_text_bytes_use_disjoint_token_ids(tokenizer):
    # regression: image/audio/lidar/signal bytes used to share the exact same ids as text bytes
    text_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "text"))
    img_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "IMG0"))
    aud_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "AUD0"))
    lid_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "LID0"))
    act_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "ACT0"))
    sta_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "STA0"))
    all_blocks = [text_ids, img_ids, aud_ids, lid_ids, act_ids, sta_ids]
    assert sum(len(b) for b in all_blocks) == len(set().union(*all_blocks))


def test_place_value_bytes_use_distinct_blocks_not_a_duplicated_one(tokenizer):
    # the hi and lo byte of a 16-bit sample must NOT share a block (e.g. "LLRR" -> distinct
    # L1/L2/R1/R2 tokens, not the same "L"/"R" token repeated) - each cycle position is disjoint
    hi_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "AUD0"))
    lo_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "AUD1"))
    assert hi_ids.isdisjoint(lo_ids)


def test_rgb_channels_use_distinct_blocks(tokenizer):
    r_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "IMG0"))
    g_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "IMG1"))
    b_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "IMG2"))
    assert r_ids.isdisjoint(g_ids) and g_ids.isdisjoint(b_ids) and r_ids.isdisjoint(b_ids)


def test_action_and_state_use_disjoint_token_blocks(tokenizer):
    # action and state are semantically different control channels, must not share ids either
    act_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "ACT0"))
    sta_ids = set(tokenizer._bytes_to_ids(bytes(range(256)), "STA0"))
    assert act_ids.isdisjoint(sta_ids)


def test_signal_roundtrip_precision_beats_8bit(tokenizer):
    # 16-bit place-value quantization must resolve a step far finer than the old 8-bit scheme
    rng = np.random.default_rng(0)
    values = rng.uniform(-1, 1, 200).astype(np.float32)
    markers = KairosTokenizer.encode_signal(values, family="ACT")
    ids = tokenizer._resolve_markers(markers)
    recon = tokenizer.decode_signal(ids, family="ACT")
    assert np.max(np.abs(recon - values)) < (2 / 255)  # much tighter than the old 8-bit step


def test_signal_and_audio_use_different_token_blocks(tokenizer):
    values = np.zeros(4, dtype=np.float32)
    signal_ids = tokenizer._resolve_markers(KairosTokenizer.encode_signal(values, family="ACT"))
    audio_ids = tokenizer._resolve_markers(KairosTokenizer.encode_audio(values))
    byte_signal_ids = [i for i in signal_ids if i != tokenizer._tick_id]
    byte_audio_ids = [i for i in audio_ids if i != tokenizer._tick_id]
    assert set(byte_signal_ids).isdisjoint(byte_audio_ids)


def test_plain_text_encode_decode_unchanged(tokenizer):
    text = "Paris is the capital of France."
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids, skip_special_tokens=True) == text


def test_structural_and_modality_tokens_present(tokenizer):
    for tag in (
        "<IMG>",
        "</IMG>",
        "<VIDEO>",
        "</VIDEO>",
        "<AUDIO>",
        "</AUDIO>",
        "<LIDAR>",
        "</LIDAR>",
        "<ENDLINE>",
        "<ENDFRAME>",
        "<TICK>",
        "<PTSEP>",
    ):
        tid = tokenizer.convert_tokens_to_ids(tag)
        assert tid is not None and tid != tokenizer.unk_token_id, f"{tag} missing from vocab"


# ========================= IMAGE: row-delimited, no header =========================
def test_image_roundtrip(tokenizer, sample_image):
    markers = KairosTokenizer.encode_image(sample_image)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.IMAGE, markers)])
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    recon = tokenizer.decode_image(decoded[0].data)
    assert np.array_equal(recon, sample_image)


def test_image_endline_every_row(tokenizer, sample_image):
    """The whole point of the design: <ENDLINE> recurs locally, every W*C tokens,."""
    markers = KairosTokenizer.encode_image(sample_image)
    ids = tokenizer._resolve_markers(markers)
    endline_positions = [i for i, tid in enumerate(ids) if tid == tokenizer._endline_id]
    h, w, c = sample_image.shape
    assert len(endline_positions) == h
    gaps = [endline_positions[0] + 1] + [
        endline_positions[i] - endline_positions[i - 1] for i in range(1, len(endline_positions))
    ]
    assert all(g == w * c + 1 for g in gaps)  # w*c bytes + the marker


def test_image_rejects_bad_dtype():
    with pytest.raises(ValueError):
        KairosTokenizer.encode_image(np.zeros((2, 2, 3), dtype=np.float32))


def test_image_decode_detects_truncated_row(tokenizer, sample_image):
    """Simulates a drifted generation with a short last row; must raise a."""
    markers = KairosTokenizer.encode_image(sample_image)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.IMAGE, markers)])
    truncated = torch.tensor(out["input_ids"].tolist()[:-5])  # cut mid last row
    decoded = tokenizer.decode_multimodal(truncated)
    with pytest.raises(ValueError, match="inconsistent row lengths"):
        tokenizer.decode_image(decoded[0].data)


# ========================= VIDEO: rows + <ENDFRAME>, fps supplied =========================
def test_video_roundtrip_with_stride(tokenizer, sample_video):
    markers = KairosTokenizer.encode_video(sample_video, stride=2)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.VIDEO, markers)])
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    recon, duration = tokenizer.decode_video(decoded[0].data, fps=12.0)
    assert np.array_equal(recon, sample_video[::2])
    assert duration == pytest.approx(recon.shape[0] / 12.0)


def test_video_endframe_count_matches_num_frames(tokenizer, sample_video):
    markers = KairosTokenizer.encode_video(sample_video)
    ids = tokenizer._resolve_markers(markers)
    num_endframes = sum(1 for tid in ids if tid == tokenizer._endframe_id)
    assert num_endframes == sample_video.shape[0]


def test_video_rejects_inconsistent_frame_shapes(tokenizer):
    """Frames of different H can be *encoded* independently, but decode_video must reject."""
    frame1 = np.zeros((2, 3, 3), dtype=np.uint8)
    frame2 = np.zeros((4, 3, 3), dtype=np.uint8)
    markers = (
        KairosTokenizer._encode_frame_rows(frame1)
        + [("marker", "<ENDFRAME>")]
        + KairosTokenizer._encode_frame_rows(frame2)
        + [("marker", "<ENDFRAME>")]
    )
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.VIDEO, markers)])
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    with pytest.raises(ValueError):
        tokenizer.decode_video(decoded[0].data)


# ========================= AUDIO: periodic <TICK>, duration from tick =========================
def test_audio_roundtrip_and_duration(tokenizer, sample_audio):
    markers = KairosTokenizer.encode_audio(sample_audio, tick_samples=16_000)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.AUDIO, markers)])
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    recon, duration = tokenizer.decode_audio(decoded[0].data, tick_samples=16_000)
    assert len(recon) == len(sample_audio)
    assert duration == pytest.approx(len(sample_audio) / KairosTokenizer.AUDIO_SAMPLE_RATE)
    max_step = 2 / (256**2 - 1)  # 16-bit place-value quantization over [-1, 1]
    assert np.max(np.abs(recon - sample_audio)) < max_step + 1e-6


def test_audio_tick_count(tokenizer, sample_audio):
    markers = KairosTokenizer.encode_audio(sample_audio, tick_samples=16_000)
    ids = tokenizer._resolve_markers(markers)
    num_ticks = sum(1 for tid in ids if tid == tokenizer._tick_id)
    assert num_ticks == -(-len(sample_audio) // 16_000)  # ceil division


# ======================= LIDAR: <PTSEP>, fixed quantization bounds ========================
def test_lidar_roundtrip_within_fixed_range_precision(tokenizer, sample_lidar):
    markers = KairosTokenizer.encode_lidar(sample_lidar)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.LIDAR, markers)])
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    recon = tokenizer.decode_lidar(decoded[0].data)
    assert recon.shape == sample_lidar.shape
    # quantization step over the fixed xyz range
    xyz_lo, xyz_hi = KairosTokenizer.LIDAR_XYZ_RANGE
    max_step = (xyz_hi - xyz_lo) / (256**2 - 1)  # 16-bit place-value quantization
    assert np.max(np.abs(recon[:, :3] - sample_lidar[:, :3])) <= max_step + 1e-3


def test_lidar_rejects_wrong_shape():
    with pytest.raises(ValueError):
        KairosTokenizer.encode_lidar(np.zeros((10, 3), dtype=np.float32))


# ==================== encode_multimodal / decode_multimodal: mixed seqs =====================
def test_mixed_text_image_video_audio(tokenizer, sample_image, sample_video, sample_audio):
    segs = [
        MultimodalSegment(Modality.TEXT, b"a scene:"),
        MultimodalSegment(Modality.IMAGE, KairosTokenizer.encode_image(sample_image)),
        MultimodalSegment(Modality.VIDEO, KairosTokenizer.encode_video(sample_video)),
        MultimodalSegment(Modality.AUDIO, KairosTokenizer.encode_audio(sample_audio)),
    ]
    out = tokenizer.encode_multimodal(segs)
    assert out["input_ids"].shape == out["modality_ids"].shape
    assert set(out["modality_ids"].tolist()) == {
        int(Modality.TEXT),
        int(Modality.IMAGE),
        int(Modality.VIDEO),
        int(Modality.AUDIO),
    }


def test_multimodal_padding_uses_text_modality(tokenizer, sample_image):
    markers = KairosTokenizer.encode_image(sample_image)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.IMAGE, markers)], max_len=2048)
    assert out["input_ids"].shape[0] == 2048
    pad_region_mod = out["modality_ids"][-10:]
    assert torch.all(pad_region_mod == int(Modality.TEXT))
    assert torch.all(out["input_ids"][-10:] == tokenizer.pad_token_id)


def test_multimodal_truncation(tokenizer):
    long_text = "a" * 1000
    segs = [MultimodalSegment(Modality.TEXT, long_text.encode("utf-8"))]
    out = tokenizer.encode_multimodal(segs, max_len=32)
    assert out["input_ids"].shape[0] == 32
    assert out["modality_ids"].shape[0] == 32


def test_empty_segments_returns_empty_tensors(tokenizer):
    out = tokenizer.encode_multimodal([])
    assert out["input_ids"].shape[0] == 0
    assert out["modality_ids"].shape[0] == 0


def test_image_decode_rejects_stream_with_no_endline(tokenizer):
    with pytest.raises(ValueError, match="no <ENDLINE> markers"):
        tokenizer.decode_image([])


def test_image_decode_rejects_width_not_divisible_by_channels(tokenizer, sample_image):
    markers = KairosTokenizer.encode_image(sample_image)
    ids = tokenizer._resolve_markers(markers)
    with pytest.raises(ValueError, match="not divisible by channels"):
        tokenizer.decode_image(ids, channels=5)


def test_encode_video_rejects_bad_dtype():
    frames = np.zeros((2, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8 array"):
        KairosTokenizer.encode_video(frames)


def test_video_decode_rejects_stream_with_no_endframe(tokenizer):
    with pytest.raises(ValueError, match="no <ENDFRAME> markers"):
        tokenizer.decode_video([])


def test_video_decode_keeps_trailing_frame_without_final_endframe_marker(tokenizer, sample_video):
    markers = KairosTokenizer.encode_video(sample_video)
    ids = tokenizer._resolve_markers(markers)
    ids_no_trailing_marker = ids[:-1]  # drop the last <ENDFRAME>
    frames, _ = tokenizer.decode_video(ids_no_trailing_marker)
    assert frames.shape[0] == sample_video.shape[0]


def test_encode_audio_rejects_bad_dtype():
    waveform = np.zeros(100, dtype=np.float64)
    with pytest.raises(ValueError, match="float32 waveform"):
        KairosTokenizer.encode_audio(waveform)


def test_audio_decode_keeps_trailing_samples_without_final_tick_marker(tokenizer, sample_audio):
    markers = KairosTokenizer.encode_audio(sample_audio, tick_samples=4000)
    ids = tokenizer._resolve_markers(markers)
    ids_no_trailing_marker = ids[:-1]  # drop the last <TICK>
    waveform, _ = tokenizer.decode_audio(ids_no_trailing_marker)
    assert waveform.shape[0] == sample_audio.shape[0]


def test_decode_lidar_rejects_payload_not_multiple_of_eight(tokenizer):
    ids = tokenizer._bytes_to_ids(b"\x00\x01\x02")  # 3 bytes, not a multiple of 4 channels x 2 bytes
    with pytest.raises(ValueError, match="not a multiple of 8"):
        tokenizer.decode_lidar(ids)


def test_encode_multimodal_wraps_segment_in_channel_tags(tokenizer, sample_image):
    markers = KairosTokenizer.encode_image(sample_image)
    seg = MultimodalSegment(Modality.IMAGE, markers, channel="R")
    out = tokenizer.encode_multimodal([seg])
    tokens = tokenizer.convert_ids_to_tokens(out["input_ids"].tolist())
    assert tokens[0] == "<IMG>"
    assert tokens[1] == "<R>"
    assert "</R>" in tokens
    assert tokens[-1] == "</IMG>"


def test_decode_multimodal_skips_unrecognized_leading_tokens(tokenizer):
    stray_id = tokenizer.convert_tokens_to_ids("<SEP>")
    text_out = tokenizer.encode_multimodal([MultimodalSegment(Modality.TEXT, b"hi")])
    ids = [stray_id, stray_id] + text_out["input_ids"].tolist()
    segments = tokenizer.decode_multimodal(torch.tensor(ids))
    assert len(segments) == 1
    assert segments[0].modality is Modality.TEXT
    assert segments[0].data == b"hi"


def test_octet_family_ids_shape_and_range(tokenizer):
    assert tokenizer.octet_family_ids.shape == (len(tokenizer),)
    assert tokenizer.octet_family_ids.max().item() == KairosTokenizer.NUM_OCTET_FAMILIES - 1


def test_octet_family_ids_distinguish_rgb_and_place_value(tokenizer):
    # R, G, B and audio hi/lo must map to different family ids, matching their distinct token blocks
    r_id = tokenizer._bytes_to_ids(bytes([0]), "IMG0")[0]
    g_id = tokenizer._bytes_to_ids(bytes([0]), "IMG1")[0]
    aud_hi_id = tokenizer._bytes_to_ids(bytes([0]), "AUD0")[0]
    aud_lo_id = tokenizer._bytes_to_ids(bytes([0]), "AUD1")[0]
    families = tokenizer.octet_family_ids[[r_id, g_id, aud_hi_id, aud_lo_id]]
    assert len(set(families.tolist())) == 4


def test_octet_family_ids_text_bytes_are_family_zero(tokenizer):
    text_id = tokenizer.convert_tokens_to_ids(chr(65))
    assert tokenizer.octet_family_ids[text_id].item() == 0
