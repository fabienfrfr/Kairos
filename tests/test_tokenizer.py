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


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ========================= Vocab / backward compatibility =========================
def test_vocab_size_is_291(tokenizer):
    """Regression: 259 base + 30 modality/channel tags == 289 (+SEP+MASK=291)."""
    assert len(tokenizer) == 291


def test_no_native_bos(tokenizer):
    assert tokenizer.bos_token_id is None


def test_byte_offset_contiguous(tokenizer):
    assert tokenizer.convert_tokens_to_ids(chr(0)) == tokenizer._byte_offset
    assert tokenizer.convert_tokens_to_ids(chr(255)) == tokenizer._byte_offset + 255


def test_byte_value_ids_are_shared_across_modalities(tokenizer):
    # by design: the *value* vocab is small and shared: a byte value 200 gets the same id whether
    # it's a text byte, an image R byte, or an audio hi byte - only octet_family_ids disambiguates
    ids = tokenizer._bytes_to_ids(bytes(range(256)))
    assert len(set(ids)) == 256


def test_place_value_bytes_use_distinct_families_not_a_duplicated_one(tokenizer):
    # the hi and lo byte of a 16-bit sample must get distinct family ids (e.g. "LLRR" -> distinct
    # L1/L2/R1/R2 families, not the same "L"/"R" family repeated)
    markers = KairosTokenizer.encode_audio(np.zeros(4, dtype=np.float32), tick_samples=4)
    _, families = tokenizer._resolve_markers(markers)
    byte_families = [f for f in families if f != 0]
    assert byte_families[0] != byte_families[1]  # hi byte family != lo byte family


def test_rgb_channels_use_distinct_families(tokenizer, sample_image):
    markers = KairosTokenizer.encode_image(sample_image[:1])  # single row: R,G,B,R,G,B,...
    _, families = tokenizer._resolve_markers(markers)
    row_families = families[: sample_image.shape[1] * 3]
    assert row_families[0:3] == list(dict.fromkeys(row_families[0:3]))  # R,G,B all distinct
    assert len(set(row_families[0:3])) == 3


def test_action_and_state_use_disjoint_families(tokenizer):
    # action and state are semantically different control channels, must not share a family either
    act_markers = KairosTokenizer.encode_signal(np.zeros(2, dtype=np.float32), family="ACT")
    sta_markers = KairosTokenizer.encode_signal(np.zeros(2, dtype=np.float32), family="STA")
    _, act_families = tokenizer._resolve_markers(act_markers)
    _, sta_families = tokenizer._resolve_markers(sta_markers)
    act_family_set = {f for f in act_families if f}
    sta_family_set = {f for f in sta_families if f}
    assert act_family_set.isdisjoint(sta_family_set)


def test_signal_roundtrip_precision_beats_8bit(tokenizer):
    # 16-bit place-value quantization must resolve a step far finer than the old 8-bit scheme
    rng = np.random.default_rng(0)
    values = rng.uniform(-1, 1, 200).astype(np.float32)
    markers = KairosTokenizer.encode_signal(values, family="ACT", scale_factor=1)
    ids, _ = tokenizer._resolve_markers(markers)
    recon = tokenizer.decode_signal(ids)
    assert np.max(np.abs(recon - values)) < (2 / 255)  # much tighter than the old 8-bit step


def test_encode_signal_ticks_matches_flattened_encode_signal(tokenizer):
    rng = np.random.default_rng(0)
    values = rng.uniform(-1, 1, 30_000).astype(np.float32)
    ticks = KairosTokenizer.encode_signal_ticks(values, family="STA", tick_samples=8000, scale_factor=1)
    flat = KairosTokenizer.encode_signal(values, family="STA", tick_samples=8000, scale_factor=1)
    assert len(ticks) > 1  # multiple ticks given a signal longer than tick_samples
    joined = [m for tick in ticks for m in tick]
    assert joined == flat  # concatenating ticks must reproduce the flat encoding exactly


def test_signal_and_audio_share_the_small_value_vocab_but_not_family(tokenizer):
    # values are shared (that's the point - small vocab); family is what disambiguates them
    values = np.zeros(4, dtype=np.float32)
    signal_ids, signal_families = tokenizer._resolve_markers(KairosTokenizer.encode_signal(values, family="ACT"))
    audio_ids, audio_families = tokenizer._resolve_markers(KairosTokenizer.encode_audio(values))
    byte_signal_families = [f for f in signal_families if f != 0]
    byte_audio_families = [f for f in audio_families if f != 0]
    assert set(byte_signal_families).isdisjoint(byte_audio_families)


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


def test_image_scale_factor_downsamples_hw(tokenizer, sample_image):
    """scale_factor=2 block-means H,W down (with edge padding for odd dims), C untouched."""
    markers = KairosTokenizer.encode_image(sample_image, scale_factor=2)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.IMAGE, markers)])
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    recon = tokenizer.decode_image(decoded[0].data)
    h, w, c = sample_image.shape
    assert recon.shape == (-(-h // 2), -(-w // 2), c)


def test_image_endline_every_row(tokenizer, sample_image):
    """The whole point of the design: <ENDLINE> recurs locally, every W*C tokens,."""
    markers = KairosTokenizer.encode_image(sample_image)
    ids, _ = tokenizer._resolve_markers(markers)
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
    ids, _ = tokenizer._resolve_markers(markers)
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
def test_audio_decimated_by_pcm_scale_factor(tokenizer, sample_audio):
    """encode_audio block-means every PCM_SCALE_FACTOR raw samples into 1 by default."""
    markers = KairosTokenizer.encode_audio(sample_audio, tick_samples=16_000)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.AUDIO, markers)])
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    recon, _ = tokenizer.decode_audio(decoded[0].data, tick_samples=16_000)
    factor = KairosTokenizer.PCM_SCALE_FACTOR
    expected = sample_audio.reshape(-1, factor).mean(axis=1).astype(np.float32)
    assert len(recon) == len(expected) == len(sample_audio) // factor
    max_step = 2 / (256**2 - 1)  # 16-bit place-value quantization over [-1, 1]
    assert np.max(np.abs(recon - expected)) < max_step + 1e-6


def test_audio_scale_factor_one_disables_decimation(tokenizer):
    """scale_factor=1 is a no-op: bit-exact roundtrip, same as before decimation was added."""
    rng = np.random.default_rng(1)
    short_audio = rng.uniform(-1, 1, 512).astype(np.float32)
    markers = KairosTokenizer.encode_audio(short_audio, tick_samples=16_000, scale_factor=1)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.AUDIO, markers)])
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    recon, duration = tokenizer.decode_audio(decoded[0].data, tick_samples=16_000)
    assert len(recon) == len(short_audio)
    assert duration == pytest.approx(len(short_audio) / KairosTokenizer.AUDIO_SAMPLE_RATE)
    max_step = 2 / (256**2 - 1)
    assert np.max(np.abs(recon - short_audio)) < max_step + 1e-6


def test_audio_tick_count(tokenizer, sample_audio):
    markers = KairosTokenizer.encode_audio(sample_audio, tick_samples=16_000)
    ids, _ = tokenizer._resolve_markers(markers)
    num_ticks = sum(1 for tid in ids if tid == tokenizer._tick_id)
    decimated_len = len(sample_audio) // KairosTokenizer.PCM_SCALE_FACTOR
    assert num_ticks == -(-decimated_len // 16_000)  # ceil division


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


def test_lidar_scale_factor_keeps_every_nth_point(tokenizer, sample_lidar):
    markers = KairosTokenizer.encode_lidar(sample_lidar, scale_factor=3)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.LIDAR, markers)])
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    recon = tokenizer.decode_lidar(decoded[0].data)
    assert recon.shape[0] == len(sample_lidar[::3])


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
    ids, _ = tokenizer._resolve_markers(markers)
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
    ids, _ = tokenizer._resolve_markers(markers)
    ids_no_trailing_marker = ids[:-1]  # drop the last <ENDFRAME>
    frames, _ = tokenizer.decode_video(ids_no_trailing_marker)
    assert frames.shape[0] == sample_video.shape[0]


def test_encode_audio_rejects_bad_dtype():
    waveform = np.zeros(100, dtype=np.float64)
    with pytest.raises(ValueError, match="float32 waveform"):
        KairosTokenizer.encode_audio(waveform)


def test_audio_decode_keeps_trailing_samples_without_final_tick_marker(tokenizer, sample_audio):
    markers = KairosTokenizer.encode_audio(sample_audio, tick_samples=4000)
    ids, _ = tokenizer._resolve_markers(markers)
    ids_no_trailing_marker = ids[:-1]  # drop the last <TICK>
    waveform, _ = tokenizer.decode_audio(ids_no_trailing_marker)
    assert waveform.shape[0] == len(sample_audio) // KairosTokenizer.PCM_SCALE_FACTOR


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


def test_octet_family_id_range(tokenizer):
    markers = KairosTokenizer.encode_image(np.zeros((1, 1, 3), dtype=np.uint8))
    _, families = tokenizer._resolve_markers(markers)
    assert max(families) < KairosTokenizer.NUM_OCTET_FAMILIES
    assert min(families) >= 0


def test_octet_family_ids_distinguish_rgb_and_place_value(tokenizer):
    img_markers = KairosTokenizer.encode_image(np.zeros((1, 1, 3), dtype=np.uint8))
    _, img_families = tokenizer._resolve_markers(img_markers)
    aud_markers = KairosTokenizer.encode_audio(np.zeros(2, dtype=np.float32), tick_samples=2)
    _, aud_families = tokenizer._resolve_markers(aud_markers)
    r, g, b = img_families[0], img_families[1], img_families[2]
    aud_hi, aud_lo = [f for f in aud_families if f][0], [f for f in aud_families if f][1]
    assert len({r, g, b, aud_hi, aud_lo}) == 5


def test_octet_family_ids_text_bytes_are_family_zero(tokenizer):
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.TEXT, b"hi")])
    assert torch.all(out["octet_family_ids"] == 0)


def test_encode_multimodal_returns_octet_family_ids_same_shape_as_input_ids(tokenizer, sample_image):
    markers = KairosTokenizer.encode_image(sample_image)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.IMAGE, markers)])
    assert out["octet_family_ids"].shape == out["input_ids"].shape
    assert out["octet_family_ids"].max().item() > 0  # image bytes got a real (nonzero) family


# ========================= reconstruct_segments =========================
def test_reconstruct_segments_roundtrips_image(tokenizer, sample_image):
    markers = KairosTokenizer.encode_image(sample_image)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.IMAGE, markers)])
    result = tokenizer.reconstruct_segments(out["input_ids"])
    assert len(result) == 1
    assert result[0]["modality"] == "IMAGE"
    assert "error" not in result[0]
    assert np.array_equal(result[0]["decoded"], sample_image)


def test_reconstruct_segments_roundtrips_audio(tokenizer, sample_audio):
    markers = KairosTokenizer.encode_audio(sample_audio, scale_factor=1)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.AUDIO, markers)])
    result = tokenizer.reconstruct_segments(out["input_ids"])
    assert result[0]["modality"] == "AUDIO"
    assert "duration_s" in result[0]
    assert result[0]["decoded"].shape[0] == sample_audio.shape[0]


def test_reconstruct_segments_roundtrips_video(tokenizer, sample_video):
    """Regression: decode_video returns (frames, duration) - reconstruct_segments used to store
    the raw tuple under "decoded" instead of unpacking it (unlike the AUDIO branch just above),
    so any caller doing decoded.shape crashed with AttributeError on a tuple."""
    markers = KairosTokenizer.encode_video(sample_video)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.VIDEO, markers)])
    result = tokenizer.reconstruct_segments(out["input_ids"])
    assert result[0]["modality"] == "VIDEO"
    assert "error" not in result[0]
    assert "duration_s" in result[0]
    assert np.array_equal(result[0]["decoded"], sample_video)


def test_reconstruct_segments_roundtrips_control_state_and_action(tokenizer, rng):
    state = rng.uniform(-1, 1, 480).astype(np.float32)
    action = rng.uniform(-1, 1, 480).astype(np.float32)
    segs = [
        MultimodalSegment(Modality.STATE, KairosTokenizer.encode_signal(state, family="STA", scale_factor=1)),
        MultimodalSegment(Modality.ACTION, KairosTokenizer.encode_signal(action, family="ACT", scale_factor=1)),
    ]
    out = tokenizer.encode_multimodal(segs)
    result = tokenizer.reconstruct_segments(out["input_ids"])
    modalities = [r["modality"] for r in result]
    assert modalities == ["STATE", "ACTION"]  # order preserved: the alternation is visible here
    assert result[0]["decoded"].shape[0] == result[1]["decoded"].shape[0]  # same length back out


def test_reconstruct_segments_decodes_text_as_str(tokenizer):
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.TEXT, "hello world".encode("utf-8"))])
    result = tokenizer.reconstruct_segments(out["input_ids"])
    assert result[0]["modality"] == "TEXT"
    assert result[0]["decoded"] == "hello world"


def test_reconstruct_segments_reports_error_instead_of_raising_on_truncated_segment(tokenizer, sample_image):
    """A window boundary can truncate a segment mid-byte-plane; this must be surfaced as an
    error entry, not crash the whole diagnostic over one corrupted row."""
    markers = KairosTokenizer.encode_image(sample_image)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.IMAGE, markers)])
    truncated = out["input_ids"][:-5]  # cut off before the closing tag / mid-payload
    result = tokenizer.reconstruct_segments(truncated)
    assert result  # decode_multimodal still finds the (now unterminated) segment
    assert result[0]["decoded"] is None
    assert "error" in result[0]


def test_reconstruct_segments_n_tokens_matches_segment_length(tokenizer, sample_lidar):
    markers = KairosTokenizer.encode_lidar(sample_lidar)
    out = tokenizer.encode_multimodal([MultimodalSegment(Modality.LIDAR, markers)])
    result = tokenizer.reconstruct_segments(out["input_ids"])
    decoded_ids = tokenizer.decode_multimodal(out["input_ids"])
    assert result[0]["n_tokens"] == len(decoded_ids[0].data)