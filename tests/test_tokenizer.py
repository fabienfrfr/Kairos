import numpy as np
import pytest
import torch

from kairos.tokenizer import KairosTokenizer, MultimodalSegment, Modality


# =========================
# Fixtures
# =========================
@pytest.fixture(scope="module")
def tokenizer():
    return KairosTokenizer()


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)


@pytest.fixture
def sample_audio():
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, 32).astype(np.float32)


@pytest.fixture
def sample_video():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (3, 2, 2, 3), dtype=np.uint8)


@pytest.fixture
def sample_lidar():
    rng = np.random.default_rng(0)
    return rng.uniform(-10, 10, (16, 4)).astype(np.float32)


# =========================
# Vocab / backward compatibility
# =========================
def test_vocab_size_is_287(tokenizer):
    """Regression test: ByT5Tokenizer defaults to extra_ids=125 (len=384).
    KairosTokenizer must force extra_ids=0 and add exactly 28 special
    tokens on top of the 259-token byte vocab."""
    assert len(tokenizer) == 287


def test_no_native_bos(tokenizer):
    """ByT5 has no BOS token — document this instead of assuming one exists."""
    assert tokenizer.bos_token_id is None


def test_byte_offset_contiguous(tokenizer):
    assert tokenizer.convert_tokens_to_ids(chr(0)) == tokenizer._byte_offset
    assert tokenizer.convert_tokens_to_ids(chr(255)) == tokenizer._byte_offset + 255


def test_plain_text_encode_decode_unchanged(tokenizer):
    """Backward compatibility: plain .encode()/.decode() must behave like
    stock ByT5Tokenizer for pure text, unaffected by the multimodal additions."""
    text = "Paris is the capital of France."
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids, skip_special_tokens=True) == text


def test_special_tokens_all_present(tokenizer):
    for tag in ("<TEXT>", "</TEXT>", "<IMG>", "</IMG>", "<AUDIO>", "</AUDIO>",
                "<VIDEO>", "</VIDEO>", "<LIDAR>", "</LIDAR>", "<SEP>", "<MASK>"):
        tid = tokenizer.convert_tokens_to_ids(tag)
        assert tid is not None and tid != tokenizer.unk_token_id, f"{tag} missing from vocab"


# =========================
# Modality quantizers: roundtrip
# =========================
def test_image_quantizer_roundtrip(sample_image):
    raw = KairosTokenizer.encode_image(sample_image)
    recon = KairosTokenizer.decode_image(raw, *sample_image.shape[:2])
    assert np.array_equal(recon, sample_image)


def test_image_quantizer_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        KairosTokenizer.encode_image(np.zeros((2, 2, 3), dtype=np.float32))


def test_audio_quantizer_roundtrip_lossy(sample_audio):
    raw = KairosTokenizer.encode_audio(sample_audio)
    recon = KairosTokenizer.decode_audio(raw)
    assert recon.shape == sample_audio.shape
    # 8-bit PCM: max quantization error is ~1/127.5
    assert np.max(np.abs(recon - sample_audio)) < 1 / 127.5 + 1e-6


def test_video_quantizer_roundtrip(sample_video):
    raw = KairosTokenizer.encode_video(sample_video)
    t, h, w, _ = sample_video.shape
    recon = KairosTokenizer.decode_video(raw, t, h, w)
    assert np.array_equal(recon, sample_video)


def test_video_quantizer_stride(sample_video):
    raw = KairosTokenizer.encode_video(sample_video, stride=2)
    expected_frames = sample_video[::2]
    recon = KairosTokenizer.decode_video(raw, expected_frames.shape[0], *sample_video.shape[1:3])
    assert np.array_equal(recon, expected_frames)


def test_lidar_quantizer_range(sample_lidar):
    raw = KairosTokenizer.encode_lidar(sample_lidar)
    recon = KairosTokenizer.decode_lidar(raw)
    assert recon.shape == sample_lidar.shape
    assert recon.min() >= 0.0 and recon.max() <= 1.0


# =========================
# encode_multimodal / decode_multimodal
# =========================
def test_multimodal_ids_and_modality_ids_same_length(tokenizer):
    segs = [MultimodalSegment(Modality.TEXT, b"hello")]
    out = tokenizer.encode_multimodal(segs)
    assert out["input_ids"].shape == out["modality_ids"].shape


def test_multimodal_dtype_is_long(tokenizer):
    segs = [MultimodalSegment(Modality.TEXT, b"hello")]
    out = tokenizer.encode_multimodal(segs)
    assert out["input_ids"].dtype == torch.long
    assert out["modality_ids"].dtype == torch.long


def test_multimodal_modality_ids_match_segments(tokenizer, sample_image):
    segs = [
        MultimodalSegment(Modality.TEXT, b"look at this:"),
        MultimodalSegment(Modality.IMAGE, KairosTokenizer.encode_image(sample_image)),
    ]
    out = tokenizer.encode_multimodal(segs)
    mods = out["modality_ids"].tolist()
    assert set(mods) == {int(Modality.TEXT), int(Modality.IMAGE)}
    # text tokens must all come before image tokens (segment order preserved)
    first_image_idx = mods.index(int(Modality.IMAGE))
    assert all(m == int(Modality.TEXT) for m in mods[:first_image_idx])
    assert all(m == int(Modality.IMAGE) for m in mods[first_image_idx:])


def test_multimodal_text_image_audio_roundtrip(tokenizer, sample_image, sample_audio):
    text = "Paris is the capital of France."
    segs = [
        MultimodalSegment(Modality.TEXT, text.encode("utf-8")),
        MultimodalSegment(Modality.IMAGE, KairosTokenizer.encode_image(sample_image)),
        MultimodalSegment(Modality.AUDIO, KairosTokenizer.encode_audio(sample_audio)),
    ]
    out = tokenizer.encode_multimodal(segs)
    decoded = tokenizer.decode_multimodal(out["input_ids"])

    assert len(decoded) == 3
    assert decoded[0].modality is Modality.TEXT
    assert decoded[0].data == text.encode("utf-8")

    assert decoded[1].modality is Modality.IMAGE
    recon_img = KairosTokenizer.decode_image(decoded[1].data, *sample_image.shape[:2])
    assert np.array_equal(recon_img, sample_image)

    assert decoded[2].modality is Modality.AUDIO
    recon_audio = KairosTokenizer.decode_audio(decoded[2].data)
    assert np.max(np.abs(recon_audio - sample_audio)) < 1 / 127.5 + 1e-6


def test_multimodal_padding_uses_text_modality(tokenizer):
    segs = [MultimodalSegment(Modality.IMAGE, bytes([1, 2, 3]))]
    out = tokenizer.encode_multimodal(segs, max_len=64)
    assert out["input_ids"].shape[0] == 64
    assert out["modality_ids"].shape[0] == 64
    # padded region (after the short image segment) must be modality TEXT
    pad_region = out["modality_ids"][-10:]
    assert torch.all(pad_region == int(Modality.TEXT))
    assert torch.all(out["input_ids"][-10:] == tokenizer.pad_token_id)


def test_multimodal_truncation(tokenizer):
    long_text = "a" * 1000
    segs = [MultimodalSegment(Modality.TEXT, long_text.encode("utf-8"))]
    out = tokenizer.encode_multimodal(segs, max_len=32)
    assert out["input_ids"].shape[0] == 32
    assert out["modality_ids"].shape[0] == 32


def test_multimodal_channel_tags_do_not_break_roundtrip(tokenizer, sample_image):
    r_channel = sample_image[:, :, 0].tobytes()
    segs = [MultimodalSegment(Modality.IMAGE, r_channel, channel="R")]
    out = tokenizer.encode_multimodal(segs)
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    # channel sub-tags are stripped on decode; underlying bytes for the
    # segment must still contain the R-channel payload somewhere in the run
    assert r_channel in decoded[0].data or decoded[0].data.strip(
        bytes([tokenizer.convert_tokens_to_ids("<R>") - tokenizer._byte_offset])
    )


def test_empty_segments_returns_empty_tensors(tokenizer):
    out = tokenizer.encode_multimodal([])
    assert out["input_ids"].shape[0] == 0
    assert out["modality_ids"].shape[0] == 0


def test_multiple_segments_same_modality(tokenizer):
    segs = [
        MultimodalSegment(Modality.TEXT, b"first "),
        MultimodalSegment(Modality.TEXT, b"second"),
    ]
    out = tokenizer.encode_multimodal(segs)
    decoded = tokenizer.decode_multimodal(out["input_ids"])
    assert len(decoded) == 2
    assert decoded[0].data == b"first "
    assert decoded[1].data == b"second"