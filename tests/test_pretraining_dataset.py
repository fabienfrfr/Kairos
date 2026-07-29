import numpy as np
import pytest
import torch

from kairos.dataset import KairosPretrainingDataset
from kairos.tokenizer import KairosTokenizer, Modality


@pytest.fixture(scope="module")
def tokenizer():
    return KairosTokenizer()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def all_kinds_examples(rng):
    return [
        {"kind": "text", "text": "Paris is the capital of France."},
        {"kind": "image_caption", "image": rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), "caption": "a dog"},
        {
            "kind": "audio_caption",
            "audio": rng.uniform(-1, 1, 4000).astype(np.float32),
            "sample_rate": 8000,
            "caption": "a bark",
        },
        {"kind": "video_caption", "video": rng.integers(0, 255, (4, 8, 8, 3), dtype=np.uint8), "caption": "running"},
        {"kind": "lidar", "points": rng.uniform(-10, 10, (50, 4)).astype(np.float32)},
        {"kind": "imu", "signal": rng.uniform(-5, 5, (50, 6)).astype(np.float32)},
        {
            "kind": "control",
            "action": rng.uniform(-1, 1, 200).astype(np.float32),
            "state": rng.uniform(-1, 1, 200).astype(np.float32),
            "sample_rate": 8000,
            "context": "method=bang-bang",
        },
    ]


# =========================
# text-only path unchanged
# =========================
def test_text_only_schema(tokenizer):
    ds = KairosPretrainingDataset(texts=["hello world"], tokenizer=tokenizer, max_len=64, stride=1)
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "modality_ids", "mask", "prompt_len"}
    assert set(item["modality_ids"].tolist()) <= {int(Modality.TEXT)}


def test_text_only_mask_matches_padding(tokenizer):
    ds = KairosPretrainingDataset(texts=["hi"], tokenizer=tokenizer, max_len=32, stride=1)
    item = ds[0]
    assert torch.equal(item["mask"] == 0, item["input_ids"] == tokenizer.pad_token_id)


# =========================
# multimodal path: all 6 kinds
# =========================
def test_all_kinds_build_without_error(tokenizer, all_kinds_examples):
    ds = KairosPretrainingDataset(multimodal_examples=all_kinds_examples, tokenizer=tokenizer, max_len=4096, stride=1)
    assert len(ds) >= len(all_kinds_examples)


def test_all_kinds_cover_expected_modalities(tokenizer, all_kinds_examples):
    ds = KairosPretrainingDataset(multimodal_examples=all_kinds_examples, tokenizer=tokenizer, max_len=4096, stride=1)
    seen = set()
    for i in range(len(ds)):
        seen |= set(ds[i]["modality_ids"][ds[i]["mask"] == 1].tolist())
    assert seen == {
        int(Modality.TEXT),
        int(Modality.IMAGE),
        int(Modality.AUDIO),
        int(Modality.VIDEO),
        int(Modality.LIDAR),
        int(Modality.STATE),
        int(Modality.ACTION),
    }


def test_lidar_has_no_text_caption(tokenizer, rng):
    ex = [{"kind": "lidar", "points": rng.uniform(-10, 10, (50, 4)).astype(np.float32)}]
    ds = KairosPretrainingDataset(multimodal_examples=ex, tokenizer=tokenizer, max_len=4096, stride=1)
    item = ds[0]
    assert set(item["modality_ids"][item["mask"] == 1].tolist()) == {int(Modality.LIDAR)}


def test_imu_uses_state_modality(tokenizer, rng):
    ex = [{"kind": "imu", "signal": rng.uniform(-5, 5, (50, 6)).astype(np.float32)}]
    ds = KairosPretrainingDataset(multimodal_examples=ex, tokenizer=tokenizer, max_len=4096, stride=1)
    item = ds[0]
    assert set(item["modality_ids"][item["mask"] == 1].tolist()) == {int(Modality.STATE)}


def test_control_uses_action_and_state_disjoint(tokenizer, rng):
    ex = [
        {
            "kind": "control",
            "action": rng.uniform(-1, 1, 200).astype(np.float32),
            "state": rng.uniform(-1, 1, 200).astype(np.float32),
            "sample_rate": 8000,
            "context": "bang-bang",
        }
    ]
    ds = KairosPretrainingDataset(multimodal_examples=ex, tokenizer=tokenizer, max_len=4096, stride=1)
    item = ds[0]
    mods = item["modality_ids"].tolist()
    action_pos = [i for i, m in enumerate(mods) if m == int(Modality.ACTION)]
    state_pos = [i for i, m in enumerate(mods) if m == int(Modality.STATE)]
    assert action_pos and state_pos
    assert set(action_pos).isdisjoint(state_pos)


def test_long_example_gets_chunked(tokenizer, rng):
    big_image = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    ex = [{"kind": "image_caption", "image": big_image, "caption": "big"}]
    ds = KairosPretrainingDataset(multimodal_examples=ex, tokenizer=tokenizer, max_len=256, stride=1)
    assert len(ds) > 1


def test_multimodal_padding_uses_text_modality(tokenizer, all_kinds_examples):
    ds = KairosPretrainingDataset(
        multimodal_examples=all_kinds_examples[:1], tokenizer=tokenizer, max_len=4096, stride=1
    )
    item = ds[0]
    pad_region = item["modality_ids"][item["mask"] == 0]
    assert torch.all(pad_region == int(Modality.TEXT))


def test_unknown_kind_raises(tokenizer):
    with pytest.raises(ValueError, match="unknown example kind"):
        KairosPretrainingDataset(multimodal_examples=[{"kind": "bogus"}], tokenizer=tokenizer, max_len=64, stride=1)


def test_multimodal_path_from_pt_file(tmp_path, tokenizer, all_kinds_examples):
    path = tmp_path / "mini.pt"
    torch.save(all_kinds_examples, path)
    ds = KairosPretrainingDataset(multimodal_path=str(path), tokenizer=tokenizer, max_len=512, stride=1)
    assert len(ds) > 0


def test_multimodal_dataset_creates_default_tokenizer_when_none_given(all_kinds_examples):
    ds = KairosPretrainingDataset(multimodal_examples=all_kinds_examples, tokenizer=None, max_len=512, stride=1)
    assert isinstance(ds.tokenizer, KairosTokenizer)
    assert len(ds) > 0


def test_text_dataset_defaults_to_cosmopedia_when_no_texts_given(tokenizer, monkeypatch):
    def fake_config_names(name):
        return ["sample"]

    def fake_load_dataset(name, config, split):
        from datasets import Dataset as HFDataset

        return HFDataset.from_dict({"text": ["Paris is the capital of France."]})

    def fake_concatenate(parts):
        return parts[0]

    monkeypatch.setattr("kairos.dataset.get_dataset_config_names", fake_config_names)
    monkeypatch.setattr("kairos.dataset.load_dataset", fake_load_dataset)
    monkeypatch.setattr("kairos.dataset.concatenate_datasets", fake_concatenate)
    ds = KairosPretrainingDataset(texts=None, tokenizer=tokenizer, max_len=32, stride=4)
    assert len(ds) > 0
