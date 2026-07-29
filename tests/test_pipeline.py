import math
import os

import numpy as np
import pytest
import torch

from kairos.modeling import KairosConfig
from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig
from kairos.tokenizer import Modality


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def text_examples():
    return [
        {"kind": "text", "text": "Paris is the capital of France."},
        {"kind": "text", "text": "The Earth orbits the Sun."},
    ]


@pytest.fixture
def multimodal_examples(rng):
    return [
        {"kind": "image_caption", "image": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), "caption": "a red square"},
        {
            "kind": "audio_caption",
            "audio": rng.uniform(-1, 1, 2000).astype(np.float32),
            "sample_rate": 4000,
            "caption": "a beep",
        },
        {"kind": "lidar", "points": rng.uniform(-10, 10, (32, 4)).astype(np.float32)},
    ]


@pytest.fixture
def model_config():
    return KairosConfig(d_model=32, n_heads=4, n_layers=4, num_modalities=8, attnres_block_size=2)


@pytest.fixture
def built_pipeline(tmp_path, model_config, text_examples, multimodal_examples):
    data_config = DataConfig(
        text_examples=text_examples,
        multimodal_examples=multimodal_examples,
        max_len=256,
        batch_size=2,
    )
    train_config = TrainConfig(epochs=1, save_every=3, run_dir=str(tmp_path / "run"))
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()
    return pipe


def test_build_creates_all_components(built_pipeline):
    assert built_pipeline.model is not None
    assert built_pipeline.dataset is not None
    assert built_pipeline.loader is not None
    assert built_pipeline.optimizer is not None
    assert built_pipeline.scheduler is not None
    assert built_pipeline.hf_trainer is not None


def test_dataset_config_requires_data():
    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=2, num_modalities=8)
    pipe = KairosMultimodalPipeline(model_config, DataConfig(), TrainConfig(run_dir="unused"))
    with pytest.raises(ValueError):
        pipe.build()


def test_train_returns_log_rows(built_pipeline):
    logs = built_pipeline.train()
    assert len(logs) > 0
    assert all("loss" in row and "step" in row and "epoch" in row for row in logs)


def test_train_no_nan_losses(built_pipeline):
    logs = built_pipeline.train()
    assert all(not math.isnan(row["loss"]) for row in logs)


def test_train_produces_checkpoints(built_pipeline):
    built_pipeline.train()
    ckpts = os.listdir(built_pipeline.ckpt_dir)
    assert "best.pt" in ckpts
    assert any(f.startswith("step_") for f in ckpts)


def test_checkpoint_contains_config(built_pipeline):
    built_pipeline.train()
    ckpt = torch.load(built_pipeline.ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    assert "model_state" in ckpt
    assert "config" in ckpt
    assert ckpt["config"]["hidden_size"] == built_pipeline.model_config.hidden_size


def test_load_checkpoint_restores_step(built_pipeline):
    built_pipeline.train()
    step_before = built_pipeline.global_step
    ckpt_path = built_pipeline.ckpt_dir / "best.pt"

    data_config = built_pipeline.data_config
    train_config = built_pipeline.train_config
    fresh = KairosMultimodalPipeline(built_pipeline.model_config, data_config, train_config)
    fresh.build()
    assert fresh.global_step == 0
    ckpt = fresh.load_checkpoint(str(ckpt_path))
    assert fresh.global_step == ckpt["step"]
    assert fresh.global_step <= step_before


def test_per_modality_loss_keys_are_present_modalities(built_pipeline):
    built_pipeline.train()
    per_mod = built_pipeline.check_per_modality_loss(n_batches=10)
    assert set(per_mod.keys()) <= {m.name for m in Modality}
    assert len(per_mod) > 0
    assert all(not math.isnan(v) for v in per_mod.values())


def test_per_modality_loss_before_train_does_not_crash(model_config, text_examples, multimodal_examples, tmp_path):
    data_config = DataConfig(
        text_examples=text_examples, multimodal_examples=multimodal_examples, max_len=256, batch_size=2
    )
    train_config = TrainConfig(epochs=1, run_dir=str(tmp_path / "run2"))
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()
    per_mod = pipe.check_per_modality_loss(n_batches=1)
    assert isinstance(per_mod, dict)


def test_check_per_modality_before_build_raises():
    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=2, num_modalities=8)
    pipe = KairosMultimodalPipeline(
        model_config, DataConfig(text_examples=[{"kind": "text", "text": "hi"}]), TrainConfig(run_dir="unused")
    )
    with pytest.raises(RuntimeError):
        pipe.check_per_modality_loss()


def test_multimodal_path_variant(tmp_path, model_config, multimodal_examples):
    pt_path = tmp_path / "mini.pt"
    torch.save(multimodal_examples, pt_path)
    data_config = DataConfig(multimodal_path=str(pt_path), max_len=256, batch_size=2)
    train_config = TrainConfig(epochs=1, save_every=100, run_dir=str(tmp_path / "run3"))
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()
    logs = pipe.train()
    assert len(logs) > 0
