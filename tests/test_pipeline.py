import copy
import json
import math
import os

import numpy as np
import pytest
import torch

from kairos.dataset import pack_multimodal_data
from kairos.modeling import KairosConfig
from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig
from kairos.tokenizer import Modality
from kairos.utils import TrainingSummary, count_parameters


def make_example(modality, caption=None, source="test", **fields):
    """Build a generic-schema row: numpy-array fields go into `data`, everything else into `meta`."""
    arrays = {k: v for k, v in fields.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in fields.items() if not isinstance(v, np.ndarray)}
    return {
        "modality": modality,
        "caption": caption,
        "source": source,
        "data": pack_multimodal_data(arrays) if arrays else None,
        "meta": json.dumps(meta) if meta else None,
    }


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def text_examples():
    return [
        {"modality": "text", "text": "Paris is the capital of France."},
        {"modality": "text", "text": "The Earth orbits the Sun."},
    ]


@pytest.fixture
def multimodal_examples(rng):
    return [
        make_example("image_caption", caption="a red square", image=rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)),
        make_example(
            "audio_caption",
            caption="a beep",
            audio=rng.uniform(-1, 1, 2000).astype(np.float32),
            sample_rate=4000,
        ),
        make_example("lidar", points=rng.uniform(-10, 10, (32, 4)).astype(np.float32)),
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
        model_config, DataConfig(text_examples=[{"modality": "text", "text": "hi"}]), TrainConfig(run_dir="unused")
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


# ------------------------------------------------------------------- summary
def test_summary_without_benchmark(built_pipeline):
    summary = built_pipeline.summary(benchmark=False)
    assert isinstance(summary, TrainingSummary)
    total_params, _ = count_parameters(built_pipeline.model)
    assert summary.total_params == total_params
    assert summary.steps_per_epoch == len(built_pipeline.loader)
    assert summary.total_steps == built_pipeline.train_config.epochs * len(built_pipeline.loader)
    assert summary.avg_step_time_sec is None
    assert summary.estimated_total_time_sec is None


def test_summary_with_benchmark_estimates_time(built_pipeline):
    summary = built_pipeline.summary(benchmark=True, n_bench_steps=1)
    assert summary.avg_step_time_sec is not None
    assert summary.avg_step_time_sec > 0
    assert summary.estimated_total_time_sec == pytest.approx(summary.avg_step_time_sec * summary.total_steps)


def test_summary_active_params_equals_total_when_dense(built_pipeline):
    summary = built_pipeline.summary(benchmark=False)
    assert built_pipeline.model_config.use_moe is False
    assert summary.active_params == summary.total_params


def test_summary_active_params_less_than_total_when_moe(tmp_path, text_examples):
    model_config = KairosConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        num_modalities=8,
        attnres_block_size=2,
        use_moe=True,
        n_routed_experts=4,
        num_local_experts=4,
        num_experts_per_tok=1,
        n_shared_experts=1,
    )
    data_config = DataConfig(text_examples=text_examples, max_len=256, batch_size=2)
    train_config = TrainConfig(epochs=1, run_dir=str(tmp_path / "run"))
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    summary = pipe.summary(benchmark=False)
    assert summary.active_params < summary.total_params


def test_summary_benchmark_does_not_change_model_or_optimizer_state(built_pipeline):
    before_model = {k: v.clone() for k, v in built_pipeline.model.state_dict().items()}
    before_optim = copy.deepcopy(built_pipeline.optimizer.state_dict())

    built_pipeline.summary(benchmark=True, n_bench_steps=2)

    after_model = built_pipeline.model.state_dict()
    for key, val in before_model.items():
        assert torch.equal(val, after_model[key]), f"param {key} changed after summary(benchmark=True)"

    after_optim = built_pipeline.optimizer.state_dict()
    assert list(before_optim["state"].keys()) == list(after_optim["state"].keys())


def test_summary_benchmark_does_not_advance_global_step(built_pipeline):
    step_before = built_pipeline.global_step
    built_pipeline.summary(benchmark=True, n_bench_steps=2)
    assert built_pipeline.global_step == step_before


def test_summary_before_build_raises():
    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=2, num_modalities=8)
    pipe = KairosMultimodalPipeline(
        model_config, DataConfig(text_examples=[{"modality": "text", "text": "hi"}]), TrainConfig(run_dir="unused")
    )
    with pytest.raises(RuntimeError):
        pipe.summary()


# --------------------------------------------------------------------- hub
def test_build_creates_hub_repo_when_configured(tmp_path, model_config, text_examples, monkeypatch):
    calls = []
    monkeypatch.setattr("huggingface_hub.HfApi.create_repo", lambda self, repo_id, **kw: calls.append((repo_id, kw)))

    data_config = DataConfig(text_examples=text_examples, max_len=256, batch_size=2)
    train_config = TrainConfig(
        epochs=1, run_dir=str(tmp_path / "run"), hub_repo_id="me/kairos-test", hub_push_every_ckpt=True
    )
    KairosMultimodalPipeline(model_config, data_config, train_config).build()

    assert calls == [("me/kairos-test", {"private": False, "exist_ok": True})]


def test_train_pushes_checkpoints_to_hub_at_save_every(tmp_path, model_config, text_examples, monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi.create_repo", lambda self, repo_id, **kw: None)
    pushed = []
    monkeypatch.setattr(
        "huggingface_hub.HfApi.upload_file",
        lambda self, path_or_fileobj, path_in_repo, repo_id: pushed.append(path_in_repo),
    )

    data_config = DataConfig(text_examples=text_examples, max_len=256, batch_size=2)
    train_config = TrainConfig(
        epochs=1, save_every=1, run_dir=str(tmp_path / "run"), hub_repo_id="me/kairos-test", hub_push_every_ckpt=True
    )
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()
    pipe.train()

    assert any(p.endswith("last.pt") for p in pushed)
    assert any("step_" in p for p in pushed)


def test_train_does_not_push_when_hub_push_disabled(built_pipeline, monkeypatch):
    pushed = []
    monkeypatch.setattr("huggingface_hub.HfApi.upload_file", lambda self, **kw: pushed.append(kw))
    built_pipeline.train_config.hub_repo_id = "me/kairos-test"  # set but hub_push_every_ckpt stays False
    built_pipeline.train()
    assert pushed == []


def test_load_checkpoint_from_hub_downloads_then_loads(built_pipeline, monkeypatch, tmp_path):
    ckpt_path = tmp_path / "last.pt"
    built_pipeline._save(ckpt_path, loss_val=0.5, epoch=1)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo_id, filename: str(ckpt_path))

    step_before_save = built_pipeline.global_step
    built_pipeline.global_step = -1  # force a visible change so we can assert the load actually applied
    ckpt = built_pipeline.load_checkpoint_from_hub("me/kairos-test")

    assert ckpt["step"] == step_before_save
    assert built_pipeline.global_step == step_before_save


def test_train_resumes_from_hub_when_no_local_checkpoint(tmp_path, model_config, text_examples, monkeypatch):
    # simulate a previous run's checkpoint living only on the hub, nothing local
    donor_config = DataConfig(text_examples=text_examples, max_len=256, batch_size=2)
    donor = KairosMultimodalPipeline(model_config, donor_config, TrainConfig(epochs=1, run_dir=str(tmp_path / "donor")))
    donor.build()
    hub_ckpt_path = tmp_path / "hub_last.pt"
    donor.global_step = 999
    donor._save(hub_ckpt_path, loss_val=0.1, epoch=2)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo_id, filename: str(hub_ckpt_path))

    data_config = DataConfig(text_examples=text_examples, max_len=256, batch_size=2)
    train_config = TrainConfig(epochs=2, run_dir=str(tmp_path / "run"), hub_repo_id="me/kairos-test")
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()
    pipe.train(resume=True)

    assert pipe.global_step > 999  # started counting up from the hub checkpoint's step, not from 0


def test_train_starts_fresh_when_hub_has_no_checkpoint(built_pipeline, monkeypatch):
    def _raise(repo_id, filename):
        raise Exception("404")  # noqa: TRY002 — mirrors a real hf_hub_download failure

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _raise)
    built_pipeline.train_config.hub_repo_id = "me/kairos-test"
    logs = built_pipeline.train(resume=True)
    assert len(logs) > 0


def test_train_starts_fresh_when_local_checkpoint_is_incompatible(
    tmp_path, text_examples
):  # a checkpoint saved with a different model_config (e.g. different n_layers/experts) must not
    # crash the run: warn and start fresh instead
    old_config = KairosConfig(d_model=32, n_heads=4, n_layers=6, num_modalities=8, attnres_block_size=2)
    data_config = DataConfig(text_examples=text_examples, max_len=256, batch_size=2)
    old_pipe = KairosMultimodalPipeline(old_config, data_config, TrainConfig(epochs=1, run_dir=str(tmp_path / "old")))
    old_pipe.build()
    old_pipe.global_step = 999
    old_pipe._save(old_pipe.ckpt_dir / "last.pt", loss_val=0.1, epoch=2)

    new_config = KairosConfig(d_model=32, n_heads=4, n_layers=3, num_modalities=8, attnres_block_size=2)
    new_pipe = KairosMultimodalPipeline(new_config, data_config, TrainConfig(epochs=1, run_dir=str(tmp_path / "old")))
    new_pipe.build()  # same run_dir -> sees the old, incompatible last.pt

    with pytest.warns(UserWarning, match="incompatible"):
        logs = new_pipe.train(resume=True)

    assert len(logs) > 0
    assert new_pipe.global_step != 999  # did not pick up the incompatible checkpoint's step


def test_train_skips_nonfinite_loss_batches(built_pipeline, monkeypatch):
    real_compute_loss = built_pipeline.hf_trainer.compute_loss
    calls = {"n": 0}

    def _flaky_compute_loss(model, batch, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return torch.tensor(float("nan"), requires_grad=True)
        return real_compute_loss(model, batch, **kw)

    monkeypatch.setattr(built_pipeline.hf_trainer, "compute_loss", _flaky_compute_loss)

    with pytest.warns(UserWarning, match="non-finite"):
        logs = built_pipeline.train(resume=False)

    assert built_pipeline.skipped_nonfinite_steps == 1
    assert all(math.isfinite(row["loss"]) for row in logs)  # the nan batch never made it into the logs


def test_train_does_not_step_optimizer_on_nonfinite_loss(built_pipeline, monkeypatch):
    before = {k: v.clone() for k, v in built_pipeline.model.state_dict().items()}

    def _always_nan(model, batch, **kw):
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(built_pipeline.hf_trainer, "compute_loss", _always_nan)

    with pytest.warns(UserWarning, match="non-finite"):
        built_pipeline.train(resume=False)

    after = built_pipeline.model.state_dict()
    for key, val in before.items():
        assert torch.equal(val, after[key]), f"param {key} changed despite every batch being non-finite"
