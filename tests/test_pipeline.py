import copy
import json
import math
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from kairos.dataset import pack_multimodal_data
from kairos.modeling import KairosConfig, KairosDiffusionLLM
from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig
from kairos.tokenizer import Modality
from kairos.utils import TrainingSummary, count_parameters


def make_example(modality, caption=None, source="test", **fields):
    """Build a generic-schema row: numpy-array fields go into `data`, everything else into."""
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


def test_training_converges_with_moe_enabled(tmp_path):
    # regression: use_moe=True used to NaN or block convergence; keep MoE small here
    model_config = KairosConfig(
        d_model=64, n_heads=4, n_layers=3, use_moe=True, num_local_experts=2, num_experts_per_tok=1
    )
    texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog " * 10}] * 8
    data_config = DataConfig(text_examples=texts, max_len=128, batch_size=2)
    train_config = TrainConfig(epochs=6, lr=1e-2, save_every=1000, run_dir=str(tmp_path / "run"), max_consecutive_nan=5)
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    logs = pipe.train(resume=False)

    assert pipe.skipped_nonfinite_steps == 0
    assert all(math.isfinite(row["loss"]) for row in logs)
    first_epochs_avg = sum(r["loss"] for r in logs if r["epoch"] <= 2) / sum(1 for r in logs if r["epoch"] <= 2)
    last_epochs_avg = sum(r["loss"] for r in logs if r["epoch"] > 4) / sum(1 for r in logs if r["epoch"] > 4)
    assert last_epochs_avg < first_epochs_avg


def test_training_converges_with_attnres_block_size_four(tmp_path):
    # cover attnres_block_size=4 (defaults to 1 elsewhere)
    model_config = KairosConfig(d_model=64, n_heads=4, n_layers=3, attnres_block_size=4)
    texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog " * 10}] * 8
    data_config = DataConfig(text_examples=texts, max_len=128, batch_size=2)
    train_config = TrainConfig(epochs=6, lr=1e-2, save_every=1000, run_dir=str(tmp_path / "run"), max_consecutive_nan=5)
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    logs = pipe.train(resume=False)

    assert pipe.skipped_nonfinite_steps == 0
    assert all(math.isfinite(row["loss"]) for row in logs)


def test_training_converges_with_moe_and_attnres_block_size_four(tmp_path):
    # regression: the exact MoE + attnres_block_size combination the user reported
    model_config = KairosConfig(
        d_model=64,
        n_heads=4,
        n_layers=3,
        use_moe=True,
        attnres_block_size=4,
        num_local_experts=2,
        num_experts_per_tok=1,
    )
    texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog " * 10}] * 8
    data_config = DataConfig(text_examples=texts, max_len=128, batch_size=2)
    train_config = TrainConfig(epochs=6, lr=1e-2, save_every=1000, run_dir=str(tmp_path / "run"), max_consecutive_nan=5)
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    logs = pipe.train(resume=False)

    assert pipe.skipped_nonfinite_steps == 0
    assert all(math.isfinite(row["loss"]) for row in logs)
    first_epochs_avg = sum(r["loss"] for r in logs if r["epoch"] <= 2) / sum(1 for r in logs if r["epoch"] <= 2)
    last_epochs_avg = sum(r["loss"] for r in logs if r["epoch"] > 4) / sum(1 for r in logs if r["epoch"] > 4)
    assert last_epochs_avg < first_epochs_avg


def test_training_converges_at_realistic_width_shallow_depth(tmp_path):
    # full-width (d_model=768, matching the real KairosConfig) but trimmed batch/epochs/seq-len to stay fast in CI
    model_config = KairosConfig(d_model=768, n_heads=12, n_layers=2)
    texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog "}] * 4
    data_config = DataConfig(text_examples=texts, max_len=32, batch_size=2)
    train_config = TrainConfig(epochs=2, lr=1e-3, save_every=1000, run_dir=str(tmp_path / "run"))
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    logs = pipe.train(resume=False)

    assert pipe.skipped_nonfinite_steps == 0
    assert all(math.isfinite(row["loss"]) for row in logs)
    first_epoch_avg = sum(r["loss"] for r in logs if r["epoch"] == 1) / sum(1 for r in logs if r["epoch"] == 1)
    last_epoch_avg = sum(r["loss"] for r in logs if r["epoch"] == 2) / sum(1 for r in logs if r["epoch"] == 2)
    assert last_epoch_avg < first_epoch_avg


def test_memory_gate_params_receive_gradient_during_train(tmp_path):
    model_config = KairosConfig(d_model=32, n_heads=2, n_layers=2, use_memory_gate=True)
    texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog"}] * 4
    data_config = DataConfig(text_examples=texts, max_len=32, batch_size=2)
    train_config = TrainConfig(epochs=2, lr=1e-2, save_every=1000, run_dir=str(tmp_path / "run"), max_consecutive_nan=5)
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    gate = pipe.model.memory_gate
    assert gate is not None
    before = [p.clone() for p in gate.parameters()]

    pipe.train(resume=False)

    after = [p.clone() for p in gate.parameters()]
    assert all(not torch.equal(b, a) for b, a in zip(before, after))


def test_memory_gate_is_noop_without_memory():
    from kairos.modeling import KairosMemoryGate

    gate = KairosMemoryGate(state_dim=16)
    state_t = torch.randn(3, 16)
    out = gate(state_t, memory=None)
    assert torch.equal(out, state_t)


def test_memory_gate_passes_through_single_memory_unchanged():
    from kairos.modeling import KairosMemoryGate

    gate = KairosMemoryGate(state_dim=16)
    state_t = torch.randn(3, 16)
    memory = torch.randn(1, 16)
    out = gate(state_t, memory=memory)
    assert torch.allclose(out, memory[0].expand_as(state_t))


def test_memory_gate_blends_across_multiple_memories():
    from kairos.modeling import KairosMemoryGate

    gate = KairosMemoryGate(state_dim=16)
    state_t = torch.randn(3, 16)
    memory = torch.randn(4, 16)
    out = gate(state_t, memory=memory)
    assert out.shape == state_t.shape
    assert not torch.allclose(out, memory[0].expand_as(state_t))
    assert not torch.equal(out, state_t)


def test_memory_gate_bottleneck_keeps_param_count_small():
    from kairos.modeling import KairosMemoryGate

    state_dim = 3872  # matches d_model=88, n_heads=4 in the notebook config
    gate = KairosMemoryGate(state_dim=state_dim)
    n_params = sum(p.numel() for p in gate.parameters())
    assert n_params < 200_000, f"memory gate should be a tiny bottleneck, got {n_params} params"


def test_memory_gate_is_shared_across_layers_and_scales():
    model_config = KairosConfig(d_model=32, n_heads=2, n_layers=3, use_memory_gate=True)
    model = KairosDiffusionLLM(model_config)
    n_gate_modules = sum(1 for m in model.modules() if type(m).__name__ == "KairosMemoryGate")
    assert n_gate_modules == 1


def test_memory_gate_blends_state_t_with_external_bank(tmp_path):
    from kairos.modeling import KairosMultiCache, gate_memory_bank

    model_config = KairosConfig(d_model=32, n_heads=2, n_layers=2, use_memory_gate=True)
    texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog"}] * 4
    data_config = DataConfig(text_examples=texts, max_len=32, batch_size=2)
    train_config = TrainConfig(epochs=1, lr=1e-2, save_every=1000, run_dir=str(tmp_path / "run"), max_consecutive_nan=5)
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    layer_idx = pipe.model.backbones[0].deltanet_layer_indices[0]
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    n_heads = model_config.num_attention_heads
    real_shape = (n_heads, head_dim, 2 * head_dim)

    bank = KairosMultiCache(model_config)
    bank.caches[0].ssm_caches[layer_idx] = torch.randn(5, *real_shape)

    out = gate_memory_bank(pipe.model, [bank], batch_size=2)
    assert out.caches[0].ssm_caches[layer_idx].shape == (2, *real_shape)

    logs = pipe.train(resume=False, memory_bank=bank)
    assert all(math.isfinite(row["loss"]) for row in logs)


def test_training_converges_on_easy_repeated_text(tmp_path, model_config):
    # a small, highly repetitive corpus should converge fast
    texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog"}] * 8
    data_config = DataConfig(text_examples=texts, max_len=32, batch_size=4)
    train_config = TrainConfig(
        epochs=15, lr=1e-2, save_every=1000, run_dir=str(tmp_path / "run"), max_consecutive_nan=5
    )
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    logs = pipe.train(resume=False)

    assert pipe.skipped_nonfinite_steps == 0
    assert all(math.isfinite(row["loss"]) for row in logs)

    first_epochs_avg = sum(r["loss"] for r in logs if r["epoch"] <= 2) / sum(1 for r in logs if r["epoch"] <= 2)
    last_epochs_avg = sum(r["loss"] for r in logs if r["epoch"] > 13) / sum(1 for r in logs if r["epoch"] > 13)
    assert last_epochs_avg < first_epochs_avg


def test_run_config_dict_contains_train_and_data_params(built_pipeline):
    d = built_pipeline.run_config_dict()
    assert d["train_config"]["lr"] == built_pipeline.train_config.lr
    assert d["data_config"]["batch_size"] == built_pipeline.data_config.batch_size
    assert "model_config" in d


def test_run_config_dict_sanitizes_raw_examples(built_pipeline):
    d = built_pipeline.run_config_dict()
    json.dumps(d)  # must be JSON-serializable, not raise
    assert "omitted" in str(d["data_config"].get("text_examples") or d["data_config"].get("multimodal_examples"))


def test_training_config_json_written_at_build(built_pipeline):
    p = Path(built_pipeline.train_config.run_dir) / "training_config.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert "train_config" in d and "model_config" in d


def test_checkpoint_embeds_train_config(built_pipeline):
    built_pipeline.train_config.epochs = 1
    built_pipeline.train(resume=False)
    ckpt = built_pipeline.load_checkpoint(str(built_pipeline.ckpt_dir / "best.pt"))
    assert ckpt["train_config"]["lr"] == built_pipeline.train_config.lr


def test_push_to_hub_respects_subfolder(built_pipeline, monkeypatch):
    mock_api = MagicMock()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: mock_api)

    built_pipeline.push_to_hub("user/repo", subfolder="run-42")

    paths = [c.kwargs.get("path_in_repo") for c in mock_api.upload_folder.call_args_list]
    paths += [c.kwargs.get("path_in_repo") for c in mock_api.upload_file.call_args_list]
    assert "run-42/checkpoints" in paths
    assert "run-42/tensorboard" in paths
    assert "run-42/README.md" in paths
    assert "run-42/training_config.json" in paths


def test_push_to_hub_without_subfolder_uses_repo_root(built_pipeline, monkeypatch):
    mock_api = MagicMock()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: mock_api)

    built_pipeline.push_to_hub("user/repo")

    paths = [c.kwargs.get("path_in_repo") for c in mock_api.upload_folder.call_args_list]
    paths += [c.kwargs.get("path_in_repo") for c in mock_api.upload_file.call_args_list]
    assert "checkpoints" in paths
    assert "README.md" in paths


def test_model_card_renders_from_template(built_pipeline):
    card = built_pipeline._model_card("user/my-model", "apache-2.0")
    assert "license: apache-2.0" in card
    assert "my-model" in card
    assert "{" not in card.split("```bibtex")[0]  # no leftover unformatted placeholders


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
    built_pipeline.global_step = -1  # force a visible change so
    ckpt = built_pipeline.load_checkpoint_from_hub("me/kairos-test")

    assert ckpt["step"] == step_before_save
    assert built_pipeline.global_step == step_before_save


def test_train_resumes_from_hub_when_no_local_checkpoint(tmp_path, model_config, text_examples, monkeypatch):
    # simulate a previous run's checkpoint on the hub
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

    assert pipe.global_step > 999  # started counting up from the


def test_train_starts_fresh_when_hub_has_no_checkpoint(built_pipeline, monkeypatch):
    def _raise(repo_id, filename):
        raise Exception("404")  # noqa: TRY002 — mirrors a real hf_hub_download failure

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _raise)
    built_pipeline.train_config.hub_repo_id = "me/kairos-test"
    logs = built_pipeline.train(resume=True)
    assert len(logs) > 0


def test_train_starts_fresh_when_local_checkpoint_is_incompatible(tmp_path, text_examples):
    # warn and start fresh
    old_config = KairosConfig(d_model=32, n_heads=4, n_layers=6, num_modalities=8, attnres_block_size=2)
    data_config = DataConfig(text_examples=text_examples, max_len=256, batch_size=2)
    old_pipe = KairosMultimodalPipeline(old_config, data_config, TrainConfig(epochs=1, run_dir=str(tmp_path / "old")))
    old_pipe.build()
    old_pipe.global_step = 999
    old_pipe._save(old_pipe.ckpt_dir / "last.pt", loss_val=0.1, epoch=2)

    new_config = KairosConfig(d_model=32, n_heads=4, n_layers=3, num_modalities=8, attnres_block_size=2)
    new_pipe = KairosMultimodalPipeline(new_config, data_config, TrainConfig(epochs=1, run_dir=str(tmp_path / "old")))
    new_pipe.build()  # same run_dir -> sees the

    with pytest.warns(UserWarning, match="incompatible"):
        logs = new_pipe.train(resume=True)

    assert len(logs) > 0
    assert new_pipe.global_step != 999  # did not pick up the


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
    assert all(math.isfinite(row["loss"]) for row in logs)  # the nan batch never made


def test_nan_log_captures_diagnostics(built_pipeline, monkeypatch):
    def _always_nan(model, batch, **kw):
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(built_pipeline.hf_trainer, "compute_loss", _always_nan)
    built_pipeline.train_config.max_consecutive_nan = 1000  # don't trip the circuit breaker

    with pytest.warns(UserWarning, match="non-finite"):
        built_pipeline.train(resume=False)

    assert len(built_pipeline.nan_log) == built_pipeline.skipped_nonfinite_steps
    assert all("step" in row and "loss" in row for row in built_pipeline.nan_log)


def test_nan_log_includes_trainer_diagnostics_for_real_forward_pass(built_pipeline, monkeypatch):
    # don't mock compute_loss: make forward emit NaN so the real check fires
    real_forward = built_pipeline.model.forward

    def _nan_forward(*args, **kw):
        out = real_forward(*args, **kw)
        out.logits[:] = float("nan")
        return out

    monkeypatch.setattr(built_pipeline.model, "forward", _nan_forward)
    built_pipeline.train_config.max_consecutive_nan = 1000

    with pytest.warns(UserWarning, match="non-finite"):
        built_pipeline.train(resume=False)

    assert built_pipeline.nan_log
    row = built_pipeline.nan_log[0]
    assert "logits_nan_frac" in row
    assert "batch_size" in row


def test_training_aborts_after_too_many_consecutive_nans(built_pipeline, monkeypatch):
    def _always_nan(model, batch, **kw):
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(built_pipeline.hf_trainer, "compute_loss", _always_nan)
    built_pipeline.train_config.max_consecutive_nan = 3

    with pytest.warns(UserWarning, match="non-finite"), pytest.raises(RuntimeError, match="consecutive non-finite"):
        built_pipeline.train(resume=False)

    # circuit breaker must fire at the configured limit
    assert built_pipeline.skipped_nonfinite_steps == 3


def test_consecutive_nan_counter_resets_on_a_good_batch(built_pipeline, monkeypatch):
    real_compute_loss = built_pipeline.hf_trainer.compute_loss
    calls = {"n": 0}

    def _nan_every_other(model, batch, **kw):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return torch.tensor(float("nan"), requires_grad=True)
        return real_compute_loss(model, batch, **kw)

    monkeypatch.setattr(built_pipeline.hf_trainer, "compute_loss", _nan_every_other)
    built_pipeline.train_config.max_consecutive_nan = 2  # would trip if the counter

    with pytest.warns(UserWarning, match="non-finite"):
        built_pipeline.train(resume=False)  # should complete without raising


def test_circuit_breaker_error_includes_nan_source(built_pipeline, monkeypatch):
    real_forward = built_pipeline.model.forward

    def _nan_forward(*args, **kw):
        out = real_forward(*args, **kw)
        out.logits[:] = float("nan")
        return out

    monkeypatch.setattr(built_pipeline.model, "forward", _nan_forward)
    built_pipeline.train_config.max_consecutive_nan = 2

    with (
        pytest.warns(UserWarning, match="non-finite"),
        pytest.raises(RuntimeError, match="First non-finite module") as exc_info,
    ):
        built_pipeline.train(resume=False)

    assert "lm_head" in str(exc_info.value) or "module" in str(exc_info.value)


def test_inspect_batch_reports_real_tokenized_input(built_pipeline):
    reports = built_pipeline.inspect_batch(n=1)

    assert len(reports) == built_pipeline.data_config.batch_size
    row = reports[0]
    assert row["seq_len"] == built_pipeline.data_config.max_len
    assert set(row["out_of_bounds"]) == {"token_ids", "modality_ids"}
    assert isinstance(row["modality_counts"], dict)
    assert isinstance(row["text_preview"], str)


def test_inspect_batch_flags_out_of_range_token_id(built_pipeline):
    real_batch = next(iter(built_pipeline.loader))
    real_batch["input_ids"][0, 0] = 99999  # far past vocab_size

    class _FakeLoader:
        def __iter__(self):
            yield real_batch

    built_pipeline.loader = _FakeLoader()
    reports = built_pipeline.inspect_batch(n=1)

    assert reports[0]["out_of_bounds"]["token_ids"] == [0]
    assert reports[0]["token_id_range"][1] == 99999


def test_inspect_batch_flags_out_of_range_modality_id(built_pipeline):
    real_batch = next(iter(built_pipeline.loader))
    real_batch["modality_ids"][0, 0] = 999  # far past num_modalities

    class _FakeLoader:
        def __iter__(self):
            yield real_batch

    built_pipeline.loader = _FakeLoader()
    reports = built_pipeline.inspect_batch(n=1)

    assert reports[0]["out_of_bounds"]["modality_ids"] == [0]


def test_inspect_batch_exposes_raw_numeric_ids(built_pipeline):
    reports = built_pipeline.inspect_batch(n=1)
    row = reports[0]

    assert isinstance(row["input_ids"], list)
    assert len(row["input_ids"]) == row["seq_len"]
    assert isinstance(row["modality_ids"], list)
    assert len(row["modality_ids"]) == row["seq_len"]
    assert isinstance(row["top_token_ids"], list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in row["top_token_ids"])
    assert set(row["max_repeat_run"]) == {"id", "length"}


def test_inspect_batch_flags_a_degenerate_repeated_run(built_pipeline):
    real_batch = next(iter(built_pipeline.loader))
    real_batch["input_ids"][0, 5:25] = 42  # 20 identical ids in a

    class _FakeLoader:
        def __iter__(self):
            yield real_batch

    built_pipeline.loader = _FakeLoader()
    reports = built_pipeline.inspect_batch(n=1)

    assert reports[0]["max_repeat_run"] == {"id": 42, "length": 20}


def test_inspect_batch_ignores_padding_tail_in_repeat_run(built_pipeline):
    # a padded short example must not be reported as a long repeat run
    real_batch = next(iter(built_pipeline.loader))
    seq_len = real_batch["input_ids"].size(1)
    real_len = 8
    real_batch["mask"][0] = 0
    real_batch["mask"][0, :real_len] = 1
    real_batch["input_ids"][0, real_len:] = built_pipeline.tokenizer.pad_token_id

    class _FakeLoader:
        def __iter__(self):
            yield real_batch

    built_pipeline.loader = _FakeLoader()
    reports = built_pipeline.inspect_batch(n=1)

    assert reports[0]["max_repeat_run"]["length"] < seq_len - real_len


def test_locate_nan_source_returns_none_before_any_skip(built_pipeline):
    assert built_pipeline.locate_nan_source() is None


def test_progress_callback_still_called_on_nonfinite_loss(built_pipeline, monkeypatch):
    def _always_nan(model, batch, **kw):
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(built_pipeline.hf_trainer, "compute_loss", _always_nan)

    seen_steps = []
    with pytest.warns(UserWarning, match="non-finite"):
        built_pipeline.train(resume=False, progress_callback=lambda step, total, loss: seen_steps.append(step))

    # every batch was skipped, but progress still advances
    assert len(seen_steps) == len(built_pipeline.loader) * built_pipeline.train_config.epochs


def test_last_ckpt_not_written_every_step(built_pipeline, monkeypatch):
    save_calls = []
    real_save = built_pipeline._save

    def _tracking_save(path, loss_val, epoch=1):
        save_calls.append(path.name)
        return real_save(path, loss_val, epoch)

    monkeypatch.setattr(built_pipeline, "_save", _tracking_save)
    built_pipeline.train_config.last_ckpt_every = 1000  # higher than total steps in
    built_pipeline.train(resume=False)

    # last.pt is only written at epoch boundaries, not every step
    last_pt_saves = save_calls.count("last.pt")
    assert last_pt_saves <= built_pipeline.train_config.epochs


def test_train_logs_periodic_eval_loss(tmp_path, model_config):
    texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog " * 10}] * 8
    eval_texts = [{"modality": "text", "text": "a completely different held-out sentence " * 8}] * 4
    data_config = DataConfig(text_examples=texts, max_len=64, batch_size=2)
    eval_data_config = DataConfig(text_examples=eval_texts, max_len=64, batch_size=2)
    train_config = TrainConfig(
        epochs=1,
        save_every=1000,
        eval_every=3,
        eval_batches=1,
        run_dir=str(tmp_path / "run"),
    )
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config, eval_data_config=eval_data_config)
    pipe.build()

    logs = pipe.train(resume=False)

    assert pipe.eval_loader is not None
    assert len(logs) > 3
    assert len(pipe.eval_log_rows) >= 2  # mid-training evals + a final one
    assert all(math.isfinite(row["loss"]) for row in pipe.eval_log_rows)
    assert all(row["batches"] <= 1 for row in pipe.eval_log_rows)
    assert pipe.best_eval_loss == min(row["loss"] for row in pipe.eval_log_rows)
    # mid-training evals land on the eval_every boundary (the final one may not)
    assert all(row["step"] % 3 == 0 for row in pipe.eval_log_rows[:-1])
    assert pipe.eval_log_rows[-1]["step"] == pipe.global_step


def test_overfit_test_drives_loss_down_and_restores_state(tmp_path, model_config):
    texts = [{"modality": "text", "text": "the quick brown fox jumps over the lazy dog " * 10}] * 16
    data_config = DataConfig(text_examples=texts, max_len=64, batch_size=2)
    train_config = TrainConfig(epochs=1, run_dir=str(tmp_path / "run"))
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    before = copy.deepcopy(pipe.model.state_dict())
    step_before = pipe.global_step
    loader_before = pipe.loader

    logs = pipe.overfit_test(n_examples=16, steps=60, lr=1e-2)

    assert len(logs) == 60
    assert logs[-1]["loss"] < logs[0]["loss"]  # memorization must be happening
    # non-destructive: weights/step/loader all restored for the real run
    for key, val in before.items():
        assert torch.equal(val, pipe.model.state_dict()[key])
    assert pipe.global_step == step_before
    assert pipe.loader is loader_before


def test_train_without_eval_config_skips_eval(tmp_path, model_config, text_examples):
    data_config = DataConfig(text_examples=text_examples, max_len=64, batch_size=2)
    train_config = TrainConfig(epochs=1, save_every=1000, eval_every=1, run_dir=str(tmp_path / "run"))
    pipe = KairosMultimodalPipeline(model_config, data_config, train_config)
    pipe.build()

    pipe.train(resume=False)

    assert pipe.eval_loader is None
    assert pipe.eval_log_rows == []


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
