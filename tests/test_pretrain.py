"""Tests for the one-command pretraining path: HF data loading, config, `kairos pretrain`, make target."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import kairos.cli as cli_mod
from kairos.cli import _apply_run_name, _data_configs, _load_raw_config, _model_kwargs
from kairos.hf_data import load_hf_examples
from kairos.modeling import KairosConfig
from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

_ROOT = Path(__file__).parents[1]
_CONFIG = _ROOT / "configs" / "pretrain.yaml"


def _fake_loader(calls):
    def loader(repo, split):
        calls.append((repo, split))
        if repo == "text/repo":
            return [{"prompt": "a", "text": "b"}] * 4
        return [{"modality": "image_caption", "id": i} for i in range(6)]

    return loader


def test_load_hf_examples_merges_text_and_multimodal_and_splits():
    calls = []
    spec = {"text_repo": "text/repo", "multimodal_repo": "mm/repo", "text_pct": 50, "eval_pct": 10}

    train, evals = load_hf_examples(spec, loader=_fake_loader(calls))

    assert calls == [("text/repo", "train[:50%]"), ("mm/repo", "train")]
    assert len(train) + len(evals) == 10
    assert len(evals) == 1
    assert {"modality": "text", "text": "a b"} in train + evals


def test_model_kwargs_casts_scale_keys_and_defaults_vocab():
    out = _model_kwargs({"modality_scales": {"0": [0, 1], "8": [0]}})

    assert out["modality_scales"] == {0: [0, 1], 8: [0]}
    assert out["vocab_size"] > 0


def test_data_configs_fills_examples_from_hf_and_keeps_eval_unpacked(monkeypatch):
    monkeypatch.setattr(cli_mod, "load_hf_examples", lambda spec: ([{"m": 1}], [{"m": 2}]))
    cfg = {"hf_data": {"text_repo": "x"}, "data": {"batch_size": 4, "pack": True}}

    data, eval_data = _data_configs(cfg)

    assert data.multimodal_examples == [{"m": 1}] and data.pack is True
    assert eval_data.multimodal_examples == [{"m": 2}]
    assert eval_data.pack is False and eval_data.shuffle is False and eval_data.batch_size == 4


def test_pretrain_yaml_builds_valid_configs():
    cfg = _load_raw_config(_CONFIG)

    model = KairosConfig(**_model_kwargs(cfg["model"]))
    DataConfig(**cfg["data"])
    train = TrainConfig(**cfg["train"])

    assert model.use_moe and model.n_routed_experts == 7
    assert train.hub_repo_id == "ffurfaro/kairos"
    assert set(cfg["hf_data"]) <= {"multimodal_repo", "text_repo", "text_pct", "eval_pct", "seed"}


def _run_pretrain(monkeypatch, repo, args=()):
    pushed = []
    train_cfg = SimpleNamespace(hub_repo_id=repo, hub_private=True)
    pipe = SimpleNamespace(train_config=train_cfg, push_to_hub=lambda r, private: pushed.append((r, private)))
    monkeypatch.setattr(cli_mod, "build_pipeline", lambda config, overrides, run_name=None: pipe)
    monkeypatch.setattr(cli_mod, "train_with_progress", lambda p, force_restart, mo: [])
    result = CliRunner().invoke(cli_mod.app, ["pretrain", str(_CONFIG), *args])
    return result, pushed


def test_pretrain_pushes_model_after_training(monkeypatch):
    result, pushed = _run_pretrain(monkeypatch, "me/model")

    assert result.exit_code == 0, result.output
    assert pushed == [("me/model", True)]


@pytest.mark.parametrize(("repo", "args"), [("me/model", ["--no-push"]), (None, [])])
def test_pretrain_skips_push(monkeypatch, repo, args):
    result, pushed = _run_pretrain(monkeypatch, repo, args)

    assert result.exit_code == 0, result.output
    assert pushed == []


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_make_pretrain_runs_cli_with_default_config():
    cmd = ["make", "-n", "pretrain"]
    out = subprocess.run(cmd, check=False, cwd=_ROOT, capture_output=True, text=True)

    assert "kairos pretrain configs/pretrain.yaml" in out.stdout


def test_apply_run_name_sets_sibling_run_dir_and_hub_subfolder():
    cfg = {"train": {"run_dir": "checkpoints/kairos-multimodal/run_01"}}

    out = _apply_run_name(cfg, "d64-e7")

    assert out["train"]["run_dir"] == "checkpoints/kairos-multimodal/d64-e7"
    assert out["train"]["hub_subfolder"] == "d64-e7"


def test_apply_run_name_without_configured_run_dir_uses_default_parent():
    out = _apply_run_name({}, "exp1")

    assert out["train"]["run_dir"].endswith("/exp1")
    assert out["train"]["hub_subfolder"] == "exp1"


def test_pretrain_rejects_invalid_run_name(monkeypatch):
    result, _ = _run_pretrain(monkeypatch, "me/model", ["--name", "bad/name"])

    assert result.exit_code != 0


def test_pretrain_passes_run_name_to_pipeline(monkeypatch):
    seen = {}
    pipe = SimpleNamespace(train_config=SimpleNamespace(hub_repo_id=None, hub_private=False))

    def fake_build(config, overrides, run_name=None):
        seen["run_name"] = run_name
        return pipe

    monkeypatch.setattr(cli_mod, "build_pipeline", fake_build)
    monkeypatch.setattr(cli_mod, "train_with_progress", lambda p, force_restart, mo: [])
    result = CliRunner().invoke(cli_mod.app, ["pretrain", str(_CONFIG), "--name", "run-A"])

    assert result.exit_code == 0, result.output
    assert seen["run_name"] == "run-A"


@pytest.mark.parametrize(
    ("subfolder", "expected"), [("run-A", "run-A/checkpoints/last.pt"), (None, "checkpoints/last.pt")]
)
def test_hub_resume_reads_checkpoint_under_run_subfolder(subfolder, expected):
    asked = []
    fake = SimpleNamespace(
        train_config=SimpleNamespace(hub_subfolder=subfolder),
        load_checkpoint_from_hub=lambda repo, filename: asked.append(filename),
    )

    KairosMultimodalPipeline._try_resume_from_hub(fake, "me/model")

    assert asked == [expected]


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_make_pretrain_forwards_run_name():
    cmd = ["make", "-n", "pretrain", "NAME=run-A"]
    out = subprocess.run(cmd, check=False, cwd=_ROOT, capture_output=True, text=True)

    assert "--name run-A" in out.stdout
