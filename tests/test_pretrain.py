"""Tests for the one-command pretraining path: HF data loading, config, `kairos pretrain`, make target."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import kairos.cli as cli_mod
from kairos.cli import _apply_run_name, _data_configs, _load_raw_config, _model_kwargs, _resolve_config
from kairos.hf_data import load_hf_examples
from kairos.modeling import KairosConfig
from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

_ROOT = Path(__file__).parents[1]
_CONFIG = _ROOT / "kairos" / "configs" / "pretrain.yaml"


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

    assert "kairos pretrain pretrain" in out.stdout


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


def test_data_configs_keeps_inline_examples_when_hf_returns_nothing(monkeypatch):
    monkeypatch.setattr(cli_mod, "load_hf_examples", lambda spec: ([], []))
    inline = [{"modality": "text", "text": "x"}]
    cfg = {"hf_data": {"text_repo": None}, "data": {"text_examples": inline, "batch_size": 2}}

    data, eval_data = _data_configs(cfg)

    assert data.text_examples == inline
    assert eval_data is None


def test_minimal_yaml_builds_valid_configs_with_inline_data():
    cfg = _load_raw_config(_ROOT / "kairos" / "configs" / "minimal.yaml")

    data, eval_data = _data_configs(cfg)
    KairosConfig(**_model_kwargs(cfg["model"]))
    train = TrainConfig(**cfg["train"])

    assert len(data.text_examples) == 4 and eval_data is None
    assert train.compile_model is False and train.hub_repo_id is None


@pytest.mark.integration
def test_minimal_config_trains_end_to_end_on_cpu(tmp_path):
    args = ["pretrain", str(_ROOT / "kairos" / "configs" / "minimal.yaml"), "--no-push"]
    for key, value in {
        "run_dir": tmp_path / "run",
        "mae_epochs": 2,
        "transition_epochs": 2,
        "diffusion_epochs": 2,
    }.items():
        args += ["--set", f"train.{key}={value}"]

    result = CliRunner().invoke(cli_mod.app, args)

    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "checkpoints" / "best.pt").exists()


def test_resolve_config_prefers_existing_file_then_bundled_name(tmp_path):
    local = tmp_path / "minimal"
    local.write_text("model: {}\n", encoding="utf-8")

    assert _resolve_config(str(local)) == local
    assert _resolve_config("minimal").name == "minimal.yaml"
    assert _resolve_config("pretrain.yaml").name == "pretrain.yaml"


def test_resolve_config_unknown_name_lists_bundled_configs():
    with pytest.raises(typer.BadParameter, match="minimal, pretrain"):
        _resolve_config("nope")


def test_pretrain_accepts_bundled_config_name(monkeypatch):
    seen = {}
    pipe = SimpleNamespace(train_config=SimpleNamespace(hub_repo_id=None, hub_private=False))

    def fake_build(config, overrides, run_name=None):
        seen["config"] = config
        return pipe

    monkeypatch.setattr(cli_mod, "build_pipeline", fake_build)
    monkeypatch.setattr(cli_mod, "train_with_progress", lambda p, force_restart, mo: [])
    result = CliRunner().invoke(cli_mod.app, ["pretrain", "minimal", "--no-push"])

    assert result.exit_code == 0, result.output
    assert seen["config"].name == "minimal.yaml"


def test_pretrain_unknown_config_name_fails_cleanly():
    result = CliRunner().invoke(cli_mod.app, ["pretrain", "nope"])

    assert result.exit_code == 2
    assert "bundled config" in result.output


def test_init_config_prints_and_saves_bundled_config(tmp_path):
    runner = CliRunner()
    expected = (_ROOT / "kairos" / "configs" / "pretrain.yaml").read_text(encoding="utf-8")

    printed = runner.invoke(cli_mod.app, ["init-config", "pretrain"])
    saved = runner.invoke(cli_mod.app, ["init-config", "pretrain", "-o", str(tmp_path / "my.yaml")])

    assert printed.output == expected
    assert saved.exit_code == 0 and (tmp_path / "my.yaml").read_text(encoding="utf-8") == expected


def test_pyproject_ships_configs_and_cli_deps_in_core():
    import tomllib

    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(project["project"]["dependencies"])

    assert "configs/*.yaml" in project["tool"]["setuptools"]["package-data"]["kairos"]
    assert "typer" in deps and "pyyaml" in deps
    assert project["project"]["optional-dependencies"]["cli"] == []
