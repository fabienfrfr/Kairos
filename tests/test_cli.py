import json

import pytest

import kairos.cli as cli_mod
from kairos.cli import _apply_overrides, _load_raw_config, _looks_like_json


def test_load_raw_config_reads_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"train": {"lr": 0.001}}))

    assert _load_raw_config(path) == {"train": {"lr": 0.001}}


def test_load_raw_config_reads_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("train:\n  lr: 0.001\n")

    assert _load_raw_config(path) == {"train": {"lr": 0.001}}


@pytest.mark.parametrize("value", ["true", "false", "null", "3", "1.5", "-2"])
def test_looks_like_json_true_for_json_scalars(value):
    assert _looks_like_json(value) is True


@pytest.mark.parametrize("value", ["hello", "/tmp/my_run", "checkpoints/run_01"])
def test_looks_like_json_false_for_plain_strings(value):
    assert _looks_like_json(value) is False


def test_apply_overrides_sets_nested_value_with_type_coercion():
    cfg = {"train": {"lr": 3e-4}}

    result = _apply_overrides(cfg, ["train.lr=0.001", "train.compile_model=false"])

    assert result["train"]["lr"] == 0.001
    assert result["train"]["compile_model"] is False


def test_apply_overrides_creates_missing_section():
    cfg = {}

    result = _apply_overrides(cfg, ["data.batch_size=4"])

    assert result["data"]["batch_size"] == 4


def test_apply_overrides_keeps_plain_strings_as_strings():
    cfg = {}

    result = _apply_overrides(cfg, ["train.run_dir=/tmp/my_run"])

    assert result["train"]["run_dir"] == "/tmp/my_run"


def test_moe_bias_check_rejects_unknown_regime():
    from typer.testing import CliRunner

    result = CliRunner().invoke(cli_mod.app, ["moe-bias-check", "not_a_regime", "--rate", "1e-3"])

    assert result.exit_code != 0


def test_moe_bias_check_wires_run_and_report(monkeypatch):
    from typer.testing import CliRunner

    monkeypatch.setattr(cli_mod, "_run_moe_bias_check", lambda *a, **k: ([1.0, 0.5], [3.0]))
    monkeypatch.setattr(cli_mod, "_format_moe_report", lambda losses, usage: f"report:{losses}:{usage}")

    result = CliRunner().invoke(cli_mod.app, ["moe-bias-check", "curriculum", "--rate", "1e-3", "--steps", "2"])

    assert result.exit_code == 0
    assert "report:[1.0, 0.5]:[3.0]" in result.stdout
