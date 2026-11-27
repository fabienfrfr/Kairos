import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from kairos.dataset import unpack_multimodal_data

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


readiness = _load("check_pretraining_readiness")
fetch = _load("fetch_datasets")


def test_synthetic_rows_follow_the_builder_schema_and_shapes():
    rows = readiness.synthetic_rows(2)
    assert set(rows) == set(readiness.MODALITIES)
    shapes = {
        "image_caption": {"image": (32, 32, 3)},
        "audio_caption": {"audio": (8000,)},
        "video_caption": {"video": (6, 16, 16, 3)},
        "lidar": {"points": (300, 4)},
        "imu": {"signal": (128, 3)},
        "control": {"state": (16000,), "action": (16000,)},
    }
    for modality, expected in shapes.items():
        row = rows[modality][0]
        assert {"modality", "caption", "source", "data", "meta"} <= set(row) and row["modality"] == modality
        arrays = unpack_multimodal_data(row["data"])
        assert {k: v.shape for k, v in arrays.items()} == expected
        assert all(np.isfinite(v).all() for v in arrays.values())
    assert rows["text"][0]["modality"] == "text" and "text" in rows["text"][0]


def test_default_scale_factors_make_images_video_and_lidar_fit_one_window():
    rows = readiness.synthetic_rows(2)
    plain = readiness.token_stats(rows)
    scaled = readiness.token_stats(rows, scale_factors=readiness.DEFAULT_SCALE_FACTORS)
    for name in ("image_caption", "video_caption", "lidar"):
        assert plain[name]["tokens_per_example"] > 1024 and scaled[name]["tokens_per_example"] < 1024
        assert scaled[name]["windows"] == scaled[name]["examples"]
    assert plain["control"]["tokens_per_example"] > 16000 > scaled["control"]["tokens_per_example"]


def test_sample_config_is_consistent_with_the_scripts_and_builds():
    from kairos.cli import _model_kwargs
    from kairos.modeling import KairosConfig
    from kairos.pipeline import DataConfig, TrainConfig

    cfg = yaml.safe_load((_ROOT / "configs" / "pretrain_sample.yaml").read_text())
    KairosConfig(**_model_kwargs(cfg["model"]))
    assert DataConfig(**cfg["data"]).modality_scale_factors == readiness.DEFAULT_SCALE_FACTORS
    assert TrainConfig(**cfg["train"]).lr == pytest.approx(0.008)
    dirs = fetch.local_dirs(Path("data"))
    assert {k: Path(cfg["hf_data"][k]) for k in fetch.REPOS} == dirs


def test_fetch_downloads_both_dataset_repos(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw))
    dirs = fetch.fetch(tmp_path, token="t")
    assert [c["repo_id"] for c in calls] == list(fetch.REPOS.values())
    assert all(c["repo_type"] == "dataset" and c["token"] == "t" for c in calls)
    assert dirs["text_repo"] == tmp_path / "keep-it-simple"


def _args(**kw):
    import argparse

    base = {
        "source": "synthetic",
        "per_modality": 2,
        "batch_size": 2,
        "lr": 0.008,
        "no_compile": True,
        "scale_factors": __import__("json").dumps(readiness.DEFAULT_SCALE_FACTORS),
        "run_dir": None,
        "max_per_modality": 5,
        "spec": None,
        "text_repo": None,
        "multimodal_repo": None,
        "batch_sizes": "1",
        "steps": 1,
    }
    return argparse.Namespace(**{**base, **kw})


@pytest.mark.integration
def test_train_summary_reports_source_device_and_all_checks(tmp_path):
    summary = readiness.run_train(_args(run_dir=str(tmp_path / "run")))
    assert summary["source"] == "synthetic" and summary["device"]["device"] in ("cpu", "cuda")
    assert (
        summary["skipped_nonfinite"] == 0
        and summary["checkpoint_reload_matches"]
        and summary["generate_returns_tokens"]
    )
    assert set(summary["ce_before"]) == set(summary["ce_after"]) == {"train", "heldout"}
    assert set(summary["modality_token_share"]) >= {"TEXT", "IMAGE", "AUDIO", "VIDEO", "LIDAR", "CONTROL"}


def test_real_rows_are_read_from_hub_layout_parquet_folders(tmp_path):
    pd = pytest.importorskip("pandas")
    rows = readiness.synthetic_rows(6)
    for name, frame in {
        "keep-it-simple": pd.DataFrame([{"prompt": "Q:", "text": r["text"]} for r in rows["text"]]),
        "keep-it-simple-multimodal": pd.DataFrame([r for m in readiness.MODALITIES if m != "text" for r in rows[m]]),
    }.items():
        (tmp_path / name).mkdir()
        frame.to_parquet(tmp_path / name / "train-00000-of-00001.parquet")
    spec = {
        "text_repo": str(tmp_path / "keep-it-simple"),
        "multimodal_repo": str(tmp_path / "keep-it-simple-multimodal"),
        "text_pct": 100,
        "eval_pct": 0.2,
        "seed": 0,
    }
    train, held = readiness.load_real_rows(spec, 4)
    groups = readiness.group_by_modality(train)
    assert set(groups) == set(readiness.MODALITIES) and all(len(v) <= 4 for v in groups.values()) and len(held) >= 2
    assert readiness.token_stats(groups, scale_factors=readiness.DEFAULT_SCALE_FACTORS)["image_caption"][
        "windows"
    ] == len(groups["image_caption"])


def test_bench_times_a_step_and_reports_finite_loss(tmp_path):
    pipe = readiness.build_pipeline_from_config(
        *readiness.load_rows(_args()), 1, readiness.DEFAULT_SCALE_FACTORS, str(tmp_path / "run"), 0.008, False
    )
    pipe.build()
    result = readiness.time_steps(pipe, 1, steps=1, warmup=1)
    assert (
        result["finite"]
        and result["sec_per_step"] > 0
        and result["tokens_per_sec"] > 0
        and result["peak_cuda_mb"] is None
    )
