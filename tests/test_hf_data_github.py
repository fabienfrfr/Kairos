import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from kairos import hf_data


def _zip_with_datasets(tmp_path: Path, top: str | None = "repo-datasets") -> Path:
    src = tmp_path / "src"
    for name, frame in {
        "keep-it-simple": pd.DataFrame([{"prompt": "Q:", "text": f"text {i}", "seed_data": "x"} for i in range(30)]),
        "keep-it-simple-multimodal": pd.DataFrame(
            [{"modality": "imu", "caption": "", "source": "s", "data": b"npz", "meta": None} for _ in range(10)]
        ),
    }.items():
        (src / name).mkdir(parents=True)
        frame.to_parquet(src / name / "train-00000-of-00001.parquet")
    archive = tmp_path / "branch.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in src.rglob("*"):
            if path.is_file():
                rel = path.relative_to(src)
                zf.write(path, Path(top) / rel if top else rel)
    return archive


def test_github_spec_downloads_once_and_feeds_both_datasets(tmp_path, monkeypatch):
    archive = _zip_with_datasets(tmp_path)
    spec = {"archive_url": archive.as_uri(), "cache_dir": str(tmp_path / "cache"), "text_pct": 100, "eval_pct": 0.1}
    train, held = hf_data.load_hf_examples(spec)
    modalities = {row["modality"] for row in train + held}
    assert modalities == {"text", "imu"} and len(train) + len(held) == 40

    def boom(*args, **kwargs):
        raise AssertionError("downloaded again")

    monkeypatch.setattr(hf_data, "fetch_archive", boom)
    again_train, again_held = hf_data.load_hf_examples(spec)
    assert len(again_train) + len(again_held) == 40


def test_refresh_redownloads_and_explicit_repos_win(tmp_path, monkeypatch):
    archive = _zip_with_datasets(tmp_path, top=None)
    calls = []
    real = hf_data.fetch_archive
    monkeypatch.setattr(hf_data, "fetch_archive", lambda url, root: calls.append(url) or real(url, root))
    spec = {"archive_url": archive.as_uri(), "cache_dir": str(tmp_path / "cache")}
    hf_data.fetch_github(None, spec["archive_url"], spec["cache_dir"])
    hf_data.fetch_github(None, spec["archive_url"], spec["cache_dir"])
    hf_data.fetch_github(None, spec["archive_url"], spec["cache_dir"], refresh=True)
    assert len(calls) == 2
    seen = []
    loader = lambda repo, split: seen.append(repo) or []
    hf_data.load_hf_examples({**spec, "text_repo": "my/own-text"}, loader=loader)
    assert seen[0] == "my/own-text" and seen[1].endswith("keep-it-simple-multimodal")


def test_load_github_datasets_returns_dataset_objects(tmp_path, monkeypatch):
    archive = _zip_with_datasets(tmp_path)
    monkeypatch.setattr(hf_data, "github_archive_url", lambda spec: archive.as_uri())
    out = hf_data.load_github_datasets("me/repo@datasets", cache_dir=tmp_path / "cache")
    assert len(out["text"]) == 30 and len(out["multimodal"]) == 10
    assert set(out["text"].column_names) >= {"prompt", "text"}


def test_missing_source_and_bad_spec_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="archive_url"):
        hf_data.fetch_github(None, None, tmp_path)
    with pytest.raises(ValueError, match="owner/repo"):
        hf_data.github_archive_url("nope")


def test_github_sample_config_points_at_the_branch_and_matches_the_local_one():
    root = Path(__file__).parents[1] / "configs"
    github = yaml.safe_load((root / "pretrain_sample_github.yaml").read_text())
    local = yaml.safe_load((root / "pretrain_sample.yaml").read_text())
    assert github["hf_data"]["github"].endswith("@datasets")
    assert {k: v for k, v in github.items() if k != "hf_data"} == {k: v for k, v in local.items() if k != "hf_data"}
