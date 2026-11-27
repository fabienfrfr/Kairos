import importlib.util
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mirror = _load("mirror_datasets_to_github")
fetch = _load("fetch_datasets")


def _write_source(folder: Path, rows: int, payload: int = 1000) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    table = pa.table({"modality": ["image_caption"] * rows, "data": [os.urandom(payload) for _ in range(rows)]})
    pq.write_table(table, folder / "data" / "train-00000-of-00001.parquet") if (
        folder / "data"
    ).mkdir() is None else None
    return folder


def _rows(folder: Path) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in mirror.parquet_files(folder))


def test_mirror_folder_reshards_below_the_cap_and_keeps_every_row(tmp_path):
    src = _write_source(tmp_path / "src", rows=3000)
    shards = mirror.mirror_folder(src, tmp_path / "out", max_bytes=400_000)
    assert len(shards) > 3 and all(p.stat().st_size <= 400_000 for p in shards)
    assert _rows(tmp_path / "out") == 3000
    original = pq.read_table(mirror.parquet_files(src)[0]).column("data")[0].as_py()
    assert pq.read_table(shards[0]).column("data")[0].as_py() == original


def test_mirror_folder_can_keep_only_the_first_rows(tmp_path):
    src = _write_source(tmp_path / "src", rows=120)
    mirror.mirror_folder(src, tmp_path / "out", max_bytes=10**7, max_rows=50)
    assert _rows(tmp_path / "out") == 50


def test_mirror_folder_without_parquet_explains_how_to_export(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="to_parquet"):
        mirror.mirror_folder(tmp_path / "empty", tmp_path / "out", 10**6)


def test_check_limits_rejects_files_above_the_hard_limit(tmp_path):
    (tmp_path / "big.bin").write_bytes(b"x" * 3 * 2**20)
    with pytest.raises(SystemExit, match="limit"):
        mirror.check_limits(tmp_path, hard_mb=2)
    assert mirror.check_limits(tmp_path, hard_mb=5) == pytest.approx(3.0)


def _built(tmp_path, text_rows=None):
    sources = {
        "keep-it-simple": _write_source(tmp_path / "hf" / "keep-it-simple", 40),
        "keep-it-simple-multimodal": _write_source(tmp_path / "hf" / "keep-it-simple-multimodal", 25),
    }
    out = tmp_path / "kairos-data"
    mirror.build(sources, out, max_file_mb=1, text_rows=text_rows)
    return out


def test_build_writes_the_layout_readme_and_the_text_sample(tmp_path):
    out = _built(tmp_path, text_rows=10)
    assert _rows(out / "keep-it-simple") == 10 and _rows(out / "keep-it-simple-multimodal") == 25
    assert "first 10 rows" in (out / "README.md").read_text()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_commit_creates_one_local_commit_without_a_remote(tmp_path):
    out = _built(tmp_path)
    mirror.commit(out, remote=None, push=False)
    log = subprocess.run(["git", "-C", str(out), "log", "--oneline"], capture_output=True, text=True, check=True).stdout
    assert len(log.splitlines()) == 1


def test_github_archive_url_accepts_owner_repo_and_ref():
    assert fetch.github_archive_url("me/data") == "https://codeload.github.com/me/data/zip/HEAD"
    assert fetch.github_archive_url("me/data@v1") == "https://codeload.github.com/me/data/zip/v1"
    with pytest.raises(ValueError, match="owner/repo"):
        fetch.github_archive_url("just-a-name")


def test_fetch_archive_extracts_a_mirror_zip_and_a_github_style_zip(tmp_path):
    out = _built(tmp_path)
    for name, archive in (("mirror", mirror.make_zip(out)), ("github", tmp_path / "gh.zip")):
        if name == "github":
            with zipfile.ZipFile(archive, "w") as zf:
                for path in out.rglob("*"):
                    if path.is_file():
                        zf.write(path, Path("repo-HEAD") / path.relative_to(out))
        dirs = fetch.fetch_archive(archive.resolve().as_uri(), tmp_path / f"root-{name}")
        assert _rows(dirs["text_repo"]) == 40 and _rows(dirs["multimodal_repo"]) == 25


def test_fetch_archive_rejects_unsafe_paths_and_missing_folders(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("repo-HEAD/../../evil.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        fetch.fetch_archive(bad.resolve().as_uri(), tmp_path / "root1")
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("repo-HEAD/README.md", "x")
    with pytest.raises(ValueError, match="expected folders"):
        fetch.fetch_archive(empty.resolve().as_uri(), tmp_path / "root2")
    assert urllib.request.urlopen is not None


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_push_creates_a_separate_branch_on_the_remote(tmp_path):
    out = _built(tmp_path)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    mirror.commit(out, remote=str(remote), push=True, branch="datasets")
    branches = subprocess.run(
        ["git", "--git-dir", str(remote), "branch", "--list"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert branches == ["datasets"]
    assert fetch.github_archive_url("me/Kairos@datasets").endswith("/zip/datasets")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_rerun_is_a_no_op_and_a_change_from_another_folder_fast_forwards(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    sources = {
        "keep-it-simple": _write_source(tmp_path / "hf" / "keep-it-simple", 40),
        "keep-it-simple-multimodal": _write_source(tmp_path / "hf" / "keep-it-simple-multimodal", 25),
    }

    def run(out, srcs):
        mirror.prepare_repo(out, str(remote), "datasets")
        mirror.build(srcs, out, max_file_mb=1, text_rows=None)
        return mirror.commit(out, str(remote), True, "datasets")

    def commits():
        return subprocess.run(
            ["git", "--git-dir", str(remote), "rev-list", "--count", "datasets"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    assert run(tmp_path / "a", sources) is True and commits() == "1"
    assert run(tmp_path / "a", sources) is False and commits() == "1"  # same data: nothing to commit
    sources["keep-it-simple"] = _write_source(tmp_path / "hf2" / "keep-it-simple", 55)
    assert run(tmp_path / "b", sources) is True and commits() == "2"  # fresh folder clones the branch, then updates it


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_default_remote_reads_origin_of_the_current_repository(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", "https://github.com/me/Kairos.git"], check=True)
    monkeypatch.chdir(repo)
    assert mirror.default_remote() == "https://github.com/me/Kairos.git"
    monkeypatch.chdir(tmp_path)
    assert mirror.default_remote() is None
