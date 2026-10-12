"""Tests for README arXiv badge and the token-driven `make publish` target."""

import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_ARXIV = "https://arxiv.org/abs/2610.12495"
_MAKE = shutil.which("make")


def _publish(tmp_path: Path, env_text: str | None, token: str | None):
    work = tmp_path / "proj"
    work.mkdir()
    shutil.copy(_ROOT / "Makefile", work / "Makefile")
    if env_text is not None:
        (work / ".env").write_text(env_text, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin"}
    if token:
        env["UV_PUBLISH_TOKEN"] = token
    cmd = [_MAKE, "-n", "publish"]
    return subprocess.run(cmd, check=False, cwd=work, env=env, capture_output=True, text=True)


def test_readme_has_arxiv_badge():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert f'<a href="{_ARXIV}">' in readme
    assert "img.shields.io/badge/arXiv-kairos-blueviolet.svg" in readme
    assert "XXXX.XXXXX" not in readme


@pytest.mark.skipif(_MAKE is None, reason="make not installed")
def test_publish_fails_without_token(tmp_path):
    out = _publish(tmp_path, None, None)
    assert "UV_PUBLISH_TOKEN missing" in out.stdout


@pytest.mark.skipif(_MAKE is None, reason="make not installed")
@pytest.mark.parametrize("from_dotenv", [True, False])
def test_publish_runs_with_token(tmp_path, from_dotenv):
    env_text = "UV_PUBLISH_TOKEN=pypi-test\n" if from_dotenv else None
    out = _publish(tmp_path, env_text, None if from_dotenv else "pypi-test")
    assert "uv publish" in out.stdout


def test_readme_paths_exist_and_status_is_honest():
    import re

    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    paths = re.findall(r"`((?:kairos|docs|configs|notebook|scripts)/[\w./-]+)`", readme)
    assert paths
    assert [p for p in paths if not (_ROOT / p).exists()] == []
    assert "not yet measured" in readme


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], check=True, cwd=cwd, capture_output=True)


def _repo(tmp_path: Path, tag: str | None) -> Path:
    work = tmp_path / "repo"
    work.mkdir()
    shutil.copy(_ROOT / "Makefile", work / "Makefile")
    _git(work, "init", "-q")
    _git(work, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(work, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    if tag:
        _git(work, "tag", tag)
    return work


def _make(work: Path, *args: str):
    env = {"PATH": "/usr/bin:/bin", "UV_PUBLISH_TOKEN": "pypi-test"}
    return subprocess.run([_MAKE, *args], check=False, cwd=work, env=env, capture_output=True, text=True)


@pytest.mark.skipif(_MAKE is None or shutil.which("git") is None, reason="make/git missing")
def test_publish_refuses_untagged_commit(tmp_path):
    out = _make(_repo(tmp_path, None), "publish")
    assert "not on a release tag" in out.stdout
    assert "uv build" not in out.stdout


@pytest.mark.skipif(_MAKE is None or shutil.which("git") is None, reason="make/git missing")
def test_publish_pins_version_to_tag(tmp_path):
    out = _make(_repo(tmp_path, "v1.2.3"), "-n", "publish")
    assert "SETUPTOOLS_SCM_PRETEND_VERSION=1.2.3 uv build" in out.stdout


def test_init_fallback_version_is_stable():
    init = (_ROOT / "kairos" / "__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "0.4.0"' in init
    assert ".dev" not in init


def test_release_docs_live_outside_readme():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "make publish" not in readme
    assert "make publish" in (_ROOT / "docs" / "RELEASING.md").read_text(encoding="utf-8")
