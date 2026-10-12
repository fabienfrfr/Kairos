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
