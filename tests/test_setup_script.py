"""Static checks for the limited-sandbox torch setup script (the install itself needs network)."""

import itertools
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "setup_limited_sandbox_torch.sh"


def test_script_is_executable_and_has_no_consecutive_comment_lines():
    lines = _SCRIPT.read_text(encoding="utf-8").splitlines()[1:]
    assert _SCRIPT.stat().st_mode & 0o111
    assert not any(a.lstrip().startswith("#") and b.lstrip().startswith("#") for a, b in itertools.pairwise(lines))


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_script_syntax_is_valid():
    out = subprocess.run(["bash", "-n", str(_SCRIPT)], check=False, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_script_never_installs_nvidia_wheels():
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "--no-deps" in text
    assert "--no-config" in text
    assert "uv pip" not in text.replace("$UV pip", "")
    assert "nvidia-|triton" in text
    assert 'CUDA_VISIBLE_DEVICES=""' in text
