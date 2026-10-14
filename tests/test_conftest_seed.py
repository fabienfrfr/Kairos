import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def test_autouse_fixture_seeds_torch():
    drawn = torch.rand(4)
    torch.manual_seed(0)
    assert torch.equal(drawn, torch.rand(4))


def test_autouse_fixture_seeds_numpy_and_random():
    np_drawn, py_drawn = np.random.rand(4), random.random()
    np.random.seed(0)
    random.seed(0)
    assert np.array_equal(np_drawn, np.random.rand(4))
    assert py_drawn == random.random()


def _run_pytest(tmp_path, *args):
    (tmp_path / "conftest.py").write_text((Path(__file__).parent / "conftest.py").read_text())
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow\n")
    (tmp_path / "test_x.py").write_text("import pytest\n\n\n@pytest.mark.slow\ndef test_long():\n    pass\n")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args]
    return subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=False).stdout


def test_slow_tests_are_skipped_by_default(tmp_path):
    assert "1 skipped" in _run_pytest(tmp_path)


def test_slow_tests_run_with_runslow_flag(tmp_path):
    assert "1 passed" in _run_pytest(tmp_path, "--runslow")


def test_slow_tests_run_with_marker_expression(tmp_path):
    assert "1 passed" in _run_pytest(tmp_path, "-m", "slow")
