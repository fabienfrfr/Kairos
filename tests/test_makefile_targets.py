import re
from pathlib import Path

_MAKEFILE = (Path(__file__).resolve().parents[1] / "Makefile").read_text()


def _recipe(target):
    match = re.search(rf"^{target}:.*\n((?:\t.*\n)+)", _MAKEFILE, re.MULTILINE)
    assert match, f"missing make target: {target}"
    return match.group(1)


def test_coverage_target_includes_slow_tests():
    assert "--runslow" in _recipe("coverage")


def test_converge_target_includes_slow_tests():
    assert "--runslow" in _recipe("converge")


def test_test_slow_target_selects_slow_marker():
    assert "-m slow" in _recipe("test-slow")


def test_default_test_target_stays_fast():
    assert "--runslow" not in _recipe("test")
