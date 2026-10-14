import tomllib
from pathlib import Path

_PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
_UV = _PYPROJECT["tool"]["uv"]


def _index(name):
    return next(index for index in _UV["index"] if index["name"] == name)


def test_rocm_index_stays_non_explicit_so_transitive_amd_wheels_resolve():
    # explicit=true breaks `uv sync`: torch[device-all] needs amd-torch-device-* only AMD hosts
    assert not _index("pytorch-rocm").get("explicit", False)


def test_best_match_strategy_is_kept_for_local_version_wheels():
    assert _UV["index-strategy"] == "unsafe-best-match"


def test_rocm_sources_are_gated_behind_the_rocm_extra():
    for package in ("rocm", "torch", "triton"):
        sources = _UV["sources"][package]
        assert all(s["index"] == "pytorch-rocm" and s["extra"] == "rocm" for s in sources), package


def test_rocm_wheels_are_only_pinned_inside_the_rocm_extra():
    base = " ".join(_PYPROJECT["project"]["dependencies"])
    assert "rocm" not in base
    assert any("rocm" in dep for dep in _PYPROJECT["project"]["optional-dependencies"]["rocm"])
