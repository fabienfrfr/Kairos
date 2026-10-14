import tomllib
from pathlib import Path

_PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
_UV = _PYPROJECT["tool"]["uv"]


def _index(name):
    return next(index for index in _UV["index"] if index["name"] == name)


def test_rocm_index_is_explicit_so_default_installs_use_pypi():
    assert _index("pytorch-rocm")["explicit"] is True


def test_pypi_index_is_not_explicit():
    assert not _index("pypi").get("explicit", False)


def test_rocm_sources_are_gated_behind_the_rocm_extra():
    for package in ("rocm", "torch", "triton"):
        sources = _UV["sources"][package]
        assert all(s["index"] == "pytorch-rocm" and s["extra"] == "rocm" for s in sources), package


def test_rocm_wheels_are_only_pinned_inside_the_rocm_extra():
    base = " ".join(_PYPROJECT["project"]["dependencies"])
    assert "rocm" not in base
    assert any("rocm" in dep for dep in _PYPROJECT["project"]["optional-dependencies"]["rocm"])
