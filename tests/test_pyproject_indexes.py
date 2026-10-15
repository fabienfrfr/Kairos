import tomllib
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = tomllib.loads((_ROOT / "pyproject.toml").read_text())
_UV = _PYPROJECT["tool"]["uv"]
_CI = yaml.safe_load((_ROOT / ".github" / "workflows" / "ci.yml").read_text())
_MAKEFILE = (_ROOT / "Makefile").read_text()


def _index(name):
    return next(index for index in _UV["index"] if index["name"] == name)


def _steps(job):
    return [step["run"] for step in _CI["jobs"][job]["steps"] if "run" in step]


def test_rocm_index_stays_non_explicit_so_transitive_amd_wheels_resolve():
    # explicit=true breaks `uv sync`: torch[device-all] needs amd-torch-device-* only AMD hosts
    assert not _index("pytorch-rocm").get("explicit", False)


def test_best_match_strategy_is_kept_for_local_version_wheels():
    assert _UV["index-strategy"] == "unsafe-best-match"


def test_rocm_sources_are_gated_behind_the_rocm_extra():
    for package in ("rocm", "torch", "triton"):
        rocm_sources = [s for s in _UV["sources"][package] if s["index"] == "pytorch-rocm"]
        assert rocm_sources and all(s["extra"] == "rocm" for s in rocm_sources), package


def test_rocm_wheels_are_only_pinned_inside_the_rocm_extra():
    base = " ".join(_PYPROJECT["project"]["dependencies"])
    assert "rocm" not in base
    assert any("rocm" in dep for dep in _PYPROJECT["project"]["optional-dependencies"]["rocm"])


def test_cpu_index_is_explicit_and_points_to_pytorch_cpu_wheels():
    index = _index("pytorch-cpu")
    assert index["explicit"] is True
    assert index["url"] == "https://download.pytorch.org/whl/cpu"


def test_cpu_extra_only_adds_torch_and_maps_it_to_the_cpu_index():
    assert _PYPROJECT["project"]["optional-dependencies"]["cpu"] == ["torch>=2.10.0"]
    cpu_sources = [s for s in _UV["sources"]["torch"] if s["index"] == "pytorch-cpu"]
    assert cpu_sources == [{"index": "pytorch-cpu", "extra": "cpu"}]


def _conflict_pairs():
    return [{item.get("extra") or "group:" + item["group"] for item in pair} for pair in _UV["conflicts"]]


def test_cpu_and_rocm_extras_are_declared_mutually_exclusive():
    assert {"cpu", "rocm"} in _conflict_pairs()


def test_cuda_group_is_the_default_accelerator_build_from_pypi():
    assert "cuda" in _UV["default-groups"] and "dev" in _UV["default-groups"]
    cuda = _PYPROJECT["dependency-groups"]["cuda"]
    assert any(dep.startswith("torch") for dep in cuda) and any(dep.startswith("triton") for dep in cuda)
    for package in ("torch", "triton"):
        assert {"index": "pypi", "group": "cuda"} in _UV["sources"][package], package


def test_cuda_group_conflicts_with_cpu_and_rocm_extras():
    pairs = _conflict_pairs()
    assert {"group:cuda", "cpu"} in pairs and {"group:cuda", "rocm"} in pairs


def test_base_dependencies_keep_torch_for_pip_users():
    assert any(dep.startswith("torch") for dep in _PYPROJECT["project"]["dependencies"])


def test_ci_tests_run_on_the_cpu_wheel_and_lint_needs_no_torch():
    sync = [run for run in _steps("test") if "uv sync" in run]
    tests = [run for run in _steps("test") if "pytest" in run]
    assert sync and all("--extra cpu" in run and "--no-group cuda" in run for run in sync)
    assert tests and all("--extra cpu" in run and "--no-group cuda" in run for run in tests)
    assert all("torch" not in run and "--extra" not in run for run in _steps("lint"))
    assert any("--only-group dev" in run for run in _steps("lint"))


def test_makefile_rocm_runs_drop_the_default_cuda_group():
    assert "UV_TORCH_EXTRA := --extra rocm --no-group cuda" in _MAKEFILE
