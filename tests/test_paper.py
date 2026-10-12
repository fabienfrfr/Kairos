"""Consistency tests for docs/paper: citations, cross-references, tables and reported numbers."""

import importlib.util
import re
import statistics
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_PAPER = _ROOT / "docs" / "paper"
_LOG = _ROOT / "docs" / "experiments" / "ablations_codec_attention.md"
_spec = importlib.util.spec_from_file_location("roofline", _ROOT / "scripts" / "roofline.py")
roofline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roofline)


def _tex() -> str:
    text = (_PAPER / "kairos_paper.tex").read_text(encoding="utf-8")
    return re.sub(r"(?<!\\)%.*", "", text)


def _cited() -> set[str]:
    keys = set()
    for group in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", _tex()):
        keys |= {key.strip() for key in group.split(",")}
    return keys


def _bib_keys() -> set[str]:
    bib = (_PAPER / "kairos_references.bib").read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\{([^,\s]+),", bib))


def _seed_rows() -> list[tuple[str, list[str], str, str]]:
    section = _LOG.read_text(encoding="utf-8").split("## 1.")[1].split("## 2.")[0]
    pattern = r"^\|\s*(.+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+) ± ([\d.]+)\s*\|$"
    rows = []
    for line in section.splitlines():
        found = re.match(pattern, line)
        if found:
            name, *seeds, mean, sd = found.groups()
            rows.append((name, seeds, mean, sd))
    return rows


def test_every_citation_has_a_bib_entry():
    """No \\cite key may be missing from the bibliography."""
    assert _cited() - _bib_keys() == set()


def test_bibliography_has_no_unused_entry():
    """Every bibliography entry is cited at least once."""
    assert _bib_keys() - _cited() == set()


def test_every_reference_points_to_a_label():
    """No \\ref or \\eqref may point to a missing label."""
    labels = set(re.findall(r"\\label\{([^}]+)\}", _tex()))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", _tex()))
    assert refs - labels == set()


def test_every_label_is_referenced():
    """Every label is used by at least one reference."""
    labels = set(re.findall(r"\\label\{([^}]+)\}", _tex()))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", _tex()))
    assert labels - refs == set()


@pytest.mark.parametrize("marker", ["TBD", "TODO", "XXX", "anticipated", "not yet logged"])
def test_no_placeholder_text(marker):
    """The paper reports what was measured: no placeholders or anticipated results."""
    assert marker.lower() not in _tex().lower()


def test_compute_table_is_generated_by_the_script():
    """The included table equals the script output, so no number is transcribed by hand."""
    included = (_PAPER / "tables" / "compute.tex").read_text(encoding="utf-8")
    assert included == roofline.latex_tabular()


def test_headline_numbers_follow_from_the_model():
    """Prose numbers (FLOPs, token ratio, threshold factor) match the sizing model."""
    tex = _tex()
    flops = roofline.train_flops(roofline.TARGET_ACTIVE, roofline.TARGET_TOKENS)
    assert f"{flops / 1e18:.1f}\\times10^{{18}}" in tex
    assert f"{roofline.TARGET_TOKENS / roofline.TARGET_ACTIVE:,.0f}".replace(",", "{,}") in tex
    assert roofline.TARGET_TOTAL / roofline.TARGET_ACTIVE == 8


@pytest.mark.parametrize("row", _seed_rows(), ids=lambda row: row[0])
def test_seed_statistics_match_the_log_and_the_paper(row):
    """Mean and sd recompute from the per-seed values and appear verbatim in the paper table."""
    _, seeds, mean, sd = row
    values = [float(v) for v in seeds]
    assert statistics.mean(values) == pytest.approx(float(mean), abs=1e-3)
    assert statistics.stdev(values) == pytest.approx(float(sd), abs=1e-3)
    assert " & ".join(seeds) in _tex()
    assert f"${mean} \\pm {sd}$" in _tex()
