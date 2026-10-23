"""Consistency tests for docs/paper: citations, cross-references, tables and reported numbers."""

import importlib.util
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

from kairos.modeling import CodecOutput, PyramidalCodec

_ROOT = Path(__file__).parents[1]
_PAPER = _ROOT / "docs" / "paper"
_ARCH = _ROOT / "docs" / "architecture"
_CODEC_SVG = _ARCH / "kairos_codec.svg"
_CODEC_FIGURES = [_CODEC_SVG, _ARCH / "kairos_codec_patch.svg", _ARCH / "kairos_codec_conv.svg"]
_CODEC_TEXT = "the old cat saw the big dog"
_CODEC_CUT = (12, 15)
_CODEC_NOISE = "k7q"
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


def test_every_included_graphic_exists_in_the_paper_folder():
    """Each \\includegraphics target sits next to the .tex file, so the folder stays self-contained."""
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", _tex()):
        assert (_PAPER / name).is_file(), name


def _codec_svg_text(path=_CODEC_SVG):
    return re.sub(r"<[^>]+>", "", path.read_text(encoding="utf-8")).replace("\n", "")


@pytest.mark.parametrize("path", _CODEC_FIGURES, ids=lambda p: p.stem)
def test_codec_figure_is_well_formed_svg(path):
    """Each codec figure parses as XML."""
    assert ET.parse(path).getroot().tag.endswith("svg")


@pytest.mark.parametrize("path", _CODEC_FIGURES, ids=lambda p: p.stem)
def test_codec_figures_follow_a_real_encode(path):
    """Positions and window texts of the noisy input follow a real encode, in every variant."""
    start, stop = _CODEC_CUT
    noisy = _CODEC_TEXT[:start] + _CODEC_NOISE + _CODEC_TEXT[stop:]
    codec = PyramidalCodec(d_model=8, stride=3, num_scales=4, mode="patch")
    scales = codec.encode(torch.zeros(1, len(_CODEC_TEXT), 8)).scales
    svg = _codec_svg_text(path)
    assert len(_CODEC_TEXT) == 27 and svg.count(f"{len(_CODEC_TEXT)} bytes") == 2
    for scale, patch in enumerate(codec.patch_sizes[:2], start=1):
        assert f"scale {scale}, p = {patch}" in svg
        assert f"on {scales[scale - 1].shape[1]} positions" in svg
        for first in range(0, len(noisy), patch):
            assert noisy[first : first + patch].replace(" ", "\u00b7") in svg
    assert "27 bytes" + noisy.replace(" ", "\u00b7") in svg
    assert "27 bytes" + _CODEC_TEXT.replace(" ", "\u00b7") in svg
    assert "same weights, all scales" in svg


def test_codec_corrupted_span_is_centred_and_random():
    """The corrupted span sits in the middle of the sequence and differs from the clean bytes."""
    start, stop = _CODEC_CUT
    assert start + stop - 1 == len(_CODEC_TEXT) - 1
    assert len(_CODEC_NOISE) == stop - start and _CODEC_NOISE != _CODEC_TEXT[start:stop]


def test_generic_codec_figure_is_mode_agnostic_and_used_by_the_paper():
    """The paper includes the generic figure, which names neither conv nor patch operators."""
    assert "kairos_codec.pdf" in _tex()
    svg = _codec_svg_text()
    assert "Encode" in svg and "Decode" in svg
    assert not any(word in svg for word in ("Conv", "Linear W", "Mixer", "transposed"))


def test_patch_codec_figure_matches_the_patch_codec():
    """The patch figure draws one linear map per scale, tied decoding, and no conv-only parts."""
    codec = PyramidalCodec(d_model=8, stride=3, num_scales=4, mode="patch")
    svg = _codec_svg_text(_ARCH / "kairos_codec_patch.svg")
    for scale, patch in enumerate(codec.patch_sizes[:2], start=1):
        assert codec.encode_lin[scale - 1].in_features == patch * 8
        assert f"Linear W{scale}: {patch}\u00b7d" in svg
    assert "transposed" in svg and "Conv" not in svg and "Mixer" not in svg
    assert "codec_mode: patch" in (_ROOT / "kairos" / "configs" / "minimal.yaml").read_text(encoding="utf-8")


def test_conv_codec_figure_matches_the_conv_codec():
    """The conv figure draws kernel = stride convolutions, a shared pre-mixer and the notebook mode."""
    codec = PyramidalCodec(d_model=8, stride=3, num_scales=4, mode="conv")
    svg = _codec_svg_text(_ARCH / "kairos_codec_conv.svg")
    for scale, patch in enumerate(codec.patch_sizes[:2], start=1):
        assert codec.encoders[scale - 1].kernel_size == codec.encoders[scale - 1].stride == (patch,)
        assert f"Conv1d, k = s = {patch}" in svg
    assert "shared Mixer" in svg and "ConvTranspose1d" in svg
    pretrain = (_ROOT / "kairos" / "configs" / "pretrain.yaml").read_text(encoding="utf-8")
    assert "codec_mode: conv" in pretrain and "share_backbones: true" in pretrain


def test_codec_figure_routes_text_to_the_first_two_scales():
    """The notebook config sends the text modality (id 0) to scales 1-2 only, as drawn."""
    config = (_ROOT / "kairos" / "configs" / "pretrain.yaml").read_text(encoding="utf-8")
    assert "modality_scales: {0: [0, 1]," in config


def _decode_grads(mode):
    codec = PyramidalCodec(d_model=8, stride=3, num_scales=4, mode=mode)
    encoded = codec.encode(torch.randn(1, 27, 8))
    detached = CodecOutput(scales=[s.detach() for s in encoded.scales], length=encoded.length)
    codec.decode(detached).sum().backward()
    return codec


def test_patch_decode_reuses_encode_weights_but_conv_does_not():
    """Patch mode decodes with the transposed encoder weights; conv mode has separate decoders."""
    assert _decode_grads("patch").encode_lin[0].weight.grad is not None
    assert _decode_grads("conv").encoders[0].weight.grad is None


def test_patch_mode_has_no_shared_mixer():
    """Only conv mode has the shared pre-mixer drawn in the figure."""
    assert hasattr(PyramidalCodec(8, stride=3, num_scales=2, mode="conv"), "pre_mixer")
    assert not hasattr(PyramidalCodec(8, stride=3, num_scales=2, mode="patch"), "pre_mixer")
