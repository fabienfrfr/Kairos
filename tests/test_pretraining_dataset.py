import json

import numpy as np
import pytest
import torch
from datasets import Dataset as HFDataset

from kairos.dataset import KairosPretrainingDataset, pack_multimodal_data
from kairos.tokenizer import KairosTokenizer, Modality


def make_example(modality, caption=None, source="test", **fields):
    """Build a generic-schema row: numpy-array fields go into `data`, everything else into."""
    arrays = {k: v for k, v in fields.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in fields.items() if not isinstance(v, np.ndarray)}
    return {
        "modality": modality,
        "caption": caption,
        "source": source,
        "data": pack_multimodal_data(arrays) if arrays else None,
        "meta": json.dumps(meta) if meta else None,
    }


@pytest.fixture(scope="module")
def tokenizer():
    return KairosTokenizer()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def all_kinds_examples(rng):
    return [
        {"modality": "text", "text": "Paris is the capital of France."},
        make_example("image_caption", caption="a dog", image=rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)),
        make_example(
            "audio_caption",
            caption="a bark",
            audio=rng.uniform(-1, 1, 4000).astype(np.float32),
            sample_rate=8000,
        ),
        make_example("video_caption", caption="running", video=rng.integers(0, 255, (4, 8, 8, 3), dtype=np.uint8)),
        make_example("lidar", points=rng.uniform(-10, 10, (50, 4)).astype(np.float32)),
        make_example("imu", signal=rng.uniform(-5, 5, (50, 6)).astype(np.float32)),
        make_example(
            "control",
            caption="method=bang-bang",
            action=rng.uniform(-1, 1, 200).astype(np.float32),
            state=rng.uniform(-1, 1, 200).astype(np.float32),
            sample_rate=8000,
        ),
    ]


# ========================= text-only path unchanged =========================
def test_text_only_schema(tokenizer):
    ds = KairosPretrainingDataset(texts=["hello world"], tokenizer=tokenizer, max_len=64, stride=1)
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "modality_ids", "mask", "prompt_len", "octet_family_ids"}
    assert set(item["modality_ids"].tolist()) <= {int(Modality.TEXT)}


def test_text_only_mask_matches_padding(tokenizer):
    ds = KairosPretrainingDataset(texts=["hi"], tokenizer=tokenizer, max_len=32, stride=1)
    item = ds[0]
    assert torch.equal(item["mask"] == 0, item["input_ids"] == tokenizer.pad_token_id)


# ========================= multimodal path: all 6 kinds =========================
def test_all_kinds_build_without_error(tokenizer, all_kinds_examples):
    ds = KairosPretrainingDataset(multimodal_examples=all_kinds_examples, tokenizer=tokenizer, max_len=4096, stride=1)
    assert len(ds) >= len(all_kinds_examples)


def test_all_kinds_cover_expected_modalities(tokenizer, all_kinds_examples):
    ds = KairosPretrainingDataset(multimodal_examples=all_kinds_examples, tokenizer=tokenizer, max_len=4096, stride=1)
    seen = set()
    for i in range(len(ds)):
        seen |= set(ds[i]["modality_ids"][ds[i]["mask"] == 1].tolist())
    assert seen == {
        int(Modality.TEXT),
        int(Modality.IMAGE),
        int(Modality.AUDIO),
        int(Modality.VIDEO),
        int(Modality.LIDAR),
        int(Modality.STATE),
        int(Modality.ACTION),
    }


def test_lidar_has_no_text_caption(tokenizer, rng):
    ex = [make_example("lidar", points=rng.uniform(-10, 10, (50, 4)).astype(np.float32))]
    ds = KairosPretrainingDataset(multimodal_examples=ex, tokenizer=tokenizer, max_len=4096, stride=1)
    item = ds[0]
    assert set(item["modality_ids"][item["mask"] == 1].tolist()) == {int(Modality.LIDAR)}


def test_imu_uses_state_modality(tokenizer, rng):
    ex = [make_example("imu", signal=rng.uniform(-5, 5, (50, 6)).astype(np.float32))]
    ds = KairosPretrainingDataset(multimodal_examples=ex, tokenizer=tokenizer, max_len=4096, stride=1)
    item = ds[0]
    assert set(item["modality_ids"][item["mask"] == 1].tolist()) == {int(Modality.STATE)}


def test_control_uses_action_and_state_disjoint(tokenizer, rng):
    ex = [
        make_example(
            "control",
            caption="bang-bang",
            action=rng.uniform(-1, 1, 200).astype(np.float32),
            state=rng.uniform(-1, 1, 200).astype(np.float32),
            sample_rate=8000,
        )
    ]
    ds = KairosPretrainingDataset(multimodal_examples=ex, tokenizer=tokenizer, max_len=4096, stride=1)
    item = ds[0]
    mods = item["modality_ids"].tolist()
    action_pos = [i for i, m in enumerate(mods) if m == int(Modality.ACTION)]
    state_pos = [i for i, m in enumerate(mods) if m == int(Modality.STATE)]
    assert action_pos and state_pos
    assert set(action_pos).isdisjoint(state_pos)


# ============================= diagnose_raw_control_balance =============================
def test_diagnose_raw_control_balance_clean_data(rng):
    from kairos.dataset import diagnose_raw_control_balance

    examples = [
        make_example(
            "control",
            action=rng.uniform(-1, 1, 480).astype(np.float32),
            state=rng.uniform(-1, 1, 480).astype(np.float32),
            sample_rate=8000,
        )
        for _ in range(5)
    ]
    report = diagnose_raw_control_balance(examples)

    assert report.n_control_examples == 5
    assert report.total_state_samples == report.total_action_samples == 2400
    assert report.mismatched_examples == []


def test_diagnose_raw_control_balance_flags_source_data_mismatch(rng):
    """This is what would catch a stale/broken *source* dataset (e.g. an old published HF
    snapshot built before a pipeline fix) - before any tokenization/windowing is involved."""
    from kairos.dataset import diagnose_raw_control_balance

    examples = [
        make_example(
            "control",
            action=rng.uniform(-1, 1, 480).astype(np.float32),
            state=rng.uniform(-1, 1, 300).astype(np.float32),  # genuinely shorter, upstream
            sample_rate=8000,
        )
    ]
    report = diagnose_raw_control_balance(examples)

    assert report.n_control_examples == 1
    assert len(report.mismatched_examples) == 1
    assert report.mismatched_examples[0] == {"index": 0, "state_samples": 300, "action_samples": 480}


def test_diagnose_raw_control_balance_ignores_other_modalities(rng):
    from kairos.dataset import diagnose_raw_control_balance

    examples = [
        {"modality": "text", "text": "hello"},
        make_example("image_caption", caption="a cat", image=rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)),
    ]
    report = diagnose_raw_control_balance(examples)
    assert report.n_control_examples == 0
    assert report.total_state_samples == report.total_action_samples == 0


def test_control_state_and_action_pair_one_to_one(tokenizer, rng):
    """One row = one atomic (state, action) transition: exactly one STATE segment
    and one ACTION segment, in that order (S then A)."""
    from kairos.dataset import _segments_for

    ex = make_example(
        "control",
        action=rng.uniform(-1, 1, 480).astype(np.float32),  # ~0.06s @ 8kHz, prod-realistic
        state=rng.uniform(-1, 1, 480).astype(np.float32),
        sample_rate=8000,
    )
    segments = _segments_for(ex, {})
    modalities = [s.modality for s in segments]

    assert modalities.count(Modality.STATE) == modalities.count(Modality.ACTION) == 1
    assert modalities.index(Modality.STATE) < modalities.index(Modality.ACTION)


def test_control_warns_on_state_action_token_count_mismatch(tokenizer, rng):
    """Regression: state/action arrays of different lengths used to silently break
    the S,A pairing (via now-removed tick-splitting); must warn instead."""
    from kairos.dataset import _segments_for

    ex = make_example(
        "control",
        action=rng.uniform(-1, 1, 480).astype(np.float32),
        state=rng.uniform(-1, 1, 200).astype(np.float32),  # mismatched length
        sample_rate=8000,
    )
    with pytest.warns(UserWarning, match="mismatch"):
        _segments_for(ex, {})


def test_control_equal_length_arrays_never_warn(tokenizer, rng, recwarn):
    from kairos.dataset import _segments_for

    ex = make_example(
        "control",
        action=rng.uniform(-1, 1, 480).astype(np.float32),
        state=rng.uniform(-1, 1, 480).astype(np.float32),
        sample_rate=8000,
    )
    _segments_for(ex, {})
    assert not [w for w in recwarn.list if "mismatch" in str(w.message)]


def test_control_scale_factor_shrinks_token_count(tokenizer, rng):
    """The scale_factor knob (decimation) is how token count is reduced for control."""
    from kairos.dataset import _segments_for

    ex = make_example(
        "control",
        action=rng.uniform(-1, 1, 480).astype(np.float32),
        state=rng.uniform(-1, 1, 480).astype(np.float32),
        sample_rate=8000,
    )
    coarse = _segments_for(ex, {"control": 8})  # more decimation
    fine = _segments_for(ex, {"control": 2})  # less decimation

    coarse_bytes = sum(len(item[1]) for s in coarse if s.modality == Modality.STATE for item in s.data if item[0] == "bytes")
    fine_bytes = sum(len(item[1]) for s in fine if s.modality == Modality.STATE for item in s.data if item[0] == "bytes")
    assert coarse_bytes < fine_bytes


def test_control_state_action_totals_match_end_to_end_even_with_window_truncation(tokenizer, rng):
    """The real invariant: summed over the whole packed+chunked dataset, STATE tokens must
    equal ACTION tokens exactly - even when windows are small enough to cut mid-segment."""
    texts = [f"filler {i}" for i in range(80)]
    multimodal = [
        make_example(
            "control",
            action=rng.uniform(-1, 1, 480).astype(np.float32),
            state=rng.uniform(-1, 1, 480).astype(np.float32),
            sample_rate=8000,
        )
        for _ in range(30)
    ]
    # max_len well under one control segment's size, to force mid-segment window boundaries
    ds = KairosPretrainingDataset(texts=texts, tokenizer=tokenizer, max_len=64, multimodal_examples=multimodal, pack=True)

    state_tok = sum((ds[i]["modality_ids"] == Modality.STATE.value).sum().item() for i in range(len(ds)))
    action_tok = sum((ds[i]["modality_ids"] == Modality.ACTION.value).sum().item() for i in range(len(ds)))

    assert len(ds) > 30  # sanity: truncation actually happened, this isn't a trivial no-op
    assert state_tok == action_tok
    assert state_tok > 0


def test_diagnose_control_alternation_reports_clean_dataset_as_clean(tokenizer, rng):
    from kairos.dataset import diagnose_control_alternation

    texts = [f"filler {i}" for i in range(40)]
    multimodal = [
        make_example(
            "control",
            action=rng.uniform(-1, 1, 480).astype(np.float32),
            state=rng.uniform(-1, 1, 480).astype(np.float32),
            sample_rate=8000,
        )
        for _ in range(10)
    ]
    # max_len large enough that the whole packed stream fits in one row - no mid-segment
    # window cuts, so per-row balance is guaranteed too (not just the aggregate total).
    ds = KairosPretrainingDataset(texts=texts, tokenizer=tokenizer, max_len=100_000, multimodal_examples=multimodal, pack=True)

    report = diagnose_control_alternation(ds.ds, sample_size=len(ds), seed=0)

    assert report.n_rows_with_control > 0
    assert report.total_state_tokens == report.total_action_tokens
    assert report.mismatched_rows == []
    assert report.non_alternating_rows == []


def test_diagnose_control_alternation_per_row_mismatch_is_expected_under_truncation(tokenizer, rng):
    """A per-row mismatch is a normal side effect of a window cutting a segment in half (the
    remainder lands in the next row) - only the aggregate total is a real invariant."""
    from kairos.dataset import diagnose_control_alternation

    texts = [f"filler {i}" for i in range(40)]
    multimodal = [
        make_example(
            "control",
            action=rng.uniform(-1, 1, 480).astype(np.float32),
            state=rng.uniform(-1, 1, 480).astype(np.float32),
            sample_rate=8000,
        )
        for _ in range(10)
    ]
    ds = KairosPretrainingDataset(texts=texts, tokenizer=tokenizer, max_len=1024, multimodal_examples=multimodal, pack=True)

    report = diagnose_control_alternation(ds.ds, sample_size=len(ds), seed=0)

    assert report.total_state_tokens == report.total_action_tokens  # aggregate: still exact
    assert report.mismatched_rows  # but individual rows can legitimately disagree


def test_diagnose_control_alternation_flags_a_broken_row(tokenizer, rng):
    """Directly poking a state-only segment (no matching action) into a row must surface
    as both a token-count mismatch and (trivially) as the row having no alternation to break."""
    from kairos.dataset import diagnose_control_alternation
    from kairos.tokenizer import MultimodalSegment

    state_only = tokenizer.encode_multimodal(
        [MultimodalSegment(Modality.STATE, KairosTokenizer.encode_signal(rng.uniform(-1, 1, 100).astype(np.float32), family="STA"))]
    )
    clean = tokenizer.encode_multimodal(
        [
            MultimodalSegment(Modality.STATE, KairosTokenizer.encode_signal(rng.uniform(-1, 1, 100).astype(np.float32), family="STA")),
            MultimodalSegment(Modality.ACTION, KairosTokenizer.encode_signal(rng.uniform(-1, 1, 100).astype(np.float32), family="ACT")),
        ]
    )

    def pad_mods(row, length):
        mods = row["modality_ids"].tolist()
        return mods + [int(Modality.TEXT)] * (length - len(mods))

    length = max(len(state_only["modality_ids"]), len(clean["modality_ids"]))
    fake_ds = HFDataset.from_dict({"modality_ids": [pad_mods(state_only, length), pad_mods(clean, length)]})

    report = diagnose_control_alternation(fake_ds, sample_size=2, seed=0)

    assert report.n_rows_with_control == 2
    assert len(report.mismatched_rows) == 1  # only the state-only row is unbalanced
    assert report.mismatched_rows[0]["action_tokens"] == 0


# ============================= find_rows_with_modality =============================
def test_find_rows_with_modality_finds_the_right_rows(tokenizer, rng):
    from kairos.dataset import find_rows_with_modality

    texts = [f"filler {i}" for i in range(30)]
    multimodal = [make_example("image_caption", caption="a cat", image=rng.integers(0, 255, (8, 8, 3), dtype=np.uint8))]
    ds = KairosPretrainingDataset(texts=texts, tokenizer=tokenizer, max_len=100_000, multimodal_examples=multimodal, pack=True)

    rows = find_rows_with_modality(ds.ds, "image", n=5)
    assert rows  # the single image_caption example must be found
    for row_i in rows:
        mods = ds.ds[row_i]["modality_ids"]
        assert int(Modality.IMAGE) in mods


def test_find_rows_with_modality_control_matches_state_or_action(tokenizer, rng):
    from kairos.dataset import find_rows_with_modality

    multimodal = [
        make_example(
            "control",
            action=rng.uniform(-1, 1, 480).astype(np.float32),
            state=rng.uniform(-1, 1, 480).astype(np.float32),
            sample_rate=8000,
        )
    ]
    ds = KairosPretrainingDataset(multimodal_examples=multimodal, tokenizer=tokenizer, max_len=100_000, pack=True)

    rows = find_rows_with_modality(ds.ds, "control", n=5)
    assert rows


def test_find_rows_with_modality_returns_empty_when_absent(tokenizer):
    from kairos.dataset import find_rows_with_modality

    ds = KairosPretrainingDataset(texts=["hello world"], tokenizer=tokenizer, max_len=256)
    assert find_rows_with_modality(ds.ds, "lidar", n=5) == []


def test_find_rows_with_modality_rejects_unknown_modality(tokenizer):
    from kairos.dataset import find_rows_with_modality

    ds = KairosPretrainingDataset(texts=["hello world"], tokenizer=tokenizer, max_len=256)
    with pytest.raises(ValueError, match="unknown modality"):
        find_rows_with_modality(ds.ds, "bogus", n=5)


def test_find_rows_with_modality_is_case_insensitive(tokenizer):
    from kairos.dataset import find_rows_with_modality

    ds = KairosPretrainingDataset(texts=["hello world"], tokenizer=tokenizer, max_len=256)
    assert find_rows_with_modality(ds.ds, "TEXT", n=5) == find_rows_with_modality(ds.ds, "text", n=5)


def test_long_example_gets_chunked(tokenizer, rng):
    big_image = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    ex = [make_example("image_caption", caption="big", image=big_image)]
    ds = KairosPretrainingDataset(multimodal_examples=ex, tokenizer=tokenizer, max_len=256, stride=1)
    assert len(ds) > 1


def test_multimodal_padding_uses_text_modality(tokenizer, all_kinds_examples):
    ds = KairosPretrainingDataset(
        multimodal_examples=all_kinds_examples[:1], tokenizer=tokenizer, max_len=4096, stride=1
    )
    item = ds[0]
    pad_region = item["modality_ids"][item["mask"] == 0]
    assert torch.all(pad_region == int(Modality.TEXT))


def test_unknown_modality_raises(tokenizer):
    with pytest.raises(ValueError, match="unknown example modality"):
        KairosPretrainingDataset(multimodal_examples=[{"modality": "bogus"}], tokenizer=tokenizer, max_len=64, stride=1)


def test_multimodal_path_from_pt_file(tmp_path, tokenizer, all_kinds_examples):
    path = tmp_path / "mini.pt"
    torch.save(all_kinds_examples, path)
    ds = KairosPretrainingDataset(multimodal_path=str(path), tokenizer=tokenizer, max_len=512, stride=1)
    assert len(ds) > 0


def test_multimodal_dataset_creates_default_tokenizer_when_none_given(all_kinds_examples):
    ds = KairosPretrainingDataset(multimodal_examples=all_kinds_examples, tokenizer=None, max_len=512, stride=1)
    assert isinstance(ds.tokenizer, KairosTokenizer)
    assert len(ds) > 0


def test_pack_concatenates_sources_before_chunking(tokenizer):
    examples = [{"modality": "text", "text": "short one"}, {"modality": "text", "text": "another short bit"}]
    unpacked = KairosPretrainingDataset(multimodal_examples=examples, tokenizer=tokenizer, max_len=64, stride=1)
    packed = KairosPretrainingDataset(
        multimodal_examples=examples, tokenizer=tokenizer, max_len=64, stride=1, pack=True
    )
    assert len(packed) < len(unpacked)  # fewer, fuller chunks
    packed_pad_frac = 1 - packed[0]["mask"].float().mean().item()
    unpacked_pad_frac = 1 - unpacked[0]["mask"].float().mean().item()
    assert packed_pad_frac < unpacked_pad_frac


def test_empty_text_examples_are_skipped(tokenizer):
    ds = KairosPretrainingDataset(
        texts=["", "   ", "Paris is the capital of France."], tokenizer=tokenizer, max_len=64, stride=1
    )
    assert len(ds) > 0  # only the non-empty text produced


def test_nonfinite_multimodal_array_is_skipped_not_raised(tokenizer, rng):
    bad = make_example("lidar", points=np.full((32, 4), np.nan, dtype=np.float32))
    good = make_example("lidar", points=rng.uniform(-10, 10, (32, 4)).astype(np.float32))

    with pytest.warns(UserWarning, match="skipping corrupt example"):
        ds = KairosPretrainingDataset(multimodal_examples=[bad, good], tokenizer=tokenizer, max_len=128, stride=1)

    assert len(ds) > 0  # the good example still produced


def test_all_nonfinite_multimodal_examples_yields_empty_dataset(tokenizer):
    bad = make_example("lidar", points=np.full((32, 4), np.inf, dtype=np.float32))

    with pytest.warns(UserWarning, match="skipping corrupt example"):
        ds = KairosPretrainingDataset(multimodal_examples=[bad], tokenizer=tokenizer, max_len=128, stride=1)

    assert len(ds) == 0


def test_skip_warning_fires_on_every_call_not_just_the_first(tokenizer):
    """Regression test: from_generator caches its output on disk keyed by a fingerprint of
    the generator function. A stale cache hit would silently skip re-running our generator
    on a second, similarly-shaped call — so the skip-warning (and any construction-time
    error) would only ever fire once per process, not once per call, and the dataset build
    itself could return silently-wrong content. This must hold across repeated calls."""
    bad = make_example("lidar", points=np.full((32, 4), np.inf, dtype=np.float32))
    good = make_example("lidar", points=np.random.uniform(-10, 10, (32, 4)).astype(np.float32))

    for _ in range(3):
        with pytest.warns(UserWarning, match="skipping corrupt example"):
            ds = KairosPretrainingDataset(multimodal_examples=[bad, good], tokenizer=tokenizer, max_len=128, stride=1)
        assert len(ds) > 0


def test_large_multimodal_build_does_not_materialize_everything_in_process_memory(tokenizer):
    """Regression test: _build_multimodal must stream rows into the arrow writer via a
    memory-mapped file (the default for Dataset.from_generator), not load the whole table
    into a real in-process buffer (keep_in_memory=True). RSS growth for a few thousand
    multimodal examples should stay well under what fully materializing them would cost —
    this is a coarse smoke check, not an exact bound, so it only needs to catch a gross
    regression (e.g. keep_in_memory=True creeping back in), not tiny fluctuations."""
    import resource

    n_examples = 4000
    examples = [
        make_example("lidar", points=np.random.uniform(-10, 10, (64, 4)).astype(np.float32))
        for _ in range(n_examples)
    ]
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB, peak-so-far
    ds = KairosPretrainingDataset(multimodal_examples=examples, tokenizer=tokenizer, max_len=256, stride=1)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert len(ds) > 0
    # a fully in-memory (non-mmap) table for this many chunked+padded examples would grow
    # RSS by tens to hundreds of MB; a properly streamed/mmap'd build should stay under ~50MB.
    assert (rss_after - rss_before) / 1024 < 50, (
        f"RSS grew {(rss_after - rss_before) / 1024:.1f} MB building {n_examples} examples — "
        "looks like the dataset is being fully materialized in process memory instead of "
        "streamed to a memory-mapped arrow file (check for a stray keep_in_memory=True)."
    )


def test_modality_scale_factors_override_reduces_row_count(tokenizer, rng):
    """A larger explicit scale_factor for 'control' shrinks its token count -> fewer chunks."""
    ex = [
        make_example(
            "control",
            action=rng.uniform(-1, 1, 4000).astype(np.float32),
            state=rng.uniform(-1, 1, 4000).astype(np.float32),
            sample_rate=8000,
        )
    ]
    ds_default = KairosPretrainingDataset(multimodal_examples=ex, tokenizer=tokenizer, max_len=256, stride=1)
    ds_scaled = KairosPretrainingDataset(
        multimodal_examples=ex,
        tokenizer=tokenizer,
        max_len=256,
        stride=1,
        modality_scale_factors={"control": 16},
    )
    assert len(ds_scaled) < len(ds_default)


def test_modality_scale_factors_defaults_to_tokenizer_class_defaults(tokenizer, all_kinds_examples):
    """Omitting modality_scale_factors entirely must match passing an all-None dict explicitly."""
    ds_omitted = KairosPretrainingDataset(multimodal_examples=all_kinds_examples, tokenizer=tokenizer, max_len=128, stride=1)
    ds_explicit_none = KairosPretrainingDataset(
        multimodal_examples=all_kinds_examples,
        tokenizer=tokenizer,
        max_len=128,
        stride=1,
        modality_scale_factors={k: None for k in ["image_caption", "audio_caption", "video_caption", "lidar", "imu", "control"]},
    )
    assert len(ds_omitted) == len(ds_explicit_none)


def test_diagnose_multimodal_examples_reports_every_modality(tokenizer, rng):
    from kairos.dataset import diagnose_multimodal_examples

    examples = [
        {"modality": "text", "text": "hello world"},
        make_example("control", action=rng.uniform(-1, 1, 4000).astype(np.float32), state=rng.uniform(-1, 1, 4000).astype(np.float32), sample_rate=8000),
    ]
    report = diagnose_multimodal_examples(examples, tokenizer=tokenizer, max_len=1024)
    modalities = {r.modality for r in report.rows}
    assert modalities == {"text", "control"}
    assert report.total_examples == 2


def test_diagnose_multimodal_examples_scale_factor_shrinks_chunk_estimate(tokenizer, rng):
    from kairos.dataset import diagnose_multimodal_examples

    examples = [
        make_example("control", action=rng.uniform(-1, 1, 4000).astype(np.float32), state=rng.uniform(-1, 1, 4000).astype(np.float32), sample_rate=8000)
        for _ in range(3)
    ]
    default_report = diagnose_multimodal_examples(examples, tokenizer=tokenizer, max_len=1024)
    scaled_report = diagnose_multimodal_examples(
        examples, tokenizer=tokenizer, max_len=1024, modality_scale_factors={"control": 16}
    )
    default_chunks = next(r.chunks_total_estimate for r in default_report.rows if r.modality == "control")
    scaled_chunks = next(r.chunks_total_estimate for r in scaled_report.rows if r.modality == "control")
    assert scaled_chunks < default_chunks


def test_pipeline_data_report_matches_diagnose_multimodal_examples(tokenizer, rng):
    from kairos.dataset import diagnose_multimodal_examples
    from kairos.modeling import KairosConfig
    from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

    examples = [
        {"modality": "text", "text": "hello world"},
        make_example("control", action=rng.uniform(-1, 1, 4000).astype(np.float32), state=rng.uniform(-1, 1, 4000).astype(np.float32), sample_rate=8000),
    ]
    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=2, num_modalities=8)
    dc = DataConfig(multimodal_examples=examples, max_len=256)
    pipe = KairosMultimodalPipeline(model_config, dc, TrainConfig(run_dir="unused"), tokenizer=tokenizer)

    pipe_report = pipe.data_report(sample_size=200)
    direct_report = diagnose_multimodal_examples(examples, tokenizer=tokenizer, max_len=256, sample_size=200)
    assert {r.modality: r.chunks_total_estimate for r in pipe_report.rows} == {
        r.modality: r.chunks_total_estimate for r in direct_report.rows
    }


def test_pipeline_data_report_falls_back_to_built_dataset_after_build(tokenizer, rng):
    from kairos.dataset import BuiltDatasetReport
    from kairos.modeling import KairosConfig
    from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

    examples = [{"modality": "text", "text": "hello world foo bar"} for _ in range(5)]
    examples.append(
        make_example(
            "control",
            action=rng.uniform(-1, 1, 4000).astype(np.float32),
            state=rng.uniform(-1, 1, 4000).astype(np.float32),
            sample_rate=8000,
        )
    )
    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=2, num_modalities=8)
    dc = DataConfig(multimodal_examples=examples, max_len=256, batch_size=2, num_workers=0)
    pipe = KairosMultimodalPipeline(model_config, dc, TrainConfig(run_dir="unused"), tokenizer=tokenizer)
    pipe.build()

    assert dc.multimodal_examples is None  # build() frees the raw examples to save RAM
    report = pipe.data_report()
    assert isinstance(report, BuiltDatasetReport)
    assert report.total_rows > 0
    assert sum(report.modality_tokens.values()) > 0


def test_pipeline_data_report_raises_when_nothing_available():
    from kairos.modeling import KairosConfig
    from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=2, num_modalities=8)
    dc = DataConfig(text_examples=[])
    pipe = KairosMultimodalPipeline(model_config, dc, TrainConfig(run_dir="unused"))
    with pytest.raises(RuntimeError):
        pipe.data_report()


def test_pipeline_data_report_supports_eval_split(tokenizer, rng):
    from kairos.dataset import BuiltDatasetReport
    from kairos.modeling import KairosConfig
    from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

    train_ex = [{"modality": "text", "text": "hello world foo bar"} for _ in range(5)]
    eval_ex = [{"modality": "text", "text": "eval example only"} for _ in range(2)]
    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=2, num_modalities=8)
    dc = DataConfig(multimodal_examples=train_ex, max_len=64, batch_size=2, num_workers=0)
    edc = DataConfig(multimodal_examples=eval_ex, max_len=64, batch_size=2, num_workers=0, shuffle=False, drop_last=False)
    pipe = KairosMultimodalPipeline(model_config, dc, TrainConfig(run_dir="unused"), eval_data_config=edc, tokenizer=tokenizer)
    pipe.build()

    train_report = pipe.data_report(split="train")
    eval_report = pipe.data_report(split="eval")
    assert isinstance(train_report, BuiltDatasetReport)
    assert isinstance(eval_report, BuiltDatasetReport)
    assert train_report.total_rows != eval_report.total_rows  # different splits, different sizes


def test_pipeline_data_report_eval_split_without_eval_config_raises():
    from kairos.modeling import KairosConfig
    from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=2, num_modalities=8)
    dc = DataConfig(multimodal_examples=[{"modality": "text", "text": "hi"}])
    pipe = KairosMultimodalPipeline(model_config, dc, TrainConfig(run_dir="unused"))
    with pytest.raises(RuntimeError):
        pipe.data_report(split="eval")


def test_pipeline_data_report_invalid_split_raises():
    from kairos.modeling import KairosConfig
    from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=2, num_modalities=8)
    dc = DataConfig(multimodal_examples=[{"modality": "text", "text": "hi"}])
    pipe = KairosMultimodalPipeline(model_config, dc, TrainConfig(run_dir="unused"))
    with pytest.raises(ValueError):
        pipe.data_report(split="bogus")



def test_modality_counts_groups_and_sorts_descending():
    from kairos.dataset import modality_counts

    examples = [{"modality": "text"}] * 5 + [{"modality": "control"}] * 2 + [{"modality": "lidar"}] * 1
    counts = modality_counts(examples)
    assert counts == {"text": 5, "control": 2, "lidar": 1}
    assert list(counts.keys()) == ["text", "control", "lidar"]  # sorted by count, descending


def test_split_examples_respects_eval_pct_and_is_deterministic():
    from kairos.dataset import split_examples

    examples = [{"modality": "text", "text": str(i)} for i in range(100)]
    train_a, eval_a = split_examples(examples, eval_pct=20, seed=0)
    train_b, eval_b = split_examples(examples, eval_pct=20, seed=0)
    assert len(eval_a) == 20
    assert len(train_a) == 80
    assert train_a == train_b and eval_a == eval_b  # same seed -> same split
    assert {ex["text"] for ex in train_a}.isdisjoint(ex["text"] for ex in eval_a)


def test_split_examples_keeps_contiguous_modality_runs_adjacent_and_ordered():
    """Regression: control transitions were scattered essentially at random by a plain
    per-example shuffle, destroying the S,A,S,A pattern once examples are packed."""
    from kairos.dataset import split_examples

    examples = (
        [{"modality": "text", "id": f"t{i}"} for i in range(20)]
        + [{"modality": "control", "id": f"c{i}"} for i in range(6)]
        + [{"modality": "image_caption", "id": f"i{i}"} for i in range(20)]
    )
    # eval_pct=0: nothing to force the run apart, so we can check pure adjacency/order
    train, eval_ = split_examples(examples, eval_pct=0, seed=0, contiguous_modalities={"control"})
    control_ids = [ex["id"] for ex in train if ex["modality"] == "control"]

    assert control_ids == [f"c{i}" for i in range(6)]  # relative order preserved, all adjacent
    assert eval_ == []


def test_split_examples_contiguous_run_survives_being_split_across_train_eval():
    """A run that straddles the train/eval cutoff must still keep its relative order
    within each split (a train/eval boundary can still cut through a run)."""
    from kairos.dataset import split_examples

    examples = [{"modality": "control", "id": f"c{i}"} for i in range(6)]
    train, eval_ = split_examples(examples, eval_pct=50, seed=0, contiguous_modalities={"control"})
    train_ids = [ex["id"] for ex in train]
    eval_ids = [ex["id"] for ex in eval_]
    original_order = [f"c{i}" for i in range(6)]

    assert train_ids == [i for i in original_order if i in train_ids]
    assert eval_ids == [i for i in original_order if i in eval_ids]
    assert set(train_ids) | set(eval_ids) == set(original_order)  # nothing lost


def test_split_examples_non_grouped_modalities_still_shuffle_independently():
    """Only modalities in contiguous_modalities get block-shuffled; everything else keeps
    the ordinary per-example shuffle (no accidental grouping of unrelated examples)."""
    from kairos.dataset import split_examples

    examples = [{"modality": "text", "id": f"t{i}"} for i in range(50)]
    train, eval_ = split_examples(examples, eval_pct=50, seed=0, contiguous_modalities={"control"})
    pool_ids = [ex["id"] for ex in train + eval_]
    original_ids = [f"t{i}" for i in range(50)]
    assert set(pool_ids) == set(original_ids)  # nothing lost or duplicated
    assert pool_ids != original_ids  # actually shuffled, not left in input order


def test_split_examples_default_groups_control_without_explicit_opt_in():
    from kairos.dataset import split_examples

    examples = [{"modality": "control", "id": f"c{i}"} for i in range(6)] + [
        {"modality": "text", "id": f"t{i}"} for i in range(20)
    ]
    train, eval_ = split_examples(examples, eval_pct=10, seed=1)  # no contiguous_modalities passed
    pool = train + eval_
    control_ids = [ex["id"] for ex in pool if ex["modality"] == "control"]
    assert control_ids == [f"c{i}" for i in range(6)]


def test_text_dataset_defaults_to_cosmopedia_when_no_texts_given(tokenizer, monkeypatch):
    def fake_config_names(name):
        return ["sample"]

    def fake_load_dataset(name, config, split):
        from datasets import Dataset as HFDataset

        return HFDataset.from_dict({"text": ["Paris is the capital of France."]})

    def fake_concatenate(parts):
        return parts[0]

    monkeypatch.setattr("kairos.dataset.get_dataset_config_names", fake_config_names)
    monkeypatch.setattr("kairos.dataset.load_dataset", fake_load_dataset)
    monkeypatch.setattr("kairos.dataset.concatenate_datasets", fake_concatenate)
    ds = KairosPretrainingDataset(texts=None, tokenizer=tokenizer, max_len=32, stride=4)
    assert len(ds) > 0
