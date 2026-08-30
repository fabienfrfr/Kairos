"""Tests for the pure (no-network) helpers in build_keep_it_simple_multimodal.py.
Loaded directly from its file path via importlib since it isn't a kairos submodule."""

import importlib.util
import os

_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "pretrain", "build_keep_it_simple_multimodal.py"
)
_spec = importlib.util.spec_from_file_location("build_keep_it_simple_multimodal", _SCRIPT_PATH)
build_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_script)


def test_control_max_seconds_is_a_safety_cap_not_a_target_duration():
    """Regression: control used to reuse/derive a target duration and stretch/pad every clip
    to it. Real clips are ~0.06s; the cap must be a generous outlier guard, not a rescale."""
    assert build_script.CONTROL_MAX_SECONDS > 1.0  # generous vs. the real ~0.06s clip length
    assert build_script.CONTROL_MAX_SECONDS < 60.0  # still a real safety net, not unbounded


def test_time_stretch_to_fixed_length_still_used_for_audio_caption():
    """audio_caption legitimately wants a fixed-duration snippet (unlike control); make sure
    the warp-based helper wasn't accidentally removed along with the control-specific one."""
    import numpy as np

    out = build_script._time_stretch_to_fixed_length(np.zeros(50, dtype=np.float32), 80)
    assert out.shape[0] == 80
