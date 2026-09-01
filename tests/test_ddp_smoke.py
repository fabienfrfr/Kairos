"""2-process torchrun/gloo smoke test for the DDP train path; skipped unless KAIROS_DDP_SMOKE=1."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("KAIROS_DDP_SMOKE") != "1",
    reason="set KAIROS_DDP_SMOKE=1 to run the 2-process torchrun smoke test",
)


def test_ddp_two_process_smoke(tmp_path):
    main_script = Path(__file__).parent / "ddp_smoke_main.py"
    run_dir = tmp_path / "ddp_run"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "--rdzv-backend=c10d",
        str(main_script),
        str(run_dir),
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT rank=0 ok steps=10" in result.stdout
    assert "RESULT rank=1 ok steps=10" in result.stdout