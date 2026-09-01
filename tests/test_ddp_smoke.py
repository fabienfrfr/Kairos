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


def _run_torchrun(tmp_path, script, *args):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "--rdzv-backend=c10d",
        str(script),
        *args,
    ]
    return subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=600, check=False)


def test_ddp_two_process_smoke(tmp_path):
    main_script = Path(__file__).parent / "ddp_smoke_main.py"
    result = _run_torchrun(tmp_path, main_script, str(tmp_path / "ddp_run"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT rank=0 ok steps=10" in result.stdout
    assert "RESULT rank=1 ok steps=10" in result.stdout


def test_launch_ddp_spawns_torchrun_job(tmp_path):
    """tau sends fresh configs through launch_ddp's pickled-job path (kairos/_entry_ddp.py)."""
    from kairos.modeling import KairosConfig
    from kairos.pipeline import DataConfig, TrainConfig, launch_ddp

    texts = [{"modality": "text", "text": f"sentence {i} padded to a few dozen tokens."} for i in range(8)]
    model_config = KairosConfig(d_model=16, n_heads=2, n_layers=1, num_modalities=8, use_memory_gate=True)
    data_config = DataConfig(text_examples=texts, max_len=64, batch_size=2, num_workers=0)
    train_config = TrainConfig(epochs=1, mae_epochs=0, transition_epochs=0, run_dir=str(tmp_path / "run"))

    launch_ddp(model_config, data_config, None, train_config, n_proc=2)
    log = Path(train_config.run_dir) / "train_ddp.log"
    assert log.exists()
    assert "training complete - steps:" in log.read_text()