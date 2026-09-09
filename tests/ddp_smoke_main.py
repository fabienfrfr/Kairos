"""torchrun entrypoint for the 2-process DDP smoke test; run_dir passed as argv[1]."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from kairos.modeling import KairosConfig
from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig

run_dir = Path(sys.argv[1])
texts = [{"modality": "text", "text": f"sentence {i} padded to a few dozen tokens."} for i in range(8)]
dc = DataConfig(text_examples=texts, max_len=64, batch_size=2, num_workers=0)
edc = DataConfig(text_examples=texts[:4], max_len=64, batch_size=2, num_workers=0)
tc = TrainConfig(
    epochs=2,
    mae_epochs=0,
    transition_epochs=0,
    run_dir=str(run_dir),
    save_every=3,
    eval_batches=2,
)
cfg = KairosConfig(d_model=16, n_heads=2, n_layers=1, num_modalities=8, attnres_block_size=2, use_memory_gate=True)

pipe = KairosMultimodalPipeline(cfg, dc, tc, eval_data_config=edc)
pipe.build()
assert pipe.distributed, "expected DDP active"
assert pipe.world_size == 2 and pipe.rank == int(os.environ["RANK"])
assert isinstance(pipe.model_forward, torch.nn.parallel.DistributedDataParallel)
assert (pipe.is_main_process) == (pipe.rank == 0)

rows = pipe.train()
assert pipe.global_step == 4, pipe.global_step
if pipe.is_main_process:
    assert len(pipe.log_rows) == 4, len(pipe.log_rows)
    assert (run_dir / "checkpoints" / "step_000003.pt").exists()
else:
    assert len(rows) == 0

# clean completion deletes last.pt; simulate an interrupted run instead
if pipe.is_main_process:
    pipe._save(run_dir / "checkpoints" / "last.pt", 0.0, epoch=2)
dist.barrier()

pipe.train_config.epochs = 4
rows2 = pipe.train()  # resumes epochs 2..4 = 6 more steps; rank1 reads rank0's last.pt
assert pipe.global_step == 10, pipe.global_step
assert len(rows2) == (10 if pipe.is_main_process else 0)

dist.destroy_process_group()
print(f"RESULT rank={pipe.rank} ok steps={pipe.global_step}", flush=True)