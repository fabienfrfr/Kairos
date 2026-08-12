import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    # marimo without widget for Jupyter/Colab compatibility. (but only ok with mo.status.progress_bar) -
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # !pip install -q -e ".[notebook]"   # uncomment on a bare Colab/Kaggle
    import random
    from pathlib import Path

    import torch
    import pandas as pd

    from kairos.modeling import KairosConfig
    from kairos.tokenizer import KairosTokenizer, Modality
    from kairos.pipeline import KairosMultimodalPipeline, DataConfig, TrainConfig

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    tokenizer = KairosTokenizer()
    print(f"vocab size: {len(tokenizer)}")
    return (
        DataConfig,
        KairosConfig,
        KairosMultimodalPipeline,
        Modality,
        Path,
        TrainConfig,
        pd,
        random,
        tokenizer,
        torch,
    )


@app.cell
def _():
    # ---- data settings ----
    MULTIMODAL_SOURCE = "hf"  # "hf" or "local" (.pt built by build_keep_it_simple_multimodal.py)
    MULTIMODAL_LOCAL_PATH = "data/keep-it-simple-multimodal.pt"
    BUILD_LOCAL_IF_MISSING = False

    TEXT_SOURCE = "hf"  # "hf" (ffurfaro/keep-it-simple) or "inline" (tiny built-in sample)
    TEXT_PCT = 2  # % of keep-it-simple to load, only used if TEXT_SOURCE == "hf"

    EVAL_PCT = 10  # % held out for eval
    return (
        BUILD_LOCAL_IF_MISSING,
        EVAL_PCT,
        MULTIMODAL_LOCAL_PATH,
        MULTIMODAL_SOURCE,
        TEXT_PCT,
        TEXT_SOURCE,
    )


@app.cell
def _(
    BUILD_LOCAL_IF_MISSING,
    MULTIMODAL_LOCAL_PATH,
    MULTIMODAL_SOURCE,
    Path,
    torch,
):
    if MULTIMODAL_SOURCE == "hf":
        from datasets import load_dataset as _load_dataset

        multimodal_examples = list(_load_dataset("ffurfaro/keep-it-simple-multimodal", split="train"))
    else:
        _path = Path(MULTIMODAL_LOCAL_PATH)
        if BUILD_LOCAL_IF_MISSING and not _path.exists():
            import subprocess

            subprocess.run(["python3", "scripts/pretrain/build_keep_it_simple_multimodal.py"], check=True)
        multimodal_examples = torch.load(_path, weights_only=False) if _path.exists() else []

    print(f"multimodal examples: {len(multimodal_examples)}")
    return (multimodal_examples,)


@app.cell
def _(TEXT_PCT, TEXT_SOURCE):
    if TEXT_SOURCE == "hf":
        try:
            from datasets import load_dataset

            _ds = load_dataset("ffurfaro/keep-it-simple", split=f"train[:{TEXT_PCT}%]")
            text_examples = [{"modality": "text", "text": f"{row['prompt']} {row['text']}".strip()} for row in _ds]
        except Exception as e:  # noqa: BLE001 — network/dataset-availability failure, fall back to a tiny inline sample
            print(f"[fallback] keep-it-simple unavailable ({e}) - using inline sample")
            text_examples = [
                {"modality": "text", "text": "Paris is the capital of France."},
                {"modality": "text", "text": "The Earth orbits the Sun."},
            ]
    else:
        text_examples = [
            {"modality": "text", "text": "Paris is the capital of France."},
            {"modality": "text", "text": "The Earth orbits the Sun."},
            {"modality": "text", "text": "Water boils at 100 degrees Celsius."},
        ]
    print(f"text examples: {len(text_examples)}")
    return (text_examples,)


@app.cell
def _(multimodal_examples, pd, text_examples):
    _all_ex = list(text_examples) + list(multimodal_examples)
    print(f"total examples: {len(_all_ex)}")
    print(pd.Series([ex["modality"] for ex in _all_ex]).value_counts())
    return


@app.cell
def _():
    SHOW_PREVIEW = True  # set False to skip decoding/plotting sample content
    return (SHOW_PREVIEW,)


@app.cell
def _(SHOW_PREVIEW, multimodal_examples):
    if SHOW_PREVIEW and multimodal_examples:
        import json as _json

        import matplotlib.pyplot as plt
        import numpy as np

        from kairos.dataset import unpack_multimodal_data

        _by_modality: dict[str, list] = {}
        for _ex in multimodal_examples:
            _by_modality.setdefault(_ex["modality"], []).append(_ex)

        for _mod, _rows in _by_modality.items():
            print(f"--- {_mod} ({len(_rows)} examples) ---")
            _sample = np.random.default_rng(0).choice(len(_rows), size=min(3, len(_rows)), replace=False)
            for _i in _sample:
                _row = _rows[_i]
                _arrays = unpack_multimodal_data(_row["data"])
                _caption = (_row.get("caption") or "")[:80]
                _meta = _json.loads(_row["meta"]) if _row.get("meta") else {}
                print(f"  caption: {_caption!r}  meta: {_meta}")

                if _mod == "image_caption":
                    plt.figure(figsize=(2, 2))
                    plt.imshow(_arrays["image"])
                    plt.axis("off")
                    plt.show()
                elif _mod == "audio_caption":
                    plt.figure(figsize=(3, 1))
                    plt.plot(_arrays["audio"], linewidth=0.5)
                    plt.axis("off")
                    plt.show()
                elif _mod == "video_caption":
                    _video = _arrays["video"]
                    _n = min(4, _video.shape[0])
                    _fig, _axes = plt.subplots(1, _n, figsize=(_n * 1.2, 1.2))
                    for _j, _ax in enumerate(_axes if _n > 1 else [_axes]):
                        _ax.imshow(_video[_j])
                        _ax.axis("off")
                    plt.show()
                elif _mod == "lidar":
                    _points = np.asarray(_arrays["points"], dtype=np.float32)
                    _fig = plt.figure(figsize=(2.5, 2.5))
                    _ax = _fig.add_subplot(projection="3d") if _points.shape[1] >= 3 else _fig.add_subplot()
                    if _points.shape[1] >= 3:
                        _ax.scatter(_points[:, 0], _points[:, 1], _points[:, 2], s=1)
                    else:
                        _ax.scatter(_points[:, 0], _points[:, 1], s=1)
                    plt.show()
                elif _mod == "control":
                    plt.figure(figsize=(3, 1.2))
                    plt.plot(_arrays["state"], label="state")
                    plt.plot(_arrays["action"], label="action")
                    plt.legend(fontsize=6)
                    plt.show()
    else:
        print("preview skipped")
    return


@app.cell
def _(EVAL_PCT, multimodal_examples, random, text_examples):
    _all_ex = list(text_examples) + list(multimodal_examples)
    _rng = random.Random(0)
    _shuffled = _all_ex.copy()
    _rng.shuffle(_shuffled)
    _n_eval = int(len(_shuffled) * EVAL_PCT / 100)
    eval_examples = _shuffled[:_n_eval]
    train_examples = _shuffled[_n_eval:]
    print(f"train: {len(train_examples)} ({100 - EVAL_PCT}%)  eval: {len(eval_examples)} ({EVAL_PCT}%)")
    return eval_examples, train_examples


@app.cell
def _():
    # ---- model settings ----
    # modality_scales routes each modality to a PyramidalConvCodec scale; attnres_block_size sets
    # the v3 Block-AttnRes window (1 = classic AttnRes)
    CFG_D_MODEL = 88
    CFG_N_HEADS = 4
    CFG_N_LAYERS = 4
    CFG_STRIDE = 3
    CFG_NUM_SCALES = 4
    CFG_ATTNRES_BLOCK = 4
    CFG_EXPERTS = 7  # 0 = dense FFN
    CFG_EXPERTS_PER_TOK = 1
    CFG_SHARED_EXPERTS = 1
    CFG_INTERMEDIATE = 352
    return (
        CFG_ATTNRES_BLOCK,
        CFG_D_MODEL,
        CFG_EXPERTS,
        CFG_EXPERTS_PER_TOK,
        CFG_INTERMEDIATE,
        CFG_NUM_SCALES,
        CFG_N_HEADS,
        CFG_N_LAYERS,
        CFG_SHARED_EXPERTS,
        CFG_STRIDE,
    )


@app.cell
def _(Modality):
    # scale 0: finest temporal res (text, control) · 1: images/lidar · 2: audio/video frames · 3: coarse/META
    modality_scales = {
        int(Modality.TEXT): [0, 1],
        int(Modality.STATE): [0],
        int(Modality.ACTION): [0],
        int(Modality.IMAGE): [1, 2],
        int(Modality.LIDAR): [1],
        int(Modality.AUDIO): [2],
        int(Modality.VIDEO): [2, 3],
        int(Modality.META): [3],
    }
    return (modality_scales,)


@app.cell
def _(
    CFG_ATTNRES_BLOCK,
    CFG_D_MODEL,
    CFG_EXPERTS,
    CFG_EXPERTS_PER_TOK,
    CFG_INTERMEDIATE,
    CFG_NUM_SCALES,
    CFG_N_HEADS,
    CFG_N_LAYERS,
    CFG_SHARED_EXPERTS,
    CFG_STRIDE,
    KairosConfig,
    modality_scales,
    tokenizer,
):
    use_moe = CFG_EXPERTS > 0

    model_config = KairosConfig(
        d_model=CFG_D_MODEL,
        n_heads=CFG_N_HEADS,
        n_layers=CFG_N_LAYERS,
        stride=CFG_STRIDE,
        vocab_size=len(tokenizer),
        num_modalities=8,
        num_scales=CFG_NUM_SCALES,
        modality_scales=modality_scales,
        intermediate_size=CFG_INTERMEDIATE,
        moe_intermediate_size=CFG_INTERMEDIATE,
        n_routed_experts=CFG_EXPERTS if use_moe else 8,
        num_local_experts=CFG_EXPERTS if use_moe else 8,  # DeepseekV3MoE backend reads this one, not n_routed_experts
        num_experts_per_tok=CFG_EXPERTS_PER_TOK,
        n_shared_experts=CFG_SHARED_EXPERTS,
        use_moe=use_moe,
        attnres_block_size=CFG_ATTNRES_BLOCK,
    )
    print(f"moe: {use_moe}  block-attnres window: {CFG_ATTNRES_BLOCK}  memory_bank: {CFG_USE_MEMORY_BANK}")
    return (model_config,)


@app.cell
def _():
    # ---- training settings ----
    TRAIN_LR = 3e-4
    TRAIN_BATCH = 8
    TRAIN_EPOCHS = 3
    TRAIN_MAX_LEN = 1024
    TRAIN_STRIDE = 3
    TRAIN_SAVE_EVERY = 200
    TRAIN_RUN_DIR = "checkpoints/kairos-multimodal/run_01"  # keep unchanged across restarts to auto-resume

    # ---- packing: concatenate samples before chunking so only the last chunk is padded ----
    TRAIN_PACK = True

    # ---- HF hub push-per-checkpoint (optional) ----
    HUB_REPO_ID = None  # e.g. "ffurfaro/kairos" - set to also push each checkpoint as it's saved
    HUB_PUSH_EVERY_CKPT = False
    HUB_PRIVATE = False
    HUB_SUBFOLDER = None  # e.g. "run_01" - push under repo_id/<subfolder> so multiple runs share one repo
    return (
        HUB_PRIVATE,
        HUB_PUSH_EVERY_CKPT,
        HUB_REPO_ID,
        HUB_SUBFOLDER,
        TRAIN_BATCH,
        TRAIN_EPOCHS,
        TRAIN_LR,
        TRAIN_MAX_LEN,
        TRAIN_PACK,
        TRAIN_RUN_DIR,
        TRAIN_SAVE_EVERY,
        TRAIN_STRIDE,
    )


@app.cell
def _(
    DataConfig,
    HUB_PRIVATE,
    HUB_PUSH_EVERY_CKPT,
    HUB_REPO_ID,
    HUB_SUBFOLDER,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_MAX_LEN,
    TRAIN_PACK,
    TRAIN_RUN_DIR,
    TRAIN_SAVE_EVERY,
    TRAIN_STRIDE,
    TrainConfig,
    eval_examples,
    train_examples,
):
    data_config = DataConfig(
        text_examples=[],
        multimodal_examples=train_examples,
        max_len=TRAIN_MAX_LEN,
        stride=TRAIN_STRIDE,
        batch_size=TRAIN_BATCH,
        pack=TRAIN_PACK,
    )
    eval_data_config = DataConfig(
        text_examples=[],
        multimodal_examples=eval_examples,
        max_len=TRAIN_MAX_LEN,
        stride=TRAIN_STRIDE,
        batch_size=TRAIN_BATCH,
        shuffle=False,
        drop_last=False,
    )
    train_config = TrainConfig(
        lr=TRAIN_LR,
        epochs=TRAIN_EPOCHS,
        save_every=TRAIN_SAVE_EVERY,
        run_dir=TRAIN_RUN_DIR,
        hub_repo_id=HUB_REPO_ID,
        hub_push_every_ckpt=HUB_PUSH_EVERY_CKPT,
        hub_private=HUB_PRIVATE,
        hub_subfolder=HUB_SUBFOLDER,
    )
    return data_config, eval_data_config, train_config


@app.cell
def _(
    KairosMultimodalPipeline,
    data_config,
    model_config,
    tokenizer,
    train_config,
):
    from kairos.utils import count_active_parameters

    pipe = KairosMultimodalPipeline(model_config, data_config, train_config, tokenizer=tokenizer)
    pipe.build()

    total_params = sum(p.numel() for p in pipe.model.parameters())
    active_params = count_active_parameters(
        pipe.model,
        model_config.num_experts_per_tok if model_config.use_moe else None,
        model_config.num_local_experts if model_config.use_moe else None,
    )
    print(f"total params: {total_params / 1e6:.2f}M  active params/tok: {active_params / 1e6:.2f}M")
    print(f"device: {pipe.device}  samples: {len(pipe.dataset)}")
    return (pipe,)


@app.cell
def _():
    # compute-cost summary: params/memory instantly, plus an estimated total training time from a
    # few real timed steps (state is restored right after, so this doesn't affect training below)
    RUN_BENCHMARK = True
    N_BENCH_STEPS = 5
    return N_BENCH_STEPS, RUN_BENCHMARK


@app.cell
def _(N_BENCH_STEPS, RUN_BENCHMARK, pipe):
    cost_summary = pipe.summary(benchmark=RUN_BENCHMARK, n_bench_steps=N_BENCH_STEPS)
    print(cost_summary)
    return


@app.cell
def _(pd, pipe):
    # visualize the tokenized input exactly as the model receives it (post-tokenization,
    # post-collation) - use this to rule data in/out before suspecting the architecture.
    # note: a single row can (and often does) mix several modalities at once, since text/image/
    # audio/... segments get concatenated into one token sequence before chunking.
    _reports = pipe.inspect_batch(n=1)
    _table = pd.DataFrame(
        [
            {
                "row": r["row"],
                "modality_counts": r["modality_counts"],
                "token_id_range": r["token_id_range"],
                "top_token_ids": r["top_token_ids"],  # [(id, count), ...] - most frequent raw ids
                "max_repeat_run": r["max_repeat_run"],  # longest run of one id repeated in a row
                "out_of_bounds_tokens": len(r["out_of_bounds"]["token_ids"]),
                "out_of_bounds_modality": len(r["out_of_bounds"]["modality_ids"]),
                "pad_frac": round(r["pad_frac"], 3) if r["pad_frac"] is not None else None,
            }
            for r in _reports
        ]
    )
    n_oob = _table["out_of_bounds_tokens"].sum() + _table["out_of_bounds_modality"].sum()
    if n_oob:
        print(f"WARNING: {n_oob} out-of-bounds ids found in this batch - inspect before training")
    max_run = max(r["max_repeat_run"]["length"] for r in _reports)
    if max_run > 50:  # arbitrary but generous threshold; a real sequence rarely repeats this much
        print(f"WARNING: a row repeats the same token id {max_run} times in a row - likely corrupted")
    print(_table.to_string())

    # raw numeric view of the first row: exactly what the embedding layer indexes with
    print("\nrow 0 input_ids  :", _reports[0]["input_ids"])
    print("row 0 modality_ids:", _reports[0]["modality_ids"])
    return


@app.cell
def _():
    FORCE_RESTART = True  # True ignores any existing last.pt / hub checkpoint and starts from step 0
    return (FORCE_RESTART,)


@app.cell
def _(FORCE_RESTART, mo, pipe):
    _resumed = not FORCE_RESTART and (pipe.ckpt_dir / "last.pt").exists()
    if _resumed:
        print(f"found last.pt in {pipe.ckpt_dir} - resuming")
    elif FORCE_RESTART:
        print("FORCE_RESTART is True - ignoring any existing checkpoint")

    _total_steps = pipe.train_config.epochs * len(pipe.loader)

    if mo.running_in_notebook():
        with mo.status.progress_bar(total=_total_steps, title="training") as _bar:
            _state = {"last_step": 0}

            def _on_step(step, total, loss_val):
                _bar.update(increment=step - _state["last_step"], subtitle=f"loss={loss_val:.4f}")
                _state["last_step"] = step

            logs = pipe.train(progress_callback=_on_step, resume=not FORCE_RESTART)
    else:
        from kairos.utils import make_progress_callback

        logs = pipe.train(progress_callback=make_progress_callback(), resume=not FORCE_RESTART)

    print(f"training complete - steps: {len(logs)}  best avg-epoch loss: {pipe.best_loss:.4f}")
    print(f"skipped non-finite batches: {pipe.skipped_nonfinite_steps}")
    print(f"checkpoints: {pipe.ckpt_dir}")
    return (logs,)


@app.cell
def _(eval_data_config, pipe, torch):
    from kairos.dataset import KairosPretrainingDataset
    from torch.utils.data import DataLoader

    if len(eval_data_config.multimodal_examples) == 0:
        print("no eval examples - skipping")
    else:
        eval_dataset = KairosPretrainingDataset(
            multimodal_examples=eval_data_config.multimodal_examples,
            tokenizer=pipe.tokenizer,
            max_len=eval_data_config.max_len,
            stride=eval_data_config.stride,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=eval_data_config.batch_size, shuffle=False)

        pipe.model.eval()
        losses = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(pipe.device) for k, v in batch.items()}
                losses.append(pipe.hf_trainer.compute_loss(pipe.model, batch).item())
        pipe.model.train()
        eval_loss = sum(losses) / len(losses)
        pipe.writer.add_scalar("eval/loss", eval_loss, pipe.global_step)
        print(f"eval loss: {eval_loss:.4f} on {len(eval_dataset)} samples")
    return


@app.cell
def _(logs, pd):
    logs_df = pd.DataFrame(logs)
    print(logs_df)
    return (logs_df,)


@app.cell
def _(logs_df):
    if not logs_df.empty:
        logs_df.plot(x="step", y="loss", figsize=(8, 4), title="Multimodal diffusion loss")
    else:
        print("no logged steps - nothing to plot")
    return


@app.cell
def _(pipe):
    # diffusion loss masked per modality, so a router-ignored one can't hide behind the global average
    per_modality = pipe.check_per_modality_loss(n_batches=10)
    for _k, _v in sorted(per_modality.items(), key=lambda kv: -kv[1]):
        print(f"{_k}: {_v:.4f}")
    return


@app.cell
def _():
    # not needed for a simple crash-recovery (training already auto-resumes from last.pt or the hub) -
    # use this only to load a *different* checkpoint, e.g. best.pt or an older step_*.pt
    RESUME_CKPT_PATH = ""
    return (RESUME_CKPT_PATH,)


@app.cell
def _(RESUME_CKPT_PATH, pipe):
    if RESUME_CKPT_PATH:
        _ckpt = pipe.load_checkpoint(RESUME_CKPT_PATH)
        print(f"loaded {RESUME_CKPT_PATH} - step {_ckpt.get('step', '?')}  loss {_ckpt.get('loss', 0):.4f}")
    else:
        print("RESUME_CKPT_PATH is empty - nothing to do")
    return


@app.cell
def _():
    PUSH_FULL_MODEL_TO_HUB = False  # pushes weights+config+model card, on top of any per-checkpoint pushes above
    FULL_MODEL_HUB_REPO_ID = "ffurfaro/kairos"
    return FULL_MODEL_HUB_REPO_ID, PUSH_FULL_MODEL_TO_HUB


@app.cell
def _(FULL_MODEL_HUB_REPO_ID, HUB_PRIVATE, PUSH_FULL_MODEL_TO_HUB, pipe):
    if PUSH_FULL_MODEL_TO_HUB:
        pipe.push_to_hub(FULL_MODEL_HUB_REPO_ID, private=HUB_PRIVATE)
        print(f"pushed to https://huggingface.co/{FULL_MODEL_HUB_REPO_ID}")
    else:
        print("PUSH_FULL_MODEL_TO_HUB is False - skipping")
    return


if __name__ == "__main__":
    app.run()
