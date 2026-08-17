import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    from huggingface_hub import login

    # set via Kaggle > Secrets > Add secret (key: HF_TOKEN), or paste the token here
    HF_TOKEN = os.environ.get("HF_TOKEN", "hf_xxx")
    login(token=HF_TOKEN, add_to_git_credential=False)
    return


@app.cell
def _():
    # marimo without widget for Jupyter/Colab compatibility.
    try:
        import marimo as mo
    except ImportError:  # plain Jupyter/Colab (no marimo): disable marimo widgets

        class _MoStub:
            @staticmethod
            def running_in_notebook() -> bool:
                return False

        mo = _MoStub()
    return (mo,)


@app.cell
def _():
    # --force-reinstall is required: pip skips reinstalling if the version is unchanged.
    # !pip install -q --force-reinstall git+https://github.com/fabienfrfr/Kairos@dev
    return


@app.cell
def _():
    import os
    import random
    from pathlib import Path

    # auto: fused flex_attention on SM>=7.0 GPUs (T4), eager O(L*W) below that.
    os.environ.setdefault("KAIROS_ATTN_BACKEND", "auto")

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
    MULTIMODAL_SOURCE = "hf"  # "hf" or "local" (.pt built
    MULTIMODAL_LOCAL_PATH = "data/keep-it-simple-multimodal.pt"
    BUILD_LOCAL_IF_MISSING = False

    TEXT_SOURCE = "hf"  # "hf" (ffurfaro/keep-it-simple) or "inline" (tiny
    TEXT_PCT = 10  # % of keep-it-simple to load; was 2%, too little data for enough optimizer steps

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
    SHOW_PREVIEW = True  # set False to skip decoding/plotting
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
    # modality_scales routes each modality to a
    # the v3 Block-AttnRes window (1 =
    CFG_D_MODEL = 64  # head_dim = 64/4 = 16, power of 2 for flex_attention
    CFG_N_HEADS = 4
    CFG_N_LAYERS = 4
    CFG_STRIDE = 3
    CFG_NUM_SCALES = 4
    CFG_ATTNRES_BLOCK = 4
    CFG_EXPERTS = 7  # 0 = dense FFN. Top-1 routing (see CFG_EXPERTS_PER_TOK) is slow to converge on tiny
    # overfit-test runs since gradient only reaches the chosen expert per token; set to 0 to isolate whether
    # the backbone itself can memorize before blaming routing.
    CFG_EXPERTS_PER_TOK = 1
    CFG_SHARED_EXPERTS = 1
    CFG_INTERMEDIATE = 544  # raised to keep ~14-15M total params after d_model 88->64
    CFG_USE_MEMORY_BANK = True  # cross-session DeltaNet state gating
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
        CFG_USE_MEMORY_BANK,
    )


@app.cell
def _(Modality):
    # scale 0: finest temporal res (text,
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
    CFG_USE_MEMORY_BANK,
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
        num_local_experts=CFG_EXPERTS if use_moe else 8,  # DeepseekV3MoE backend reads this one,
        num_experts_per_tok=CFG_EXPERTS_PER_TOK,
        n_shared_experts=CFG_SHARED_EXPERTS,
        use_moe=use_moe,
        attnres_block_size=CFG_ATTNRES_BLOCK,
        use_memory_gate=CFG_USE_MEMORY_BANK,
    )
    print(f"moe: {use_moe}  block-attnres window: {CFG_ATTNRES_BLOCK}  memory_bank: {CFG_USE_MEMORY_BANK}")
    return (model_config,)


@app.cell
def _():
    # ---- training settings ----
    TRAIN_LR = 3e-4
    TRAIN_BATCH = 32
    TRAIN_MAX_LEN = 1024
    TRAIN_STRIDE = 3
    TRAIN_SAVE_EVERY = 200
    TRAIN_MASK_EPS = 1e-3  # floor of masked-diffusion rate p (CE/p); lower -> more variance, harder to overfit fast

    # ---- two-stage curriculum: Stage 1 (MAE, fixed-rate bidirectional denoising, cheap/stable) bootstraps
    # the backbone, then Stage 2 (full diffusion, p up to 1.0, CE/p-weighted) resumes from Stage 1's weights
    # and fine-tunes on the real generative objective. Set *_EPOCHS = 0 to skip a stage entirely.
    TRAIN_MAE_EPOCHS = 5  # was 1, too few steps to leave the near-random-init loss (fixed codec bottleneck; see PyramidalCodec)
    TRAIN_MAE_P_MAX = 0.3  # MAE stage ceiling on p: fixed-ish low corruption, easy/stable to optimize
    TRAIN_MAE_REWEIGHT = False  # plain CE in MAE stage: no 1/p variance blowup

    TRAIN_DIFFUSION_EPOCHS = 5  # was 3, same reasoning as TRAIN_MAE_EPOCHS
    TRAIN_MASK_P_MAX = 1.0  # Stage 2 ceiling on p: full diffusion, rows can be up to 100% noised
    TRAIN_MASK_REWEIGHT = True  # Stage 2: divide CE by p (standard masked-diffusion ELBO weighting)

    TRAIN_EVAL_EVERY = 100  # eval on held-out set every N steps (0 = off)
    TRAIN_EVAL_BATCHES = 2  # batches per eval; small keeps it cheap
    TRAIN_RUN_DIR = "checkpoints/kairos-multimodal/run_01"  # keep unchanged across restarts to
    # also bridge Stage 1 -> Stage 2: both stages write/read checkpoints/last.pt in this same directory.

    # ---- packing: concatenate samples before chunking
    TRAIN_PACK = True

    # ---- HF hub push-per-checkpoint (optional) ----
    HUB_REPO_ID = None  # e.g. "ffurfaro/kairos" - set to
    HUB_PUSH_EVERY_CKPT = False
    HUB_PRIVATE = False
    HUB_SUBFOLDER = None  # e.g. "run_01" - push under
    return (
        HUB_PRIVATE,
        HUB_PUSH_EVERY_CKPT,
        HUB_REPO_ID,
        HUB_SUBFOLDER,
        TRAIN_BATCH,
        TRAIN_DIFFUSION_EPOCHS,
        TRAIN_EVAL_BATCHES,
        TRAIN_EVAL_EVERY,
        TRAIN_LR,
        TRAIN_MAE_EPOCHS,
        TRAIN_MAE_P_MAX,
        TRAIN_MAE_REWEIGHT,
        TRAIN_MASK_EPS,
        TRAIN_MASK_P_MAX,
        TRAIN_MASK_REWEIGHT,
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
    TRAIN_DIFFUSION_EPOCHS,
    TRAIN_EVAL_BATCHES,
    TRAIN_EVAL_EVERY,
    TRAIN_LR,
    TRAIN_MAX_LEN,
    TRAIN_PACK,
    TRAIN_MAE_EPOCHS,
    TRAIN_MAE_P_MAX,
    TRAIN_MAE_REWEIGHT,
    TRAIN_MASK_EPS,
    TRAIN_MASK_P_MAX,
    TRAIN_MASK_REWEIGHT,
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
    train_config_mae = TrainConfig(
        lr=TRAIN_LR,
        mask_eps=TRAIN_MASK_EPS,
        mask_p_max=TRAIN_MAE_P_MAX,
        mask_reweight=TRAIN_MAE_REWEIGHT,
        epochs=TRAIN_MAE_EPOCHS,
        save_every=TRAIN_SAVE_EVERY,
        eval_every=TRAIN_EVAL_EVERY,
        eval_batches=TRAIN_EVAL_BATCHES,
        run_dir=TRAIN_RUN_DIR,
        hub_repo_id=HUB_REPO_ID,
        hub_push_every_ckpt=HUB_PUSH_EVERY_CKPT,
        hub_private=HUB_PRIVATE,
        hub_subfolder=HUB_SUBFOLDER,
    )
    train_config = TrainConfig(
        lr=TRAIN_LR,
        mask_eps=TRAIN_MASK_EPS,
        mask_p_max=TRAIN_MASK_P_MAX,
        mask_reweight=TRAIN_MASK_REWEIGHT,
        epochs=TRAIN_DIFFUSION_EPOCHS,
        save_every=TRAIN_SAVE_EVERY,
        eval_every=TRAIN_EVAL_EVERY,
        eval_batches=TRAIN_EVAL_BATCHES,
        run_dir=TRAIN_RUN_DIR,
        hub_repo_id=HUB_REPO_ID,
        hub_push_every_ckpt=HUB_PUSH_EVERY_CKPT,
        hub_private=HUB_PRIVATE,
        hub_subfolder=HUB_SUBFOLDER,
    )
    return data_config, eval_data_config, train_config, train_config_mae


@app.cell
def _(KairosMultimodalPipeline, data_config, mo, model_config, tokenizer, train_config_mae):
    # Stage 1 (MAE): fixed-rate bidirectional denoising, no CE/p reweighting. Cheap sanity check that the
    # backbone can learn/memorize at all, and a stable bootstrap before the harder full-diffusion objective.
    # Writes checkpoints/last.pt in train_config_mae.run_dir, picked up by Stage 2 below via resume=True.
    if train_config_mae.epochs > 0:
        pipe_mae = KairosMultimodalPipeline(model_config, data_config, train_config_mae, tokenizer=tokenizer)
        pipe_mae.build()

        _total_steps = train_config_mae.epochs * len(pipe_mae.loader)
        if mo.running_in_notebook():
            with mo.status.progress_bar(total=_total_steps, title="mae_pretrain") as _bar:
                _state = {"last_step": 0}

                def _on_mae_step(step, total, loss_val):
                    _bar.update(increment=step - _state["last_step"], subtitle=f"loss={loss_val:.4f}")
                    _state["last_step"] = step

                mae_logs = pipe_mae.train(progress_callback=_on_mae_step, resume=True)
        else:
            from kairos.utils import make_progress_callback

            mae_logs = pipe_mae.train(progress_callback=make_progress_callback(desc="mae_pretrain"), resume=True)

        print(f"MAE stage complete - steps: {len(mae_logs)}  best avg-epoch loss: {pipe_mae.best_loss:.4f}")
    else:
        print("TRAIN_MAE_EPOCHS is 0 - skipping the MAE bootstrap stage")
    return


@app.cell
def _(
    KairosMultimodalPipeline,
    data_config,
    eval_data_config,
    model_config,
    tokenizer,
    train_config,
):
    from kairos.utils import count_active_parameters

    pipe = KairosMultimodalPipeline(
        model_config, data_config, train_config, eval_data_config=eval_data_config, tokenizer=tokenizer
    )
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
    # compute-cost summary: params/memory instantly, plus an
    # few real timed steps (state is
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
    # visualize the tokenized input exactly as
    # post-collation) - use this to rule
    # note: a single row can (and
    # audio/... segments get concatenated into one
    _reports = pipe.inspect_batch(n=1)
    _table = pd.DataFrame(
        [
            {
                "row": r["row"],
                "modality_counts": r["modality_counts"],
                "token_id_range": r["token_id_range"],
                "top_token_ids": r["top_token_ids"],  # [(id, count), ...] - most
                "max_repeat_run": r["max_repeat_run"],  # longest run of one id
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
    if max_run > 50:  # arbitrary but generous threshold; a
        print(f"WARNING: a row repeats the same token id {max_run} times in a row - likely corrupted")
    print(_table.to_string())

    # raw numeric view of the first
    print("\nrow 0 input_ids  :", _reports[0]["input_ids"])
    print("row 0 modality_ids:", _reports[0]["modality_ids"])
    return


@app.cell
def _():
    OVERFIT_RUN = True  # sanity-check the model can memorize before the real run
    OVERFIT_EXAMPLES = 64  # tiny subset, repeated each epoch
    OVERFIT_STEPS = 200  # steps on that subset; loss should crash toward 0
    return OVERFIT_EXAMPLES, OVERFIT_RUN, OVERFIT_STEPS


@app.cell
def _(
    OVERFIT_EXAMPLES,
    OVERFIT_RUN,
    OVERFIT_STEPS,
    TRAIN_MAE_P_MAX,
    TRAIN_MAE_REWEIGHT,
    mo,
    pipe,
):
    # MAE-mode overfit test: fixed-rate low corruption, plain CE (matches Stage 1's objective).
    # non-destructive: model/optimizer/loader state, and pipe.hf_trainer's mask settings, are restored afterwards.
    _oflogs_mae = []
    if OVERFIT_RUN:
        if mo.running_in_notebook():
            with mo.status.progress_bar(total=OVERFIT_STEPS, title="overfit_test (MAE)") as _bar:
                _state = {"last_step": 0}

                def _on_step(step, total, loss_val):
                    _bar.update(increment=step - _state["last_step"], subtitle=f"loss={loss_val:.4f}")
                    _state["last_step"] = step

                _oflogs_mae = pipe.overfit_test(
                    n_examples=OVERFIT_EXAMPLES,
                    steps=OVERFIT_STEPS,
                    progress_callback=_on_step,
                    mask_p_max=TRAIN_MAE_P_MAX,
                    mask_reweight=TRAIN_MAE_REWEIGHT,
                )
        else:
            from kairos.utils import make_progress_callback

            _oflogs_mae = pipe.overfit_test(
                n_examples=OVERFIT_EXAMPLES,
                steps=OVERFIT_STEPS,
                progress_callback=make_progress_callback(desc="overfit_test (MAE)"),
                mask_p_max=TRAIN_MAE_P_MAX,
                mask_reweight=TRAIN_MAE_REWEIGHT,
            )
    else:
        print("OVERFIT_RUN is False - skipping MAE overfit test")
    return


@app.cell
def _(
    OVERFIT_EXAMPLES,
    OVERFIT_RUN,
    OVERFIT_STEPS,
    TRAIN_MASK_P_MAX,
    TRAIN_MASK_REWEIGHT,
    mo,
    pipe,
):
    # Diffusion-mode overfit test: full p in [eps, 1], CE/p-weighted (matches Stage 2's real objective).
    # non-destructive: model/optimizer/loader state, and pipe.hf_trainer's mask settings, are restored afterwards.
    _oflogs_diffusion = []
    if OVERFIT_RUN:
        if mo.running_in_notebook():
            with mo.status.progress_bar(total=OVERFIT_STEPS, title="overfit_test (diffusion)") as _bar:
                _state = {"last_step": 0}

                def _on_step(step, total, loss_val):
                    _bar.update(increment=step - _state["last_step"], subtitle=f"loss={loss_val:.4f}")
                    _state["last_step"] = step

                _oflogs_diffusion = pipe.overfit_test(
                    n_examples=OVERFIT_EXAMPLES,
                    steps=OVERFIT_STEPS,
                    progress_callback=_on_step,
                    mask_p_max=TRAIN_MASK_P_MAX,
                    mask_reweight=TRAIN_MASK_REWEIGHT,
                )
        else:
            from kairos.utils import make_progress_callback

            _oflogs_diffusion = pipe.overfit_test(
                n_examples=OVERFIT_EXAMPLES,
                steps=OVERFIT_STEPS,
                progress_callback=make_progress_callback(desc="overfit_test (diffusion)"),
                mask_p_max=TRAIN_MASK_P_MAX,
                mask_reweight=TRAIN_MASK_REWEIGHT,
            )
    else:
        print("OVERFIT_RUN is False - skipping diffusion overfit test")
    return


@app.cell
def _():
    FORCE_RESTART = False  # Stage 2 must resume=True (FORCE_RESTART=False) to pick up Stage 1's MAE weights;
    # setting this True skips that bridge and (re)trains Stage 2 from a freshly random-initialized model.
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
    if pipe.eval_log_rows:
        print(f"eval points: {len(pipe.eval_log_rows)}  best eval loss: {pipe.best_eval_loss:.4f}")
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
                with pipe._autocast():
                    losses.append(pipe.hf_trainer.compute_loss(pipe.model, batch).item())
        pipe.model.train()
        eval_loss = sum(losses) / len(losses)
        pipe.writer.add_scalar("eval/loss", eval_loss, pipe.global_step)
        print(f"eval loss: {eval_loss:.4f} on {len(eval_dataset)} samples")
    return


@app.cell
def _():
    # ---- generation smoke test ----
    GEN_N_EXAMPLES = 3  # number of text prompts to denoise
    GEN_PROMPT_TOKENS = 32  # keep this many tokens as the fixed prompt
    GEN_MAX_NEW_TOKENS = 64  # length of the denoised continuation (canvas)
    GEN_DENOISING_STEPS = 24  # diffusion iterations per canvas
    GEN_T_MIN = 0.4  # final temperature (cold/confident)
    GEN_T_MAX = 1.0  # initial temperature (hot/exploratory)
    GEN_ENTROPY_BOUND = 0.5  # cumulative-entropy bound: higher -> accept more tokens/step
    GEN_SEED = 0
    return (
        GEN_DENOISING_STEPS,
        GEN_ENTROPY_BOUND,
        GEN_MAX_NEW_TOKENS,
        GEN_N_EXAMPLES,
        GEN_PROMPT_TOKENS,
        GEN_SEED,
        GEN_T_MAX,
        GEN_T_MIN,
    )


@app.cell
def _(
    GEN_DENOISING_STEPS,
    GEN_ENTROPY_BOUND,
    GEN_MAX_NEW_TOKENS,
    GEN_N_EXAMPLES,
    GEN_PROMPT_TOKENS,
    GEN_SEED,
    GEN_T_MAX,
    GEN_T_MIN,
    eval_examples,
    pipe,
    text_examples,
    tokenizer,
):
    # diffusion generation via KairosDiffusionGenerationMixin (reuses the HF
    # DiffusionGemma EntropyBoundSampler + temperature schedule + adaptive stopping)
    _rows = [
        ex
        for ex in eval_examples
        if ex.get("modality") == "text"
        and len(tokenizer.encode(ex["text"], add_special_tokens=False)) > GEN_PROMPT_TOKENS
    ]
    if len(_rows) < GEN_N_EXAMPLES:
        _rows = [
            ex
            for ex in text_examples
            if ex.get("modality") == "text"
            and len(tokenizer.encode(ex["text"], add_special_tokens=False)) > GEN_PROMPT_TOKENS
        ]
    _rows = _rows[:GEN_N_EXAMPLES]

    for _i, _ex in enumerate(_rows, 1):
        _ids = tokenizer.encode(_ex["text"], add_special_tokens=False)
        _prompt = _ids[:GEN_PROMPT_TOKENS]
        _full = pipe.generate(
            _prompt,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            max_denoising_steps=GEN_DENOISING_STEPS,
            entropy_bound=GEN_ENTROPY_BOUND,
            t_min=GEN_T_MIN,
            t_max=GEN_T_MAX,
            seed=GEN_SEED + _i,
        )
        _gen = _full[GEN_PROMPT_TOKENS:]
        _reference = _ids[GEN_PROMPT_TOKENS : GEN_PROMPT_TOKENS + GEN_MAX_NEW_TOKENS]
        print(f"--- example {_i} ---")
        print("prompt:    ", tokenizer.decode(_prompt, skip_special_tokens=True))
        print("generated: ", tokenizer.decode(_gen, skip_special_tokens=True))
        print("reference: ", tokenizer.decode(_reference, skip_special_tokens=True))
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
def _(pd, pipe):
    if pipe.eval_log_rows:
        pd.DataFrame(pipe.eval_log_rows).plot(
            x="step", y="loss", figsize=(8, 4), title="Eval loss (held-out, every N steps)"
        )
    else:
        print("no eval points logged - set TRAIN_EVAL_EVERY > 0")
    return


@app.cell
def _(pipe):
    # diffusion loss masked per modality, so
    per_modality = pipe.check_per_modality_loss(n_batches=10)
    for _k, _v in sorted(per_modality.items(), key=lambda kv: -kv[1]):
        print(f"{_k}: {_v:.4f}")
    return


@app.cell
def _():
    # not needed for a simple crash-recovery
    # use this only to load a
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
    PUSH_FULL_MODEL_TO_HUB = False  # pushes weights+config+model card, on top
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
