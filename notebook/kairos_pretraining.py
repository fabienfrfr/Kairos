import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from huggingface_hub import login

    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = os.environ.get("HF_TOKEN", "hf_xxx")

    login(token=HF_TOKEN, add_to_git_credential=False)
    return (os,)


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
    # !pip install -q git+https://github.com/fabienfrfr/Kairos@dev
    return


@app.cell
def _(os):
    from pathlib import Path

    # auto: fused flex_attention on SM>=7.0 GPUs (T4), eager O(L*W) below that.
    # Multi-GPU (T4x2): flex isn't thread-safe under DataParallel; launch via torchrun (DDP).
    os.environ.setdefault("KAIROS_ATTN_BACKEND", "auto")

    import torch
    import pandas as pd

    from kairos.modeling import KairosConfig
    from kairos.tokenizer import KairosTokenizer, Modality
    from kairos.pipeline import KairosMultimodalPipeline, DataConfig, TrainConfig
    from kairos.utils import make_progress_callback
    from kairos.dataset import (
        diagnose_raw_control_balance,
        modality_counts,
        split_examples,
        preview_multimodal_examples,
    )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    tokenizer = KairosTokenizer()
    print(f"vocab size: {len(tokenizer)}")

    # ---- dev settings ----
    DEV_MODE = True
    return (
        DEV_MODE,
        DataConfig,
        KairosConfig,
        KairosMultimodalPipeline,
        Modality,
        Path,
        TrainConfig,
        diagnose_raw_control_balance,
        make_progress_callback,
        modality_counts,
        pd,
        preview_multimodal_examples,
        split_examples,
        tokenizer,
        torch,
    )


@app.cell
def _(DEV_MODE):
    # ---- data settings ----
    MULTIMODAL_SOURCE = "hf"  # "hf" or "local" (.pt built
    MULTIMODAL_LOCAL_PATH = "data/keep-it-simple-multimodal.pt"
    BUILD_LOCAL_IF_MISSING = False

    TEXT_SOURCE = "hf"  # "hf" (ffurfaro/keep-it-simple) or "inline"

    if DEV_MODE :
        TEXT_PCT = 1  # % of keep-it-simple to load
        EVAL_PCT = 1  # % held out for eval
    else :
        TEXT_PCT = 100   # % of keep-it-simple to load
        EVAL_PCT = 0.001 # % held out for eval
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
        except Exception as e:  # noqa: BLE001 - network/dataset failure, fall back to a sample
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
def _(modality_counts, multimodal_examples, text_examples):
    _all_ex = list(text_examples) + list(multimodal_examples)
    print(f"total examples: {len(_all_ex)}")
    for _modality, _count in modality_counts(_all_ex).items():
        print(f"  {_modality:<16} {_count}")
    return


@app.cell
def _():
    SHOW_PREVIEW = True  # set False to skip decoding/plotting
    return (SHOW_PREVIEW,)


@app.cell
def _(SHOW_PREVIEW, diagnose_raw_control_balance, multimodal_examples):
    # source-data sanity check, before tokenization: raw control state/action counts must match.
    if SHOW_PREVIEW and multimodal_examples:
        _raw_report = diagnose_raw_control_balance(multimodal_examples)
        print(_raw_report)
        if _raw_report.n_control_examples and _raw_report.mismatched_examples:
            print("\n^ mismatch found in the SOURCE data itself, before tokenization")
    return


@app.cell
def _(SHOW_PREVIEW, multimodal_examples, preview_multimodal_examples):
    if SHOW_PREVIEW and multimodal_examples:
        preview_multimodal_examples(multimodal_examples, n=3)
    else:
        print("preview skipped")
    return


@app.cell
def _(EVAL_PCT, multimodal_examples, split_examples, text_examples):
    _all_ex = list(text_examples) + list(multimodal_examples)
    train_examples, eval_examples = split_examples(_all_ex, eval_pct=EVAL_PCT, seed=0)
    print(f"train: {len(train_examples)} ({100 - EVAL_PCT}%)  eval: {len(eval_examples)} ({EVAL_PCT}%)")
    return eval_examples, train_examples


@app.cell
def _():
    # ---- model settings ----
    CFG_D_MODEL = 64  # head_dim = 64/4 = 16, power of 2 for flex_attention
    CFG_N_HEADS = 4
    CFG_N_LAYERS = 4
    CFG_STRIDE = 3
    CFG_NUM_SCALES = 4
    CFG_ATTNRES_BLOCK = 4
    CFG_EXPERTS = 7  # 0 = dense FFN; top-1 routing is slow to converge on tiny overfit-test runs
    # set to 0 to isolate whether the backbone itself can memorize before blaming routing
    CFG_EXPERTS_PER_TOK = 1
    CFG_SHARED_EXPERTS = 1
    CFG_INTERMEDIATE = 544  # raised to keep ~14-15M total params after d_model 88->64
    CFG_USE_MEMORY_BANK = True  # cross-session DeltaNet state gating
    CFG_SHARE_BACKBONES = True  # share one backbone across all scales (saves ~75% params)
    CFG_CODEC_MODE = "patch"  # "conv" (fast, cuDNN) or "patch" (nn.Linear per scale)
    return (
        CFG_ATTNRES_BLOCK,
        CFG_CODEC_MODE,
        CFG_D_MODEL,
        CFG_EXPERTS,
        CFG_EXPERTS_PER_TOK,
        CFG_INTERMEDIATE,
        CFG_NUM_SCALES,
        CFG_N_HEADS,
        CFG_N_LAYERS,
        CFG_SHARED_EXPERTS,
        CFG_SHARE_BACKBONES,
        CFG_STRIDE,
        CFG_USE_MEMORY_BANK,
    )


@app.cell
def _(Modality):
    # scale 0: finest temporal res (text, control); scale 3: coarsest (meta)
    modality_scales = {
        int(Modality.TEXT): [0, 1],
        int(Modality.CONTROL): [0],  # paired or observation-only (<OBS>); fuses old STATE/ACTION/imu
        int(Modality.IMAGE): [1, 2],
        int(Modality.LIDAR): [1],
        int(Modality.AUDIO): [2],
        int(Modality.VIDEO): [2, 3],
        int(Modality.META): [3],
    }
    return (modality_scales,)


@app.cell
def _():
    # per-modality encode_* scale_factor override; None uses the tokenizer's class default
    modality_scale_factors = {
        "audio_caption": None,  # None -> KairosTokenizer.PCM_SCALE_FACTOR default (4)
        "control": None,
        "imu": None,
        "image_caption": None,  # None -> KairosTokenizer.IMAGE_SCALE_FACTOR default (1, no-op)
        "video_caption": None,
        "lidar": None,  # None -> KairosTokenizer.LIDAR_SCALE_FACTOR default (1, no-op)
    }
    return (modality_scale_factors,)


@app.cell
def _(
    CFG_ATTNRES_BLOCK,
    CFG_CODEC_MODE,
    CFG_D_MODEL,
    CFG_EXPERTS,
    CFG_EXPERTS_PER_TOK,
    CFG_INTERMEDIATE,
    CFG_NUM_SCALES,
    CFG_N_HEADS,
    CFG_N_LAYERS,
    CFG_SHARED_EXPERTS,
    CFG_SHARE_BACKBONES,
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
        num_modalities=9,
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
        share_backbones=CFG_SHARE_BACKBONES,
        codec_mode=CFG_CODEC_MODE,
    )
    print(
        f"moe: {use_moe}  block-attnres window: {CFG_ATTNRES_BLOCK}  memory_bank: {CFG_USE_MEMORY_BANK}  share_backbones: {CFG_SHARE_BACKBONES}  codec_mode: {CFG_CODEC_MODE}"
    )
    return (model_config,)


@app.cell
def _():
    # ---- training settings ----
    TRAIN_LR = 3e-4
    TRAIN_BATCH = 32
    TRAIN_MAX_LEN = 1024
    TRAIN_STRIDE = 3
    TRAIN_SAVE_EVERY = 200
    TRAIN_MASK_EPS = 1e-3  # floor of masked-diffusion rate p; lower -> harder to overfit fast

    # single-pipeline MAE -> transition -> diffusion curriculum; set an *_EPOCHS to 0 to skip.
    TRAIN_MAE_EPOCHS = 1
    TRAIN_MAE_P_MAX = 0.3  # MAE stage ceiling on p: fixed-ish low corruption, stable to optimize
    TRAIN_MAE_REWEIGHT = False  # plain CE in MAE stage: no 1/p variance blowup

    TRAIN_TRANSITION_EPOCHS = 1  # ramps masking rate + reweighting from MAE to diffusion values

    TRAIN_DIFFUSION_EPOCHS = 1
    TRAIN_MASK_P_MAX = 1.0  # diffusion stage ceiling on p: full diffusion, up to 100% noised
    TRAIN_MASK_REWEIGHT = True  # diffusion stage: divide CE by p (standard ELBO weighting)

    TRAIN_EVAL_EVERY = 100  # eval on held-out set every N steps (0 = off)
    TRAIN_EVAL_BATCHES = 2  # batches per eval; small keeps it cheap
    TRAIN_RUN_DIR = "checkpoints/kairos-multimodal/run_01"  # keep unchanged across restarts -
    # resuming mid-curriculum (any stage) reads/writes checkpoints/last.pt in this same directory.

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
        TRAIN_TRANSITION_EPOCHS,
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
    TRAIN_TRANSITION_EPOCHS,
    TrainConfig,
    eval_examples,
    modality_scale_factors,
    train_examples,
):
    data_config = DataConfig(
        text_examples=[],
        multimodal_examples=train_examples,
        max_len=TRAIN_MAX_LEN,
        stride=TRAIN_STRIDE,
        batch_size=TRAIN_BATCH,
        pack=TRAIN_PACK,
        modality_scale_factors=modality_scale_factors,
    )
    eval_data_config = DataConfig(
        text_examples=[],
        multimodal_examples=eval_examples,
        max_len=TRAIN_MAX_LEN,
        stride=TRAIN_STRIDE,
        batch_size=TRAIN_BATCH,
        shuffle=False,
        drop_last=False,
        modality_scale_factors=modality_scale_factors,
    )
    # single TrainConfig for the whole MAE -> transition -> diffusion curriculum (one pipeline)
    train_config = TrainConfig(
        lr=TRAIN_LR,
        mask_eps=TRAIN_MASK_EPS,
        mae_epochs=TRAIN_MAE_EPOCHS,
        mask_mae_p_max=TRAIN_MAE_P_MAX,
        mask_mae_reweight=TRAIN_MAE_REWEIGHT,
        transition_epochs=TRAIN_TRANSITION_EPOCHS,
        diffusion_epochs=TRAIN_DIFFUSION_EPOCHS,
        mask_p_max=TRAIN_MASK_P_MAX,
        mask_reweight=TRAIN_MASK_REWEIGHT,
        save_every=TRAIN_SAVE_EVERY,
        eval_every=TRAIN_EVAL_EVERY,
        eval_batches=TRAIN_EVAL_BATCHES,
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
    # compute-cost summary: params/memory measured via a few real timed steps
    RUN_BENCHMARK = True
    N_BENCH_STEPS = 5
    return N_BENCH_STEPS, RUN_BENCHMARK


@app.cell
def _(N_BENCH_STEPS, RUN_BENCHMARK, pipe):
    cost_summary = pipe.summary(benchmark=RUN_BENCHMARK, n_bench_steps=N_BENCH_STEPS)
    print(cost_summary)
    return


@app.cell
def _(RUN_BENCHMARK, pipe):
    # one-off diagnostic: runs a single real step; don't call inside the training loop
    if RUN_BENCHMARK:
        print(pipe.memory_report())
    return


@app.cell
def _(RUN_BENCHMARK, pipe):
    # per-module wall-clock time (self-time, excludes children); shows where compute actually goes
    if RUN_BENCHMARK:
        print(pipe.profile(n_steps=3))
    return


@app.cell
def _(RUN_BENCHMARK, pipe):
    # per-modality raw-vs-tokenized breakdown: where dataset rows/steps actually come from
    if RUN_BENCHMARK:
        print(pipe.data_report(split="train"))
        if pipe.eval_data_config is not None:
            print()
            print(pipe.data_report(split="eval"))
    return


@app.cell
def _(pipe):
    # visualize the tokenized input as fed to the model, post-collation
    _reports = pipe.inspect_batch(n=1)
    _table = pipe.inspect_batch_df(n=1)
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
def _(RUN_BENCHMARK, pipe):
    # localizes a STATE/ACTION token imbalance to specific rows (not just an aggregate %)
    report = None
    if RUN_BENCHMARK:
        report = pipe.control_alternation_report(split="train")
        print(report)
        if pipe.eval_data_config is not None:
            print()
            print(pipe.control_alternation_report(split="eval"))
    return (report,)


@app.cell
def _(RUN_BENCHMARK, pipe, report):
    # visually reconstruct real tokenized rows; preview_tokenized: one row per modality present.
    if RUN_BENCHMARK and report is not None and report.mismatched_rows:
        pipe.plot_row(row=report.mismatched_rows[0]["row"], split="train")
        pipe.preview_tokenized(split="train")
    elif RUN_BENCHMARK:
        pipe.show(n=3, split="train")  # a few random rows, any modality
        pipe.preview_tokenized(split="train")  # one row per modality, CONTROL state/action overlaid
    return


@app.cell
def _():
    OVERFIT_RUN = True  # sanity-check the model can memorize before the real run
    OVERFIT_EXAMPLES = 16  # tiny subset, repeated each epoch
    OVERFIT_STEPS = 200  # steps on that subset; loss should crash toward 0
    return OVERFIT_EXAMPLES, OVERFIT_RUN, OVERFIT_STEPS


@app.cell
def _(
    OVERFIT_EXAMPLES,
    OVERFIT_RUN,
    OVERFIT_STEPS,
    make_progress_callback,
    mo,
    pipe,
):
    # walks whichever of the MAE / transition / diffusion stages are configured, proportionally
    if OVERFIT_RUN:
        if mo.running_in_notebook():
            with mo.status.progress_bar(total=OVERFIT_STEPS, title="overfit_test") as _bar:
                overfit_logs = pipe.overfit_test(
                    n_examples=OVERFIT_EXAMPLES,
                    steps=OVERFIT_STEPS,
                    progress_callback=lambda step, total, loss_val: _bar.update(
                        increment=1, subtitle=f"loss={loss_val:.4f}"
                    ),
                )
        else:
            overfit_logs = pipe.overfit_test(
                n_examples=OVERFIT_EXAMPLES,
                steps=OVERFIT_STEPS,
                progress_callback=make_progress_callback(desc="overfit_test"),
            )
    else:
        print("OVERFIT_RUN is False - skipping overfit test")
        overfit_logs = []
    return


@app.cell
def _():
    FORCE_RESTART = True  # True = restart from scratch, False = resume, landing in the right stage
    return (FORCE_RESTART,)


@app.cell
def _(FORCE_RESTART, make_progress_callback, mo, pipe):
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
    # diffusion generation via KairosDiffusionGenerationMixin (HF EntropyBoundSampler + adaptive)
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
    # not needed for simple crash-recovery; use this to load a specific saved checkpoint by path
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
