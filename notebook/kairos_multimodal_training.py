import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 🌀 Kairos — Multimodal Pretraining Notebook
    Text · Image · Video · Audio · Lidar · IMU · Control

    All logic lives in `kairos.pipeline.KairosMultimodalPipeline`
    (tested in `tests/test_pipeline.py`); this notebook only declares config and
    calls the pipeline. Every modality shares one byte-level tokenizer
    (`KairosTokenizer`, PixelBytes approach) — no per-modality VQ-VAE.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import torch
    import pandas as pd

    from kairos.modeling import KairosConfig
    from kairos.tokenizer import KairosTokenizer, Modality
    from kairos.pipeline import KairosMultimodalPipeline, DataConfig, TrainConfig

    return (
        DataConfig,
        KairosConfig,
        KairosMultimodalPipeline,
        KairosTokenizer,
        Modality,
        Path,
        TrainConfig,
        pd,
        torch,
    )


@app.cell
def _(mo, torch):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    mo.callout(mo.md(f"**Device:** `{device}`"), kind="info")
    return (device,)


@app.cell
def _(KairosTokenizer):
    # created once, shared by model config (vocab_size) and the pipeline below
    tokenizer = KairosTokenizer()
    return (tokenizer,)


@app.cell
def _(mo):
    mo.md("## 📦 1. Data")
    return


@app.cell
def _(mo):
    mo.md("""
    `build_keep_it_simple_multimodal.py` downloads and serializes a small sample from
    6 sources (Flickr8k, AudioCaps, MSR-VTT, nuScenes-mini, MotionSense,
    PixelBytes-OptimalControl) into `list[dict]` with a `kind` field — each source is
    wrapped in its own `try/except`, so one missing source doesn't block the rest.
    `keep-it-simple` supplies the text data (not covered by the multimodal set).
    """)
    return


@app.cell
def _(mo):
    multimodal_path = mo.ui.text(
        value="data/keep-it-simple-multimodal.pt",
        label="Multimodal .pt path (build_keep_it_simple_multimodal.py)",
    )
    build_button = mo.ui.run_button(label="⚙️ Build if missing")
    mo.hstack([multimodal_path, build_button])
    return build_button, multimodal_path


@app.cell
def _(Path, build_button, mo, multimodal_path):
    _path = Path(multimodal_path.value)
    if build_button.value and not _path.exists():
        with mo.status.spinner(title="Building keep-it-simple-multimodal..."):
            import subprocess

            subprocess.run(
                ["python3", "scripts/pretrain/build_keep_it_simple_multimodal.py"],
                check=True,
            )
    mo.callout(
        mo.md(f"`{_path}` {'✅ present' if _path.exists() else '❌ missing — click the button above'}"),
        kind="success" if _path.exists() else "warn",
    )
    return


@app.cell
def _(Path, mo, multimodal_path, pd, torch):
    _path = Path(multimodal_path.value)
    multimodal_examples = torch.load(_path, weights_only=False) if _path.exists() else []
    if multimodal_examples:
        _counts = pd.Series([ex["kind"] for ex in multimodal_examples]).value_counts()
        _table = mo.ui.table(
            _counts.reset_index().rename(columns={"index": "kind", 0: "count"}), label="Multimodal examples by kind"
        )
    else:
        _table = mo.md("")
    _table
    return (multimodal_examples,)


@app.cell
def _(mo):
    text_source = mo.ui.radio(
        options=["ffurfaro/keep-it-simple (2%)", "small inline sample"],
        value="ffurfaro/keep-it-simple (2%)",
        label="Text source",
    )
    text_source
    return (text_source,)


@app.cell
def _(mo, text_source):
    with mo.status.spinner(title="Loading text..."):
        if text_source.value.startswith("ffurfaro"):
            try:
                from datasets import load_dataset

                _ds = load_dataset("ffurfaro/keep-it-simple", split="train[:2%]")
                text_examples = [{"kind": "text", "text": f"{row['prompt']} {row['text']}".strip()} for row in _ds]
            except Exception as e:
                mo.callout(mo.md(f"[fallback] keep-it-simple unavailable ({e}) — using inline sample"), kind="warn")
                text_examples = [
                    {"kind": "text", "text": "Paris is the capital of France."},
                    {"kind": "text", "text": "The Earth orbits the Sun."},
                ]
        else:
            text_examples = [
                {"kind": "text", "text": "Paris is the capital of France."},
                {"kind": "text", "text": "The Earth orbits the Sun."},
                {"kind": "text", "text": "Water boils at 100 degrees Celsius."},
            ]
    mo.callout(mo.md(f"**Text examples:** `{len(text_examples)}`"), kind="success")
    return (text_examples,)


@app.cell
def _(mo):
    mo.md("## ⚙️ 2. Model configuration")
    return


@app.cell
def _(mo):
    mo.md("""
    `num_modalities=8` covers the full `Modality` enum; `modality_scales` routes each
    modality to its best-fit `PyramidalConvCodec` scale; `attnres_block_size` sets the
    v3 Block-AttnRes window (`1` = classic per-layer softmax residual, `>1` groups
    layers into blocks before aggregation, cost O(N/S)).
    """)
    return


@app.cell
def _(mo):
    cfg_d_model = mo.ui.slider(32, 768, step=32, value=128, label="d_model (hidden size)")
    cfg_n_heads = mo.ui.slider(2, 16, step=2, value=4, label="n_heads")
    cfg_n_layers = mo.ui.slider(1, 24, step=1, value=8, label="n_layers")
    cfg_stride = mo.ui.slider(1, 6, step=1, value=3, label="PyramidalConvCodec stride")
    cfg_num_scales = mo.ui.slider(2, 6, step=1, value=4, label="num_scales")
    cfg_attnres_block = mo.ui.slider(1, 8, step=1, value=4, label="attnres_block_size (v3 Block-AttnRes)")
    cfg_experts = mo.ui.slider(0, 32, step=1, value=8, label="Routed experts (0 = dense FFN)")
    cfg_experts_per_tok = mo.ui.slider(1, 8, step=1, value=2, label="Experts active per token")
    cfg_shared_experts = mo.ui.slider(0, 4, step=1, value=1, label="Shared experts")
    cfg_intermediate = mo.ui.slider(128, 2048, step=128, value=512, label="FFN/MoE intermediate size")

    mo.vstack(
        [
            mo.hstack([cfg_d_model, cfg_n_heads, cfg_n_layers]),
            mo.hstack([cfg_stride, cfg_num_scales, cfg_attnres_block]),
            mo.hstack([cfg_experts, cfg_experts_per_tok, cfg_shared_experts]),
            cfg_intermediate,
        ]
    )
    return (
        cfg_attnres_block,
        cfg_d_model,
        cfg_experts,
        cfg_experts_per_tok,
        cfg_intermediate,
        cfg_n_heads,
        cfg_n_layers,
        cfg_num_scales,
        cfg_shared_experts,
        cfg_stride,
    )


@app.cell
def _(Modality):
    # scale 0: finest temporal resolution (text, control) · 1: images/lidar
    # scale 2: audio/video frames (chunked by <TICK>/<ENDFRAME>) · 3: coarse/META
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
    KairosConfig,
    cfg_attnres_block,
    cfg_d_model,
    cfg_experts,
    cfg_experts_per_tok,
    cfg_intermediate,
    cfg_n_heads,
    cfg_n_layers,
    cfg_num_scales,
    cfg_shared_experts,
    cfg_stride,
    mo,
    modality_scales,
    tokenizer,
):
    use_moe = cfg_experts.value > 0

    model_config = KairosConfig(
        d_model=cfg_d_model.value,
        n_heads=cfg_n_heads.value,
        n_layers=cfg_n_layers.value,
        stride=cfg_stride.value,
        vocab_size=len(tokenizer),
        num_modalities=8,
        num_scales=cfg_num_scales.value,
        modality_scales=modality_scales,
        intermediate_size=cfg_intermediate.value,
        moe_intermediate_size=cfg_intermediate.value,
        n_routed_experts=cfg_experts.value if use_moe else 8,
        num_experts_per_tok=cfg_experts_per_tok.value,
        n_shared_experts=cfg_shared_experts.value,
        use_moe=use_moe,
        attnres_block_size=cfg_attnres_block.value,
    )

    mo.callout(
        mo.md(
            f"**Vocab size:** `{len(tokenizer)}`  \n"
            f"**MoE:** `{'Yes — ' + str(cfg_experts.value) + ' routed + ' + str(cfg_shared_experts.value) + ' shared' if use_moe else 'No — dense FFN'}`  \n"
            f"**Block-AttnRes window:** `{cfg_attnres_block.value}`"
        ),
        kind="success",
    )
    return (model_config,)


@app.cell
def _(mo):
    mo.md("## 🏋️ 3. Training configuration")
    return


@app.cell
def _(mo):
    train_lr = mo.ui.number(1e-5, 1e-2, step=1e-5, value=3e-4, label="Learning rate")
    train_batch = mo.ui.slider(1, 64, step=1, value=8, label="Batch size")
    train_epochs = mo.ui.slider(1, 50, step=1, value=3, label="Epochs")
    train_max_len = mo.ui.slider(64, 4096, step=64, value=1024, label="Max sequence length")
    train_stride = mo.ui.slider(1, 6, step=1, value=3, label="Chunking stride")
    train_save_every = mo.ui.slider(10, 1000, step=10, value=200, label="Save every N steps")
    train_run_dir = mo.ui.text(value="checkpoints/kairos-multimodal/run_01", label="Run directory")

    mo.vstack(
        [
            mo.hstack([train_lr, train_batch, train_epochs]),
            mo.hstack([train_max_len, train_stride, train_save_every]),
            train_run_dir,
        ]
    )
    return (
        train_batch,
        train_epochs,
        train_lr,
        train_max_len,
        train_run_dir,
        train_save_every,
        train_stride,
    )


@app.cell
def _(
    DataConfig,
    TrainConfig,
    multimodal_examples,
    text_examples,
    train_batch,
    train_epochs,
    train_lr,
    train_max_len,
    train_run_dir,
    train_save_every,
    train_stride,
):
    data_config = DataConfig(
        text_examples=text_examples,
        multimodal_examples=multimodal_examples,
        max_len=train_max_len.value,
        stride=train_stride.value,
        batch_size=train_batch.value,
    )
    train_config = TrainConfig(
        lr=train_lr.value,
        epochs=train_epochs.value,
        save_every=train_save_every.value,
        run_dir=train_run_dir.value,
    )
    return data_config, train_config


@app.cell
def _(mo):
    mo.md("## 🚂 4. Build & Train")
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="▶ Start Training")
    run_button
    return (run_button,)


@app.cell
def _(
    KairosMultimodalPipeline,
    data_config,
    mo,
    model_config,
    run_button,
    tokenizer,
    train_config,
):
    if not run_button.value:
        mo.stop(True, mo.callout(mo.md("Click **▶ Start Training** to begin."), kind="neutral"))

    pipe = KairosMultimodalPipeline(model_config, data_config, train_config, tokenizer=tokenizer)

    with mo.status.spinner(title="Building pipeline (tokenizer/dataset/model/optimizer)..."):
        pipe.build()

    total_params = sum(p.numel() for p in pipe.model.parameters())
    mo.callout(
        mo.md(
            f"**Total params:** `{total_params / 1e6:.2f}M`  |  **Device:** `{pipe.device}`  |  **Samples:** `{len(pipe.dataset)}`"
        ),
        kind="success",
    )
    return (pipe,)


@app.cell
def _(mo, pipe):
    with mo.status.spinner(title="Training..."):
        logs = pipe.train()

    mo.callout(
        mo.md(
            f"✅ **Training complete**  \n"
            f"Steps: `{len(logs)}` | Best avg-epoch loss: `{pipe.best_loss:.4f}`  \n"
            f"Checkpoints: `{pipe.ckpt_dir}`  \n"
            f"TensorBoard: `tensorboard --logdir {pipe.tb_dir}`"
        ),
        kind="success",
    )
    return (logs,)


@app.cell
def _(logs, mo, pd):
    mo.vstack(
        [
            mo.md("## 📊 5. Logs"),
            mo.ui.table(pd.DataFrame(logs), label="Step logs", pagination=True, page_size=20),
        ]
    )
    return


@app.cell
def _(logs, pd):
    pd.DataFrame(logs).plot(x="step", y="loss", figsize=(8, 4), title="Multimodal diffusion loss")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🔍 6. Per-modality check

    Diffusion loss is masked by `modality_ids` instead of averaged over the whole
    batch, so a modality the router ignores can't hide behind a healthy-looking
    global average.
    """)
    return


@app.cell
def _(mo, pipe):
    with mo.status.spinner(title="Checking per-modality loss..."):
        per_modality = pipe.check_per_modality_loss(n_batches=10)

    mo.ui.table(
        [{"modality": k, "loss": round(v, 4)} for k, v in sorted(per_modality.items(), key=lambda kv: -kv[1])],
        label="Per-modality diffusion loss",
    )
    return


@app.cell
def _(mo):
    mo.md("## ♻️ 7. Resume from checkpoint")
    return


@app.cell
def _(mo):
    resume_path = mo.ui.text(value="", label="Checkpoint path (.pt)")
    resume_button = mo.ui.run_button(label="Load checkpoint")
    mo.hstack([resume_path, resume_button])
    return resume_button, resume_path


@app.cell
def _(mo, pipe, resume_button, resume_path):
    if not resume_button.value or not resume_path.value:
        mo.stop(True)

    _ckpt = pipe.load_checkpoint(resume_path.value)
    mo.callout(
        mo.md(
            f"✅ Loaded `{resume_path.value}`  \nStep: `{_ckpt.get('step', '?')}` | Loss: `{_ckpt.get('loss', 0):.4f}`"
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
