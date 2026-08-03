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

    Config + calls into `kairos.pipeline.KairosMultimodalPipeline`.
    """)
    return


@app.cell
def _():
    import random
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
        random,
        torch,
    )


@app.cell
def _(mo, torch):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    mo.callout(mo.md(f"**Device:** `{device}`"), kind="info")
    return


@app.cell
def _(KairosTokenizer):
    # created once, shared by model config (vocab_size) and the pipeline below
    tokenizer = KairosTokenizer()
    return (tokenizer,)


@app.cell
def _(mo):
    mo.md("""
    ## 📦 1. Data
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    `build_keep_it_simple_multimodal.py` builds a small `list[dict]` sample from 6
    sources; `keep-it-simple` supplies the text data.
    """)
    return


@app.cell
def _(mo):
    multimodal_source = mo.ui.radio(
        options=["HF Hub (ffurfaro/keep-it-simple-multimodal)", "Local .pt (build_keep_it_simple_multimodal.py)"],
        value="HF Hub (ffurfaro/keep-it-simple-multimodal)",
        label="Multimodal source",
    )
    multimodal_path = mo.ui.text(
        value="data/keep-it-simple-multimodal.pt",
        label="Local .pt path (used if 'Local .pt' selected)",
    )
    build_button = mo.ui.run_button(label="⚙️ Build locally if missing")
    mo.vstack([multimodal_source, mo.hstack([multimodal_path, build_button])])
    return build_button, multimodal_path, multimodal_source


@app.cell
def _(Path, build_button, mo, multimodal_path, multimodal_source):
    _path = Path(multimodal_path.value)
    if multimodal_source.value.startswith("Local") and build_button.value and not _path.exists():
        with mo.status.spinner(title="Building keep-it-simple-multimodal..."):
            import subprocess

            subprocess.run(
                ["python3", "scripts/pretrain/build_keep_it_simple_multimodal.py"],
                check=True,
            )
    if multimodal_source.value.startswith("Local"):
        mo.callout(
            mo.md(f"`{_path}` {'✅ present' if _path.exists() else '❌ missing — click the button above'}"),
            kind="success" if _path.exists() else "warn",
        )
    return


@app.cell
def _(Path, mo, multimodal_source, pd, torch):
    with mo.status.spinner(title="Loading multimodal examples..."):
        if multimodal_source.value.startswith("HF Hub"):
            from datasets import load_dataset as _load_dataset

            multimodal_examples = list(_load_dataset("ffurfaro/keep-it-simple-multimodal", split="train"))
        else:
            _path = Path("data/keep-it-simple-multimodal.pt")
            multimodal_examples = torch.load(_path, weights_only=False) if _path.exists() else []
    if multimodal_examples:
        _counts = pd.Series([ex["modality"] for ex in multimodal_examples]).value_counts()
        _table = mo.ui.table(
            _counts.reset_index().rename(columns={"index": "modality", 0: "count"}), label="Multimodal examples by modality"
        )
    else:
        _table = mo.md("")
    _table
    return (multimodal_examples,)


@app.cell
def _(mo):
    text_source = mo.ui.radio(
        options=["ffurfaro/keep-it-simple", "small inline sample"],
        value="ffurfaro/keep-it-simple",
        label="Text source",
    )
    text_pct = mo.ui.slider(start=1, stop=100, value=2, step=1, label="% of keep-it-simple to load")
    mo.vstack([text_source, text_pct])
    return text_pct, text_source


@app.cell
def _(mo, text_pct, text_source):
    with mo.status.spinner(title="Loading text..."):
        if text_source.value.startswith("ffurfaro"):
            try:
                from datasets import load_dataset

                _ds = load_dataset("ffurfaro/keep-it-simple", split=f"train[:{text_pct.value}%]")
                text_examples = [{"modality": "text", "text": f"{row['prompt']} {row['text']}".strip()} for row in _ds]
            except Exception as e:
                mo.callout(mo.md(f"[fallback] keep-it-simple unavailable ({e}) — using inline sample"), kind="warn")
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
    mo.callout(mo.md(f"**Text examples:** `{len(text_examples)}`"), kind="success")
    return (text_examples,)


@app.cell
def _(mo):
    mo.md("""
    ## 📊 3.5 Modality overview & train/eval split
    """)
    return


@app.cell
def _(mo, multimodal_examples, pd, text_examples):
    _all_ex = list(text_examples) + list(multimodal_examples)
    _counts = pd.Series([ex["modality"] for ex in _all_ex]).value_counts().reset_index()
    _counts.columns = ["modality", "count"]

    try:
        import altair as alt

        _chart = (
            alt.Chart(_counts)
            .mark_bar()
            .encode(x=alt.X("modality:N", sort="-y"), y="count:Q", tooltip=["modality", "count"])
            .properties(title="Examples per modality (before train/eval split)", width=500)
        )
        _viz = mo.ui.altair_chart(_chart)
    except ImportError:
        _viz = mo.ui.table(_counts, label="Examples per modality (before split) — install altair for a chart")

    mo.vstack([mo.md(f"**Total examples (text + multimodal):** `{len(_all_ex)}`"), _viz])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 👀 3.6 Preview samples (real decoded content, not just counts)
    """)
    return


@app.cell
def _(mo, multimodal_examples):
    import io as _io
    import json as _json

    import matplotlib.pyplot as plt
    import numpy as np

    from kairos.dataset import unpack_multimodal_data

    def _fig_to_image(fig):
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    def _preview_row(ex):
        modality = ex["modality"]
        arrays = unpack_multimodal_data(ex["data"])
        meta = _json.loads(ex["meta"]) if ex.get("meta") else {}
        caption = (ex.get("caption") or "")[:80]

        if modality == "image_caption":
            dims = f"orig {meta['original_width']}×{meta['original_height']} → 32×32" if "original_width" in meta else ""
            return mo.vstack([mo.image(arrays["image"], width=120), mo.md(f"`{caption}`  \n*{dims}*")])

        if modality == "audio_caption":
            audio = arrays["audio"]
            rate = meta.get("sample_rate", 8000)
            fig, ax = plt.subplots(figsize=(3, 1))
            ax.plot(audio, linewidth=0.5)
            ax.axis("off")
            info = (
                f"orig {meta['original_duration_sec']:.1f}s → {len(audio) / rate:.1f}s "
                f"(stretch ×{meta['stretch_factor']:.2f}, peak {meta.get('peak_scale', 1.0):.2f})"
                if "stretch_factor" in meta
                else ""
            )
            return mo.vstack(
                [mo.image(_fig_to_image(fig), width=180), mo.audio(audio, rate=rate), mo.md(f"`{caption}`  \n*{info}*")]
            )

        if modality == "video_caption":
            video = arrays["video"]  # (T, H, W, 3)
            n = min(4, video.shape[0])
            fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 1.2))
            for i, ax in enumerate(axes if n > 1 else [axes]):
                ax.imshow(video[i])
                ax.axis("off")
            info = (
                f"orig {meta['original_width']}×{meta['original_height']} @ {meta['fps']:.0f}fps, "
                f"{meta['duration_sec']:.1f}s total — {video.shape[0]} frames spread over first 1s"
                if "original_width" in meta
                else ""
            )
            return mo.vstack([mo.image(_fig_to_image(fig), width=n * 90), mo.md(f"`{caption}`  \n*{info}*")])

        if modality == "lidar":
            points = arrays["points"]  # (N, 4) x,y,z,intensity
            fig = plt.figure(figsize=(2.5, 2.5))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 3], cmap="viridis")
            ax.set_axis_off()
            info = (
                f"{meta['n_points_original']} pts → {points.shape[0]} (azimuth-uniform, from {meta.get('components')})"
                if "n_points_original" in meta
                else ""
            )
            return mo.vstack([mo.image(_fig_to_image(fig), width=180), mo.md(f"`{ex['source']}`  \n*{info}*")])

        if modality == "control":
            state, action = arrays["state"], arrays["action"]
            fig, ax = plt.subplots(figsize=(3, 1.2))
            ax.plot(state, label="state", linewidth=0.7)
            ax.plot(action, label="action", linewidth=0.7)
            ax.legend(fontsize=6)
            ax.set_xticks([])
            try:
                params = _json.loads(caption)
                caption_display = ", ".join(f"{k}={v}" for k, v in list(params.items())[:4])
            except Exception:  # noqa: BLE001 — caption wasn't JSON, show as-is
                caption_display = caption
            peaks = f"peaks: state={meta.get('state_peak_scale', 1.0):.2f} action={meta.get('action_peak_scale', 1.0):.2f}"
            return mo.vstack([mo.image(_fig_to_image(fig), width=180), mo.md(f"`{caption_display[:80]}`  \n*{peaks}*")])

        return mo.md(f"(no preview for `{modality}`)")

    _by_modality: dict[str, list] = {}
    for _ex in multimodal_examples:
        _by_modality.setdefault(_ex["modality"], []).append(_ex)

    _sections = []
    for _mod, _rows in _by_modality.items():
        _sample = np.random.default_rng(0).choice(len(_rows), size=min(3, len(_rows)), replace=False)
        _cards = [_preview_row(_rows[i]) for i in _sample]
        _sections.append(mo.vstack([mo.md(f"### `{_mod}` ({len(_rows)} examples)"), mo.hstack(_cards, wrap=True)]))

    mo.vstack(_sections) if _sections else mo.md("*(no multimodal examples loaded)*")
    return


@app.cell
def _(mo):
    eval_pct = mo.ui.slider(start=0, stop=50, value=10, step=1, label="% held out for eval")
    eval_pct
    return (eval_pct,)


@app.cell
def _(eval_pct, mo, multimodal_examples, random, text_examples):
    _all_ex = list(text_examples) + list(multimodal_examples)
    _rng = random.Random(0)
    _shuffled = _all_ex.copy()
    _rng.shuffle(_shuffled)
    _n_eval = int(len(_shuffled) * eval_pct.value / 100)
    eval_examples = _shuffled[:_n_eval]
    train_examples = _shuffled[_n_eval:]
    mo.callout(
        mo.md(
            f"**Train:** `{len(train_examples)}` ({100 - eval_pct.value}%) · **Eval:** `{len(eval_examples)}` ({eval_pct.value}%)"
        ),
        kind="success",
    )
    return eval_examples, train_examples


@app.cell
def _(mo):
    mo.md("""
    ## ⚙️ 2. Model configuration
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    `modality_scales` routes each modality to a `PyramidalConvCodec` scale;
    `attnres_block_size` sets the v3 Block-AttnRes window (`1` = classic AttnRes).
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
    mo.md("""
    ## 🏋️ 3. Training configuration
    """)
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
    eval_examples,
    train_batch,
    train_epochs,
    train_examples,
    train_lr,
    train_max_len,
    train_run_dir,
    train_save_every,
    train_stride,
):
    data_config = DataConfig(
        text_examples=[],
        multimodal_examples=train_examples,
        max_len=train_max_len.value,
        stride=train_stride.value,
        batch_size=train_batch.value,
    )
    eval_data_config = DataConfig(
        text_examples=[],
        multimodal_examples=eval_examples,
        max_len=train_max_len.value,
        stride=train_stride.value,
        batch_size=train_batch.value,
        shuffle=False,
        drop_last=False,
    )
    train_config = TrainConfig(
        lr=train_lr.value,
        epochs=train_epochs.value,
        save_every=train_save_every.value,
        run_dir=train_run_dir.value,
    )
    return data_config, eval_data_config, train_config


@app.cell
def _(mo):
    mo.md("""
    ## 🚂 4. Build & Train
    """)
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
    _resumed = (pipe.ckpt_dir / "last.pt").exists()
    if _resumed:
        mo.callout(mo.md(f"↻ Found `last.pt` in `{pipe.ckpt_dir}` — resuming from there."), kind="info")

    _total_steps = pipe.train_config.epochs * len(pipe.loader)
    with mo.status.progress_bar(total=_total_steps, title="Training", subtitle="step 0") as _bar:

        def _on_step(step, total, loss_val):
            _bar.update(increment=1, subtitle=f"step {step}/{total} — loss {loss_val:.4f}")

        logs = pipe.train(progress_callback=_on_step, resume=True)

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
def _(eval_data_config, mo, pipe, torch):
    from kairos.dataset import KairosPretrainingDataset
    from torch.utils.data import DataLoader

    if len(eval_data_config.multimodal_examples) == 0:
        mo.callout(mo.md("No eval examples (eval % = 0) — skipping."), kind="neutral")
        eval_loss = None
    else:
        with mo.status.spinner(title="Evaluating on held-out split..."):
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

        mo.callout(mo.md(f"**Eval loss:** `{eval_loss:.4f}` on `{len(eval_dataset)}` samples"), kind="success")
    return


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
    Diffusion loss masked per modality so a router-ignored one can't hide behind the global average.
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
    mo.md("""
    ## ♻️ 7. Resume from checkpoint
    """)
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


@app.cell
def _(mo):
    mo.md("""
    ## 🚀 8. Push to Hugging Face Hub
    """)
    return


@app.cell
def _(mo):
    hub_repo_id = mo.ui.text(value="ffurfaro/kairos", label="Model repo id")
    hub_private = mo.ui.checkbox(value=False, label="Private")
    push_button = mo.ui.run_button(label="Push model + checkpoints + tensorboard")
    mo.vstack([hub_repo_id, hub_private, push_button])
    return hub_private, hub_repo_id, push_button


@app.cell
def _(hub_private, hub_repo_id, mo, pipe, push_button):
    if not push_button.value:
        mo.stop(True, mo.callout(mo.md("Click **Push** once training is done."), kind="neutral"))

    with mo.status.spinner(title=f"Pushing to {hub_repo_id.value}..."):
        pipe.push_to_hub(hub_repo_id.value, private=hub_private.value)

    mo.callout(mo.md(f"✅ Pushed to [{hub_repo_id.value}](https://huggingface.co/{hub_repo_id.value})"), kind="success")
    return


if __name__ == "__main__":
    app.run()
