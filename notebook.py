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
    # 🌀 Kairos — Training Notebook
    Build · Inspect · Train · Checkpoint
    """)
    return


@app.cell
def _():
    import os
    import math
    import time
    import json
    from pathlib import Path
    from datetime import datetime

    import torch
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import DataLoader

    from transformers import TrainingArguments

    from src.modeling import KairosConfig, KairosDiffusionLLM, ConvCodec
    from src.attentions import KairosCache
    from src.tokenizer import KairosTokenizer
    from src.dataset import KairosPretrainingDataset
    from src.trainer import KairosDiffusionTrainer

    return (
        DataLoader,
        KairosConfig,
        KairosPretrainingDataset,
        KairosDiffusionLLM,
        KairosDiffusionTrainer,
        KairosTokenizer,
        Path,
        SummaryWriter,
        TrainingArguments,
        math,
        torch,
    )


@app.cell
def _(mo, torch):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    mo.callout(
        mo.md(f"**Device:** `{device}`  |  "
              f"**CUDA:** `{torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}`"),
        kind="info",
    )
    return (device,)


@app.cell
def _(mo):
    mo.md("""
    ## ⚙️ Model Configuration
    """)
    return


@app.cell
def _(mo):
    cfg_d_model = mo.ui.slider(
        32, 768, step=32, value=256, label="d_model (hidden size)"
    )
    cfg_n_heads = mo.ui.slider(
        2, 16, step=2, value=4, label="n_heads"
    )
    cfg_n_layers = mo.ui.slider(
        1, 24, step=1, value=4, label="n_layers"
    )
    cfg_window = mo.ui.slider(
        16, 512, step=16, value=64, label="SWA window size"
    )
    cfg_stride = mo.ui.slider(
        1, 6, step=1, value=3, label="ConvCodec stride"
    )
    cfg_experts = mo.ui.slider(
        0, 32, step=2, value=3,
        label="MoE experts (0 = dense FFN)"
    )

    mo.vstack([
        mo.hstack([cfg_d_model, cfg_n_heads, cfg_n_layers]),
        mo.hstack([cfg_window, cfg_stride, cfg_experts]),
    ])
    return (
        cfg_d_model,
        cfg_experts,
        cfg_n_heads,
        cfg_n_layers,
        cfg_stride,
        cfg_window,
    )


@app.cell
def _(
    KairosConfig,
    KairosDiffusionLLM,
    cfg_d_model,
    cfg_experts,
    cfg_n_heads,
    cfg_n_layers,
    cfg_stride,
    cfg_window,
    device,
    mo,
):
    config = KairosConfig(
        d_model=cfg_d_model.value,
        n_heads=cfg_n_heads.value,
        n_layers=cfg_n_layers.value,
        window_size=cfg_window.value,
        stride=cfg_stride.value,
        num_experts=cfg_experts.value if cfg_experts.value > 0 else 8,
        num_experts_per_tok=2,
    )

    num_experts_arg = cfg_experts.value if cfg_experts.value > 0 else None

    model = KairosDiffusionLLM(
        config,
        vocab_size=259,
        num_experts=num_experts_arg,
    ).to(device)

    total_params   = sum(p.numel() for p in model.parameters())
    active_params  = sum(
        p.numel() for n, p in model.named_parameters()
        if "experts" not in n or any(f"experts.{i}" in n for i in range(config.num_experts_per_tok))
    )

    mo.callout(
        mo.md(
            f"**Total params:** `{total_params/1e6:.2f}M`  \n"
            f"**Active params (est.):** `{active_params/1e6:.2f}M`  \n"
            f"**Layers config:** `{config.layers_config}`  \n"
            f"**MoE:** `{'Yes — ' + str(cfg_experts.value) + ' experts' if num_experts_arg else 'No — dense FFN'}`"
        ),
        kind="success",
    )
    return config, model


@app.cell
def _(mo):
    mo.md("""
    ## 🔬 Architecture Inspection
    """)
    return


@app.cell
def _(mo, model):
    rows = []
    for name, param in model.named_parameters():
        rows.append({
            "Layer": name,
            "Shape": str(list(param.shape)),
            "Params": f"{param.numel():,}",
            "Trainable": "✅" if param.requires_grad else "❌",
            "dtype": str(param.dtype).replace("torch.", ""),
        })

    mo.ui.table(
        rows,
        label="Model parameters",
        pagination=True,
        page_size=100,
    )
    return


@app.cell
def _(config, device, mo, model, torch):
    # Dry-run forward to check shapes
    _x = torch.randint(0, 259, (1, 12)).to(device)
    with torch.no_grad():
        _out = model(input_ids=_x)

    mo.callout(
        mo.md(
            f"**Dry-run input:** `(1, 12)` tokens  \n"
            f"**After codec encode:** `(1, {12 // config.stride}, {config.hidden_size})`  \n"
            f"**Output logits:** `{list(_out.logits.shape)}`  \n"
            f"**No NaN:** `{not torch.isnan(_out.logits).any().item()}`"
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🏋️ Training Configuration
    """)
    return


@app.cell
def _(mo):
    train_lr          = mo.ui.number(1e-4, 1e-2, step=1e-5, value=3e-4, label="Learning rate")
    train_batch       = mo.ui.slider(1, 64, step=1, value=8, label="Batch size")
    train_epochs      = mo.ui.slider(1, 100, step=1, value=5, label="Epochs")
    train_max_len     = mo.ui.slider(32, 512, step=32, value=128, label="Max sequence length")
    train_save_steps  = mo.ui.slider(10, 500, step=10, value=50, label="Save every N steps")
    train_log_steps   = mo.ui.slider(1, 50, step=1, value=10, label="Log every N steps")
    train_output_dir  = mo.ui.text(value="checkpoints/kairos", label="Output directory")
    train_run_name    = mo.ui.text(value="run_01", label="Run name (TensorBoard)")

    mo.vstack([
        mo.hstack([train_lr, train_batch, train_epochs]),
        mo.hstack([train_max_len, train_save_steps, train_log_steps]),
        mo.hstack([train_output_dir, train_run_name]),
    ])
    return (
        train_batch,
        train_epochs,
        train_log_steps,
        train_lr,
        train_max_len,
        train_output_dir,
        train_run_name,
        train_save_steps,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 📦 Dataset
    """)
    return


@app.cell
def _(mo):
    dataset_source = mo.ui.radio(
        options=["cosmopedia (HuggingFace)", "custom texts"],
        value="cosmopedia (HuggingFace)",
        label="Dataset source",
    )
    dataset_source
    return (dataset_source,)


@app.cell
def _(dataset_source, mo):
    custom_texts_input = mo.ui.text_area(
        value="Paris is the capital of France.\nThe Earth orbits the Sun.\nWater boils at 100 degrees Celsius.",
        label="Custom texts (one per line)",
        rows=6,
    ) if dataset_source.value == "custom texts" else None

    if custom_texts_input:
        custom_texts_input
    return (custom_texts_input,)


@app.cell
def _(
    KairosPretrainingDataset,
    KairosTokenizer,
    custom_texts_input,
    dataset_source,
    mo,
    train_max_len,
):
    tokenizer = KairosTokenizer()

    texts = None
    if dataset_source.value == "custom texts" and custom_texts_input:
        texts = [t for t in custom_texts_input.value.split("\n") if t.strip()]

    with mo.status.spinner(title="Loading dataset..."):
        dataset = KairosPretrainingDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_len=train_max_len.value,
        )

    mo.callout(
        mo.md(f"**Samples:** `{len(dataset)}`  |  **Max len:** `{train_max_len.value}`  |  **Source:** `{dataset_source.value}`"),
        kind="success",
    )
    return dataset, tokenizer


@app.cell
def _(mo):
    mo.md("""
    ## 🚀 Train
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="▶ Start Training")
    run_button
    return (run_button,)


@app.cell
def _(
    DataLoader,
    KairosDiffusionTrainer,
    Path,
    SummaryWriter,
    TrainingArguments,
    config,
    dataset,
    device,
    math,
    mo,
    model,
    run_button,
    tokenizer,
    torch,
    train_batch,
    train_epochs,
    train_log_steps,
    train_lr,
    train_output_dir,
    train_run_name,
    train_save_steps,
):
    if not run_button.value:
        mo.stop(True, mo.callout(mo.md("Click **▶ Start Training** to begin."), kind="neutral"))

    # ── dirs ──
    run_dir = Path(train_output_dir.value) / train_run_name.value
    tb_dir  = run_dir / "tensorboard"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── save config ──
    import json as _json
    with open(run_dir / "config.json", "w") as _f:
        _json.dump({
            "d_model": config.hidden_size,
            "n_heads": config.num_attention_heads,
            "n_layers": config.num_hidden_layers,
            "sliding_window_size": config.sliding_window_size,
            "stride": config.stride,
            "vocab_size": 259,
        }, _f, indent=2)

    writer = SummaryWriter(log_dir=str(tb_dir))

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=train_epochs.value,
        per_device_train_batch_size=train_batch.value,
        learning_rate=train_lr.value,
        logging_steps=train_log_steps.value,
        save_steps=train_save_steps.value,
        save_total_limit=3,
        report_to=[],          # on gère TensorBoard manuellement
        dataloader_pin_memory=(device == "cuda"),
        remove_unused_columns=False,
        no_cuda=(device != "cuda"),
    )

    trainer = KairosDiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # ── custom loop with marimo progress + TensorBoard ──
    loader = DataLoader(
        dataset,
        batch_size=train_batch.value,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_lr.value)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_epochs.value * len(loader),
        eta_min=train_lr.value * 0.1,
    )

    global_step = 0
    best_loss   = float("inf")
    log_rows    = []

    model.train()

    with mo.status.progress_bar(
        total=train_epochs.value * len(loader),
        title="Training Kairos",
        subtitle="epoch 1",
    ) as _bar:

        for epoch in range(1, train_epochs.value + 1):
            epoch_loss = 0.0

            for step, batch in enumerate(loader, 1):
                # move to device
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

                optimizer.zero_grad()

                loss = trainer.compute_loss(model, batch)
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                loss_val   = loss.item()
                epoch_loss += loss_val
                global_step += 1

                # ── TensorBoard ──
                if global_step % train_log_steps.value == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    writer.add_scalar("train/loss",    loss_val,   global_step)
                    writer.add_scalar("train/lr",      lr_now,     global_step)
                    writer.add_scalar("train/epoch",   epoch,      global_step)

                    # gradient norm
                    grad_norm = math.sqrt(sum(
                        p.grad.norm().item() ** 2
                        for p in model.parameters()
                        if p.grad is not None
                    ))
                    writer.add_scalar("train/grad_norm", grad_norm, global_step)

                    log_rows.append({
                        "step":  global_step,
                        "epoch": epoch,
                        "loss":  f"{loss_val:.4f}",
                        "lr":    f"{lr_now:.2e}",
                        "grad_norm": f"{grad_norm:.3f}",
                    })

                # ── Checkpoint ──
                if global_step % train_save_steps.value == 0:
                    ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
                    torch.save({
                        "step":            global_step,
                        "epoch":           epoch,
                        "model_state":     model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "loss":            loss_val,
                        "config":          config.to_dict(),
                    }, ckpt_path)
                    writer.add_text(
                        "checkpoints",
                        f"Saved `{ckpt_path.name}` — loss {loss_val:.4f}",
                        global_step,
                    )

                _bar.update(subtitle=f"epoch {epoch}/{train_epochs.value} | loss {loss_val:.4f}")

            avg_loss = epoch_loss / len(loader)
            writer.add_scalar("train/epoch_avg_loss", avg_loss, epoch)

            # ── Best checkpoint ──
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = ckpt_dir / "best.pt"
                torch.save({
                    "step":        global_step,
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "loss":        best_loss,
                    "config":      config.to_dict(),
                }, best_path)
                writer.add_text("checkpoints", f"🏆 New best at epoch {epoch} — loss {best_loss:.4f}", global_step)

    writer.flush()
    writer.close()

    mo.callout(
        mo.md(
            f"✅ **Training complete**  \n"
            f"Steps: `{global_step}` | Best loss: `{best_loss:.4f}`  \n"
            f"Checkpoints: `{ckpt_dir}`  \n"
            f"TensorBoard: `tensorboard --logdir {tb_dir}`"
        ),
        kind="success",
    )
    return log_rows, optimizer, scheduler, tb_dir


@app.cell
def _(log_rows, mo, run_button):
    if not run_button.value or not log_rows:
        mo.stop(True)

    mo.vstack([
        mo.md("## 📊 Training Logs"),
        mo.ui.table(log_rows, label="Step logs", pagination=True, page_size=20),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 💾 Checkpoint Browser
    """)
    return


@app.cell
def _(mo):
    ckpt_browser_dir = mo.ui.text(
        value="checkpoints/kairos/run_01/checkpoints",
        label="Checkpoint directory",
    )
    ckpt_browser_dir
    return (ckpt_browser_dir,)


@app.cell
def _(Path, ckpt_browser_dir, mo, torch):
    _ckpt_path = Path(ckpt_browser_dir.value)

    if not _ckpt_path.exists():
        mo.callout(mo.md(f"Directory `{_ckpt_path}` not found."), kind="warn")
    else:
        _files = sorted(_ckpt_path.glob("*.pt"))
        if not _files:
            mo.callout(mo.md("No checkpoints found."), kind="warn")
        else:
            _rows = []
            for _f in _files:
                try:
                    _ck = torch.load(_f, map_location="cpu", weights_only=True)
                    _rows.append({
                        "File":    _f.name,
                        "Step":    _ck.get("step", "?"),
                        "Epoch":   _ck.get("epoch", "?"),
                        "Loss":    f"{_ck.get('loss', 0):.4f}",
                        "Size":    f"{_f.stat().st_size / 1e6:.1f} MB",
                    })
                except Exception as _e:
                    _rows.append({"File": _f.name, "Step": "error", "Epoch": "?", "Loss": "?", "Size": "?"})

            mo.ui.table(_rows, label="Available checkpoints")
    return


@app.cell
def _(mo):
    mo.md("""
    ## ♻️ Resume from Checkpoint
    """)
    return


@app.cell
def _(mo):
    resume_path   = mo.ui.text(value="", label="Checkpoint path (.pt)")
    resume_button = mo.ui.run_button(label="Load checkpoint")
    mo.hstack([resume_path, resume_button])
    return resume_button, resume_path


@app.cell
def _(
    Path,
    mo,
    model,
    optimizer,
    resume_button,
    resume_path,
    scheduler,
    torch,
):
    if not resume_button.value or not resume_path.value:
        mo.stop(True)

    _path = Path(resume_path.value)
    if not _path.exists():
        mo.callout(mo.md(f"File not found: `{_path}`"), kind="danger")
    else:
        _ck = torch.load(_path, map_location="cpu")
        model.load_state_dict(_ck["model_state"])
        if "optimizer_state" in _ck:
            optimizer.load_state_dict(_ck["optimizer_state"])
        if "scheduler_state" in _ck:
            scheduler.load_state_dict(_ck["scheduler_state"])

        mo.callout(
            mo.md(
                f"✅ Loaded `{_path.name}`  \n"
                f"Step: `{_ck.get('step', '?')}` | Epoch: `{_ck.get('epoch', '?')}` | Loss: `{_ck.get('loss', 0):.4f}`"
            ),
            kind="success",
        )
    return


@app.cell
def _(mo, run_button, tb_dir):
    if not run_button.value:
        mo.stop(True)

    mo.callout(
        mo.md(
            f"## 📈 TensorBoard\n\n"
            f"Lance dans un terminal :\n\n"
            f"```bash\ntensorboard --logdir {tb_dir} --port 6006\n```\n\n"
            f"Puis ouvre [http://localhost:6006](http://localhost:6006)"
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
