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
    Build · Inspect · Train · Checkpoint · ⚠️ WIP
    """)
    return


@app.cell
def _():
    import math
    from pathlib import Path

    import torch
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import DataLoader, ConcatDataset

    from transformers import TrainingArguments

    from kairos.modeling import KairosConfig, KairosDiffusionLLM
    from kairos.tokenizer import KairosTokenizer
    from kairos.dataset import KairosPretrainingDataset
    from kairos.trainer import KairosDiffusionTrainer

    return (
        ConcatDataset,
        DataLoader,
        KairosConfig,
        KairosDiffusionLLM,
        KairosDiffusionTrainer,
        KairosPretrainingDataset,
        KairosTokenizer,
        Path,
        SummaryWriter,
        TrainingArguments,
        math,
        torch,
    )


@app.cell
def _(mo, torch):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    mo.callout(mo.md(f"**Device:** `{device}`"), kind="info")
    return (device,)


@app.cell
def _(KairosTokenizer):
    # Multimodal tokenizer — created once, shared by model config (vocab_size)
    # and dataset cells below. len(tokenizer) is 291, not the old hardcoded 259.
    tokenizer = KairosTokenizer()
    return (tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""*Estimate pretraining compute time* — [Chinchilla](https://arxiv.org/abs/2203.15556)""")
    return


@app.cell
def _(mo):
    GPU_FLOPS = {"T4": 65, "RTX_5060Ti": 200, "RTX_5070Ti": 300, "A100": 312, "H100": 1000}
    GPU_BW = {"T4": 300e9, "RTX_5060Ti": 600e9, "RTX_5070Ti": 800e9, "A100": 1500e9, "H100": 3000e9}

    gpu = mo.ui.dropdown(list(GPU_FLOPS.keys()), value="T4", label="GPU")
    eff = mo.ui.slider(0.05, 0.5, step=0.05, value=0.3, label="GPU efficiency")
    params = mo.ui.slider(1, 50, value=14, label="Active params (millions)")
    tokens = mo.ui.slider(1, 50, value=1, label="Training tokens (billions)")

    mo.vstack([gpu, params, tokens, eff])
    return GPU_BW, GPU_FLOPS, eff, gpu, params, tokens


@app.cell
def _(GPU_BW, GPU_FLOPS, eff, gpu, mo, params, tokens):
    flops = GPU_FLOPS[gpu.value] * 1e12
    bw = GPU_BW[gpu.value]
    params_total = params.value * 1e6
    tokens_total = tokens.value * 1e9
    n_ops = 6

    tok_s_compute = (flops / (n_ops * params_total)) * eff.value
    bytes_per_token = 4 * params_total
    tok_s_memory = bw / bytes_per_token
    tok_s_real = min(tok_s_compute, tok_s_memory)
    time_days = tokens_total / tok_s_real / 86400

    mo.md(f"""
    ### 🧠 Compute-bound` {tok_s_compute:.0f} tok/s`
    ### 💾 Memory-bound` {tok_s_memory:.0f} tok/s`
    ### ⚖️ Real throughput` {tok_s_real:.0f} tok/s`
    ### ⏱️ Training time` {time_days:.1f} days`
    """)
    return


@app.cell
def _(mo):
    mo.md("## ⚙️ Model Configuration")
    return


@app.cell
def _(mo):
    # Target: ~14M total, 7 routed experts + 1 always-on shared expert,
    # top-1 routing => 2 experts active per token (1 shared + 1 routed).
    cfg_d_model = mo.ui.slider(32, 768, step=32, value=128, label="d_model (hidden size)")
    cfg_n_heads = mo.ui.slider(2, 16, step=2, value=4, label="n_heads")
    cfg_n_layers = mo.ui.slider(1, 24, step=1, value=6, label="n_layers")
    cfg_window = mo.ui.slider(16, 512, step=16, value=64, label="SWA window size")
    cfg_stride = mo.ui.slider(1, 6, step=1, value=3, label="PyramidalConvCodec stride")
    cfg_local_experts = mo.ui.slider(0, 32, step=1, value=7, label="Routed experts (0 = dense FFN)")
    cfg_experts_per_tok = mo.ui.slider(1, 8, step=1, value=1, label="Routed experts active per token")
    cfg_shared_experts = mo.ui.slider(0, 4, step=1, value=1, label="Shared experts (always active)")

    mo.vstack(
        [
            mo.hstack([cfg_d_model, cfg_n_heads, cfg_n_layers]),
            mo.hstack([cfg_window, cfg_stride]),
            mo.hstack([cfg_local_experts, cfg_experts_per_tok, cfg_shared_experts]),
        ]
    )
    return (
        cfg_d_model,
        cfg_experts_per_tok,
        cfg_local_experts,
        cfg_n_heads,
        cfg_n_layers,
        cfg_shared_experts,
        cfg_stride,
        cfg_window,
    )


@app.cell
def _(
    KairosConfig,
    KairosDiffusionLLM,
    cfg_d_model,
    cfg_experts_per_tok,
    cfg_local_experts,
    cfg_n_heads,
    cfg_n_layers,
    cfg_shared_experts,
    cfg_stride,
    cfg_window,
    device,
    mo,
    tokenizer,
):
    use_moe = cfg_local_experts.value > 0

    config = KairosConfig(
        d_model=cfg_d_model.value,
        n_heads=cfg_n_heads.value,
        n_layers=cfg_n_layers.value,
        window_size=cfg_window.value,
        stride=cfg_stride.value,
        vocab_size=len(tokenizer),
        # NOTE: DeepseekV3MoE (used under the hood) reads config.num_local_experts
        # for both the router and the expert weight matrices — n_routed_experts
        # is not read anywhere and has no effect on the actual model. This is
        # what caused the historical "Class values must be smaller than
        # num_classes" crash when only n_routed_experts was set.
        # kept in sync: depending on the installed transformers version, the
        # DeepseekV3 MoE backend reads n_routed_experts or num_local_experts
        # for the actual expert weight tensor size — set both or the router's
        # top-k index space and the experts tensor can disagree.
        num_local_experts=cfg_local_experts.value if use_moe else 8,
        n_routed_experts=cfg_local_experts.value if use_moe else 8,
        num_experts_per_tok=cfg_experts_per_tok.value,
        n_shared_experts=cfg_shared_experts.value,
        use_moe=use_moe,
    )

    model = KairosDiffusionLLM(config, vocab_size=len(tokenizer)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
    non_expert_params = total_params - expert_params
    active_ratio = (config.num_experts_per_tok / config.num_local_experts) if use_moe else 1.0
    active_params = non_expert_params + expert_params * active_ratio

    mo.callout(
        mo.md(
            f"**Total params:** `{total_params / 1e6:.2f}M`  \n"
            f"**Active params:** `{active_params / 1e6:.2f}M`  \n"
            f"**MoE:** `{'Yes — ' + str(cfg_local_experts.value) + ' routed + ' + str(cfg_shared_experts.value) + ' shared' if use_moe else 'No — dense FFN'}`  \n"
            f"**Vocab size:** `{len(tokenizer)}`"
        ),
        kind="success",
    )
    return config, model


@app.cell
def _(mo):
    mo.md("## 🔬 Architecture Inspection")
    return


@app.cell
def _(mo, model):
    rows = [
        {
            "Layer": name,
            "Shape": str(list(param.shape)),
            "Params": f"{param.numel():,}",
            "Trainable": "✅" if param.requires_grad else "❌",
        }
        for name, param in model.named_parameters()
    ]
    mo.ui.table(rows, label="Model parameters", pagination=True, page_size=100)
    return


@app.cell
def _(device, mo, model, tokenizer, torch):
    _x = torch.randint(0, len(tokenizer), (1, 12)).to(device)
    with torch.no_grad():
        _out = model(input_ids=_x)

    mo.callout(
        mo.md(
            f"**Dry-run input:** `(1, 12)` tokens  \n"
            f"**Output logits:** `{list(_out.logits.shape)}`  \n"
            f"**No NaN:** `{not torch.isnan(_out.logits).any().item()}`"
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md("## 🏋️ Training Configuration")
    return


@app.cell
def _(mo):
    train_lr = mo.ui.number(1e-4, 1e-2, step=1e-5, value=3e-4, label="Learning rate")
    train_batch = mo.ui.slider(1, 64, step=1, value=8, label="Batch size")
    train_epochs = mo.ui.slider(1, 100, step=1, value=5, label="Epochs")
    train_max_len = mo.ui.slider(32, 4096, step=32, value=512, label="Max sequence length")
    train_save_steps = mo.ui.slider(10, 500, step=10, value=50, label="Save every N steps")
    train_log_steps = mo.ui.slider(1, 50, step=1, value=10, label="Log every N steps")
    train_output_dir = mo.ui.text(value="checkpoints/kairos", label="Output directory")
    train_run_name = mo.ui.text(value="run_01", label="Run name (TensorBoard)")

    mo.vstack(
        [
            mo.hstack([train_lr, train_batch, train_epochs]),
            mo.hstack([train_max_len, train_save_steps, train_log_steps]),
            mo.hstack([train_output_dir, train_run_name]),
        ]
    )
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
    mo.md("## 📦 Dataset")
    return


@app.cell
def _(mo):
    dataset_source = mo.ui.radio(
        options=[
            "keep-it-simple + keep-it-simple-multimodal (recommended)",
            "keep-it-simple (text only)",
            "custom texts",
        ],
        value="keep-it-simple + keep-it-simple-multimodal (recommended)",
        label="Dataset source",
    )
    dataset_source
    return (dataset_source,)


@app.cell
def _(dataset_source, mo):
    custom_texts_input = (
        mo.ui.text_area(
            value="Paris is the capital of France.\nThe Earth orbits the Sun.",
            label="Custom texts (one per line)",
            rows=6,
        )
        if dataset_source.value == "custom texts"
        else None
    )
    custom_texts_input if custom_texts_input is not None else mo.md("")
    return (custom_texts_input,)


@app.cell
def _(dataset_source, mo):
    multimodal_path = (
        mo.ui.text(
            value="data/keep-it-simple-multimodal.pt",
            label="keep-it-simple-multimodal .pt path (from build_keep_it_simple_multimodal.py)",
        )
        if dataset_source.value.startswith("keep-it-simple + ")
        else None
    )
    multimodal_path if multimodal_path is not None else mo.md("")
    return (multimodal_path,)


@app.cell
def _(
    ConcatDataset,
    KairosPretrainingDataset,
    custom_texts_input,
    dataset_source,
    mo,
    multimodal_path,
    tokenizer,
    train_max_len,
):
    with mo.status.spinner(title="Loading dataset..."):
        if dataset_source.value == "custom texts":
            texts = [t for t in (custom_texts_input.value if custom_texts_input else "").split("\n") if t.strip()]
            dataset = KairosPretrainingDataset(texts=texts, tokenizer=tokenizer, max_len=train_max_len.value)

        elif dataset_source.value == "keep-it-simple (text only)":
            from datasets import load_dataset

            text_rows = load_dataset("ffurfaro/keep-it-simple", split="train")["text"]
            dataset = KairosPretrainingDataset(texts=text_rows, tokenizer=tokenizer, max_len=train_max_len.value)

        else:
            from datasets import load_dataset

            text_rows = load_dataset("ffurfaro/keep-it-simple", split="train")["text"]
            text_ds = KairosPretrainingDataset(texts=text_rows, tokenizer=tokenizer, max_len=train_max_len.value)
            multimodal_ds = KairosPretrainingDataset(
                multimodal_path=multimodal_path.value, tokenizer=tokenizer, max_len=train_max_len.value
            )
            dataset = ConcatDataset([text_ds, multimodal_ds])

    mo.callout(
        mo.md(
            f"**Samples:** `{len(dataset)}`  |  **Max len:** `{train_max_len.value}`  |  **Source:** `{dataset_source.value}`"
        ),
        kind="success",
    )
    return (dataset,)


@app.cell
def _(mo):
    mo.md("## 🚀 Train")
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
    config,
    dataset,
    device,
    math,
    mo,
    model,
    run_button,
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

    run_dir = Path(train_output_dir.value) / train_run_name.value
    tb_dir = run_dir / "tensorboard"
    ckpt_dir = run_dir / "checkpoints"
    for _d in (run_dir, tb_dir, ckpt_dir):
        _d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))

    loader = DataLoader(dataset, batch_size=train_batch.value, shuffle=True, pin_memory=(device == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_lr.value)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_epochs.value * len(loader),
        eta_min=train_lr.value * 0.1,
    )

    global_step = 0
    best_loss = float("inf")
    log_rows = []
    model.train()

    # batches carry input_ids/modality_ids/mask straight from
    # KairosPretrainingDataset; compute_loss reads all three from `batch`.
    trainer = KairosDiffusionTrainer(model=model)

    with mo.status.progress_bar(total=train_epochs.value * len(loader), title="Training Kairos") as _bar:
        for epoch in range(1, train_epochs.value + 1):
            epoch_loss = 0.0
            for step, batch in enumerate(loader, 1):
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

                optimizer.zero_grad()
                loss = trainer.compute_loss(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                global_step += 1

                if global_step % train_log_steps.value == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    writer.add_scalar("train/loss", loss_val, global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)
                    grad_norm = math.sqrt(
                        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
                    )
                    writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    log_rows.append(
                        {"step": global_step, "epoch": epoch, "loss": f"{loss_val:.4f}", "lr": f"{lr_now:.2e}"}
                    )

                if global_step % train_save_steps.value == 0:
                    ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
                    torch.save(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "loss": loss_val,
                            "config": config.to_dict(),
                        },
                        ckpt_path,
                    )

                _bar.update(subtitle=f"epoch {epoch}/{train_epochs.value} | loss {loss_val:.4f}")

            avg_loss = epoch_loss / len(loader)
            writer.add_scalar("train/epoch_avg_loss", avg_loss, epoch)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "loss": best_loss,
                        "config": config.to_dict(),
                    },
                    ckpt_dir / "best.pt",
                )

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
    return log_rows, tb_dir


@app.cell
def _(log_rows, mo, run_button):
    if not run_button.value or not log_rows:
        mo.stop(True)
    mo.vstack([mo.md("## 📊 Training Logs"), mo.ui.table(log_rows, label="Step logs", pagination=True, page_size=20)])
    return


@app.cell
def _(mo, run_button, tb_dir):
    if not run_button.value:
        mo.stop(True)
    mo.callout(
        mo.md(f"## 📈 TensorBoard\n\n```bash\ntensorboard --logdir {tb_dir} --port 6006\n```"),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
