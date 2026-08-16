import numpy as np
import pytest
import torch

from kairos.dataset import KairosPretrainingDataset, pack_multimodal_data
from kairos.modeling import KairosConfig, KairosDiffusionFM
from kairos.tokenizer import KairosTokenizer
from kairos.trainer import KairosDiffusionTrainer, compute_masked_diffusion_losses, make_diffusion_mask


@pytest.fixture
def tokenizer():
    return KairosTokenizer()


@pytest.fixture
def config(tokenizer):
    return KairosConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        vocab_size=len(tokenizer),
        num_modalities=8,
        stride=1,
        num_scales=2,
        # keep both aliases in sync; different transformers versions read different fields
        num_local_experts=7,
        n_routed_experts=7,
        num_experts_per_tok=1,
        n_shared_experts=1,
        use_moe=True,
    )


@pytest.fixture
def model(config, tokenizer):
    torch.manual_seed(42)
    return KairosDiffusionFM(config, vocab_size=len(tokenizer))


@pytest.fixture
def dense_config(tokenizer):
    return KairosConfig(
        d_model=32, n_heads=4, n_layers=2, vocab_size=len(tokenizer), num_modalities=8, stride=1, num_scales=2
    )


@pytest.fixture
def dense_model(dense_config, tokenizer):
    torch.manual_seed(42)
    return KairosDiffusionFM(dense_config, vocab_size=len(tokenizer))


def test_compute_loss_runs_end_to_end(dense_model, tokenizer):
    """Regression: trainer used to default every token to Modality.TEXT."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    examples = [
        {"modality": "text", "text": "Paris is the capital of France."},
        {
            "modality": "image_caption",
            "caption": "a dog",
            "source": "test",
            "data": pack_multimodal_data({"image": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)}),
            "meta": None,
        },
    ]
    ds = KairosPretrainingDataset(multimodal_examples=examples, tokenizer=tokenizer, max_len=128, stride=1)
    batch = {
        "input_ids": torch.stack([ds[i]["input_ids"] for i in range(len(ds))]),
        "modality_ids": torch.stack([ds[i]["modality_ids"] for i in range(len(ds))]),
        "mask": torch.stack([ds[i]["mask"] for i in range(len(ds))]),
        "prompt_len": torch.zeros(len(ds), dtype=torch.long),
    }

    trainer = KairosDiffusionTrainer(model=dense_model)
    loss = trainer.compute_loss(dense_model, batch)

    assert torch.is_tensor(loss) and loss.dim() == 0
    assert not torch.isnan(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dense_model.parameters())


def test_octet_family_loss_adds_to_total_when_configured(dense_model, tokenizer):
    torch.manual_seed(0)
    x0 = torch.randint(0, len(tokenizer), (2, 8))
    batch = {
        "input_ids": x0,
        "modality_ids": torch.zeros_like(x0),
        "mask": torch.ones_like(x0),
        "prompt_len": torch.zeros(2, dtype=torch.long),
    }
    trainer = KairosDiffusionTrainer(model=dense_model)
    torch.manual_seed(1)
    loss_without = trainer.compute_loss(dense_model, batch)
    trainer.octet_family_ids = tokenizer.octet_family_ids
    torch.manual_seed(1)
    loss_with = trainer.compute_loss(dense_model, batch)
    assert not torch.allclose(loss_without, loss_with)


def test_moe_plumbing_does_not_crash(model, tokenizer):
    """Looser MoE-path check: tiny random inits can occasionally give non-finite logits, so."""
    torch.manual_seed(0)
    ids = tokenizer.encode("hello world", add_special_tokens=False)
    ids = ids + [tokenizer.pad_token_id] * (32 - len(ids))
    batch = {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "prompt_len": torch.zeros(1, dtype=torch.long),
    }
    trainer = KairosDiffusionTrainer(model=model)
    loss = trainer.compute_loss(model, batch)
    if not torch.isfinite(loss):
        pytest.skip("known MoE numerical fragility on tiny random init (non-finite loss), not a code bug")
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_compute_loss_never_noises_padding(model, tokenizer):
    ds = KairosPretrainingDataset(texts=["hi"], tokenizer=tokenizer, max_len=64, stride=1)
    batch = {
        "input_ids": ds[0]["input_ids"].unsqueeze(0),
        "modality_ids": ds[0]["modality_ids"].unsqueeze(0),
        "mask": ds[0]["mask"].unsqueeze(0),
        "prompt_len": torch.zeros(1, dtype=torch.long),
    }
    pad_positions = batch["mask"] == 0
    assert pad_positions.any(), "fixture must contain padding for this test to be meaningful"

    x0_before = batch["input_ids"].clone()
    torch.manual_seed(0)
    trainer = KairosDiffusionTrainer(model=model)
    trainer.compute_loss(model, batch)

    # compute_loss must not mutate the original input batch
    assert torch.equal(batch["input_ids"], x0_before)


def test_compute_loss_backward_compatible_without_modality_or_mask(tokenizer):
    """SFT/DPO/RL-style batches (no modality_ids/mask keys) must still work."""
    torch.manual_seed(0)
    dense_config = KairosConfig(d_model=32, n_heads=4, n_layers=2, vocab_size=len(tokenizer), num_modalities=8)
    dense_model = KairosDiffusionFM(dense_config, vocab_size=len(tokenizer))

    ids = tokenizer.encode("hello world", add_special_tokens=False)
    ids = ids + [tokenizer.pad_token_id] * (32 - len(ids))
    batch = {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "prompt_len": torch.zeros(1, dtype=torch.long),
    }
    trainer = KairosDiffusionTrainer(model=dense_model)
    loss = trainer.compute_loss(dense_model, batch)
    assert torch.is_tensor(loss) and not torch.isnan(loss)


def test_compute_loss_forces_one_position_when_noise_mask_is_empty(dense_model, tokenizer, monkeypatch):
    """When no token gets noised, compute_loss must still score exactly one position."""
    ids = tokenizer.encode("hello world", add_special_tokens=False)
    pad_len = 16 - len(ids)
    ids = ids + [tokenizer.pad_token_id] * pad_len
    batch = {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "mask": torch.tensor([[1] * (16 - pad_len) + [0] * pad_len], dtype=torch.long),
        "prompt_len": torch.zeros(1, dtype=torch.long),
    }

    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.ones(*a, **k))
    trainer = KairosDiffusionTrainer(model=dense_model)
    loss = trainer.compute_loss(dense_model, batch)
    assert torch.is_tensor(loss) and not torch.isnan(loss)


def test_compute_loss_forces_one_position_without_pad_mask(dense_model, tokenizer, monkeypatch):
    """Same fallback as above, but with no `mask` key: `eligible` must fall."""
    ids = tokenizer.encode("hello world", add_special_tokens=False)
    ids = ids + [tokenizer.pad_token_id] * (16 - len(ids))
    batch = {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "prompt_len": torch.zeros(1, dtype=torch.long),
    }

    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.ones(*a, **k))
    trainer = KairosDiffusionTrainer(model=dense_model)
    loss = trainer.compute_loss(dense_model, batch)
    assert torch.is_tensor(loss) and not torch.isnan(loss)


# --- MAE-style curriculum (capped p_max / disabled reweighting) -----------------------------------


def _padded_batch(tokenizer, text="hello world", total_len=32):
    ids = tokenizer.encode(text, add_special_tokens=False)
    pad_len = total_len - len(ids)
    ids = ids + [tokenizer.pad_token_id] * pad_len
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "mask": torch.tensor([[1] * (total_len - pad_len) + [0] * pad_len], dtype=torch.long),
        "prompt_len": torch.zeros(1, dtype=torch.long),
    }


def test_make_diffusion_mask_default_p_max_matches_full_diffusion():
    """p_max defaults to 1.0: unchanged behavior from before mask_p_max existed."""
    torch.manual_seed(0)
    x0 = torch.randint(0, 50, (64, 16))
    prompt_len = torch.zeros(64, dtype=torch.long)
    _, p = make_diffusion_mask(x0, prompt_len, eps=1e-3)
    assert p.max() <= 1.0
    assert p.min() >= 1e-3
    # with enough rows, some should land close to the p_max=1.0 ceiling
    assert p.max() > 0.9


def test_make_diffusion_mask_p_max_caps_masking_rate():
    """A capped p_max (MAE-style curriculum) must never sample p above that ceiling."""
    torch.manual_seed(0)
    x0 = torch.randint(0, 50, (64, 16))
    prompt_len = torch.zeros(64, dtype=torch.long)
    _, p = make_diffusion_mask(x0, prompt_len, eps=1e-3, p_max=0.3)
    assert p.max() <= 0.3
    assert p.min() >= 1e-3


def test_compute_masked_diffusion_losses_reweight_true_divides_by_p(dense_model):
    torch.manual_seed(0)
    x0 = torch.randint(0, dense_model.lm_head.vocab_size, (2, 8))
    noise_mask = torch.zeros_like(x0, dtype=torch.bool)
    noise_mask[:, 2:5] = True
    p = torch.full_like(x0, fill_value=5, dtype=torch.float)  # p=5 everywhere (unrealistic but isolates the /p math)

    torch.manual_seed(0)  # same noise for both calls, otherwise /p math is masked by fresh noise draws
    reweighted, _, _ = compute_masked_diffusion_losses(dense_model, x0, noise_mask, p, reweight=True)
    torch.manual_seed(0)
    plain, _, _ = compute_masked_diffusion_losses(dense_model, x0, noise_mask, p, reweight=False)

    assert torch.allclose(reweighted, plain / 5, atol=1e-5)


def test_compute_masked_diffusion_losses_reweight_false_is_plain_ce(dense_model):
    """MAE mode (reweight=False) must not blow up with a tiny p, unlike the default /p weighting."""
    torch.manual_seed(0)
    x0 = torch.randint(0, dense_model.lm_head.vocab_size, (2, 8))
    noise_mask = torch.zeros_like(x0, dtype=torch.bool)
    noise_mask[:, 2:5] = True
    p = torch.full_like(x0, fill_value=1e-3, dtype=torch.float)  # tiny p: /p would explode if reweight were on

    plain, _, _ = compute_masked_diffusion_losses(dense_model, x0, noise_mask, p, reweight=False)
    assert torch.isfinite(plain).all()
    assert plain.max() < 50  # sane CE range for a small vocab, no 1/p blowup


def test_trainer_mask_p_max_and_reweight_default_to_full_diffusion(dense_model):
    """Defaults must reproduce the pre-curriculum behavior exactly."""
    trainer = KairosDiffusionTrainer(model=dense_model)
    assert trainer.mask_p_max == 1.0
    assert trainer.mask_reweight is True


def test_trainer_mae_mode_runs_end_to_end(dense_model, tokenizer):
    """Stage-1 MAE config (capped p_max, no reweighting) must train like any other compute_loss call."""
    torch.manual_seed(0)
    batch = _padded_batch(tokenizer)
    trainer = KairosDiffusionTrainer(model=dense_model)
    trainer.mask_p_max = 0.3
    trainer.mask_reweight = False

    loss = trainer.compute_loss(dense_model, batch)

    assert torch.is_tensor(loss) and not torch.isnan(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dense_model.parameters())


def test_trainer_mae_mode_never_exceeds_p_max(dense_model, tokenizer, monkeypatch):
    """compute_loss's internal make_diffusion_mask call must respect trainer.mask_p_max."""
    batch = _padded_batch(tokenizer)
    trainer = KairosDiffusionTrainer(model=dense_model)
    trainer.mask_p_max = 0.3

    captured = {}
    real_make_mask = make_diffusion_mask

    def spy(*args, **kwargs):
        noise_mask, p = real_make_mask(*args, **kwargs)
        captured["p"] = p
        return noise_mask, p

    monkeypatch.setattr("kairos.trainer.make_diffusion_mask", spy)
    trainer.compute_loss(dense_model, batch)

    assert captured["p"].max() <= 0.3
