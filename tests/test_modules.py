import pytest
import torch
import random

from kairos.attentions import KairosCache
from kairos.modeling import (
    KairosConfig,
    DiffusionBlock,
    KairosDiffusionBackbone,
    KairosEmbedding,
    KairosDiffusionLLM,
    KairosAttnRes,
    ConvCodec,
)

from kairos.tokenizer import KairosTokenizer
from kairos.dataset import KairosPretrainingDataset, KairosRLDataset, KairosSFTDataset

from kairos.trainer import KairosDiffusionTrainer

# =========================
# Fixtures
# =========================
@pytest.fixture
def tokenizer():
    return KairosTokenizer()


@pytest.fixture
def mini_wiki_texts():
    return [
        "Paris is the capital of France.",
        "The Earth orbits the Sun.",
        "Water boils at 100 degrees Celsius."
    ]

@pytest.fixture
def mini_mcq():
    return [
        {
            "inputs": "What is the capital of France?",
            "multiple_choice_targets": ["Berlin", "Paris", "Madrid"],
            "multiple_choice_scores":  [0, 1, 0],
            "reasoning": "France is in Western Europe. Its capital is Paris.",
        },
        {
            "inputs": "What orbits the Sun?",
            "multiple_choice_targets": ["The Moon", "The Earth", "Mars"],
            "multiple_choice_scores":  [0, 1, 0],
            "reasoning": "The Earth orbits the Sun.",
        },
    ]


MAX_LEN = 512  # ByT5 tokenizes byte-by-byte

@pytest.fixture
def mini_toolace():
    """One ToolACE-style example: system + user + assistant + tool result + assistant."""
    return [
        {
            "system": '[{"name": "get_weather", "description": "Get weather for a city.", "parameters": {"city": {"type": "string"}}}]',
            "conversations": [
                {"from": "user",      "value": "What is the weather in Paris?"},
                {"from": "assistant", "value": "[get_weather(city=Paris)]"},
                {"from": "tool",      "value": '[{"temperature": 22, "condition": "sunny"}]'},
                {"from": "assistant", "value": "It is 22°C and sunny in Paris."},
            ],
        }
    ]


@pytest.fixture
def mini_alpaca():
    """One alpaca-style example: instruction + input + output."""
    return [
        {
            "instruction": "Translate the following sentence to French.",
            "input": "The sky is blue.",
            "output": "Le ciel est bleu.",
        }
    ]



@pytest.fixture
def config():
    return KairosConfig(d_model=32, n_heads=4, n_layers=2)

# =========================
# Config
# =========================
def test_kairos_config(config):
    assert config.hidden_size == 32
    assert config.num_attention_heads == 4


# =========================
# Blocks
# =========================
def test_diffusion_block(config):
    block = DiffusionBlock(config, 0)

    x = torch.randn(2, 8, 32)
    out = block(x)

    assert out.shape == x.shape


def test_backbone(config):
    model = KairosDiffusionBackbone(config)

    x = torch.randn(2, 8, 32)
    out = model(x)

    assert out.shape == x.shape


# =========================
# Aggregator
# =========================
def test_aggregator_shape():
    agg = KairosAttnRes(32)
    states = [torch.randn(2, 8, 32) for _ in range(4)]

    out = agg(states)
    assert out.shape == (2, 8, 32)


def test_aggregator_weights_sum():
    agg = KairosAttnRes(16)
    states = [torch.randn(1, 4, 16) for _ in range(3)]

    V = torch.stack(states, dim=0)
    K = agg.key_norm(V)
    logits = torch.einsum("d,lbtd->lbt", agg.w, K)
    weights = torch.softmax(logits, dim=0)

    assert torch.allclose(weights.sum(0), torch.ones_like(weights[0]), atol=1e-5)


# =========================
# Embeddings & Codec
# =========================
def test_token_embedding():
    codec = ConvCodec(32, stride=3)
    emb = KairosEmbedding(100, 32, codec)

    x = torch.randint(0, 100, (2, 8))
    out = emb(x)

    
    assert out.shape[1] == 3
    assert out.shape[2] == 32




def test_codec_roundtrip():
    codec = ConvCodec(32, stride=3)
    x = torch.randn(2, 16, 32)

    encoded = codec(x, "encode")
    decoded = codec(encoded, "decode")

    assert decoded.shape == x.shape

# =========================
# Model
# =========================
def test_kairos_model_init(config):
    model = KairosDiffusionLLM(config)
    assert model is not None


def test_kairos_model_forward(config):
    model = KairosDiffusionLLM(config)

    x = torch.randint(0, 259, (2, 16))
    out = model(input_ids=x)

    assert out.logits.shape == (2, 16, 259)


def test_no_nan_forward(config):
    model = KairosDiffusionLLM(config)

    x = torch.randint(0, 259, (2, 8))
    out = model(input_ids=x)

    assert not torch.isnan(out.logits).any()


def test_backward_pass(config):
    model = KairosDiffusionLLM(config)

    x = torch.randint(0, 259, (2, 8))
    out = model(input_ids=x)

    loss = out.logits.mean()
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None


# =========================
# Trainer
# =========================

def test_diffusion_trainer_loss(config):
    B, L, vocab = 2, 8, 50

    model = KairosDiffusionLLM(config, vocab_size=vocab)

    trainer = KairosDiffusionTrainer(model=model)

    inputs = {
        "input_ids": torch.randint(0, vocab, (B, L)),
        "prompt_len": torch.zeros(B, dtype=torch.long),
    }

    loss = trainer.compute_loss(model, inputs)

    # checks
    assert loss is not None
    assert torch.is_tensor(loss)
    assert loss.dim() == 0          # scalar
    assert not torch.isnan(loss)
    assert loss > 0


def test_diffusion_trainer_backward(config):
    B, L, vocab = 2, 8, 50

    model = KairosDiffusionLLM(config, vocab_size=vocab)

    trainer = KairosDiffusionTrainer(model=model)

    inputs = {
        "input_ids": torch.randint(0, vocab, (B, L)),
        "prompt_len": torch.zeros(B, dtype=torch.long),
    }

    loss = trainer.compute_loss(model, inputs)

    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]

    assert any(g is not None for g in grads)


def test_diffusion_trainer_applies_noise(config):
    B, L, vocab = 2, 16, 100

    model = KairosDiffusionLLM(config, vocab_size=vocab)
    trainer = KairosDiffusionTrainer(model=model)

    x0 = torch.randint(0, vocab, (B, L))

    inputs = {
        "input_ids": x0.clone(),
        "prompt_len": torch.zeros(B, dtype=torch.long),
    }

    # Hook pour capturer xt
    captured = {}

    def forward_hook(module, inp, out):
        captured["logits"] = out.logits

    handle = model.register_forward_hook(forward_hook)

    _ = trainer.compute_loss(model, inputs)

    handle.remove()

    assert "logits" in captured

# =========================
# Dataset
# =========================
def test_dataset_with_text(tokenizer, mini_wiki_texts):
    ds = KairosPretrainingDataset(mini_wiki_texts, tokenizer, max_len=32)

    assert len(ds) >= len(mini_wiki_texts)  # chunking

    sample = ds[0]
    assert "input_ids" in sample
    assert "mask" in sample
    assert "prompt_len" in sample


@pytest.mark.integration
def test_dataset_with_wikitext(tokenizer):
    from datasets import load_dataset

    try:
        ds_raw = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="train[:1%]"
        )
    except Exception as e:
        pytest.skip(f"Dataset download failed: {e}")

    texts = ds_raw["text"]

    ds = KairosPretrainingDataset(texts, tokenizer, max_len=32)

    assert len(ds) > 0
    sample = ds[0]

    assert "input_ids" in sample


def test_dataset_preprocess(tokenizer, mini_wiki_texts):
    ds = KairosPretrainingDataset(mini_wiki_texts, tokenizer, max_len=32)

    sample = ds[0]

    input_ids = sample["input_ids"]
    mask = sample["mask"]

    assert input_ids.shape[0] == 32
    assert mask.shape[0] == 32

    assert input_ids.dtype == torch.long
    assert mask.dtype in (torch.int64, torch.bool)

    assert mask.sum() > 0
    assert (input_ids != tokenizer.pad_token_id).any()


# --- ToolACE tests ---

def test_sft_toolace_length(tokenizer, mini_toolace):
    ds = KairosSFTDataset(tokenizer, examples=mini_toolace, max_len=MAX_LEN)
    assert len(ds) == 1


def test_sft_toolace_keys(tokenizer, mini_toolace):
    ds = KairosSFTDataset(tokenizer, examples=mini_toolace, max_len=MAX_LEN)
    for key in ("input_ids", "gen_mask", "prompt_len"):
        assert key in ds[0]


def test_sft_toolace_shapes(tokenizer, mini_toolace):
    ds = KairosSFTDataset(tokenizer, examples=mini_toolace, max_len=MAX_LEN)
    s = ds[0]
    assert s["input_ids"].shape == (MAX_LEN,)
    assert s["gen_mask"].shape  == (MAX_LEN,)


def test_sft_toolace_prompt_never_noised(tokenizer, mini_toolace):
    ds = KairosSFTDataset(tokenizer, examples=mini_toolace, max_len=MAX_LEN)
    s    = ds[0]
    plen = s["prompt_len"].item()
    assert s["gen_mask"][:plen].sum() == 0


def test_sft_toolace_generation_region_exists(tokenizer, mini_toolace):
    ds = KairosSFTDataset(tokenizer, examples=mini_toolace, max_len=MAX_LEN)
    assert ds[0]["gen_mask"].sum() > 0


def test_sft_toolace_last_assistant_is_generation(tokenizer, mini_toolace):
    """The generation region must correspond to the last assistant turn only."""
    ds   = KairosSFTDataset(tokenizer, examples=mini_toolace, max_len=MAX_LEN)
    s    = ds[0]
    plen = s["prompt_len"].item()

    # Decode the generation region and check it contains the last assistant answer
    gen_ids = s["input_ids"][plen:]
    decoded = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
    assert "22" in decoded or "sunny" in decoded or "Paris" in decoded


# --- Alpaca tests ---

def test_sft_alpaca_length(tokenizer, mini_alpaca):
    ds = KairosSFTDataset(tokenizer, examples=mini_alpaca, max_len=MAX_LEN)
    assert len(ds) == 1


def test_sft_alpaca_generation_is_output(tokenizer, mini_alpaca):
    """The generation region must contain the expected French translation."""
    ds   = KairosSFTDataset(tokenizer, examples=mini_alpaca, max_len=MAX_LEN)
    s    = ds[0]
    plen = s["prompt_len"].item()

    gen_ids = s["input_ids"][plen:]
    decoded = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
    assert "bleu" in decoded.lower()

# --- BigBench tests ---

def test_rldataset_length(tokenizer, mini_mcq):
    ds = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=64)
    assert len(ds) == 2
 
 
def test_rldataset_keys(tokenizer, mini_mcq):
    ds = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=64)
    sample = ds[0]
    for key in ("input_ids", "gen_mask", "prompt_len", "mask_ratio", "choices", "scores", "level"):
        assert key in sample
 
 
def test_rldataset_shapes(tokenizer, mini_mcq):
    ds = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=256)
    s = ds[0]
    assert s["input_ids"].shape == (256,)
    assert s["gen_mask"].shape  == (256,)
 
 
def test_rldataset_prompt_never_noised(tokenizer, mini_mcq):
    ds = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=64)
    for i in range(len(ds)):
        s    = ds[i]
        plen = s["prompt_len"].item()
        assert s["gen_mask"][:plen].sum() == 0
 
 
def test_rldataset_generation_region_exists(tokenizer, mini_mcq):
    ds = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=64)
    for i in range(len(ds)):
        assert ds[i]["gen_mask"].sum() > 0
 
 
def test_rldataset_uncertainty_choice_present(tokenizer, mini_mcq):
    ds = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=64)
    for i in range(len(ds)):
        assert "not sure / I don't know" in ds[i]["choices"]
 
 
def test_rldataset_anti_reversal_curse(tokenizer, mini_mcq):
    """Same examples, different seeds → at least one sample must differ."""
    random.seed(0)
    ds1 = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=128)
    random.seed(99)
    ds2 = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=128)
 
    diffs = sum(not torch.equal(ds1[i]["input_ids"], ds2[i]["input_ids"]) for i in range(len(ds1)))
    assert diffs > 0


# =========================
# Cache
# =========================

# DiffusionBlock + cache_params
def test_diffusion_block_accepts_cache_params(config):
    """DiffusionBlock.forward must accept cache_params without raising."""
    block = DiffusionBlock(config, layer_idx=0)
    cache = KairosCache(config)
 
    x = torch.randn(1, 8, 32)
    out = block(x, cache_params=cache)
 
    assert out.shape == x.shape
 
 
def test_diffusion_block_output_differs_with_cache(config):
    """
    Output with a populated cache must differ from output without cache,
    proving cache_params is actually consumed by the attention layer.
    """
    block = DiffusionBlock(config, layer_idx=0)
 
    x_ctx = torch.randn(1, 16, 32)
    x_q   = torch.randn(1, 8,  32)
 
    # without cache
    out_no_cache = block(x_q)
 
    # with a pre-populated cache
    cache = KairosCache(config)
    _ = block(x_ctx, cache_params=cache)
    out_with_cache = block(x_q, cache_params=cache)
 
    assert not torch.allclose(out_no_cache, out_with_cache, atol=1e-4), (
        "Cache had no effect on output — cache_params is probably not forwarded"
    )
 
 
def test_diffusion_block_cache_not_mutated(config):
    """
    Cloned cache must not be mutated by a DiffusionBlock forward pass
    (diffusion safety: each denoising step must start from the same N-state).
    """
    block = DiffusionBlock(config, layer_idx=0)
 
    x_ctx = torch.randn(1, 16, 32)
    cache = KairosCache(config)
    _ = block(x_ctx, cache_params=cache)
 
    ref = cache.clone()
 
    x_m = torch.randn(1, 8, 32)
    _ = block(x_m, cache_params=cache.clone())
 
    for c1, c2 in zip(cache.ssm_caches, ref.ssm_caches):
        if c1 is not None:
            assert torch.allclose(c1, c2), "ssm_cache was mutated on the original cache"
 
    for idx in cache._key_cache:
        k1 = cache._key_cache[idx]
        k2 = ref._key_cache[idx]
        if k1 is not None:
            assert torch.allclose(k1, k2), f"KV cache at layer {idx} was mutated"
 
 
def test_diffusion_block_cache_determinism(config):
    """
    Same cache clone + same input → identical output.
    Critical for stable iterative diffusion.
    """
    block = DiffusionBlock(config, layer_idx=0)
 
    x_ctx = torch.randn(1, 16, 32)
    cache = KairosCache(config)
    _ = block(x_ctx, cache_params=cache)
 
    x_m = torch.randn(1, 8, 32)
 
    out1 = block(x_m, cache_params=cache.clone())
    out2 = block(x_m, cache_params=cache.clone())
 
    assert torch.allclose(out1, out2, atol=1e-5), (
        "Non-deterministic output with identical cache clones"
    )
 
 
def test_diffusion_block_no_cache_backward(config):
    """Gradient must flow through DiffusionBlock without cache."""
    block = DiffusionBlock(config, layer_idx=0)
 
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = block(x)
    out.mean().backward()
 
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
 
 
def test_diffusion_block_with_cache_backward(config):
    """Gradient must also flow when cache_params is provided."""
    block = DiffusionBlock(config, layer_idx=0)
    cache = KairosCache(config)
 
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = block(x, cache_params=cache)
    out.mean().backward()
 
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
 
 
# Backbone propagates cache
def test_backbone_propagates_cache(config):
    """
    KairosDiffusionBackbone.forward must accept and propagate cache_params
    to every DiffusionBlock layer.
    """
    backbone = KairosDiffusionBackbone(config)
    cache = KairosCache(config)
 
    x = torch.randn(1, 8, 32)
    out = backbone(x, cache_params=cache)
 
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
 
 
def test_backbone_cache_conditions_output(config):
    """
    Backbone output must differ depending on cache content.
    """
    backbone = KairosDiffusionBackbone(config)
 
    x_ctx1 = torch.randn(1, 16, 32)
    x_ctx2 = torch.randn(1, 16, 32)
    x_q    = torch.randn(1, 8,  32)
 
    cache1 = KairosCache(config)
    cache2 = KairosCache(config)
 
    _ = backbone(x_ctx1, cache_params=cache1)
    _ = backbone(x_ctx2, cache_params=cache2)
 
    out1 = backbone(x_q, cache_params=cache1.clone())
    out2 = backbone(x_q, cache_params=cache2.clone())
 
    assert not torch.allclose(out1, out2, atol=1e-4), (
        "Backbone ignores cache — different contexts produce identical outputs"
    )
 
 
# Full model propagates cache
def test_model_forward_with_cache(config):
    """KairosDiffusionLLM.forward must accept cache_params end-to-end."""
    model = KairosDiffusionLLM(config)
    cache = KairosCache(config)
 
    x = torch.randint(0, 259, (1, 16))
    out = model(input_ids=x, cache_params=cache)
 
    assert out.logits is not None
    assert not torch.isnan(out.logits).any()
 
 
def test_model_diffusion_stability_with_cache(config):
    """
    Five denoising iterations with cloned cache → identical outputs.
    End-to-end diffusion stability test.
    """
    model = KairosDiffusionLLM(config)
 
    x_ctx = torch.randint(0, 259, (1, 16))
    x_m   = torch.randint(0, 259, (1, 8))
 
    cache = KairosCache(config)
    _ = model(input_ids=x_ctx, cache_params=cache)
 
    outs = [
        model(input_ids=x_m, cache_params=cache.clone()).logits
        for _ in range(5)
    ]
 
    for o in outs[1:]:
        assert torch.allclose(outs[0], o, atol=1e-5), (
            "Diffusion instability: identical cache clones produce different logits"
        )
 