import random

import pytest
import torch

from kairos.dataset import KairosPretrainingDataset, KairosRLDataset, KairosSFTDataset
from kairos.modeling import (
    DiffusionBlock,
    KairosAttnRes,
    KairosCache,
    KairosConfig,
    KairosDiffusionBackbone,
    KairosDiffusionLLM,
    KairosEmbedding,
    KairosMultiCache,
    PyramidalConvCodec,
)
from kairos.tokenizer import KairosTokenizer
from kairos.trainer import KairosDiffusionTrainer


@pytest.fixture
def tokenizer():
    return KairosTokenizer()


@pytest.fixture
def mini_wiki_texts():
    return [
        "Paris is the capital of France.",
        "The Earth orbits the Sun.",
        "Water boils at 100 degrees Celsius.",
    ]


@pytest.fixture
def mini_mcq():
    return [
        {
            "inputs": "What is the capital of France?",
            "multiple_choice_targets": ["Berlin", "Paris", "Madrid"],
            "multiple_choice_scores": [0, 1, 0],
            "reasoning": "France is in Western Europe. Its capital is Paris.",
        },
        {
            "inputs": "What orbits the Sun?",
            "multiple_choice_targets": ["The Moon", "The Earth", "Mars"],
            "multiple_choice_scores": [0, 1, 0],
            "reasoning": "The Earth orbits the Sun.",
        },
    ]


MAX_LEN = 512


@pytest.fixture
def mini_toolace():
    return [
        {
            "system": '[{"name": "get_weather", "description": "Get weather for a city.", "parameters": {"city": {"type": "string"}}}]',
            "conversations": [
                {"from": "user", "value": "What is the weather in Paris?"},
                {"from": "assistant", "value": "[get_weather(city=Paris)]"},
                {"from": "tool", "value": '[{"temperature": 22, "condition": "sunny"}]'},
                {"from": "assistant", "value": "It is 22°C and sunny in Paris."},
            ],
        }
    ]


@pytest.fixture
def mini_alpaca():
    return [
        {
            "instruction": "Translate the following sentence to French.",
            "input": "The sky is blue.",
            "output": "Le ciel est bleu.",
        }
    ]


@pytest.fixture
def config():
    return KairosConfig(d_model=32, n_heads=4, n_layers=2, vocab_size=259, num_modalities=2)


def test_kairos_config(config):
    assert config.hidden_size == 32
    assert config.num_attention_heads == 4


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


def test_backbone_block_size_default_is_one():
    cfg = KairosConfig(d_model=32, n_heads=4, n_layers=4, vocab_size=259, num_modalities=2)
    assert cfg.attnres_block_size == 1


def test_backbone_block_size_one_matches_original_graph():
    # S=1 must reproduce the pre-blocking AttnRes graph term for term.
    torch.manual_seed(0)
    cfg = KairosConfig(d_model=16, n_heads=2, n_layers=4, vocab_size=259, num_modalities=2, attnres_block_size=1)
    torch.manual_seed(42)
    model = KairosDiffusionBackbone(cfg)
    x = torch.randn(2, 6, 16)

    # Reference: the original states=[x]; h=agg(states); x=layer(h); states.append(x) graph.
    states = [x]
    xr = x
    for layer in model.layers:
        h = model.aggregator(states)
        xr = layer(h)
        states.append(xr)
    expected = model.norm(xr)

    out = model(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_backbone_block_size_greater_than_one_shape():
    cfg = KairosConfig(d_model=32, n_heads=4, n_layers=6, vocab_size=259, num_modalities=2, attnres_block_size=3)
    model = KairosDiffusionBackbone(cfg)
    x = torch.randn(2, 8, 32)
    out = model(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_backbone_block_size_changes_output():
    torch.manual_seed(0)
    cfg1 = KairosConfig(d_model=32, n_heads=4, n_layers=6, vocab_size=259, num_modalities=2, attnres_block_size=1)
    cfg3 = KairosConfig(d_model=32, n_heads=4, n_layers=6, vocab_size=259, num_modalities=2, attnres_block_size=3)
    torch.manual_seed(42)
    model1 = KairosDiffusionBackbone(cfg1)
    torch.manual_seed(42)
    model3 = KairosDiffusionBackbone(cfg3)
    x = torch.randn(2, 8, 32)
    assert not torch.allclose(model1(x), model3(x), atol=1e-5)


def test_backbone_block_size_backward():
    cfg = KairosConfig(d_model=16, n_heads=2, n_layers=5, vocab_size=259, num_modalities=2, attnres_block_size=2)
    model = KairosDiffusionBackbone(cfg)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out = model(x)
    out.mean().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_backbone_block_size_uneven_layers_no_nan():
    # n_layers not a multiple of attnres_block_size => trailing partial block.
    cfg = KairosConfig(d_model=16, n_heads=2, n_layers=5, vocab_size=259, num_modalities=2, attnres_block_size=3)
    model = KairosDiffusionBackbone(cfg)
    x = torch.randn(1, 4, 16)
    out = model(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_token_embedding():
    emb = KairosEmbedding(vocab_size=100, num_modalities=7, d_model=32)
    x = torch.randint(0, 100, (2, 16))
    m = torch.zeros_like(x)
    y = emb(token_ids=x, modality_ids=m)
    assert y.shape[0] == 2


def test_codec_roundtrip():
    codec = PyramidalConvCodec(32, stride=3)
    x = torch.randn(2, 16, 32)
    encoded = codec.encode(x)
    decoded = codec.decode(encoded)
    assert decoded.shape == x.shape


def test_kairos_model_init(config):
    model = KairosDiffusionLLM(config)
    assert model is not None


def test_post_init_is_called_and_initializes_all_parameters():
    # regression test for the actual NaN root cause: KairosDiffusionLLM used to skip self.post_init() (DeepseekV3Experts')
    config = KairosConfig(
        d_model=32, n_heads=4, n_layers=2, vocab_size=259, use_moe=True, num_local_experts=2, num_experts_per_tok=1
    )
    model = KairosDiffusionLLM(config)

    non_finite = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert non_finite == [], f"non-finite parameters right after construction: {non_finite}"


def test_moe_expert_weights_are_not_uninitialized_memory():
    # more targeted than the finite-check above: torch.empty() garbage happens to often be xactly 0.0 (freshly-allocated/zeroed pages)
    config = KairosConfig(
        d_model=32, n_heads=4, n_layers=2, vocab_size=259, use_moe=True, num_local_experts=2, num_experts_per_tok=1
    )
    model = KairosDiffusionLLM(config)

    found_experts = False
    for module in model.modules():
        if type(module).__name__ == "DeepseekV3Experts":
            found_experts = True
            assert module.gate_up_proj.data.std().item() > 0, "gate_up_proj looks untouched (all-zero/constant)"
            assert module.down_proj.data.std().item() > 0, "down_proj looks untouched (all-zero/constant)"
    assert found_experts, "test setup didn't actually build a KairosMoE - config wiring changed?"


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
    total_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.abs().sum().item()
    assert total_grad > 0


def test_diffusion_trainer_loss(config):
    B, L, vocab = 2, 8, 50
    model = KairosDiffusionLLM(config, vocab_size=vocab)
    trainer = KairosDiffusionTrainer(model=model)
    inputs = {"input_ids": torch.randint(0, vocab, (B, L)), "prompt_len": torch.zeros(B, dtype=torch.long)}
    loss = trainer.compute_loss(model, inputs)
    assert loss is not None
    assert torch.is_tensor(loss)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss > 0


def test_diffusion_trainer_backward(config):
    B, L, vocab = 2, 8, 50
    model = KairosDiffusionLLM(config, vocab_size=vocab)
    trainer = KairosDiffusionTrainer(model=model)
    inputs = {"input_ids": torch.randint(0, vocab, (B, L)), "prompt_len": torch.zeros(B, dtype=torch.long)}
    loss = trainer.compute_loss(model, inputs)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_diffusion_trainer_applies_noise(config):
    B, L, vocab = 2, 16, 100
    model = KairosDiffusionLLM(config, vocab_size=vocab)
    trainer = KairosDiffusionTrainer(model=model)
    x0 = torch.randint(0, vocab, (B, L))
    inputs = {"input_ids": x0.clone(), "prompt_len": torch.zeros(B, dtype=torch.long)}
    captured = {}

    def forward_hook(module, inp, out):
        captured["logits"] = out.logits

    handle = model.register_forward_hook(forward_hook)
    _ = trainer.compute_loss(model, inputs)
    handle.remove()
    assert "logits" in captured


def test_dataset_with_text(tokenizer, mini_wiki_texts):
    ds = KairosPretrainingDataset(mini_wiki_texts, tokenizer, max_len=32)
    assert len(ds) >= len(mini_wiki_texts)
    sample = ds[0]
    assert "input_ids" in sample
    assert "mask" in sample
    assert "prompt_len" in sample


@pytest.mark.integration
def test_dataset_with_wikitext(tokenizer):
    from datasets import load_dataset

    try:
        ds_raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    except Exception as e:  # noqa: BLE001 — any network/HF-hub failure should skip, not fail the suite
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
    assert s["gen_mask"].shape == (MAX_LEN,)


def test_sft_toolace_prompt_never_noised(tokenizer, mini_toolace):
    ds = KairosSFTDataset(tokenizer, examples=mini_toolace, max_len=MAX_LEN)
    s = ds[0]
    plen = s["prompt_len"].item()
    assert s["gen_mask"][:plen].sum() == 0


def test_sft_toolace_generation_region_exists(tokenizer, mini_toolace):
    ds = KairosSFTDataset(tokenizer, examples=mini_toolace, max_len=MAX_LEN)
    assert ds[0]["gen_mask"].sum() > 0


def test_sft_toolace_last_assistant_is_generation(tokenizer, mini_toolace):
    ds = KairosSFTDataset(tokenizer, examples=mini_toolace, max_len=MAX_LEN)
    s = ds[0]
    plen = s["prompt_len"].item()
    gen_ids = s["input_ids"][plen:]
    decoded = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
    assert "22" in decoded or "sunny" in decoded or "Paris" in decoded


def test_sft_alpaca_length(tokenizer, mini_alpaca):
    ds = KairosSFTDataset(tokenizer, examples=mini_alpaca, max_len=MAX_LEN)
    assert len(ds) == 1


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
    assert s["gen_mask"].shape == (256,)


def test_rldataset_prompt_never_noised(tokenizer, mini_mcq):
    ds = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=64)
    for i in range(len(ds)):
        s = ds[i]
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
    random.seed(0)
    ds1 = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=128)
    random.seed(99)
    ds2 = KairosRLDataset(tokenizer, examples=mini_mcq, max_len=128)
    diffs = sum(not torch.equal(ds1[i]["input_ids"], ds2[i]["input_ids"]) for i in range(len(ds1)))
    assert diffs > 0


def test_diffusion_block_accepts_cache_params(config):
    block = DiffusionBlock(config, layer_idx=0)
    cache = KairosCache(config)
    x = torch.randn(1, 8, 32)
    out = block(x, cache_params=cache)
    assert out.shape == x.shape


def test_diffusion_block_output_differs_with_cache(config):
    block = DiffusionBlock(config, layer_idx=0)
    x_ctx = torch.randn(1, 16, 32)
    x_q = torch.randn(1, 8, 32)
    out_no_cache = block(x_q)
    cache = KairosCache(config)
    _ = block(x_ctx, cache_params=cache)
    out_with_cache = block(x_q, cache_params=cache)
    assert not torch.allclose(out_no_cache, out_with_cache, atol=1e-4)


def test_diffusion_block_cache_not_mutated(config):
    block = DiffusionBlock(config, layer_idx=0)
    x_ctx = torch.randn(1, 16, 32)
    cache = KairosCache(config)
    _ = block(x_ctx, cache_params=cache)
    ref = cache.clone()
    x_m = torch.randn(1, 8, 32)
    _ = block(x_m, cache_params=cache.clone())
    for c1, c2 in zip(cache.ssm_caches, ref.ssm_caches):
        if c1 is not None:
            assert torch.allclose(c1, c2)
    for idx in cache._key_cache:
        k1 = cache._key_cache[idx]
        k2 = ref._key_cache[idx]
        if k1 is not None:
            assert torch.allclose(k1, k2)


def test_diffusion_block_cache_determinism(config):
    block = DiffusionBlock(config, layer_idx=0)
    x_ctx = torch.randn(1, 16, 32)
    cache = KairosCache(config)
    _ = block(x_ctx, cache_params=cache)
    x_m = torch.randn(1, 8, 32)
    out1 = block(x_m, cache_params=cache.clone())
    out2 = block(x_m, cache_params=cache.clone())
    assert torch.allclose(out1, out2, atol=1e-5)


def test_diffusion_block_no_cache_backward(config):
    block = DiffusionBlock(config, layer_idx=0)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = block(x)
    out.mean().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_diffusion_block_with_cache_backward(config):
    block = DiffusionBlock(config, layer_idx=0)
    cache = KairosCache(config)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = block(x, cache_params=cache)
    out.mean().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_backbone_propagates_cache(config):
    backbone = KairosDiffusionBackbone(config)
    cache = KairosCache(config)
    x = torch.randn(1, 8, 32)
    out = backbone(x, cache_params=cache)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_backbone_cache_conditions_output(config):
    backbone = KairosDiffusionBackbone(config)
    x_ctx1 = torch.randn(1, 16, 32)
    x_ctx2 = torch.randn(1, 16, 32)
    x_q = torch.randn(1, 8, 32)
    cache1 = KairosCache(config)
    cache2 = KairosCache(config)
    _ = backbone(x_ctx1, cache_params=cache1)
    _ = backbone(x_ctx2, cache_params=cache2)
    out1 = backbone(x_q, cache_params=cache1.clone())
    out2 = backbone(x_q, cache_params=cache2.clone())
    assert not torch.allclose(out1, out2, atol=1e-4)


def test_model_forward_with_cache(config):
    model = KairosDiffusionLLM(config)
    cache = KairosMultiCache(config)
    x = torch.randint(0, 259, (1, 16))
    out = model(input_ids=x, cache_params=cache)
    assert out.logits is not None
    assert not torch.isnan(out.logits).any()


def test_model_diffusion_stability_with_cache(config):
    model = KairosDiffusionLLM(config)
    x_ctx = torch.randint(0, 259, (1, 16))
    x_m = torch.randint(0, 259, (1, 8))
    cache = KairosMultiCache(config)
    _ = model(input_ids=x_ctx, cache_params=cache)
    outs = [model(input_ids=x_m, cache_params=cache.clone()).logits for _ in range(5)]
    for o in outs[1:]:
        assert torch.allclose(outs[0], o, atol=1e-5)
