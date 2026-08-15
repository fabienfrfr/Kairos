import os

import pytest
import torch
from transformers.models.diffusion_gemma.generation_diffusion_gemma import DiffusionGemmaGenerationConfig

os.environ.setdefault("KAIROS_ATTN_BACKEND", "eager")

from kairos.modeling import KairosConfig, KairosDiffusionFM
from kairos.pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig
from kairos.tokenizer import KairosTokenizer, Modality


def make_config(canvas_length=32, use_memory_gate=False):
    return KairosConfig(
        d_model=64,
        n_heads=4,
        n_layers=4,
        stride=3,
        vocab_size=291,
        num_modalities=8,
        num_scales=4,
        modality_scales={m: [0] for m in range(8)},
        intermediate_size=544,
        moe_intermediate_size=544,
        num_local_experts=7,
        num_experts_per_tok=1,
        n_shared_experts=1,
        use_moe=True,
        attnres_block_size=4,
        use_memory_gate=use_memory_gate,
        canvas_length=canvas_length,
    )


@pytest.fixture
def tokenizer():
    return KairosTokenizer()


@pytest.fixture
def model(tokenizer):
    torch.manual_seed(0)
    return KairosDiffusionFM(make_config(), vocab_size=len(tokenizer))


@pytest.fixture
def text_prompt(tokenizer):
    return torch.tensor([tokenizer.encode("Paris is the capital", add_special_tokens=False)], dtype=torch.long)


@pytest.fixture
def text_modality(text_prompt):
    return torch.full_like(text_prompt, int(Modality.TEXT))


# ---------------------------------------------------------------- model.generate
def test_generate_output_shape(model, text_prompt, text_modality):
    out = model.generate(
        input_ids=text_prompt,
        modality_ids=text_modality,
        max_new_tokens=8,
        max_denoising_steps=3,
        entropy_bound=0.5,
    )
    seqs = out.sequences
    assert seqs.shape == (1, text_prompt.shape[1] + 8)
    assert seqs.dtype == torch.long


def test_generate_returns_generation_output_by_default(model, text_prompt, text_modality):
    out = model.generate(
        input_ids=text_prompt, modality_ids=text_modality, max_new_tokens=8, max_denoising_steps=3, entropy_bound=0.5
    )
    assert hasattr(out, "sequences")


def test_generate_return_dict_false_returns_raw_tensor(model, text_prompt, text_modality):
    raw = model.generate(
        input_ids=text_prompt,
        modality_ids=text_modality,
        max_new_tokens=8,
        max_denoising_steps=3,
        entropy_bound=0.5,
        return_dict_in_generate=False,
    )
    assert isinstance(raw, torch.Tensor)
    assert raw.shape == (1, text_prompt.shape[1] + 8)


def test_generate_defaults_modality_ids_to_text(model, text_prompt):
    out = model.generate(input_ids=text_prompt, max_new_tokens=8, max_denoising_steps=3, entropy_bound=0.5)
    assert out.sequences.shape == (1, text_prompt.shape[1] + 8)


def test_generate_batch_padded_prompts(model, tokenizer, text_modality):
    p1 = tokenizer.encode("Paris is the capital", add_special_tokens=False)
    p2 = tokenizer.encode("The Earth orbits", add_special_tokens=False)
    max_pl = max(len(p1), len(p2))
    batch = torch.cat(
        [
            torch.cat(
                [
                    torch.tensor([p1], dtype=torch.long),
                    torch.full((1, max_pl - len(p1)), tokenizer.pad_token_id, dtype=torch.long),
                ],
                -1,
            ),
            torch.cat(
                [
                    torch.tensor([p2], dtype=torch.long),
                    torch.full((1, max_pl - len(p2)), tokenizer.pad_token_id, dtype=torch.long),
                ],
                -1,
            ),
        ]
    )
    batch_mod = torch.full_like(batch, int(Modality.TEXT))
    out = model.generate(
        input_ids=batch, modality_ids=batch_mod, max_new_tokens=8, max_denoising_steps=3, entropy_bound=0.5
    )
    assert out.sequences.shape == (2, max_pl + 8)


def test_generate_multi_canvas_exceeds_canvas_length(model, text_prompt, text_modality):
    assert make_config().canvas_length == 32
    out = model.generate(
        input_ids=text_prompt,
        modality_ids=text_modality,
        max_new_tokens=40,
        max_denoising_steps=3,
        entropy_bound=0.5,
    )
    assert out.sequences.shape == (1, text_prompt.shape[1] + 40)


def test_generate_non_multiple_canvas_truncates_to_max_new_tokens(model, text_prompt, text_modality):
    out = model.generate(
        input_ids=text_prompt,
        modality_ids=text_modality,
        max_new_tokens=37,
        max_denoising_steps=3,
        entropy_bound=0.5,
    )
    assert out.sequences.shape == (1, text_prompt.shape[1] + 37)


def test_generate_empty_prompt(model):
    out = model.generate(
        input_ids=torch.zeros(1, 0, dtype=torch.long),
        modality_ids=torch.zeros(1, 0, dtype=torch.long),
        max_new_tokens=8,
        max_denoising_steps=3,
        entropy_bound=0.5,
    )
    assert out.sequences.shape == (1, 8)


def test_generate_without_temperature_schedule_or_stopping(model, text_prompt, text_modality):
    out = model.generate(
        input_ids=text_prompt,
        modality_ids=text_modality,
        max_new_tokens=8,
        max_denoising_steps=3,
        entropy_bound=0.5,
        t_min=None,
        t_max=None,
        stability_threshold=None,
        confidence_threshold=None,
    )
    assert out.sequences.shape == (1, text_prompt.shape[1] + 8)


def test_generate_accepts_generation_config_object(model, text_prompt, text_modality):
    gen_cfg = DiffusionGemmaGenerationConfig()
    gen_cfg.max_new_tokens = 8
    out = model.generate(
        input_ids=text_prompt,
        modality_ids=text_modality,
        generation_config=gen_cfg,
        max_denoising_steps=3,
        entropy_bound=0.5,
    )
    assert out.sequences.shape == (1, text_prompt.shape[1] + 8)


def test_generate_deterministic_with_seed(model, text_prompt, text_modality):
    torch.manual_seed(0)
    first = model.generate(
        input_ids=text_prompt, modality_ids=text_modality, max_new_tokens=8, max_denoising_steps=3, entropy_bound=0.5
    ).sequences
    torch.manual_seed(0)
    second = model.generate(
        input_ids=text_prompt, modality_ids=text_modality, max_new_tokens=8, max_denoising_steps=3, entropy_bound=0.5
    ).sequences
    assert torch.equal(first, second)


def test_generate_preserves_prompt_prefix(model, text_prompt, text_modality):
    out = model.generate(
        input_ids=text_prompt, modality_ids=text_modality, max_new_tokens=8, max_denoising_steps=3, entropy_bound=0.5
    ).sequences
    assert torch.equal(out[:, : text_prompt.shape[1]], text_prompt)


def test_generate_produces_valid_token_ids(model, text_prompt, text_modality):
    out = model.generate(
        input_ids=text_prompt, modality_ids=text_modality, max_new_tokens=16, max_denoising_steps=3, entropy_bound=0.5
    ).sequences
    assert out.ge(0).all()
    assert out.lt(model.config.vocab_size).all()


def test_generate_respects_configured_canvas_length(tokenizer, text_prompt):
    torch.manual_seed(0)
    big_canvas = KairosDiffusionFM(make_config(canvas_length=64), vocab_size=len(tokenizer))
    text_modality = torch.full_like(text_prompt, int(Modality.TEXT))
    out = big_canvas.generate(
        input_ids=text_prompt,
        modality_ids=text_modality,
        max_new_tokens=16,
        max_denoising_steps=3,
        entropy_bound=0.5,
    )
    assert out.sequences.shape == (1, text_prompt.shape[1] + 16)


# --------------------------------------------------------------- pipeline.generate
@pytest.fixture
def built_pipe(tmp_path, tokenizer):
    cfg = make_config()
    texts = [
        {"modality": "text", "text": "Paris is the capital of France and it is a beautiful city."},
        {"modality": "text", "text": "The Earth orbits the Sun once every year."},
    ]
    dc = DataConfig(text_examples=[], multimodal_examples=texts, max_len=256, stride=3, batch_size=2, pack=True)
    tc = TrainConfig(epochs=1, run_dir=str(tmp_path / "run"))
    pipe = KairosMultimodalPipeline(cfg, dc, tc, tokenizer=tokenizer)
    pipe.build()
    return pipe


@pytest.fixture
def prompt_ids(tokenizer):
    return tokenizer.encode("Paris is the capital", add_special_tokens=False)


def test_pipeline_generate_returns_prompt_plus_generated(built_pipe, prompt_ids):
    full = built_pipe.generate(prompt_ids, max_new_tokens=16, max_denoising_steps=3, entropy_bound=0.5, seed=42)
    assert isinstance(full, list)
    assert all(isinstance(i, int) for i in full)
    assert len(full) == len(prompt_ids) + 16
    assert full[: len(prompt_ids)] == prompt_ids


def test_pipeline_generate_deterministic_with_seed(built_pipe, prompt_ids):
    first = built_pipe.generate(prompt_ids, max_new_tokens=16, max_denoising_steps=3, entropy_bound=0.5, seed=42)
    second = built_pipe.generate(prompt_ids, max_new_tokens=16, max_denoising_steps=3, entropy_bound=0.5, seed=42)
    assert first == second


def test_pipeline_generate_accepts_modality(built_pipe, prompt_ids):
    full = built_pipe.generate(
        prompt_ids, max_new_tokens=8, max_denoising_steps=3, entropy_bound=0.5, modality=Modality.TEXT, seed=0
    )
    assert len(full) == len(prompt_ids) + 8


def test_pipeline_generate_restores_training_mode(built_pipe, prompt_ids):
    assert built_pipe.model.training is True
    built_pipe.generate(prompt_ids, max_new_tokens=8, max_denoising_steps=3, entropy_bound=0.5, seed=0)
    assert built_pipe.model.training is True


def test_pipeline_generate_before_build_raises(tokenizer):
    cfg = make_config()
    dc = DataConfig(
        text_examples=[],
        multimodal_examples=[{"modality": "text", "text": "The Earth orbits the Sun once every year."}],
        max_len=256,
        batch_size=1,
    )
    tc = TrainConfig(epochs=1, run_dir="unused")
    pipe = KairosMultimodalPipeline(cfg, dc, tc, tokenizer=tokenizer)
    with pytest.raises(RuntimeError):
        pipe.generate([1, 2, 3])
