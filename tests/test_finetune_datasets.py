import pytest
import torch

from kairos.dataset import KairosDPODataset, KairosRLDataset, KairosSFTDataset
from kairos.tokenizer import KairosTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return KairosTokenizer()


# =========================
# SFT
# =========================


@pytest.fixture
def sft_examples():
    return [
        {
            "system": "You are helpful.",
            "conversations": [
                {"from": "user", "value": "Hi"},
                {"from": "assistant", "value": "Hello!"},
            ],
        },
        {
            "conversations": [
                {"from": "human", "value": "What's 2+2?"},
                {"from": "gpt", "value": "4"},
                {"from": "tool", "value": "calc(2+2)=4"},
            ],
        },
    ]


def test_sft_schema(tokenizer, sft_examples):
    ds = KairosSFTDataset(tokenizer, max_len=64, examples=sft_examples)
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "gen_mask", "prompt_len"}
    assert item["input_ids"].shape == (64,)
    assert item["gen_mask"].shape == (64,)
    assert len(ds) == 2


def test_sft_only_last_assistant_turn_is_generated(tokenizer, sft_examples):
    ds = KairosSFTDataset(tokenizer, max_len=64, examples=sft_examples)
    item = ds[0]
    prompt_len = item["prompt_len"].item()
    gen_mask = item["gen_mask"]
    assert gen_mask[:prompt_len].sum() == 0
    assert gen_mask[prompt_len:].sum() > 0


def test_sft_padding_is_masked_out(tokenizer, sft_examples):
    ds = KairosSFTDataset(tokenizer, max_len=64, examples=sft_examples)
    item = ds[0]
    total_real = item["prompt_len"].item() + item["gen_mask"].sum().item()
    assert total_real <= 64
    assert item["gen_mask"][int(total_real) :].sum() == 0


def test_sft_truncates_to_max_len(tokenizer):
    long_value = "word " * 500
    ex = {"conversations": [{"from": "user", "value": long_value}, {"from": "assistant", "value": long_value}]}
    ds = KairosSFTDataset(tokenizer, max_len=32, examples=[ex])
    item = ds[0]
    assert item["input_ids"].shape == (32,)


def test_sft_alpaca_style_example(tokenizer):
    # _process_alpaca is exercised directly to avoid a network dataset download
    raw = KairosSFTDataset(tokenizer, max_len=64, examples=[{"conversations": []}])
    processed = raw._process_alpaca({"instruction": "Say hi", "input": "", "output": "Hi there"})
    assert processed["prompt_len"] > 0
    assert sum(processed["gen_mask"]) > 0


def test_sft_alpaca_style_example_with_input(tokenizer):
    raw = KairosSFTDataset(tokenizer, max_len=64, examples=[{"conversations": []}])
    processed = raw._process_alpaca(
        {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
    )
    assert processed["prompt_len"] > 0
    assert sum(processed["gen_mask"]) > 0


def test_sft_unknown_source_raises(tokenizer):
    with pytest.raises(ValueError, match="Unknown source"):
        KairosSFTDataset(tokenizer, max_len=64, examples=None, source="bogus")


def test_sft_toolace_dispatch_uses_load_dataset(tokenizer, monkeypatch):
    calls = {}

    def fake_load_dataset(name, split):
        calls["name"], calls["split"] = name, split
        return [{"conversations": [{"from": "user", "value": "hi"}, {"from": "assistant", "value": "yo"}]}]

    monkeypatch.setattr("kairos.dataset.load_dataset", fake_load_dataset)
    ds = KairosSFTDataset(tokenizer, max_len=32, examples=None, source="toolace")
    assert calls["name"] == "Team-ACE/ToolACE"
    assert len(ds) == 1


def test_sft_alpaca_dispatch_uses_load_dataset(tokenizer, monkeypatch):
    def fake_load_dataset(name, split):
        return [{"instruction": "Say hi", "input": "", "output": "Hi!"}]

    monkeypatch.setattr("kairos.dataset.load_dataset", fake_load_dataset)
    ds = KairosSFTDataset(tokenizer, max_len=32, examples=None, source="alpaca")
    assert len(ds) == 1


# =========================
# DPO
# =========================


@pytest.fixture
def dpo_examples():
    return [
        {
            "prompt": "Tell me a joke.",
            "chosen": [{"role": "assistant", "content": "Why did the chicken cross the road?"}],
            "rejected": [{"role": "assistant", "content": "I don't know jokes."}],
        }
    ]


def test_dpo_schema(tokenizer, dpo_examples):
    ds = KairosDPODataset(tokenizer, max_len=64, examples=dpo_examples)
    item = ds[0]
    assert set(item.keys()) == {"chosen_ids", "chosen_mask", "rejected_ids", "rejected_mask", "prompt_len"}
    assert item["chosen_ids"].shape == (64,)
    assert item["rejected_ids"].shape == (64,)
    assert len(ds) == 1


def test_dpo_chosen_and_rejected_share_the_same_prompt(tokenizer, dpo_examples):
    ds = KairosDPODataset(tokenizer, max_len=64, examples=dpo_examples)
    item = ds[0]
    plen = item["prompt_len"].item()
    assert torch.equal(item["chosen_ids"][:plen], item["rejected_ids"][:plen])


def test_dpo_response_is_masked_as_generation(tokenizer, dpo_examples):
    ds = KairosDPODataset(tokenizer, max_len=64, examples=dpo_examples)
    item = ds[0]
    plen = item["prompt_len"].item()
    assert item["chosen_mask"][:plen].sum() == 0
    assert item["chosen_mask"][plen:].sum() > 0
    assert item["rejected_mask"][plen:].sum() > 0


# =========================
# RL
# =========================


@pytest.fixture
def rl_examples():
    return [
        {
            "inputs": "What color is the sky?",
            "multiple_choice_targets": ["blue", "green", "red"],
            "multiple_choice_scores": [1, 0, 0],
            "reasoning": "The sky scatters blue light most.",
        }
    ]


def test_rl_schema(tokenizer, rl_examples):
    ds = KairosRLDataset(tokenizer, max_len=128, examples=rl_examples)
    item = ds[0]
    expected_keys = {"input_ids", "gen_mask", "prompt_len", "mask_ratio", "choices", "scores", "level"}
    assert set(item.keys()) == expected_keys
    assert item["input_ids"].shape == (128,)
    assert len(ds) == 1


def test_rl_adds_not_sure_option(tokenizer, rl_examples):
    ds = KairosRLDataset(tokenizer, max_len=128, examples=rl_examples)
    item = ds[0]
    assert "not sure / I don't know" in item["choices"]
    assert len(item["choices"]) == len(item["scores"]) == 4


def test_rl_mask_ratio_within_bounds(tokenizer, rl_examples):
    ds = KairosRLDataset(tokenizer, max_len=128, examples=rl_examples)
    item = ds[0]
    assert 0.0 <= item["mask_ratio"].item() <= 1.0
    assert item["level"] in ("low", "medium", "flex")


def test_rl_prompt_tokens_are_never_generated(tokenizer, rl_examples):
    ds = KairosRLDataset(tokenizer, max_len=128, examples=rl_examples)
    item = ds[0]
    plen = item["prompt_len"].item()
    assert item["gen_mask"][:plen].sum() == 0


def test_dpo_dispatch_uses_load_dataset(tokenizer, monkeypatch):
    def fake_load_dataset(name, split):
        return [
            {
                "prompt": "Tell me a joke.",
                "chosen": [{"role": "assistant", "content": "Chosen answer"}],
                "rejected": [{"role": "assistant", "content": "Rejected answer"}],
            }
        ]

    monkeypatch.setattr("kairos.dataset.load_dataset", fake_load_dataset)
    ds = KairosDPODataset(tokenizer, max_len=32, examples=None)
    assert len(ds) == 1


def test_rl_dispatch_uses_streaming_load_dataset(tokenizer, monkeypatch):
    def fake_config_names(name):
        return ["config_a"]

    def fake_load_dataset(name, config, split, streaming):
        return [
            {
                "inputs": "2+2?",
                "multiple_choice_targets": ["4", "5"],
                "multiple_choice_scores": [1, 0],
                "reasoning": "basic math",
            }
        ]

    monkeypatch.setattr("kairos.dataset.get_dataset_config_names", fake_config_names)
    monkeypatch.setattr("kairos.dataset.load_dataset", fake_load_dataset)
    ds = KairosRLDataset(tokenizer, max_len=64, examples=None, max_samples=1)
    assert len(ds) == 1
