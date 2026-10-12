"""Loads the Hugging Face pretraining datasets named in a config's `hf_data` section."""

from __future__ import annotations

from .dataset import split_examples


def _text_rows(rows) -> list[dict]:
    return [{"modality": "text", "text": f"{r['prompt']} {r['text']}".strip()} for r in rows]


def load_hf_examples(spec: dict, loader=None) -> tuple[list, list]:
    """Returns (train, eval) examples from the text and multimodal HF repos named in `spec`."""
    if loader is None:
        from datasets import load_dataset as loader

    pct = spec.get("text_pct", 100)
    text = _text_rows(loader(spec["text_repo"], split=f"train[:{pct}%]")) if spec.get("text_repo") else []
    multimodal = list(loader(spec["multimodal_repo"], split="train")) if spec.get("multimodal_repo") else []
    return split_examples(text + multimodal, eval_pct=spec.get("eval_pct", 0.001), seed=spec.get("seed", 0))
