"""Kairos: a hybrid MoE diffusion language model (DeltaNet + sliding-window attention) for multimodal edge AI."""

from .modeling import KairosConfig, KairosDiffusionLLM, KairosMemoryBank
from .pipeline import DataConfig, KairosMultimodalPipeline, TrainConfig
from .tokenizer import KairosTokenizer, Modality

__version__ = "1.0.0"

__all__ = [
    "DataConfig",
    "KairosConfig",
    "KairosDiffusionLLM",
    "KairosMemoryBank",
    "KairosMultimodalPipeline",
    "KairosTokenizer",
    "Modality",
    "TrainConfig",
    "__version__",
]
