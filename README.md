# Kairos

**Kairos** is an experimental 1B/100M hybrid MoE multimodal model. It explores the combination of linear-attention (**DeltaNet**) and sliding-window attention (**SWA**), stabilized by **Attention Residuals** (**AttnRes**).

The objective is to achieve **universal edge-capability**. By using a shared latent space for text, audio, image, video, and lidar data, Kairos prioritizes architectural efficiency over brute-force compute, leveraging linear-attention for infinite context compression.

## Architecture

* **Total Params:** 1B
* **Active Params:** 100M
* **Unified Latent Space:** Cross-modal projection layers mapping all inputs to the DeltaNet/SWA backbone.
* **Core Components:**
* **DeltaNet:** Linear SSM backbone for cross-modal state compression.
* **SWA:** Sliding Window Attention for modality-specific precision.
* **MoE:** Sparse routing across heterogeneous modal experts.
* **AttnRes:** Signal stability for multi-modal residual flow ([arXiv:2603.15031](https://arxiv.org/abs/2603.15031)).


## Roadmap: Toward Universal Intelligence

* **Multimodal Integration:** Early-stage training for image, video, audio & lidar tokens (1% of training).
* **Unified Tokenizer:** Learning a cross-modal embedding space.
* **Generative Capabilities:** Researching native diffusion decoding within the SSM framework.

## Data & Training

* **Dataset:** Inspired by *SmolLM* (high-quality, filtered educational and reasoning data) + cross-modal alignment sets.
* **Procedure:** Compact, curriculum-based training focused on cross-modal grounding and agentic reasoning within a 1B budget.

## References

* *Attention Residuals for Deep Signal Stability* ([arXiv:2603.15031](https://arxiv.org/abs/2603.15031))
* *DeltaNet / SWA* implementation standards