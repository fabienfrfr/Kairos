---
language: en
license: {license}
library_name: transformers
tags:
  - kairos
  - diffusion
  - multimodal
  - moe
  - trust_remote_code
pipeline_tag: text-generation
datasets:
  - ffurfaro/keep-it-simple
  - ffurfaro/keep-it-simple-multimodal
---

<h1 align="center"><p>🌀 {name}</p></h1>

<p align="center">
<a href="https://pypi.org/project/kairos-fm/">
<img alt="PyPI" src="https://img.shields.io/pypi/v/kairos-fm?color=orange&logo=pypi">
</a>
<a href="https://github.com/fabienfrfr/Kairos">
<img alt="GitHub" src="https://img.shields.io/badge/github-fabienfrfr%2FKairos-black?logo=github">
</a>
<a href="https://huggingface.co/ffurfaro">
<img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-model-yellow?logo=huggingface">
</a>
</p>

<h3 align="center">
<p>Kairos Foundation Model — less parameters, more signal.</p>
</h3>

KairosFM is a hybrid MoE diffusion language model combining **DeltaNet** (linear attention),
**Sliding Window Attention**, and **Attention Residuals (AttnRes)**, trained on text, image,
video, audio, lidar, and control (state/action) modalities through a shared multimodal
conv-byte tokenizer. See [github.com/fabienfrfr/Kairos](https://github.com/fabienfrfr/Kairos)
for the full architecture writeup.

## This checkpoint

| | |
|---|---|
| Total params | {d_model}-dim, {n_layers} layers |
| Experts | {n_routed_experts} routed / {n_shared_experts} shared, top-{num_experts_per_tok} |
| Vocab size | {vocab_size} |
| Best training loss | `{best_loss}` |
| Steps trained | `{global_step}` |

Note: this repo currently tracks best-training-loss only (`checkpoints/best.pt`) — no held-out
validation split is evaluated during training yet.

## Files

- `checkpoints/` — `best.pt` (lowest avg training loss) + periodic `step_*.pt`
- `tensorboard/` — `events.out.tfevents.*`, viewable in the Hub's **Training Metrics** tab
- `config.json`, `model.safetensors` — native HF format, loadable via `trust_remote_code`

## Usage

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
```

Requires the `kairos` package importable (custom architecture, not upstream `transformers`) —
install from [github.com/fabienfrfr/Kairos](https://github.com/fabienfrfr/Kairos) first, or add
it to `PYTHONPATH`. Alternatively, skip `Auto*` and import the class directly:

```python
from kairos.modeling import KairosDiffusionFM

model = KairosDiffusionFM.from_pretrained("{repo_id}")
```

## Generation

Kairos is a masked-diffusion model: generation denoises a fresh canvas of
`max_new_tokens` tokens after a fixed prompt (block diffusion, no KV cache).

Via the pipeline wrapper (handles device placement, modality ids and AMP):

```python
from kairos.tokenizer import KairosTokenizer

tokenizer = KairosTokenizer()
prompt = tokenizer.encode("The capital of France is", add_special_tokens=False)
generated = pipe.generate(
    prompt,
    max_new_tokens=64,
    max_denoising_steps=24,
    entropy_bound=0.5,  # higher accepts more tokens per denoising step
    t_min=0.4,
    t_max=1.0,
)
print(tokenizer.decode(generated, skip_special_tokens=True))
```

Directly on the model, reusing the DiffusionGemma knobs:

```python
from kairos.modeling import KairosDiffusionFM
from kairos.tokenizer import KairosTokenizer, Modality

model = KairosDiffusionFM.from_pretrained("{repo_id}")
tokenizer = KairosTokenizer()
prompt = tokenizer.encode("The capital of France is", add_special_tokens=False)
prompt = torch.as_tensor(prompt, dtype=torch.long).unsqueeze(0)
modality_ids = torch.full_like(prompt, int(Modality.TEXT))
out = model.generate(
    input_ids=prompt,
    modality_ids=modality_ids,
    max_new_tokens=64,
    max_denoising_steps=24,
    entropy_bound=0.5,
    t_min=0.4,
    t_max=1.0,
)
print(tokenizer.decode(out.sequences[0], skip_special_tokens=True))
```

## Limitations

Experimental, low-compute-budget training run — expect uneven quality across modalities
(multimodal data is a small fraction of total training). Not evaluated for safety-critical use.

## Citation

```bibtex
@misc{{kairos,
  title  = {{KairosFM: less parameters, more signal — a multimodal MoE diffusion model for edge AI}},
  author = {{Fabien Furfaro}},
  url    = {{https://github.com/fabienfrfr/Kairos}}
}}
```
