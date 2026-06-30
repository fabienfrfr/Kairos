---
language:
- en
- fr
license: mit
task_categories:
- text-generation
- translation
pretty_name: Keep it simple !
size_categories:
- 100K<n<1M
---


# keep-it-simple

**Objective:** An ultra-minimalist dataset for pre-training tiny language models. The logic relies on **bidirectional symmetry** (**A is B and B is A]**) to foster deep semantic understanding. By training the model to predict the "prompt" from the "text" and vice versa, we maximize the utility of every pair.

## Data Sources

* **Simple English Wikipedia**: Simplified encyclopedic articles.
* **Vikidia (FR)**: Educational content for younger audiences.
* **[OPUS Books (en-fr)](https://aclanthology.org/L12-1246/)**: Aligned English-French literary translations.
* **[Cosmopedia-100k](https://huggingface.co/blog/cosmopedia)**: Synthetic educational content.

## Structure

| Column | Description |
| --- | --- |
| `prompt` | Input (concept, title, or English translation). |
| `text` | Output (explanation, summary, or French translation). |
| `seed_data` | Origin identifier (traceability). |

## Context & Usage

* **Bidirectional Training**: Each source item yields two training entries (`prompt` $\rightarrow$ `text` and `text` $\rightarrow$ `prompt`). This enforces semantic symmetry, [reversal curve](https://arxiv.org/abs/2309.12288) and limitate span corruption.
* **Minimalism**: More compact than the **[BabyLM](https://arxiv.org/abs/2602.20092)** challenge; focused on density and the purity of pairs to maximize efficiency on tiny, resource-constrained architectures.
* **Goal**: Rapid testing of alignment theories and training "pocket" models for fundamental, bidirectional interactions.

*This dataset is a minimalist research tool.*