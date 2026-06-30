---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - synthetic
  - tools
  - instruction-finetuning
pretty_name: SimpleTool-Instruct
size_categories:
  - 10K<n<100K
---

# SimpleTool-Instruct

**SimpleTool-Instruct** is a merged instruction dataset combining:

- **ToolACE** — real multi-turn conversations with function-calling examples  
- **Alpaca-cleaned** — clean, simple instruction–response pairs  

The goal is to provide a unified dataset for training models that can handle both:

- classic instruction-following  
- tool-augmented interactions (function calling)


## Intended Use

This dataset is designed for:

- Training instruction-following LLMs  
- Training models with **function-calling** capabilities  
- Fine-tuning lightweight models for tool-augmented reasoning  
- Research on hybrid conversational + tool-use datasets  
