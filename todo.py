import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaPreTrainedModel
)
from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet, 
    Qwen3_5MoeMLP
)


class KairosConfig(DiffusionGemmaConfig):
    model_type = "kairos"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 1. Vision Config: Drastically reduced
        if self.vision_config is not None:
            self.vision_config.hidden_size = 512
            self.vision_config.intermediate_size = 2048
            self.vision_config.num_hidden_layers = 12
            self.vision_config.num_attention_heads = 8
            self.vision_config.num_key_value_heads = 8
            self.vision_config.head_dim = 64
            
        # 2. Text Config: Overrides
        # We ensure these align with your specific architecture requirements
        self.hidden_size = 512
        self.num_hidden_layers = 16
        self.intermediate_size = 256
        self.moe_intermediate_size = 256
        self.num_experts = 32
        self.top_k_experts = 4
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 64
        self.vocab_size = 32768
        
        # 3. Force sliding window pattern
        self.layer_types = ["sliding_attention"] * self.num_hidden_layers
        
        # Compatibility
        self.tie_word_embeddings = True


class AttnResAggregator(nn.Module):
    """
    Implements Layer-Attention Residual Aggregation.
    Weights contributions from all preceding sublayers.
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_embd))
        self.key_norm = nn.RMSNorm(n_embd)

    def forward(self, prior_values: list[torch.Tensor]) -> torch.Tensor:
        # Stack history: [Layers, Batch, SeqLen, HiddenSize]
        V = torch.stack(prior_values, dim=0)
        K = self.key_norm(V)
        
        # Calculate attention logits across layers
        logits = torch.einsum("d,lbtd->lbt", self.w, K)
        weights = F.softmax(logits, dim=0)
        
        # Compute weighted sum
        return (weights.unsqueeze(-1) * V).sum(dim=0)

class KairosLayer(nn.Module):
    """
    Hybrid Kairos Layer: DeltaNet + MoE MLP with Attention-Residual gating.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.is_causal = False # TODO
        self.deltanet = Qwen3_5MoeGatedDeltaNet(config, layer_idx=layer_idx)
        self.mlp = Qwen3_5MoeMLP(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.aggregator = AttnResAggregator(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, prior_outputs: list[torch.Tensor], **kwargs) -> torch.Tensor:
        # 1. DeltaNet branch
        out_ssm = self.deltanet(hidden_states, **kwargs)
        
        # 2. MLP branch (with normalization)
        out_mlp = self.mlp(self.norm(hidden_states))
        
        # 3. Aggregate: Input + DeltaNet Out + MLP Out
        return self.aggregator(prior_outputs + [hidden_states, out_ssm, out_mlp])

class KairosModel(DiffusionGemmaPreTrainedModel):
    """
    Custom Kairos Model architecture for from-scratch training.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Define embeddings (Required for from-scratch model) --> NO ! use pretrained tokenizer ?
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Define hybrid layers
        self.layers = nn.ModuleList([
            KairosLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        
        self.post_init()

    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kwargs) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        history = [hidden_states]
        
        for layer in self.layers:
            # Pass full history to the layer for aggregation
            hidden_states = layer(hidden_states, history, attention_mask=attention_mask, **kwargs)
            history.append(hidden_states)
            
        return hidden_states