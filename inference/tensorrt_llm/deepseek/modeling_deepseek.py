from typing import Optional
import math
import torch
from torch import nn

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

from configuration_deepseek import DeepSeekV3Config

class DeepSeekV3Attention(Attention):
    def __init__(self, model_config: ModelConfig[DeepSeekV3Config], layer_idx: Optional[int] = None):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            dtype=torch.float16,
            layer_idx=layer_idx,
            q_scaling=1.0,
        )
        self.latent_dim = config.latent_dim
        self.latent_tokens = Linear(
            config.max_position_embeddings * config.hidden_size,
            config.latent_dim * config.hidden_size,
            dtype=torch.float16,
            mapping=model_config.mapping, gather_output=True
        )

    def forward(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create latent tokens
        x_flat = hidden_states.view(batch_size, -1)
        latent_compressed = self.latent_tokens(x_flat)
        latent = latent_compressed.view(batch_size, self.latent_dim, hidden_size)
        
        # Use cross-attention: Q from input, K,V from latent
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=None,
            past_key_value=None,
            attn_metadata=attn_metadata,
            **kwargs
        )

class DeepSeekV3Expert(nn.Module):
    def __init__(self, model_config: ModelConfig[DeepSeekV3Config]):
        super().__init__()
        config = model_config.pretrained_config
        self.gate_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=False,
            dtype=torch.float16, mapping=model_config.mapping, gather_output=False
        )
        self.up_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=False,
            dtype=torch.float16, mapping=model_config.mapping, gather_output=False
        )
        self.down_proj = Linear(
            config.intermediate_size, config.hidden_size, bias=False,
            dtype=torch.float16, mapping=model_config.mapping, gather_output=True
        )

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class DeepSeekV3MoE(nn.Module):
    def __init__(self, model_config: ModelConfig[DeepSeekV3Config]):
        super().__init__()
        config = model_config.pretrained_config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        
        self.gate = Linear(
            config.hidden_size, config.num_experts, bias=False,
            dtype=torch.float16, mapping=model_config.mapping, gather_output=True
        )
        self.experts = nn.ModuleList([
            DeepSeekV3Expert(model_config) for _ in range(config.num_experts)
        ])

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        
        # Router computation
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # Expert computation
        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx)
            expert_weights = routing_weights * expert_mask
            
            if expert_weights.sum() > 0:
                expert_output = self.experts[expert_idx](hidden_states)
                final_hidden_states += expert_output * expert_weights.sum(dim=-1, keepdim=True)
        
        return final_hidden_states.view(batch_size, seq_len, hidden_size)

class DeepSeekV3DecoderLayer(DecoderLayer):
    def __init__(self, model_config: ModelConfig[DeepSeekV3Config], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=torch.float16
        )
        self.self_attn = DeepSeekV3Attention(model_config, layer_idx)
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=torch.float16
        )
        self.mlp = DeepSeekV3MoE(model_config)

    def forward(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata, **kwargs):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attn_metadata, **kwargs)
        hidden_states = residual + hidden_states
        
        # MoE with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class DeepSeekV3Model(DecoderModel):
    def __init__(self, model_config: ModelConfig[DeepSeekV3Config]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        
        from tensorrt_llm._torch.modules.linear import TensorParallelMode
        self.embed_tokens = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=torch.float16,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True
        )
        
        self.layers = nn.ModuleList([
            DeepSeekV3DecoderLayer(model_config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=torch.float16
        )

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: Optional[torch.IntTensor] = None,
                position_ids: Optional[torch.IntTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None):
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_metadata)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states

@register_auto_model("DeepSeekV3ForCausalLM")
class DeepSeekV3ForCausalLM(DecoderModelForCausalLM[DeepSeekV3Model, DeepSeekV3Config]):
    def __init__(self, model_config: ModelConfig[DeepSeekV3Config]):
        super().__init__(
            DeepSeekV3Model(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size
        )
        # Override lm_head to match embedding TP mode
        from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
        self.lm_head = Linear(
            model_config.pretrained_config.hidden_size,
            model_config.pretrained_config.vocab_size,
            bias=False,
            dtype=torch.float16,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True
        )

    def load_weights(self, weights: dict):
        # Custom weight loading logic for DeepSeek V3
        for name, param in self.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])
            else:
                # Handle weight mapping if needed
                mapped_name = self._map_weight_name(name)
                if mapped_name in weights:
                    param.data.copy_(weights[mapped_name])
    
    def _map_weight_name(self, name: str) -> str:
        # Map TensorRT-LLM parameter names to original checkpoint names
        name = name.replace("model.", "")
        return name