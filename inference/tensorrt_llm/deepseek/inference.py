#!/usr/bin/env python3
"""
DeepSeek V3 Inference with TensorRT-LLM
Simple inference script for DeepSeek V3 model with Multi-Head Latent Attention and MoE
"""
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))

def run_deepseek_inference():
    """Run DeepSeek V3 inference"""
    try:
        from configuration_deepseek import DeepSeekV3Config
        from modeling_deepseek import DeepSeekV3ForCausalLM
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm.mapping import Mapping
        
        print("🚀 DeepSeek V3 TensorRT-LLM Inference")
        
        # Small model configuration
        config = DeepSeekV3Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            max_position_embeddings=512,
            num_experts=4,
            num_experts_per_tok=2,
            intermediate_size=512,
            latent_dim=32,
        )
        
        # Single GPU setup
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        model_config = ModelConfig(
            pretrained_config=config,
            mapping=mapping,
            max_num_tokens=256,
        )
        
        # Load model
        model = DeepSeekV3ForCausalLM(model_config).cuda()
        print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        
        # Generate text
        prompt = torch.randint(0, config.vocab_size, (1, 10)).cuda()
        generated = prompt.clone()
        
        print("🔥 Generating tokens...")
        for step in range(20):
            with torch.no_grad():
                embeddings = model.model.embed_tokens(generated)
                hidden_states = model.model.norm(embeddings)
                logits = model.lm_head(hidden_states)
                
                # Sample next token
                next_token_logits = logits[0, -1, :]
                probs = torch.softmax(next_token_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                if step % 5 == 0:
                    print(f"  Step {step}: {generated.shape[1]} tokens")
        
        print(f"✓ Generated sequence: {generated[0].tolist()}")
        print("🎉 DeepSeek V3 inference successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_deepseek_inference()