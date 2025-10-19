
import onnx
import os
import torch
import numpy as np
import sys
import tiktoken

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.python.model.moe.deepseek.pytorch.deepseek_v3_pytorch import DeepSeekV3Model, generate_text_simple, text_to_tokens, token_ids_to_text

test_cfg = {
    "vocab_size": 50257,  # Full GPT-2 vocab to handle real tokens
    "emb_dim": 64,        # Small embedding
    "n_layers": 1,        # Only 1 layer
    "batch": 1,
    "num_heads": 4,       # 4 heads (64/4 = 16 head_dim)
    "seq_len": 8,         # Short sequence
    "num_experts": 4,     # 4 experts
    "top_k": 2,           # Top-2 routing
    "expert_dim": 128,     # Small expert hidden dim
    "context_length": 256
}

onnx_path = "deepseek_v3.onnx"
if os.path.exists(onnx_path):
    model = DeepSeekV3Model(test_cfg)
    onnx_model = onnx.load(onnx_path)
    # Extract weights from ONNX initializers
    onnx_weights = {}
    for initializer in onnx_model.graph.initializer:
        onnx_weights[initializer.name] = torch.from_numpy(
            np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
        )
    
    # Map ONNX weights to PyTorch state dict
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in onnx_weights and param.shape == onnx_weights[name].shape:
            state_dict[name] = onnx_weights[name]
    
    model.load_state_dict(state_dict)
    print("Loaded weights from ONNX model")

    tokenizer = tiktoken.get_encoding('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    prompt = "Hello world, my"
    out = generate_text_simple(model, text_to_tokens(prompt, tokenizer), 10, test_cfg, 50256)
    output = token_ids_to_text(out, tokenizer)
    print(output)