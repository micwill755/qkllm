import torch
import tensorrt as trt
import onnx
import os
import tiktoken
import sys

# Add path to access your model utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.python.model.moe.deepseek.pytorch.deepseek_v3_pytorch import text_to_tokens, token_ids_to_text

# setup a TensorRT pipeline for converting an ONNX model 
def convert_to_tensorrt(model, input_shape, logger, fp16=True):
    # create builder object that will construct the optimized engine
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model.SerializeToString()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # build the engine
    engine = builder.build_serialized_network(network, config)
    return engine

# Use same config as ONNX script
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

# show warning error messages and above
logger = trt.Logger(trt.Logger.WARNING)
input_shape = (1, test_cfg["seq_len"])  # Use seq_len from config
engine_path = 'deepseek_v3.trt'

if os.path.exists(engine_path):
    with open(engine_path, "rb") as f:
        engine = f.read()
else:
    #convert from onnx
    model = onnx.load("/home/ubuntu/llm/interview/deepseek_v3.onnx")
    engine = convert_to_tensorrt(model, input_shape, logger, fp16=True)
    with open (engine_path, 'wb') as f:
        f.write(engine)

# once we have compiled the engine we can now inference
# Deserialize and create runtime
runtime = trt.Runtime(logger)
engine_obj = runtime.deserialize_cuda_engine(engine)

# Create execution context
context = engine_obj.create_execution_context()

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load tokenizer - same as ONNX script
tokenizer = tiktoken.get_encoding('gpt2')

# Use SAME prompt as ONNX script
prompt = "Hello world, my"
tokens = text_to_tokens(prompt, tokenizer)
print('Raw tokens:', tokens)

# Extract the actual token list from the 2D tensor
if isinstance(tokens, torch.Tensor):
    if tokens.dim() == 2:
        token_list = tokens[0].tolist()  # Get first row
    else:
        token_list = tokens.tolist()
else:
    token_list = tokens

print('Token list:', token_list)

# Pad or truncate to seq_len
token_list = token_list[:test_cfg["seq_len"]]
while len(token_list) < test_cfg["seq_len"]:
    token_list.append(50256)  # EOS token for padding

input_tensor = torch.tensor([token_list], dtype=torch.long)
print('Input tensor shape:', input_tensor.shape)

# Fix memory allocation
input_size = np.prod(input_shape) * 4
output_shape = engine_obj.get_tensor_shape(engine_obj.get_tensor_name(1))
output_size = np.prod(output_shape) * 4

d_input = cuda.mem_alloc(int(input_size))
d_output = cuda.mem_alloc(int(output_size))

# Use tokenized input
input_data = input_tensor.numpy().astype(np.float32)
cuda.memcpy_htod(d_input, input_data)

bindings = [int(d_input), int(d_output)]
context.execute_v2(bindings)

output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_data, d_output)

# Process output same way as ONNX
logits = output_data.reshape(-1, test_cfg["vocab_size"])
predicted_tokens = np.argmax(logits, axis=-1)
print('Input tokens:', token_list)
print('Predicted tokens:', predicted_tokens.tolist())

# Convert to tensor for token_ids_to_text function
predicted_tensor = torch.tensor([predicted_tokens.tolist()])
output = token_ids_to_text(predicted_tensor, tokenizer)
print('Output:', output)