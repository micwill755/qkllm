# Disaggregated Prefill/Decode: A Beginner's Guide

## What is Disaggregated P/D?

**Disaggregated Prefill/Decode** separates the two phases of LLM inference across different GPUs or nodes to optimize resource utilization and performance.

### Traditional vs Disaggregated Inference

**Traditional (Single GPU):**
```
GPU: Prefill → Decode → Prefill → Decode → ...
     (Process prompt) (Generate tokens) (Next prompt) (Generate tokens)
```

**Disaggregated:**
```
GPU 0 (Prefill):  Prefill → Prefill → Prefill → ...
                   (Process prompts in parallel)

GPU 1 (Decode):   Decode → Decode → Decode → ...
                   (Generate tokens sequentially)
```

## Why Disaggregate?

Tuning time-to-first-token (TTFT) and inter-token-latency (ITL) separately. Disaggregated prefilling put prefill and decode phase of LLM inference inside different vLLM instances. This gives you the flexibility to assign different parallel strategies (e.g. tp and pp) to tune TTFT without affecting ITL, or to tune ITL without affecting TTFT.

Controlling tail ITL. Without disaggregated prefilling, vLLM may insert some prefill jobs during the decoding of one request. This results in higher tail latency. Disaggregated prefilling helps you solve this issue and control tail ITL. Chunked prefill with a proper chunk size also can achieve the same goal, but in practice it's hard to figure out the correct chunk size value. So disaggregated prefilling is a much more reliable way to control tail ITL.

## Example

The example below is a disaggregated serving proxy for vLLM that implements a distributed architecture separating prefill and decode operations. 

Disaggregated Design: Splits LLM inference into two phases:
- Prefill instances: Handle initial prompt processing and KV cache preparation
- Decode instances: Handle token generation/decoding

Request Flow
Client Request → Proxy → Prefill Instance (KV prep) → Decode Instance (generation) → Client
For each request:
- Prefill stage: Sends request with max_tokens=1 to prepare KV cache
- Decode stage: Sends full request to decode instance for token generation

Terminal 1: Prefill Instance (GPUs 0-3)

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8100 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching

Terminal 2: Decode Instance (GPUs 4-7)

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8200 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching

Terminal 3: Start Proxy

- Routes requests between prefill and decode instances
- Uses round-robin scheduling via itertools.cycle
- Manages instance pools and health validation

python3 disaggregated_pd_serving.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --prefill localhost:8100 \
  --decode localhost:8200 \
  --port 8000

# Test the Setup

1. Check Status

curl http://localhost:8000/status

2. Test Completion

curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0.7
  }'

3. Test Chat Completion

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'

  # Benchmark

  Use vLLM bench command

  vllm bench latency --model meta-llama/Meta-Llama-3.1-8B-Instruct --batch-size 8 --input-len 32 --output-len 128

  Then benchmark our disaggregated setup on 8 GPUs

