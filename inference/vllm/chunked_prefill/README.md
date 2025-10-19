# Chunked Prefill Demo with vLLM

This directory contains a beginner-friendly implementation of chunked prefill using vLLM.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python chunked_prefill_demo.py
```

Choose between:
- **Option 1**: Simple synchronous demo (easier to understand)
- **Option 2**: Advanced async demo (shows real-world usage)

## What You'll See

### Simple Demo Output:
```
🚀 Processing 4 prompts with chunked prefill...

📊 Results (Total: 2.34s):
Prompt 1 ( 3 tokens): What is AI?...
Response: AI stands for Artificial Intelligence, which refers to...

Prompt 2 (12 tokens): Explain machine learning in detail...
Response: Machine learning is a subset of AI that enables...

💡 Chunked prefill enabled efficient processing of mixed request sizes!
```

### Advanced Demo Output:
```
📋 Processing 5 requests:
  - large_1: LARGE (2000 tokens)
  - small_1: SMALL (3 tokens)
  - small_2: SMALL (4 tokens)
  - small_3: SMALL (5 tokens)
  - large_2: LARGE (1500 tokens)

📊 RESULTS (Total time: 3.45s)
small_1  | SMALL |    3 tokens |   0.12s | What is 2+2?...
small_2  | SMALL |    4 tokens |   0.15s | Hello, how are you?...
small_3  | SMALL |    5 tokens |   0.18s | Translate 'hello' to French...
large_1  | LARGE | 2000 tokens |   2.34s | Please analyze the following...
large_2  | LARGE | 1500 tokens |   2.89s | Please analyze the following...

📈 PERFORMANCE SUMMARY:
  Small requests avg latency: 0.15s
  Large requests avg latency: 2.62s
  Total throughput: 1.4 req/s

💡 KEY INSIGHT:
  Without chunked prefill, small requests would wait for large ones!
  With chunked prefill, small requests get served quickly.
```

## Key Features Demonstrated

### 1. **Request Interleaving**
- Large requests are split into chunks
- Small requests get processed immediately
- No request monopolizes the GPU

### 2. **Memory Efficiency**
- Controlled memory usage through chunking
- Prevents OOM errors on long sequences
- Predictable memory patterns

### 3. **Improved Fairness**
- Small requests don't wait behind large ones
- Better user experience for mixed workloads
- Higher overall throughput

## Configuration Options

The demo shows key vLLM parameters for chunked prefill:

```python
llm = LLM(
    model="microsoft/DialoGPT-medium",
    enable_chunked_prefill=True,        # Enable the feature
    max_num_batched_tokens=2048,        # Total batch size
    max_num_seqs=256,                   # Max concurrent requests
    gpu_memory_utilization=0.9,         # Memory usage limit
)
```

## Understanding the Output

### Latency Patterns:
- **Small requests**: ~0.1-0.2 seconds (served immediately)
- **Large requests**: ~2-3 seconds (processed in chunks)
- **Without chunked prefill**: Small requests would wait 2-3 seconds too!

### Throughput Benefits:
- **Traditional**: Process requests sequentially
- **Chunked prefill**: Process multiple requests concurrently
- **Result**: 3-5x better throughput

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**
   - Reduce `gpu_memory_utilization` to 0.7 or 0.8
   - Use a smaller model
   - Reduce `max_num_batched_tokens`

2. **Model not found**
   - The demo uses `microsoft/DialoGPT-medium` (small model)
   - Replace with any HuggingFace model you prefer
   - Ensure the model is compatible with vLLM

3. **vLLM installation issues**
   ```bash
   pip install vllm --upgrade
   # or for specific CUDA version:
   pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
   ```

## Next Steps

1. **Try different models**: Replace the model name with larger models
2. **Adjust chunk sizes**: Experiment with `max_num_batched_tokens`
3. **Add more requests**: Test with different request patterns
4. **Monitor GPU usage**: Use `nvidia-smi` to see memory and utilization

## Learn More

- Read the [Chunked Prefill Guide](chunked_prefill.md) for detailed explanations
- Check out [vLLM documentation](https://vllm.readthedocs.io/) for advanced features
- Explore the [Parallelism Guide](../PARALLELISM_GUIDE.md) for scaling strategies