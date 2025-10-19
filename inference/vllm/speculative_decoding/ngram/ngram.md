# N-gram Speculative Decoding: A Beginner's Guide

## What is Speculative Decoding?

Speculative decoding speeds up text generation by:
1. **Draft**: Quickly generating candidate tokens
2. **Verify**: Having the main model check these candidates in parallel
3. **Accept**: Using verified tokens, rejecting bad ones

Think of it like autocomplete - we guess what comes next, then verify if it's correct.

## What are N-grams?

An **n-gram** is a sequence of n consecutive tokens:
- **1-gram**: "hello"
- **2-gram**: "hello world" 
- **3-gram**: "hello world today"

## How N-gram Speculative Decoding Works

### The Process
1. **Look back**: Take the last few tokens you just generated
2. **Search**: Find where you've seen this exact pattern before in the conversation
3. **Predict**: Use the tokens that followed that pattern as candidates
4. **Verify**: Let the main model check if these candidates are good

### Example
```
Current conversation: "The weather is nice today. The weather is"
                                                    ↑ generating here

1. Take last 4 tokens: ["The", "weather", "is", "nice"]
2. Search earlier: Found at start → "The weather is nice today"
3. Predict next 2 tokens: ["today", "."] 
4. Main model verifies these candidates
```

## Key Parameters

- **`prompt_lookup_max`**: Maximum n-gram size to try (e.g., 4 tokens)
- **`prompt_lookup_min`**: Minimum n-gram size (e.g., 1 token)  
- **`k`**: Number of candidate tokens to propose (e.g., 2 tokens)

## Setting Up with vLLM

### Basic Setup
```python
from vllm import LLM, SamplingParams

# Initialize model with n-gram speculative decoding
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    speculative_model=None,  # No separate draft model needed
    use_v2_block_manager=True,
    speculative_draft_tensor_parallel_size=1,
    # N-gram specific parameters
    prompt_lookup_max=4,     # Try up to 4-token patterns
    prompt_lookup_min=1,     # Fall back to 1-token patterns
    max_model_len=4096
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    speculative_tokens=3     # Number of tokens to speculate
)

# Generate
prompt = "The weather forecast shows"
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

### Advanced Configuration
```python
# For better performance
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    # N-gram tuning
    prompt_lookup_max=6,     # Longer patterns for repetitive text
    prompt_lookup_min=2,     # Skip single tokens
    use_v2_block_manager=True,
    block_size=16,
    swap_space=4
)

sampling_params = SamplingParams(
    temperature=0.1,         # Lower temp works better with speculation
    speculative_tokens=5,    # More aggressive speculation
    max_tokens=200
)
```

## When N-gram Works Best

### Good Use Cases
- **Repetitive text**: Code, documentation, structured formats
- **Conversations**: Repeated phrases and patterns
- **Templates**: Forms, reports, standard formats
- **Long contexts**: More opportunities to find patterns

### Example Scenarios
```python
# Code generation (lots of repetitive patterns)
prompt = "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    else:"

# Chat conversations (repeated conversational patterns)  
prompt = "User: What's the weather like?\nAssistant: The weather is sunny today.\nUser: What about tomorrow?\nAssistant:"

# Documentation (structured, repetitive format)
prompt = "## Installation\n\nTo install this package:\n\n```bash\npip install"
```

## Performance Tips

### Optimal Parameters
- **Short contexts**: `prompt_lookup_max=3-4`
- **Long contexts**: `prompt_lookup_max=6-8` 
- **Structured text**: `prompt_lookup_min=2` (skip single tokens)
- **Creative writing**: `speculative_tokens=2-3`
- **Code/docs**: `speculative_tokens=4-6`

### Monitoring Performance
```python
# Check if speculation is helping
import time

start = time.time()
outputs = llm.generate([prompt], sampling_params)
end = time.time()

print(f"Generation time: {end - start:.2f}s")
print(f"Tokens generated: {len(outputs[0].outputs[0].token_ids)}")
print(f"Tokens/second: {len(outputs[0].outputs[0].token_ids)/(end-start):.1f}")
```

## Comparison with Other Methods

| Method | Speed | Memory | Setup Complexity |
|--------|-------|---------|------------------|
| **N-gram** | Fast | Low | Simple |
| Eagle | Faster | Medium | Model surgery needed |
| Medusa | Fastest | High | Additional training |
| Small draft model | Medium | High | Two models required |

## Troubleshooting

### Low Speedup?
- Increase `prompt_lookup_max` for longer patterns
- Increase `speculative_tokens` for more aggressive speculation
- Check if your text has repetitive patterns

### High Memory Usage?
- Decrease `max_model_len`
- Lower `gpu_memory_utilization`
- Reduce `speculative_tokens`

### Poor Quality?
- Lower `temperature` (speculation works better with focused generation)
- Decrease `speculative_tokens`
- Check if n-gram speculation suits your use case

## Getting Started

1. **Install vLLM**: `pip install vllm`
2. **Start simple**: Use default parameters first
3. **Test your use case**: See if your text has repetitive patterns
4. **Tune parameters**: Adjust based on your specific needs
5. **Monitor performance**: Track tokens/second improvements

N-gram speculative decoding is the easiest way to speed up text generation - no additional models, no training, just pattern matching on your existing context!