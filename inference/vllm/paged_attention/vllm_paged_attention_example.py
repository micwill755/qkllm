#!/usr/bin/env python3
"""
Simple vLLM PagedAttention Usage Example
PagedAttention is enabled by default in vLLM
"""

from vllm import LLM, SamplingParams

def basic_vllm_example():
    """Basic vLLM usage - PagedAttention is automatic"""
    
    # Initialize LLM - PagedAttention enabled by default
    llm = LLM(
        model="microsoft/DialoGPT-medium",  # Smaller model for demo
        # PagedAttention parameters (optional tuning)
        block_size=16,                     # Tokens per block (default: 16)
        max_num_seqs=256,                  # Max concurrent sequences
        max_num_batched_tokens=2048,       # Max tokens per batch
        gpu_memory_utilization=0.9,        # Use 90% GPU memory for KV cache
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100
    )
    
    # Multiple prompts - PagedAttention manages memory automatically
    prompts = [
        "The future of artificial intelligence",
        "In a world where robots",
        "The most important discovery",
        "Once upon a time in a distant galaxy",
        "The secret to happiness is"
    ]
    
    print(f"Processing {len(prompts)} prompts with PagedAttention...")
    
    # Generate - PagedAttention handles memory pool automatically
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

def advanced_vllm_config():
    """Advanced vLLM configuration with PagedAttention tuning"""
    
    llm = LLM(
        model="microsoft/DialoGPT-medium",
        
        # PagedAttention memory configuration
        block_size=32,                     # Larger blocks for longer sequences
        max_num_seqs=128,                  # Fewer concurrent sequences
        max_num_batched_tokens=4096,       # Larger batches
        gpu_memory_utilization=0.95,       # Use more GPU memory
        swap_space=2,                      # 2GB CPU swap space
        
        # Other optimizations that work with PagedAttention
        trust_remote_code=True,
        dtype="float16",                   # Use FP16 for memory efficiency
    )
    
    # Test with varying length prompts to show PagedAttention efficiency
    prompts = [
        "Hi",  # Very short
        "Tell me about the weather today and what I should wear",  # Medium
        "Write a detailed story about a space explorer who discovers an ancient alien civilization on a distant planet, including their culture, technology, and the challenges faced during first contact",  # Long
        "Hello there",  # Short again
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=150
    )
    
    print("Testing PagedAttention with variable-length prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs):
        print(f"\nSequence {i+1} (length: {len(output.prompt.split())} words):")
        print(f"Input: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")

def streaming_example():
    """Streaming generation with PagedAttention"""
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model="microsoft/DialoGPT-medium",
        block_size=16,
        max_num_seqs=64,
    )
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=200,
        # Streaming-friendly settings
        top_p=0.9,
    )
    
    prompts = ["Tell me a story about", "The future of technology"]
    
    print("Streaming generation with PagedAttention...")
    
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs):
        print(f"\nStream {i+1}:")
        print(f"Prompt: {output.prompt}")
        print(f"Response: {output.outputs[0].text}")

if __name__ == "__main__":
    print("=== Basic vLLM with PagedAttention ===")
    basic_vllm_example()
    
    print("\n" + "="*50)
    print("=== Advanced PagedAttention Configuration ===")
    advanced_vllm_config()
    
    print("\n" + "="*50)
    print("=== Streaming with PagedAttention ===")
    streaming_example()