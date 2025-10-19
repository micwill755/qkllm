#!/usr/bin/env python3
"""
Chunked Prefill Demo with vLLM
==============================

This script demonstrates how chunked prefill works in vLLM to enable
request interleaving and improve serving performance.

Key Concepts:
- Large requests are split into chunks
- Small requests get served immediately
- Mixed batches improve GPU utilization
"""

import asyncio
import time
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


class ChunkedPrefillDemo:
    def __init__(self, model_name: str = "facebook/opt-125m"):
        # Configure vLLM with chunked prefill
        self.engine_args = AsyncEngineArgs(
            model=model_name,
            # Enable chunked prefill
            enable_chunked_prefill=True,
            # Set chunk size (tokens per chunk)
            max_num_batched_tokens=2048,  # Total batch size
            max_num_seqs=256,             # Max concurrent requests
            # Memory optimization
            gpu_memory_utilization=0.9,
            # Disable unnecessary features for demo
            disable_log_stats=False,
        )
        
        self.engine = None
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )
    
    async def start_engine(self):
        """Start the async vLLM engine"""
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        print("✅ vLLM engine started with chunked prefill enabled")
    
    async def generate_response(self, prompt: str, request_id: str) -> Dict[str, Any]:
        """Generate response for a single request"""
        start_time = time.time()
        
        # Submit request to vLLM
        results_generator = self.engine.generate(
            prompt, 
            self.sampling_params, 
            request_id
        )
        
        # Collect results
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        end_time = time.time()
        
        return {
            "request_id": request_id,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "response": final_output.outputs[0].text if final_output else "",
            "latency": end_time - start_time,
            "prompt_tokens": len(prompt.split()),
        }
    
    async def demo_traditional_vs_chunked(self):
        # Create test requests - mix of large and small
        requests = [
            {
                "id": "large_1",
                "prompt": self.create_long_prompt(800),  # ~800 tokens
                "type": "LARGE"
            },
            {
                "id": "small_1", 
                "prompt": "What is 2+2?",
                "type": "SMALL"
            },
            {
                "id": "small_2",
                "prompt": "Hello, how are you?", 
                "type": "SMALL"
            },
            {
                "id": "small_3",
                "prompt": "Translate 'hello' to French",
                "type": "SMALL"
            },
            {
                "id": "large_2",
                "prompt": self.create_long_prompt(600),  # ~600 tokens
                "type": "LARGE"
            },
        ]
        
        print(f"\nProcessing {len(requests)} requests:")
        for req in requests:
            tokens = len(req["prompt"].split())
            print(f"  - {req['id']}: {req['type']} ({tokens} tokens)")
        
        # Process all requests concurrently
        print(f"\nStarting concurrent processing with chunked prefill...")
        start_time = time.time()
        
        # Submit all requests at once
        tasks = [
            self.generate_response(req["prompt"], req["id"]) 
            for req in requests
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Display results
        print(f"\nRESULTS (Total time: {total_time:.2f}s)")
        
        small_latencies = []
        large_latencies = []
        
        for result in sorted(results, key=lambda x: x["latency"]):
            req_type = "LARGE" if result["prompt_tokens"] > 100 else "SMALL"
            
            print(f"{result['request_id']:>8} | {req_type:>5} | "
                  f"{result['prompt_tokens']:>4} tokens | "
                  f"{result['latency']:>6.2f}s | "
                  f"{result['prompt']}")
            
            if req_type == "SMALL":
                small_latencies.append(result["latency"])
            else:
                large_latencies.append(result["latency"])
        
        # Summary statistics
        print("\n📈 PERFORMANCE SUMMARY:")
        print(f"  Small requests avg latency: {sum(small_latencies)/len(small_latencies):.2f}s")
        print(f"  Large requests avg latency: {sum(large_latencies)/len(large_latencies):.2f}s")
        print(f"  Total throughput: {len(requests)/total_time:.1f} req/s")
    
    def create_long_prompt(self, target_tokens: int) -> str:
        base_text = """
        Please analyze the following business scenario in detail. Consider all aspects 
        including market conditions, competitive landscape, financial implications, 
        operational challenges, strategic opportunities, risk factors, and potential 
        outcomes. Provide a comprehensive analysis with specific recommendations.
        
        The scenario involves a technology startup that has developed an innovative 
        AI-powered solution for healthcare diagnostics. The company has completed 
        initial testing and is now considering various go-to-market strategies.
        """
        
        # Repeat and expand to reach target token count
        repeated_text = base_text
        while len(repeated_text.split()) < target_tokens:
            repeated_text += base_text
        
        # Trim to approximate target
        words = repeated_text.split()[:target_tokens]
        return " ".join(words)
    
    async def monitor_engine_stats(self):
        if hasattr(self.engine, 'get_model_config'):
            print("\n📊 Engine Configuration:")
            print(f"  Max batch tokens: {self.engine_args.max_num_batched_tokens}")
            print(f"  Max sequences: {self.engine_args.max_num_seqs}")
            print(f"  Chunked prefill: {self.engine_args.enable_chunked_prefill}")


async def main():
    print("🔧 Initializing Chunked Prefill Demo...")
    
    demo = ChunkedPrefillDemo()
    
    try:
        # Start the engine
        await demo.start_engine()
        # Show configuration
        await demo.monitor_engine_stats()
        # Run the main demo
        await demo.demo_traditional_vs_chunked()
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Cleanup
        if demo.engine:
            print("\n🧹 Cleaning up...")


def simple_sync_demo():    
    try:
        # Initialize vLLM with chunked prefill
        llm = LLM(
            model="facebook/opt-125m",
            # Enable chunked prefill
            enable_chunked_prefill=True,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.8,
        )
        
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        
        # Test prompts - mix of sizes
        prompts = [
            "What is AI?",  # Small
            "Explain machine learning in detail with examples and applications",  # Medium
            "Hello",  # Small
            "How does chunked prefill work in large language models?",  # Medium
        ]
        
        print(f"\nProcessing {len(prompts)} prompts with chunked prefill...")
        
        start_time = time.time()
        
        # Generate responses (vLLM handles chunking internally)
        outputs = llm.generate(prompts, sampling_params)
        
        total_time = time.time() - start_time
        
        # Display results
        print(f"\nResults (Total: {total_time:.2f}s):")
        print("-" * 50)
        
        for i, output in enumerate(outputs):
            prompt_len = len(output.prompt.split())
            response = output.outputs[0].text.strip()
            
            print(f"Prompt {i+1} ({prompt_len:2d} tokens): {output.prompt[:30]}...")
            print(f"Response: {response[:60]}...")
            print()
        
        print(f"Chunked prefill enabled efficient processing of mixed request sizes!")
        
    except Exception as e:
        print(f"Error in simple demo: {e}")


if __name__ == "__main__":
    print("🎯 Chunked Prefill Demo with vLLM")
    print("Choose demo mode:")
    print("1. Simple synchronous demo")
    print("2. Advanced async demo with detailed analysis")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        simple_sync_demo()
    else:
        # Run async demo
        asyncio.run(main())