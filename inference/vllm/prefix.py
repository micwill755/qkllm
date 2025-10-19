from vllm import LLM, SamplingParams

long_prefix = "<a piece of text that is encoded into more than block_size tokens>"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Batch all prompts with prefix for chunked prefill
    prefixed_prompts = [long_prefix + prompt for prompt in prompts]
    outputs = llm.generate(prefixed_prompts, sampling_params)
    
    for output in outputs:
        print(f"Generated: {output.outputs[0].text}")

if __name__ == "__main__":
    main()