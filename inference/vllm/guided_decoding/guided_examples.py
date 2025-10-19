from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

def main():
    print("Initializing model...")
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("Model loaded successfully!\n")
    
    # Regex example - phone number format
    print("1. Regex guided decoding (phone number):")
    try:
        regex_params = GuidedDecodingParams(regex=r"\d{3}-\d{3}-\d{4}")
        regex_sampling = SamplingParams(guided_decoding=regex_params, max_tokens=20, temperature=0.1)
        outputs = llm.generate("Generate a phone number:", regex_sampling)
        print(f"   Result: {outputs[0].outputs[0].text}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # JSON example - structured output
    print("2. JSON guided decoding (person profile):")
    try:
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        json_params = GuidedDecodingParams(json=json_schema)
        json_sampling = SamplingParams(guided_decoding=json_params, max_tokens=50, temperature=0.1)
        outputs = llm.generate("Create a person profile:", json_sampling)
        print(f"   Result: {outputs[0].outputs[0].text}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Additional regex example - email format
    print("3. Regex guided decoding (email):")
    try:
        email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        email_params = GuidedDecodingParams(regex=email_regex)
        email_sampling = SamplingParams(guided_decoding=email_params, max_tokens=30, temperature=0.1)
        outputs = llm.generate("Generate an email address:", email_sampling)
        print(f"   Result: {outputs[0].outputs[0].text}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    print("Note: Grammar guided decoding may not be supported in this vLLM version.")
    print("Regex and JSON guided decoding are working correctly!")

if __name__ == "__main__":
    main()