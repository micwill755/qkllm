import onnxruntime as ort
from vllm import SamplingParams
import numpy as np
import tiktoken

class ONNXEngine:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        
    def generate(self, prompts, sampling_params):
        results = []
        for prompt in prompts:
            tokens = self._tokenize(prompt)
            outputs = self.session.run(None, {"input_ids": tokens})
            text = self._decode(outputs[0])
            results.append(type('Output', (), {'prompt': prompt, 'outputs': [type('Text', (), {'text': text})()]})())
        return results
    
    def _tokenize(self, text):
        tokenizer = tiktoken.get_encoding("gpt2")
        return np.array([tokenizer.encode(text)[:512]], dtype=np.int64)
    
    def _decode(self, tokens):
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = np.clip(tokens.flatten(), 0, tokenizer.n_vocab - 1).astype(int)
        return tokenizer.decode(tokens)

# Usage
engine = ONNXEngine("../../model/python/model/moe/deepseek/pytorch/deepseek_v3.onnx")
# right now the deep seek model is only taking 8 tokens
outputs = engine.generate(["Hello llm my question is can you"], SamplingParams())
print(outputs[0].outputs[0].text)

