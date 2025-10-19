# gpt2_server.py
import asyncio
import time
from collections import deque
import threading
from typing import List, Dict, Any
import numpy as np
import tiktoken
import sys
import os

# Add parent directory to path to import GPT2 modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'model'))
from model.gpt2 import GPT2Model, GPT_CONFIG_124M

class GPT2InferenceModel:
    def __init__(self):
        self.model = GPT2Model(GPT_CONFIG_124M)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self._running = False

    @property
    def max_context_length(self) -> int:
        return GPT_CONFIG_124M["context_length"]

    @property
    def max_batch_size(self) -> int:
        return 4  # Process up to 4 requests concurrently

    def generate_batch(self, prompts: List[str], max_tokens: int = 50, temperature: float = 1.0) -> List[str]:
        if self._running:
            raise RuntimeError("Model should not be running concurrently")
        
        self._running = True
        
        try:
            # Tokenize all prompts
            input_batches = []
            for prompt in prompts:
                tokens = self.tokenizer.encode(prompt)
                if len(tokens) + max_tokens > self.max_context_length:
                    raise ValueError(f"Prompt too long: {len(tokens)} tokens")
                input_batches.append(np.array([tokens]))
            
            results = []
            for input_ids in input_batches:
                # Generate for each sequence (your model doesn't support batching yet)
                generated = self._generate_single(input_ids, max_tokens, temperature)
                results.append(generated)
            
            return results
        finally:
            self._running = False

    def _generate_single(self, input_ids: np.ndarray, max_tokens: int, temperature: float) -> str:
        original_length = input_ids.shape[1]
        
        for _ in range(max_tokens):
            # Forward pass
            logits = self.model.forward(input_ids)
            
            # Get next token logits (last position)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probs = self._softmax(next_token_logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            # Append to sequence
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
            
            # Stop if context length exceeded
            if input_ids.shape[1] >= self.max_context_length:
                break
        
        # Decode only the generated part
        generated_ids = input_ids[0, original_length:]
        return self.tokenizer.decode(generated_ids.tolist())

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class GPT2Server:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._model = GPT2InferenceModel()
        self._loop = loop
        self._request_queue = deque()
        self._lock = threading.Lock()
        self._running = True
        self._model_lock = None
        self._processor_task = None
    
    async def start(self):
        self._model_lock = asyncio.Lock()
        self._processor_task = asyncio.create_task(self._process_requests())

    def shutdown(self):
        self._running = False
        self._processor_task.cancel()

    async def generate(self, prompts: List[str], max_tokens: int = 50, temperature: float = 1.0) -> List[str]:
        # Validate input
        if len(prompts) > self._model.max_batch_size:
            raise ValueError(f"Batch size {len(prompts)} exceeds maximum {self._model.max_batch_size}")
        
        # Queue request for serialized processing
        future = asyncio.Future()
        with self._lock:
            self._request_queue.append((prompts, max_tokens, temperature, future))
        
        return await future

    async def _generate_direct(self, prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
        if self._model_lock is None:
            self._model_lock = asyncio.Lock()
        async with self._model_lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._model.generate_batch, prompts, max_tokens, temperature
            )

    async def _process_requests(self):
        while self._running:
            request = None
            
            with self._lock:
                if self._request_queue:
                    request = self._request_queue.popleft()
            
            if request:
                prompts, max_tokens, temperature, future = request
                try:
                    if not future.cancelled():
                        result = await self._process_large_batch(prompts, max_tokens, temperature)
                        future.set_result(result)
                except Exception as e:
                    if not future.cancelled():
                        future.set_exception(e)
            else:
                await asyncio.sleep(0.001)

    async def _process_large_batch(self, prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
        # Break large batches into smaller ones
        results = []
        i = 0
        
        while i < len(prompts):
            # Take up to max_batch_size prompts
            batch_end = min(i + self._model.max_batch_size, len(prompts))
            sub_batch = prompts[i:batch_end]
            
            result = await self._generate_direct(sub_batch, max_tokens, temperature)
            results.extend(result)
            i = batch_end
        
        return results

# Flask API wrapper
from flask import Flask, request, jsonify

app = Flask(__name__)
loop = asyncio.new_event_loop()
gpt2_server = GPT2Server(loop)

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompts = data.get('prompts', [])
    if isinstance(prompts, str):
        prompts = [prompts]
    
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 1.0)
    
    try:
        results = run_async(gpt2_server.generate(prompts, max_tokens, temperature))
        return jsonify({
            'prompts': prompts,
            'generated_texts': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Start asyncio loop in background thread
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(gpt2_server.start())
        loop.run_forever()
    
    threading.Thread(target=run_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8000)