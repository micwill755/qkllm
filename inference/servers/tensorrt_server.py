# tensorrt_server.py
import asyncio
import time
from collections import deque
import threading
from typing import List, Dict, Any
import numpy as np
import tiktoken
import sys
import os
from flask import Flask, request, jsonify

class TensorRTInferenceModel:
    def __init__(self, engine_path: str):
        # import tensorrt_llm
        # from tensorrt_llm.runtime import ModelRunner
        # self.runner = ModelRunner.from_dir(engine_path)
        # self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # For now, placeholder implementation
        self.engine_path = engine_path
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self._running = False
        print(f"TensorRT-LLM engine loaded from: {engine_path}")

    @property
    def max_context_length(self) -> int:
        return 2048  # TensorRT-LLM typically supports longer contexts

    @property
    def max_batch_size(self) -> int:
        return 64  # TensorRT-LLM supports larger batches

    def generate_batch(self, prompts: List[str], max_tokens: int = 50, temperature: float = 1.0) -> List[str]:
        if self._running:
            raise RuntimeError("Model should not be running concurrently")
        
        self._running = True
        
        try:
            # TensorRT-LLM batch generation
            # batch_input_ids = [self.tokenizer.encode(prompt) for prompt in prompts]
            # outputs = self.runner.generate(
            #     batch_input_ids,
            #     max_new_tokens=max_tokens,
            #     temperature=temperature,
            #     do_sample=True
            # )
            # return [self.tokenizer.decode(output) for output in outputs]
            
            # Placeholder implementation
            return [f"TensorRT generated response for: {prompt[:50]}..." for prompt in prompts]
        finally:
            self._running = False

class TensorRTServer:
    def __init__(self, loop: asyncio.AbstractEventLoop, engine_path: str):
        self._model = TensorRTInferenceModel(engine_path)
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
        if self._processor_task:
            self._processor_task.cancel()

    async def generate(self, prompts: List[str], max_tokens: int = 50, temperature: float = 1.0) -> List[str]:
        if len(prompts) > self._model.max_batch_size:
            raise ValueError(f"Batch size {len(prompts)} exceeds maximum {self._model.max_batch_size}")
        
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
        results = []
        i = 0
        
        while i < len(prompts):
            batch_end = min(i + self._model.max_batch_size, len(prompts))
            sub_batch = prompts[i:batch_end]
            
            result = await self._generate_direct(sub_batch, max_tokens, temperature)
            results.extend(result)
            i = batch_end
        
        return results

# Flask API
app = Flask(__name__)
loop = asyncio.new_event_loop()

# Initialize with engine path
ENGINE_PATH = os.getenv("TENSORRT_ENGINE_PATH", "/path/to/tensorrt/engine")
tensorrt_server = TensorRTServer(loop, ENGINE_PATH)

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
        results = run_async(tensorrt_server.generate(prompts, max_tokens, temperature))
        return jsonify({
            'prompts': prompts,
            'generated_texts': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'backend': 'tensorrt-llm'})

if __name__ == '__main__':
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tensorrt_server.start())
        loop.run_forever()
    
    threading.Thread(target=run_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8001)  # Different port