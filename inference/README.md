# GPT-2 Inference Server

A concurrent inference server for the custom GPT-2 implementation.

## Features

- **Concurrent request handling**: Queues and processes requests asynchronously
- **Batch processing**: Handles multiple prompts in one request
- **Resource limits**: Respects context length and batch size constraints
- **Thread-safe**: Uses locks to prevent concurrent model execution

## Usage

### Start the server:
```bash
cd inference
pip install -r requirements.txt
python gpt2_server.py
```

### Run tests:
```bash
# Install dependencies first
cd "/Users/michaelwilliams/Documents/code/deep learning/llm/gpt2/python/inference"
pip install -r requirements.txt

# Unit tests
python -m pytest test_gpt2_server.py -v

# Integration tests (requires server running in separate terminal)
# Terminal 1: Start server
python gpt2_server.py
# Terminal 2: Run integration tests
python -m pytest test_integration.py -v

# All tests
python -m pytest -v

# Alternative with unittest
python test_gpt2_server.py
```

### Make requests:
```bash
# Single prompt
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": "The future of AI is", "max_tokens": 30, "temperature": 0.8}'

# Multiple prompts
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["The future is", "AI will"], "max_tokens": 30}'

# Health check
curl http://localhost:8000/health
```

## API Endpoints

### POST /generate
Generate text completions for given prompts.

**Request:**
```json
{
  "prompts": ["prompt1", "prompt2"],  // string or array of strings
  "max_tokens": 50,                   // optional, default: 50
  "temperature": 1.0                  // optional, default: 1.0
}
```

**Response:**
```json
{
  "prompts": ["prompt1", "prompt2"],
  "generated_texts": ["completion1", "completion2"]
}
```

### GET /health
Check server health status.

**Response:**
```json
{
  "status": "healthy"
}
```