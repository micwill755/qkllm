import unittest
import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from gpt2_server import GPT2InferenceModel, GPT2Server

class TestGPT2InferenceModel(unittest.TestCase):
    def setUp(self):
        self.model = GPT2InferenceModel()

    def test_properties(self):
        self.assertEqual(self.model.max_context_length, 256)
        self.assertEqual(self.model.max_batch_size, 4)

    @patch('gpt2_server.tiktoken.get_encoding')
    def test_prompt_too_long(self, mock_tokenizer):
        mock_tokenizer.return_value.encode.return_value = [1] * 250  # 250 tokens
        
        with self.assertRaises(ValueError) as context:
            self.model.generate_batch(["very long prompt"], max_tokens=50)
        
        self.assertIn("Prompt too long", str(context.exception))

    def test_concurrent_execution_blocked(self):
        self.model._running = True
        
        with self.assertRaises(RuntimeError) as context:
            self.model.generate_batch(["test"])
        
        self.assertIn("should not be running concurrently", str(context.exception))

    @patch('gpt2_server.tiktoken.get_encoding')
    @patch.object(GPT2InferenceModel, '_generate_single')
    def test_generate_batch_success(self, mock_generate, mock_tokenizer):
        mock_tokenizer.return_value.encode.return_value = [1, 2, 3]
        mock_generate.return_value = "generated text"
        
        result = self.model.generate_batch(["test prompt"], max_tokens=10)
        
        self.assertEqual(result, ["generated text"])
        mock_generate.assert_called_once()

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        result = self.model._softmax(x)
        
        # Check probabilities sum to 1
        self.assertAlmostEqual(np.sum(result), 1.0, places=6)
        # Check highest input gets highest probability
        self.assertEqual(np.argmax(result), 2)

class TestGPT2Server(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.server = GPT2Server(self.loop)

    def tearDown(self):
        self.server.shutdown()
        self.loop.close()

    def test_batch_size_validation(self):
        async def test():
            with self.assertRaises(ValueError) as context:
                await self.server.generate(["prompt"] * 10)  # Exceeds max_batch_size
            
            self.assertIn("exceeds maximum", str(context.exception))
        
        self.loop.run_until_complete(test())

    @patch.object(GPT2Server, '_generate_direct')
    def test_generate_success(self, mock_generate):
        mock_generate.return_value = asyncio.Future()
        mock_generate.return_value.set_result(["generated text"])
        
        async def test():
            result = await self.server.generate(["test prompt"])
            self.assertEqual(result, ["generated text"])
        
        self.loop.run_until_complete(test())

    @patch.object(GPT2Server, '_generate_direct')
    def test_large_batch_processing(self, mock_generate):
        # Mock returns for each sub-batch
        mock_generate.side_effect = [
            asyncio.Future(),
            asyncio.Future()
        ]
        mock_generate.return_value.set_result(["text1", "text2"])
        for future in mock_generate.side_effect:
            future.set_result(["text1", "text2"])
        
        async def test():
            # Test with 8 prompts (should split into 2 batches of 4)
            prompts = [f"prompt{i}" for i in range(8)]
            result = await self.server._process_large_batch(prompts, 10, 1.0)
            self.assertEqual(len(result), 8)
        
        self.loop.run_until_complete(test())

class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        from gpt2_server import app
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')

    @patch('gpt2_server.run_async')
    def test_generate_endpoint_single_prompt(self, mock_run_async):
        mock_run_async.return_value = ["generated text"]
        
        response = self.app.post('/generate', 
                                json={'prompts': 'test prompt', 'max_tokens': 20})
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['prompts'], ['test prompt'])
        self.assertEqual(data['generated_texts'], ['generated text'])

    @patch('gpt2_server.run_async')
    def test_generate_endpoint_multiple_prompts(self, mock_run_async):
        mock_run_async.return_value = ["text1", "text2"]
        
        response = self.app.post('/generate', 
                                json={'prompts': ['prompt1', 'prompt2']})
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data['generated_texts']), 2)

    @patch('gpt2_server.run_async')
    def test_generate_endpoint_error_handling(self, mock_run_async):
        mock_run_async.side_effect = ValueError("Test error")
        
        response = self.app.post('/generate', 
                                json={'prompts': 'test'})
        
        self.assertEqual(response.status_code, 500)
        data = response.get_json()
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()