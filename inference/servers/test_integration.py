import asyncio
import threading
import time
import requests
import pytest
from gpt2_server import app, loop, gpt2_server

class TestIntegration:
    @classmethod
    def setup_class(cls):
        """Start the server in a separate thread for integration tests"""
        def run_server():
            def run_loop():
                asyncio.set_event_loop(loop)
                loop.run_forever()
            
            threading.Thread(target=run_loop, daemon=True).start()
            app.run(host='127.0.0.1', port=8001, debug=False, use_reloader=False)
        
        cls.server_thread = threading.Thread(target=run_server, daemon=True)
        cls.server_thread.start()
        time.sleep(2)  # Wait for server to start
        cls.base_url = "http://127.0.0.1:8001"

    def test_health_check(self):
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_single_prompt_generation(self):
        payload = {
            "prompts": "The future of AI",
            "max_tokens": 5,
            "temperature": 1.0
        }
        
        response = requests.post(f"{self.base_url}/generate", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "prompts" in data
        assert "generated_texts" in data
        assert len(data["generated_texts"]) == 1

    def test_multiple_prompts_generation(self):
        payload = {
            "prompts": ["Hello", "World"],
            "max_tokens": 3
        }
        
        response = requests.post(f"{self.base_url}/generate", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["generated_texts"]) == 2

    def test_concurrent_requests(self):
        """Test that multiple concurrent requests are handled properly"""
        import concurrent.futures
        
        def make_request():
            payload = {"prompts": "Test", "max_tokens": 2}
            return requests.post(f"{self.base_url}/generate", json=payload)
        
        # Make 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            responses = [future.result() for future in futures]
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert len(response.json()["generated_texts"]) == 1

if __name__ == "__main__":
    pytest.main([__file__])