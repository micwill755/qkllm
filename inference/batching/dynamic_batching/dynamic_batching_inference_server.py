import asyncio
import threading
import time
from flask import Flask, request, jsonify

class DynamicBatchingInferenceServer:
    def __init__(self):
        self.request_queue = []
        self.lock = threading.Lock()
        self.max_batch_size = 1
        self.batch_wait_time = 1
        self.running = False
        self.loop = None

    def add_request(self, sequence):
        with self.lock:
            received_time = time.time()
            req_idx = len(self.request_queue) + 1
            print(f'add_request: request {req_idx} received {received_time}')
            
            request = {
                'req_idx': req_idx,
                'sequence': sequence,
                'received_time': received_time
            }
            self.request_queue.append(request)
    
    async def _process_batch(self, batch):
        for b in batch:
            process_time = time.time() 
            print(f'_process_batch: processing request {b["req_idx"]} at {process_time}')

    async def process(self):
        while self.running:
            batch = []
            with self.lock:
                current_time = time.time()
                if self.request_queue:
                    # Check if we should process: hit max size OR oldest request expired
                    if len(self.request_queue) >= self.max_batch_size or (current_time - self.request_queue[0]['received_time']) >= self.batch_wait_time:
                        m = min(len(self.request_queue), self.max_batch_size)
                        batch = self.request_queue[:m]
                        self.request_queue = self.request_queue[m:]

            if batch:
                process_time = time.time() 
                print(f'_process: processing batch of size {len(batch)} at {process_time}')
                await self._process_batch(batch)

            await asyncio.sleep(0.01)

    def start_background(self, batch_size=1, batch_wait_time=1):
        if not self.running:
            received_time = time.time()
            print(f'start_background: starting background {received_time}')
            self.max_batch_size = batch_size
            self.batch_wait_time = batch_wait_time
            self.running = True

app = Flask(__name__)
server = DynamicBatchingInferenceServer()

def start_inference_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server.loop = loop
    server.start_background(batch_size=4, batch_wait_time=0.5)
    loop.create_task(server.process())
    loop.run_forever()

threading.Thread(target=start_inference_server, daemon=True).start()
time.sleep(0.1)  # Give thread time to start

@app.route('/add', methods=['POST'])
def predict():
    data = request.json
    sequence = data.get('sequence', [])
    server.add_request(sequence)
    return jsonify({"status": "queued", "message": "Request added to processing queue"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)