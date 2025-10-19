import time
import asyncio

class BatchInferenceServer:
    def __init__(self):
        self.request_queue = []
        self.lock = asyncio.Lock()
        self.background = None
        self.background_interval = 1
        self.max_batch_size = 1
        self.batch_wait_time = 1
        self.running = False

    async def add_request(self, sequence):
        async with self.lock:
            received_time = time.time()
            req_idx = len(self.request_queue) + 1
            print (f'add_request: request {req_idx} received {received_time}')
            
            request = {
                'req_idx': req_idx,
                'sequence': sequence,
                'received_time': received_time
            }

            self.request_queue.append(request)
    
    async def _process_batch(self, batch):
        for b in batch:
            # process sequence
            process_time = time.time() 
            print (f'_process_batch: processing request {process_time}')

    async def _process(self):
        while self.running:
            batch = []

            async with self.lock:
                batch = self.request_queue[:self.batch_size]  
                self.request_queue = self.request_queue[self.batch_size:]
            
            # process outside of lock
            process_time = time.time() 
            print (f'_process: processing batch of size {len(batch)} at {process_time}')
            if batch:
                await self._process_batch(batch)

            await asyncio.sleep(self.background_interval)

    def start_background(self, interval=1, batch_size=1, batch_wait_time=1):
        if not self.running:
            received_time = time.time()
            print (f'start_background: starting background {received_time}')
            self.background_interval = interval
            self.batch_size = batch_size
            self.batch_wait_time = batch_wait_time
            self.background = asyncio.create_task(self._process())
            self.running = True
    
    def stop_background(self):
        if self.running and self.background:
            received_time = time.time()
            print (f'stop_background: stopping background {received_time}')
            self.running = False
            self.background.cancel()
            self.background = None

async def main():
    server = BatchInferenceServer()
    await server.add_request([10, 1, 532, 2])
    server.start_background(1, 5)
    await server.add_request([10, 1, 532, 2])
    await server.add_request([10, 1, 532, 2])
    await asyncio.sleep(100)

asyncio.run(main())