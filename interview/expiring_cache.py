''' """
Implement a Key-Value Expiring Cache
Create a class ExpiringCache that functions as a key-value store where each entry has a 
time-to-live (TTL) in seconds. 
When a TTL of an entry expires, it should no longer be accessible from the cache.
"""

"""
Implement a Size-Bounded Expiring Cache with Access Tracking
Now, extend the ExpiringCache to create a SizeBoundedExpiringCache with an additional constraint:
1. When the number of entries exceeds max_size, the cache should automatically remove the least recently accessed entry.
2. If an entry is accessed (via get), it should update the access time so that it’s considered recently accessed.
The TTL functionality should still apply, so an entry may expire due to TTL even if it hasn’t been removed due to size.
"""


"""
Implement a Thread-Safe, Asynchronous Expiring LRU Cache
Extend the SizeBoundedExpiringCache with the following requirements:
    1. Thread Safety: Multiple threads should be able to safely call get and set without causing race conditions or data corruption.
        ○ You may use synchronization primitives (threading.Lock, RLock, etc.).
    2. Background Expiry Task (Async):
        ○ Implement a background task that periodically removes expired entries (instead of only cleaning up on get/set).
        ○ This task should be implemented using asyncio, so the cache can run in asynchronous applications.
    3. Graceful Shutdown: Provide a method to stop the background cleanup task when the cache is no longer needed.
""" '''

import time
import asyncio
import threading

class ExpiringCache:
    def __init__(self, max_size):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.background_task = None

    def add(self, key, ttl):
        with self.lock:
            if len(self.cache) >= self.max_size:
                least_accessed_key = min(self.cache.keys(), key=lambda k: self.cache[k]['accessed'])
                print(f"Removing least accessed {least_accessed_key}")
                # remove least accessed
                del self.cache[least_accessed_key]
                
            received_time = time.time()
            expiry_time = received_time + ttl

            request = {
                'expiry': expiry_time,
                'received': received_time,
                'accessed': received_time
            }
            self.cache[key] = request
    
    def get(self, key):
        with self.lock:
            access_time = time.time()
            if key in self.cache:
                self.cache[key]['accessed'] = access_time
                return self.cache[key]
            return None
    
    def remove(self, key):
        with self.lock:
            if key in self.cache:
                print('removed', key)
                del self.cache[key]

    async def check_expired(self):
        while True:
            with self.lock:
                current_time = time.time()
                expired_keys = [k for k in self.cache.keys() if current_time > self.cache[k]['expiry']]
                for k in expired_keys:
                    if k in self.cache:
                        del self.cache[k]

            await asyncio.sleep(1)
            print(f'Cache size {len(self.cache)}, Expired {len(expired_keys)}')

    def start_background(self):
        self.background_task = asyncio.create_task(self.check_expired())
        print("background started")
    
    def stop_background(self):
        if self.background_task:
            self.background_task.cancel()
            print("background stopped")

async def main():
    cache.start_background()
    await asyncio.sleep(10)
    cache.stop_background()

cache = ExpiringCache(4)

cache.add('0', 5)
cache.add('1', 5)
cache.add('2', 5)
cache.add('3', 5)

asyncio.run(main())