import numpy as np
import psutil
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return None
        else:
            # Move the accessed item to the end of the OrderedDict to mark it as recently used.
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update the value and move it to the end.
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove the first key-value pair which is the least recently used.
            self.cache.popitem(last=False)

class Permute:
    permutations = LRUCache()
    def __init__(self, N):
        self.N = N
        cache_size = int(psutil.virtual_memory().total / 100 * 0.2 / N / 8) # 10% of memory: vocab_size * 8 bytes per long
        self.permutations.capacity = cache_size

    def get_permutation(self, prev_tok, id = None, cache = False):
        assert not (id is None), "id must be provided to permute"
        key = (id, *prev_tok.cpu().numpy())
        if cache:
            permutation = self.permutations.get(key)
            if permutation is None:
                permutation = np.random.RandomState(key).permutation(self.N)
                self.permutations.put(key, permutation)
        else:
            permutation = np.random.RandomState(key).permutation(self.N)
        return permutation