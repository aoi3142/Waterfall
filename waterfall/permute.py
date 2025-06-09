import numpy as np
import psutil
from collections import OrderedDict
import gc
from typing import TypeVar, Tuple

T = TypeVar('T')

class LRUCache:
    def __init__(self, capacity: int = 1000) -> None:
        self.cache = OrderedDict()
        self.capacity = capacity
        self.cache_hits : int = 0
        self.cache_misses : int = 0

    def get(self, key: Tuple) -> T | None:
        if key not in self.cache:
            self.cache_misses += 1
            return None
        else:
            self.cache_hits += 1
            # Move the accessed item to the end of the OrderedDict to mark it as recently used.
            self.cache.move_to_end(key)
            return self.cache[key]

    def __str__(self) -> str:
        gc.collect()
        return f"Cache hits: {self.cache_hits}, misses: {self.cache_misses}, rate: {self.cache_hits/(max(self.cache_hits+self.cache_misses, 1)):.2f}"

    def put(self, key: Tuple, value: T) -> None:
        if key in self.cache:
            # Update the value and move it to the end.
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove the first key-value pair which is the least recently used.
            del self.cache[next(iter(self.cache))]

    def clear(self) -> None:
        self.cache.clear()
        gc.collect()

class Permute:
    permutations = LRUCache()
    def __init__(self, N : int = 128000) -> None:
        self.N = N
        self.dtype = np.min_scalar_type(self.N)
        assert self.dtype.kind == 'u', "N must be a positive integer"
        size_per_permutation_in_bytes = N * self.dtype.itemsize
        cache_size = int(psutil.virtual_memory().total * 0.02 / size_per_permutation_in_bytes)  # 2% of total memory
        self.permutations.capacity = cache_size

    def get_permutation(self, prev_tok, id : int, cache : bool = False) -> np.ndarray:
        key = (id, *prev_tok)
        if cache:
            permutation = self.permutations.get(key)
            if permutation is None:
                permutation = np.random.RandomState(key).permutation(self.N).astype(self.dtype)
                self.permutations.put(key, permutation)
        else:
            permutation = np.random.RandomState(key).permutation(self.N).astype(self.dtype)
        return permutation

    def get_unshuffled_indices(self, ids, args) -> dict[int, np.ndarray]:
        key, indices = args
        permutation = np.stack([self.get_permutation(key, id) for id in ids])
        return {k: v for k, v in zip(indices, permutation[:,indices].T)}