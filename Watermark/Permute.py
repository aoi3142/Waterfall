import numpy as np

class Permute:
    def __init__(self, N):
        self.N = N
        self.permutations = {}

    def get_permutation(self, prev_tok, id = None, cache = False):
        assert not (id is None), "id must be provided to permute"
        key = (id, *prev_tok)
        if cache:
            if key in self.permutations:
                permutation = self.permutations[permutation]
            else:
                permutation = np.random.RandomState(key).permutation(self.N)
                self.permutations[permutation] = permutation
        else:
            permutation = np.random.RandomState(key).permutation(self.N)
        return permutation