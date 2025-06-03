from waterfall.WatermarkingFn import *
import numpy as np

class WatermarkingFnSquare(WatermarkingFn):
    def __init__(self, id : int = 0, k_p : int = 1, N : int = 32000, kappa : float = 1.) -> None:
        super().__init__(id = id, k_p = k_p, N = N, kappa = kappa)

        self.k_N = 0

        if (self.N%2)==1:
            self.k_N = 1
            self.phi = np.ones(self.N) * self.kappa
            self.phi[self.N//2:] *= -1
            self.phis = [self.phi]
            return

        N = self.N
        while not (N & 0b1):
            N >>= 1
            self.k_N += 1
        assert (k_p > 0) and (k_p < (self.k_N * 2)), f"k_p {k_p} larger than available number of fns {self.k_N*2-1}"

        self.phis = np.empty((self.k_N*2-1, self.N), dtype=np.float32)
        for i in range(self.k_N*2-1):
            k_p = i+1
            if k_p <= self.k_N:
                self.phis[i] = (-1)**(np.floor(np.arange(self.N)*2**k_p/self.N)) * self.kappa
            else:
                k_p -= self.k_N
                self.phis[i] = (-1)**(np.floor(np.arange(self.N)*2**k_p/self.N+0.5)) * self.kappa

        self.phi = self.phis[self.k_p-1]

        self.scaling_factor = 1 / self.kappa

    def _q(self, bins : np.ndarray | spmatrix, k_p : List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = bins.dot(self.phis.T)
        q *= self.scaling_factor
        k_p_strength = q[:,np.array(k_p)-1]
        k_p_ranking = ((q[...,None,:] > k_p_strength[...,None]).sum(axis=-1)).astype(self.dtype)
        k_p_extracted = (np.argmax(q, axis=-1) + 1).astype(self.dtype)
        return k_p_strength, k_p_ranking, k_p_extracted