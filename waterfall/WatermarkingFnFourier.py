from waterfall.WatermarkingFn import *
import numpy as np
from scipy.fft import rfft
from scipy.sparse import isspmatrix

class WatermarkingFnFourier(WatermarkingFn):
    def __init__(self, id : int = 0, k_p : int = 1, N : int = 32000, kappa : float = 1.) -> None:
        super().__init__(id = id, k_p = k_p, N = N, kappa = kappa)

        self.max_freq = self.N // 2 - 1 if self.N % 2 == 0 else self.N // 2
        self.num_fns = self.max_freq * 2
        assert 0 < self.k_p <= self.num_fns

        if self.k_p <= self.max_freq:
            freq = self.k_p
            self.phi = np.cos(np.arange(self.N) / self.N * 2 * np.pi * freq)
        else:
            freq = self.k_p - self.max_freq
            self.phi = np.sin(np.arange(self.N) / self.N * 2 * np.pi * freq)
        self.phi *= self.kappa

        self.scaling_factor = 1

    def _q(self, bins : np.ndarray | spmatrix, k_p : List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isspmatrix(bins):
            bins = bins.todense()
        q = rfft(bins, axis=-1)
        q = q[:, 1:-1] if self.N % 2 == 0 else q[:, 1:]
        q = q.astype(np.complex64)
        q = np.concatenate((np.real(q), -np.imag(q)), axis=1)
        q *= self.scaling_factor
        k_p_strength = q[:,np.array(k_p)-1]
        k_p_ranking = ((q[...,None,:] > k_p_strength[...,None]).sum(axis=-1)).astype(self.dtype)
        k_p_extracted = (np.argmax(q, axis=-1) + 1).astype(self.dtype)
        return k_p_strength, k_p_ranking, k_p_extracted