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

    def _q(self, avg_cumulative_token_distribution : np.ndarray | spmatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isspmatrix(avg_cumulative_token_distribution):
            avg_cumulative_token_distribution = avg_cumulative_token_distribution.todense()
        q = rfft(avg_cumulative_token_distribution, axis=-1)
        q = q[:, 1:-1] if self.N % 2 == 0 else q[:, 1:]
        q = q.astype(np.complex64)
        q = np.concatenate((np.real(q), -np.imag(q)), axis=1)
        q *= self.scaling_factor
        return q