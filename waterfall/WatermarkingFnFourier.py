from waterfall.WatermarkingFn import *
import numpy as np
from scipy.fft import rfft
from scipy.sparse import isspmatrix

class WatermarkingFnFourier(WatermarkingFn):
    def __init__(self, id : int = 0, k_p : int = 1, N : int = 32000, kappa : float = 1.) -> None:
        super().__init__(id = id, k_p = k_p, N = N, kappa = kappa)

        freq = self.k_p
        assert (freq > 0) and (freq < self.N), f"k_p must be 0<k_p<{self.N}, value provided is {freq}"

        half_N = int(self.N/2)
        if freq <= half_N:
            self.phi = np.cos(np.arange(self.N)/self.N*2*np.pi*freq)
        else:
            freq -= half_N
            self.phi = np.sin(np.arange(self.N)/self.N*2*np.pi*freq)
        self.phi *= self.kappa

        self.scaling_factor = 1

    def _q(self, bins : np.ndarray | spmatrix, k_p : List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isspmatrix(bins):
            bins = bins.todense()
        q = rfft(bins, axis=-1)[:,1:-1].astype(np.complex64)
        q = np.concatenate((np.real(q), np.imag(q)), axis=1)
        q *= self.scaling_factor
        k_p_strength = q[:,np.array(k_p)-1]
        k_p_ranking = ((q[...,None,:] > k_p_strength[...,None]).sum(axis=-1)).astype(self.dtype)
        k_p_extracted = (np.argmax(q, axis=-1) + 1).astype(self.dtype)
        return k_p_strength, k_p_ranking, k_p_extracted