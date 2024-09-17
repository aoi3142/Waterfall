from Watermark.WatermarkingFn import WatermarkingFn
import numpy as np

class WatermarkingFnFourier(WatermarkingFn):
    def __init__(self, id = 0, k_p = 1, N = 32000, kappa = 1):
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

        self.scaling_factor = np.ones((self.N//2 - 1)*2)

    def q(self, bins):
        if bins.ndim == 1:
            bins = bins[None,:]
        fft_res = np.fft.rfft(bins / bins.sum(axis=1))[:,1:-1]
        fft_res = np.concatenate((np.real(fft_res), np.imag(fft_res)), axis=1)
        fft_res *= self.scaling_factor

        return fft_res