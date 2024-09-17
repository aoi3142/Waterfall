from Watermark.WatermarkingFn import WatermarkingFn
import numpy as np

class WatermarkingFnSquare(WatermarkingFn):
    def __init__(self, id = 0, k_p = 1, N = 32000, kappa = 1):
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

        self.phis = [None] * (self.k_N*2-1)
        for i in range(self.k_N*2-1):
            k_p = i+1
            if k_p <= self.k_N:
                self.phis[i] = (-1)**(np.floor(np.arange(self.N)*2**k_p/self.N)) * self.kappa
            else:
                k_p -= self.k_N
                self.phis[i] = (-1)**(np.floor(np.arange(self.N)*2**k_p/self.N+0.5)) * self.kappa

        self.phi = self.phis[self.k_p-1]

    def q(self, bins):
        if bins.ndim == 1:
            bins = bins[None,:]
        nomalized_bins = bins / bins.sum(axis = 1)[:, None]
        res = np.zeros((nomalized_bins.shape[0], len(self.phis)))
        for i, k_p in enumerate(self.phis):
            res[:,i] = np.dot(nomalized_bins, k_p/self.kappa)
        return res