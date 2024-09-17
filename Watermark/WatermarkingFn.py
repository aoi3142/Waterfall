class WatermarkingFn:
    def __init__(self, id = 0, k_p = 1, N = 32000, kappa = 1):
        self.id = id
        self.k_p = k_p
        self.N = N
        self.kappa = kappa
        self.phi = None

    def q(self, bins):
        raise NotImplementedError