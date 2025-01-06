import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from typing import List, Tuple
from scipy.sparse import spmatrix

class WatermarkingFn:
    def __init__(self, id = 0, k_p = 1, N = 32000, kappa = 1):
        self.id = id
        self.k_p = k_p
        self.N = N
        self.kappa = kappa
        self.phi = None

    def _q(self, bins : np.ndarray | spmatrix, k_p : List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def q(self, 
          bins : np.ndarray | spmatrix, 
          k_p : List[int],   # If set, only return the k_p-th element of the dot product and its ranking
          batch : int = 2**8, 
          use_tqdm : bool = False, 
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if bins.ndim == 1:
            bins = bins[None,:]
        res = []
        bins_sum = bins.sum(axis=1).reshape(-1,1)
        bins_sum[bins_sum == 0] = 1
        batch_range = range(0, bins.shape[0], batch)
        if use_tqdm:
            batch_range = tqdm(batch_range, desc="Preparing batches for q", leave=False)
        batched = [bins[i:i+batch] / bins_sum[i:i+batch] for i in batch_range]
        with Pool() as p:
            res = p.imap(partial(self._q, k_p=k_p), batched)
            if use_tqdm:
                res_ = []
                with tqdm(total=bins.shape[0], desc="Calculating dot product", leave=False) as pbar:
                    for r in res:
                        res_.append(r)
                        pbar.update(len(r))
                res = res_
            else:
                res = list(res)
        k_p_strength, k_p_ranking, k_p_extracted = list(map(np.concatenate, zip(*res)))
        return k_p_strength, k_p_ranking, k_p_extracted