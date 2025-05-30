import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os
from functools import partial
from typing import List, Tuple
from scipy.sparse import spmatrix

class WatermarkingFn:
    def __init__(self, id : int = 0, k_p : int = 1, N : int = 32000, kappa : float = 1.) -> None:
        self.id = id
        self.k_p = k_p
        self.N = N
        self.kappa = kappa
        self.phi = None
        self.dtype = np.min_scalar_type(self.N)

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
        batched = (bins[i:i+batch] / bins_sum[i:i+batch] for i in batch_range)
        use_mp = len(batch_range) > 4
        if use_mp:
            p = Pool(len(os.sched_getaffinity(0))-1)
            pool_map = p.imap
        else:
            pool_map = map
        res = pool_map(partial(self._q, k_p=k_p), batched)
        if use_tqdm:
            res_ = []
            with tqdm(total=bins.shape[0], desc="Calculating dot product") as pbar:
                for r in res:
                    res_.append(r)
                    pbar.update(len(r[0]))
            res = res_
        else:
            res = list(res)
        if use_mp:
            p.close()
            p.join()
        k_p_strength, k_p_ranking, k_p_extracted = list(map(np.concatenate, zip(*res)))
        return k_p_strength, k_p_ranking, k_p_extracted