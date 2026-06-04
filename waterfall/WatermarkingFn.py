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

    def _q(self, avg_cumulative_token_distribution : np.ndarray | spmatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def stats_from_q(self, q : np.ndarray, k_p : List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_p_strength = q[:,np.array(k_p)-1]
        k_p_ranking = ((q[...,None,:] > k_p_strength[...,None]).sum(axis=-1)).astype(self.dtype)
        k_p_extracted = (np.argmax(q, axis=-1) + 1).astype(self.dtype)
        return k_p_strength, k_p_ranking, k_p_extracted
    
    def stats_from_avg_cumulative_token_distribution(self, avg_cumulative_token_distribution : np.ndarray | spmatrix, k_p : List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self._q(avg_cumulative_token_distribution)
        return self.stats_from_q(q, k_p)

    def q(self,
          cumulative_token_counts : np.ndarray | spmatrix,
          k_p : List[int],   # If set, only return the k_p-th element of the dot product and its ranking
          batch : int = 2**8,
          use_tqdm : bool = False,
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cumulative_token_counts.ndim == 1:
            cumulative_token_counts = cumulative_token_counts[None,:]
        res = []
        cumulative_token_counts_sum = cumulative_token_counts.sum(axis=1).reshape(-1,1)
        cumulative_token_counts_sum[cumulative_token_counts_sum == 0] = 1
        batch_range = range(0, cumulative_token_counts.shape[0], batch)
        # Lazy evaluation of the batches to save memory, and parallelize if there are many batches
        batched = (cumulative_token_counts[i:i+batch] / cumulative_token_counts_sum[i:i+batch] for i in batch_range)
        use_mp = len(batch_range) > 4
        if use_mp:
            p = Pool(len(os.sched_getaffinity(0))-1)
            pool_map = p.imap
        else:
            pool_map = map
        res = pool_map(partial(self.stats_from_avg_cumulative_token_distribution, k_p=k_p), batched)
        if use_tqdm:
            res_ = []
            with tqdm(total=cumulative_token_counts.shape[0], desc="Calculating dot product") as pbar:
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