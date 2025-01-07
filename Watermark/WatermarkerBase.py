from transformers import LogitsProcessor
import numpy as np
import torch
from .Permute import Permute
from .WatermarkingFn import WatermarkingFn
from scipy.sparse import vstack
from tqdm import tqdm
import gc
import time
from scipy.sparse import dok_matrix
from typing import List, Tuple
import torch
from multiprocessing import Pool
import os
from collections import defaultdict

class PerturbationProcessor(LogitsProcessor):
    def __init__(self, 
                 N = 32000,     # Vocab size
                 id = 0,        # Watermark ID
                 ):

        self.id = id
        self.N = N
        self.init_token_count = None
        self.phi = np.ones(N)
        self.n_gram = None

        self.skip_watermark = False

        self.permute = Permute(self.N)

    def reset(self, n_gram = 2):
        self.n_gram = n_gram
        self.init_token_count = None
        if np.allclose(self.phi,np.median(self.phi)):
            self.skip_watermark = True
            print(f"Generating without watermark as watermarking function is flat")
        else:
            self.skip_watermark = False

    def set_phi(self, phi):
        self.phi = phi

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.skip_watermark:
            return scores

        if self.init_token_count is None:
            self.init_token_count = input_ids.shape[1]

        # Insufficient tokens generated for n-gram
        if self.init_token_count + self.n_gram - 1 > input_ids.shape[1]:
            return scores

        prev_tokens = input_ids[:,-self.n_gram+1:].cpu().numpy()
        permutations = [self.permute.get_permutation(prev_tokens[i,:], self.id, cache=True) for i in range(prev_tokens.shape[0])]
        scores[:,:self.N] += torch.tensor(self.phi[permutations], device=scores.device)
        return scores

class Watermarker:
    def __init__(self, model, tokenizer, id = 0, kappa = 6, k_p = 1, n_gram = 2, watermarkingFnClass = None):
        assert kappa >= 0, f"kappa must be >= 0, value provided is {kappa}"

        self.model = model
        self.tokenizer = tokenizer
        self.id = id
        self.k_p = k_p
        self.n_gram = n_gram
        self.kappa = kappa

        self.N = self.tokenizer.vocab_size
        self.logits_processor = PerturbationProcessor(N = self.N, id = id)

        self.watermarking_fn: WatermarkingFn = watermarkingFnClass(id = id, k_p = k_p, N = self.N, kappa = kappa)
        self.phi = self.watermarking_fn.phi

        self.logits_processor.set_phi(self.phi)

    def generate(
            self, 
            prompt = None, 
            n_gram = None, 
            max_new_tokens = 1000, 
            return_text=True, 
            return_tokens=False, 
            return_scores=False, 
            do_sample=True, 
            **kwargs
            ) -> List[str] | dict:
        if n_gram is None:
            n_gram = self.n_gram
        tokd_input = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            self.logits_processor.reset(n_gram)
            output = self.model.generate(
                **tokd_input, 
                max_new_tokens=max_new_tokens, 
                do_sample=do_sample,
                logits_processor=([self.logits_processor] if self.kappa > 0 else None),
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
                )
        output = output[:,tokd_input["input_ids"].shape[-1]:].cpu()

        return_dict = {}

        if return_scores:
            cumulative_token_count = self.get_cumulative_token_count(self.id, output, n_gram = n_gram, return_dense=False)[0]
            q_score, _, _ = self.watermarking_fn.q(cumulative_token_count, k_p = [self.k_p], use_tqdm=False)
            return_dict["q_score"] = q_score[:,0]

        if return_tokens:
            return_dict["tokens"] = output

        if return_text:
            decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            decoded_output = [i.strip() for i in decoded_output]
            return_dict["text"] = decoded_output

        if len(output) == 1:
            for k, v in return_dict.items():
                return_dict[k] = v[0]

        if return_text and len(return_dict) == 0:
            return decoded_output

        return return_dict

    def get_cumulative_token_count(
            self, 
            ids : List[int] | int,
            all_tokens : List[torch.Tensor] | torch.Tensor | List[np.ndarray] | np.ndarray | List[List[int]] | List[int], 
            n_gram : int = 2, 
            return_unshuffled_indices : bool = False, 
            use_tqdm : bool = False, 
            return_dense : bool = True
            ) -> List[dok_matrix] | List[np.ndarray] | Tuple[List[dok_matrix], List[List[np.ndarray]]] | Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        if isinstance(ids, int):
            ids = [ids]
        if isinstance(all_tokens[0], int) or (isinstance(all_tokens, (np.ndarray, torch.Tensor)) and all_tokens.ndim == 1):
            all_tokens = [all_tokens]
        all_tokens = list(map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, all_tokens))
        max_length = max(map(len, all_tokens))
        window = n_gram - 1
        cumulative_token_count = [dok_matrix((len(all_tokens), self.N), dtype=np.min_scalar_type(max_length)) for _ in ids]
        unshuffled_indices: List[List[np.ndarray]] = [[np.empty(max(len(i)-1, 0), dtype=np.uint32) for i in all_tokens] for _ in ids]

        # Collect all unique seeds for psuedo-random number generation
        key_index_dict = defaultdict(set)
        for i, tokens in enumerate(all_tokens):
            for j in range(window, len(tokens)):
                prev_token = tokens[j-window:j]
                t = tokens[j]
                if t >= self.N:
                    continue
                for k, id in enumerate(ids):
                    key = (id, *prev_token)
                    key_index_dict[key].add(t)

        # Generate permutations for all unique seeds
        with Pool(len(os.sched_getaffinity(0))-1) as p:
            permutations = p.imap(self.logits_processor.permute.get_unshuffled_indices, key_index_dict.items(), chunksize=1000)
            if use_tqdm:
                permutations = tqdm(permutations, total=len(key_index_dict), mininterval=5, desc="Getting permutations")
            for k, value in zip(key_index_dict, permutations):
                key_index_dict[k] = value

        # Assign indices to unshuffled_indices
        for i, tokens in enumerate(all_tokens):
            for j in range(window, len(tokens)):
                prev_token = tokens[j-window:j]
                t = tokens[j]
                if t >= self.N:
                    continue
                for k, id in enumerate(ids):
                    key = (id, *prev_token)
                    x = key_index_dict[key][t]
                    unshuffled_indices[k][i][j-window] = x
                    cumulative_token_count[k][i,x] += 1

        if return_dense:
            cumulative_token_count = list(map(lambda x: x.toarray(), cumulative_token_count))
        else:
            cumulative_token_count = list(map(lambda x: x.tocsr(), cumulative_token_count))

        if return_unshuffled_indices:
            return cumulative_token_count, unshuffled_indices
        return cumulative_token_count

    def verify(
            self, 
            text : str | List[str], 
            id: int | List[int] | None = None, 
            k_p : int | List[int] | None = None, 
            return_ranking : bool = False,
            return_extracted_k_p : bool = False, 
            return_counts : bool = False, 
            return_unshuffled_indices : bool = False, 
            use_tqdm : bool = False
            ) -> np.ndarray | dict:
        begin_time = time.time()

        if id is None:
            id = self.id

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        tokens = [np.array(self.tokenizer.encode(text, add_special_tokens=False), dtype=np.uint32) for text in texts]

        if isinstance(id, int):
            ids = [id]
        else:
            ids = id

        if isinstance(k_p, int):
            k_ps = [k_p]
        else:
            k_ps = k_p

        # Get cummulative token counts
        start_time = time.time()
        results = self.get_cumulative_token_count(ids, tokens, self.n_gram, return_unshuffled_indices, use_tqdm=use_tqdm, return_dense=False)
        gc.collect()
        if return_unshuffled_indices:
            results, unshuffled_indices = results
            unshuffled_indices = list(zip(*unshuffled_indices))
        results = vstack(results, format="csr")
        if use_tqdm:
            tqdm.write(f"Cummulative token counts done in {time.time() - start_time:.2f} seconds")

        # Calculate Q score via dot product
        start_time = time.time()
        q_score, ranking, k_p_extracted = self.watermarking_fn.q(results, k_p = k_ps, use_tqdm = use_tqdm)
        q_score, ranking = [i.reshape(len(ids), -1, i.shape[-1]).transpose(1,0,2) for i in (q_score, ranking)]  # [text x id x k_p for i in (score, rank)]
        k_p_extracted = k_p_extracted.reshape(len(ids), -1).transpose()  # [text x id]
        if use_tqdm:
            tqdm.write(f"Q score calculated in {time.time() - start_time:.2f} seconds")

        res = q_score

        if return_ranking or return_extracted_k_p or return_counts or return_unshuffled_indices:
            res = {
                "q_score": q_score, 
                }
            if return_ranking:
                res["ranking"] = ranking
            if return_extracted_k_p:
                res["k_p_extracted"] = k_p_extracted
            if return_counts:
                res["counts"] = results
            if return_unshuffled_indices:
                res["unshuffled_indices"] = unshuffled_indices

        if use_tqdm:
            tqdm.write(f"Total time taken for verify: {time.time() - begin_time:.2f} seconds")

        return res