import gc
import logging
import os
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessor, TopKLogitsWarper, TopPLogitsWarper

from waterfall.permute import Permute
from waterfall.WatermarkingFn import WatermarkingFn
from waterfall.WatermarkingFnFourier import WatermarkingFnFourier

class PerturbationProcessor(LogitsProcessor):
    def __init__(self,
                 N : int = 32000,     # Vocab size
                 id : int = 0,        # Watermark ID
                 ) -> None:

        self.id = id
        self.N = N
        self.init_token_count = None
        self.phi = np.ones(N)
        self.n_gram = 2

        self.skip_watermark = False

        self.permute = Permute(self.N)

    def reset(self, n_gram : int = 2) -> None:
        self.n_gram = n_gram
        self.init_token_count = None
        if np.allclose(self.phi,np.median(self.phi)):
            self.skip_watermark = True
            logging.warning(f"Generating without watermark as watermarking function is flat")
        else:
            self.skip_watermark = False

    def set_phi(self, phi : np.ndarray) -> None:
        self.phi = phi

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:

        if self.skip_watermark:
            return scores

        if self.init_token_count is None:
            self.init_token_count = input_ids.shape[1]

        # Insufficient tokens generated for n-gram
        if self.init_token_count + self.n_gram - 1 > input_ids.shape[1]:
            return scores

        prev_tokens = input_ids[:,-self.n_gram+1:].cpu().numpy()
        permutations = [self.permute.get_permutation(prev_tokens[i,:], self.id, cache=True) for i in range(prev_tokens.shape[0])]

        scores[:,:self.N] += torch.tensor(self.phi[permutations],
                                          device=scores.device,
                                          dtype=scores.dtype)
        return scores

def indices_to_counts(N : int, dtype : np.dtype, indices : np.ndarray) -> csr_matrix:
    counts = csr_matrix([np.bincount(j, minlength=N).astype(dtype) for j in indices])
    return counts

class Watermarker:
    def __init__(self,
                 tokenizer : PreTrainedTokenizerBase,
                 model : Optional[PreTrainedModel] = None,
                 id : int = 0,
                 kappa : float = 6,
                 k_p : int = 1,
                 n_gram : int = 2,
                 watermarkingFnClass = WatermarkingFnFourier
                 ) -> None:
        assert kappa >= 0, f"kappa must be >= 0, value provided is {kappa}"

        assert (model is None) or isinstance(model, PreTrainedModel), f"model must be a transformers model, value provided is {type(model)}" # argument order for tokenizer and model were swapped since the original code

        self.tokenizer = tokenizer
        self.model = model
        self.id = id
        self.k_p = k_p
        self.n_gram = n_gram
        self.kappa = kappa

        self.N = self.tokenizer.vocab_size
        self.logits_processor = PerturbationProcessor(N = self.N, id = id)

        self.compute_phi(watermarkingFnClass)

    def compute_phi(self, watermarkingFnClass = WatermarkingFnFourier) -> None:
        self.watermarking_fn: WatermarkingFn = watermarkingFnClass(id = id, k_p = self.k_p, N = self.N, kappa = self.kappa)
        self.phi = self.watermarking_fn.phi

        self.logits_processor.set_phi(self.phi)

    def generate(
            self,
            prompt : Optional[str] = None,
            tokd_input : Optional[torch.Tensor] = None,
            n_gram : Optional[int] = None,
            max_new_tokens : int = 1000,
            return_text : bool =True,
            return_tokens : bool =False,
            return_scores : bool =False,
            do_sample : bool =True,
            **kwargs
            ) -> List[str] | dict:

        assert self.model is not None, "Model is not loaded. Please load the model before generating text."

        if n_gram is None:
            n_gram = self.n_gram
        if tokd_input is None:
            assert prompt is not None, "Either prompt or tokd_input must be provided."
            tokd_input = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        tokd_input = tokd_input.to(self.model.device)
        logits_processor = []
        if "top_k" in kwargs and kwargs["top_k"] is not None and kwargs["top_k"] != 0:
            logits_processor.append(TopKLogitsWarper(kwargs.pop("top_k")))
        if "top_p" in kwargs and kwargs["top_p"] is not None and kwargs["top_p"] < 1.0:
            logits_processor.append(TopPLogitsWarper(kwargs.pop("top_p")))
        if self.kappa != 0:
            logits_processor.append(self.logits_processor)

        with torch.no_grad():
            self.logits_processor.reset(n_gram)
            output = self.model.generate(
                **tokd_input,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
                )
        output = output[:,tokd_input["input_ids"].shape[-1]:].cpu()

        return_dict = {}

        if return_scores:
            cumulative_token_count = self.get_cumulative_token_count(self.id, output, n_gram = n_gram, return_dense=False)
            cumulative_token_count = vstack([i[0] for i in cumulative_token_count], format="csr")
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
            return_dense : bool = True,
            batch_size : int = 2**8,
            ) -> List[csr_matrix] | List[np.ndarray] | Tuple[List[csr_matrix], List[List[np.ndarray]]] | Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        if isinstance(ids, int):
            ids = [ids]
        if isinstance(all_tokens[0], int) or (isinstance(all_tokens, (np.ndarray, torch.Tensor)) and all_tokens.ndim == 1):
            all_tokens = [all_tokens]
        all_tokens = list(map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, all_tokens))
        max_length = max(map(len, all_tokens))
        window = n_gram - 1

        # Collect all unique seeds for psuedo-random number generation
        key_index_dict = defaultdict(set)
        all_keys = []
        for i, tokens in enumerate(tqdm(all_tokens, desc="Collecting unique n-grams", disable=not use_tqdm)):
            all_keys.append([])
            for j in range(window, len(tokens)):
                prev_token = tuple(tokens[j-window:j])
                t = tokens[j]
                if t >= self.N:
                    break
                key_index_dict[prev_token].add(t)
                all_keys[i].append((prev_token, t))
        key_index_dict = {k:tuple(v) for k,v in key_index_dict.items()}

        use_mp = len(all_tokens) > batch_size * 4
        if use_mp:
            p = Pool(len(os.sched_getaffinity(0))-1)
            pool_map = partial(p.imap, chunksize=batch_size)
        else:
            pool_map = map

        # Generate permutations for all unique seeds
        permutations = pool_map(
            partial(self.logits_processor.permute.get_unshuffled_indices, ids),
            key_index_dict.items())
        permutations = tqdm(permutations, total=len(key_index_dict), desc="Getting permutations", disable=not use_tqdm)
        for k, value in zip(key_index_dict.keys(), permutations):
            key_index_dict[k] = value

        # Assign indices to unshuffled_indices
        unshuffled_indices: List[np.ndarray] = []  # [text x id x length]
        for keys in tqdm(all_keys, desc="Assigning indices", disable=not use_tqdm):
            if len(keys) == 0:
                unshuffled_indices.append(np.zeros((len(ids), 0), dtype=np.min_scalar_type(self.N)))
            else:
                unshuffled_indices.append(np.stack([key_index_dict[key][t] for key, t in keys]).T)  # [id x length]

        # Convert indices to counts
        cumulative_token_count = pool_map(
            partial(indices_to_counts, self.N, np.min_scalar_type(max_length)),
            unshuffled_indices
            )
        cumulative_token_count = list(tqdm(cumulative_token_count, total=len(unshuffled_indices), desc="Counting tokens", disable=not use_tqdm))

        if use_mp:
            p.close()
            p.join()

        if return_dense:
            cumulative_token_count = list(map(lambda x: x.toarray(), cumulative_token_count))

        if return_unshuffled_indices:
            return cumulative_token_count, unshuffled_indices
        return cumulative_token_count

    def verify(
            self,
            text : str | List[str],
            id: Optional[int | List[int]] = None,
            k_p : Optional[int | List[int]] = None,
            return_ranking : bool = False,
            return_extracted_k_p : bool = False,
            return_counts : bool = False,
            return_unshuffled_indices : bool = False,
            use_tqdm : bool = False,
            batch_size : int = 2**8,
            ) -> np.ndarray | dict:
        begin_time = time.time()

        if id is None:
            id = self.id

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        tokens = [np.array(self.tokenizer.encode(text, add_special_tokens=False), dtype=np.uint32) for text in tqdm(texts, desc="Tokenizing", disable=not use_tqdm)]

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
        results = self.get_cumulative_token_count(ids, tokens, self.n_gram, return_unshuffled_indices, use_tqdm=use_tqdm, return_dense=False, batch_size=batch_size)
        gc.collect()
        if return_unshuffled_indices:
            results, unshuffled_indices = results
        results = vstack(results, format="csr")
        if use_tqdm:
            tqdm.write(f"Cummulative token counts done in {time.time() - start_time:.2f} seconds")

        # Calculate Q score via dot product
        start_time = time.time()
        q_score, ranking, k_p_extracted = self.watermarking_fn.q(results, k_p = k_ps, batch = batch_size, use_tqdm = use_tqdm)
        q_score, ranking = [i.reshape(-1, len(ids), i.shape[-1]) for i in (q_score, ranking)]  # [text x ids x k_p for i in (score, rank)]
        k_p_extracted = k_p_extracted.reshape(-1, len(ids))  # [text x ids]
        if use_tqdm:
            tqdm.write(f"Q score calculated in {time.time() - start_time:.2f} seconds")

        res = q_score # [text x ids x k_p]

        if return_ranking or return_extracted_k_p or return_counts or return_unshuffled_indices:
            res = {
                "q_score": q_score,                             # [text x ids x k_p]
                }
            if return_ranking:
                res["ranking"] = ranking                        # [text x ids x k_p]
            if return_extracted_k_p:
                res["k_p_extracted"] = k_p_extracted            # [text x ids]
            if return_counts:
                res["counts"] = results                         # [text x ids x k_p]
            if return_unshuffled_indices:
                res["unshuffled_indices"] = unshuffled_indices  # [text x ids x length]

        if use_tqdm:
            tqdm.write(f"Total time taken for verify: {time.time() - begin_time:.2f} seconds")

        return res