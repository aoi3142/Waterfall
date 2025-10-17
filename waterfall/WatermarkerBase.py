import gc
import logging
import os
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Optional
from itertools import repeat

import numpy as np
import torch
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from transformers.generation.logits_process import LogitsProcessor, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper
from transformers.generation.configuration_utils import GenerationConfig

from waterfall.permute import Permute
from waterfall.WatermarkingFn import WatermarkingFn
from waterfall.WatermarkingFnFourier import WatermarkingFnFourier

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PerturbationProcessor(LogitsProcessor):
    def __init__(self,
                 N : int = 32000,     # Vocab size
                 id : int = 0,        # Watermark ID
                 ) -> None:

        self.id = id
        self.N = N
        self.init_token_count = None
        self.phi = torch.zeros(N)
        self.n_gram = 2

        self.skip_watermark = False

        self.permute = Permute(self.N)

    def reset(self, n_gram : int = 2) -> None:
        self.n_gram = n_gram
        self.init_token_count = None
        if torch.allclose(self.phi,torch.median(self.phi)):
            self.skip_watermark = True
            logging.warning(f"Generating without watermark as watermarking function is flat")
        else:
            self.skip_watermark = False

    def set_phi(self, phi : np.ndarray) -> None:
        self.phi = torch.from_numpy(phi)

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:

        if self.skip_watermark:
            return scores

        if self.init_token_count is None:
            self.init_token_count = input_ids.shape[1]

        # Insufficient tokens generated for n-gram
        if self.init_token_count + self.n_gram - 1 > input_ids.shape[1]:
            return scores

        # using numpy as PyTorch tensors doesn't hash properly for rng and dict key
        prev_tokens = input_ids[:,-self.n_gram+1:].cpu().numpy()

        permutations = (
            self.permute.get_permutation(prev_tokens[i,:], self.id, cache=True)
            for i in range(prev_tokens.shape[0])
        )
        perturbations = torch.stack([
            self.phi[permutation] for permutation in permutations
        ])
        scores[:,:self.N] += perturbations.to(device=scores.device, dtype=scores.dtype)
        return scores

def indices_to_counts(N : int, dtype : np.dtype, indices : np.ndarray) -> csr_matrix:
    counts = csr_matrix([np.bincount(j, minlength=N).astype(dtype) for j in indices])
    return counts

class Watermarker:
    def __init__(self,
                 tokenizer : Optional[PreTrainedTokenizerBase | str] = None,
                 model : Optional[PreTrainedModel | str] = None,
                 id : int = 0,
                 kappa : float = 6,
                 k_p : int = 1,
                 n_gram : int = 2,
                 watermarkingFnClass = WatermarkingFnFourier,
                 device = None,
                 dtype = torch.bfloat16,
                 ) -> None:
        assert kappa >= 0, f"kappa must be >= 0, value provided is {kappa}"
        self.k_p = k_p
        self.n_gram = n_gram
        self.kappa = kappa

        if tokenizer is None:
            if isinstance(model, str):
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            elif isinstance(model, PreTrainedModel):
                self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            else:
                raise NotImplementedError("tokenizer must be provided or model must be a string or PreTrainedModel")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self.N = self.tokenizer.vocab_size

        if isinstance(model, str):
            self.load_model(model, device_map=device if device is not None else "auto", dtype=dtype)
        else:
            self.model = model

        assert (self.model is None) or isinstance(self.model, PreTrainedModel), f"model must be a transformers model, value provided is {type(self.model)}" # argument order for tokenizer and model were swapped since the original code

        self.watermarkingFnClass = watermarkingFnClass
        self.set_id(id)

    def set_id(self, id : int):
        self.id = id
        self.logits_processor = PerturbationProcessor(N = self.N, id = self.id)
        self.compute_phi(self.watermarkingFnClass)

    def load_model(self, model_name_or_path : str, device_map : str = "auto", dtype = torch.bfloat16):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=dtype,
        )

    def compute_phi(self, watermarkingFnClass = WatermarkingFnFourier) -> None:
        self.watermarking_fn: WatermarkingFn = watermarkingFnClass(id = self.id, k_p = self.k_p, N = self.N, kappa = self.kappa)
        self.phi = self.watermarking_fn.phi

        self.logits_processor.set_phi(self.phi)

    # Format prompt(s) into chat template
    def format_prompt(
            self, 
            T_os : str | List[str],
            system_prompt : Optional[str] = None,
            assistant_prefill : Optional[str | List[str]] = "",
            ) -> str | List[str]:
        if isinstance(system_prompt, str):
            _system_prompt = {"role":"system", "content":system_prompt}
        is_single = isinstance(T_os, str)
        if is_single:
            T_os = [T_os]
        if not isinstance(assistant_prefill, list):
            assistant_prefill = repeat(assistant_prefill, len(T_os))
        else:
            assert len(assistant_prefill) == len(T_os), "Length of assistant_prefill must match length of T_os"
        formatted_prompts = []
        for T_o, prefill in zip(T_os, assistant_prefill):
            formatted_prompt : str = self.tokenizer.apply_chat_template(
                [
                    _system_prompt,
                    {"role":"user", "content":T_o},
                ], tokenize=False, add_generation_prompt = True)
            if prefill is not None:
                formatted_prompt += prefill
            formatted_prompts.append(formatted_prompt)
        if is_single:
            return formatted_prompts[0]
        return formatted_prompts

    # Find the largest batch size that fits in GPU memory
    def find_largest_batch_size(
            self,
            tokd_inputs : List[BatchEncoding],
            logits_processor : List[LogitsProcessor] = [],
            **kwargs,
        ):
        longest_idx = np.argmax([tokd_input["input_ids"].shape[-1] for tokd_input in tokd_inputs])
        if "generation_config" in kwargs:
            generation_config = GenerationConfig(**kwargs["generation_config"].to_dict()) # copy
            max_new_tokens = generation_config.max_new_tokens
        else:
            generation_config = GenerationConfig(**kwargs)
            max_new_tokens = kwargs.get("max_new_tokens", 2048)
        generation_config.update(max_new_tokens=1)
        input_ids = tokd_inputs[longest_idx]["input_ids"]
        input_ids = torch.zeros(
            (1, max_new_tokens + input_ids.shape[-1] - 1), 
            dtype=input_ids.dtype,
            device=self.model.device
            )
        max_batch_size = 1
        with torch.no_grad():
            while max_batch_size < min(16, len(tokd_inputs)):
                torch.cuda.empty_cache()
                try:
                    _ = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        logits_processor=logits_processor,
                        generation_config=generation_config,
                        pad_token_id=self.tokenizer.eos_token_id,
                        tokenizer=self.tokenizer,
                    )
                    max_batch_size = input_ids.shape[0]
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        break
                    else:
                        raise e
                input_ids = torch.cat([input_ids, input_ids], dim=0)
        torch.cuda.empty_cache()
        return max_batch_size

    def generate(
            self,
            prompts : Optional[str | List[str]] = None,
            tokd_inputs : Optional[torch.Tensor | List[torch.Tensor] | BatchEncoding | List[BatchEncoding]] = None,
            n_gram : Optional[int] = None,
            return_text : bool = True,
            return_tokens : bool = False,
            return_scores : bool = False,
            use_tqdm : bool = False,
            batched_generate : bool = True,
            discard_incomplete : bool = True,
            **kwargs    # Other generate parameters
            ) -> List[str] | dict:  # Returns flattened list of query x beam

        assert self.model is not None, "Model is not loaded. Please load the model before generating text."

        is_single = isinstance(prompts, str) or isinstance(tokd_inputs, torch.Tensor)
        if is_single:
            prompts = [prompts] if prompts is not None else None
            tokd_inputs = [tokd_inputs] if tokd_inputs is not None else None

        if n_gram is None:
            n_gram = self.n_gram
        if tokd_inputs is None:
            assert prompts is not None, "Either prompt or tokd_input must be provided."
            tokd_inputs = [self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False) for prompt in prompts]

        # If tokd_input is a tensor, convert it to a BatchEncoding
        squeezed_tokd_inputs = []
        for tokd_input in tokd_inputs:
            if isinstance(tokd_input, torch.Tensor):
                input_ids = tokd_input
                attention_mask = torch.ones_like(tokd_input)
            else:
                input_ids = tokd_input["input_ids"]
                attention_mask = tokd_input["attention_mask"]
            if input_ids.ndim == 2:
                input_ids = input_ids.squeeze()
                attention_mask = attention_mask.squeeze()
            squeezed_tokd_inputs.append(BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask}))
        tokd_inputs = squeezed_tokd_inputs

        logits_processor = []
        # Ensure top_k and top_p happens before watermarking
        if "generation_config" in kwargs:
            generation_config: GenerationConfig = kwargs["generation_config"]
            top_k = generation_config.top_k
            top_p = generation_config.top_p
            temperature = generation_config.temperature
            num_beams = generation_config.num_beams
            diversity_penalty = generation_config.diversity_penalty
            if num_beams <= 1:
                diversity_penalty = None
            generation_config.update(top_p=1.0, temperature=None, diversity_penalty=diversity_penalty)
        else:
            top_k = kwargs.pop("top_k", None)
            top_p = kwargs.pop("top_p", None)
            temperature = kwargs.pop("temperature", 1.0)
            num_beams = kwargs.pop("num_beams", 1)
            diversity_penalty = kwargs.pop("diversity_penalty", None)
            if num_beams <= 1:
                kwargs["diversity_penalty"] = None

        if num_beams > 1 and temperature is not None and temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(float(temperature)))
        if top_k is not None and top_k != 0:
            logits_processor.append(TopKLogitsWarper(top_k))
        if top_p is not None and top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p))
        if self.kappa != 0:
            logits_processor.append(self.logits_processor)

        if batched_generate and len(tokd_inputs) >= 8:
            max_batch_size = self.find_largest_batch_size(tokd_inputs, logits_processor=logits_processor, **kwargs)
        else:
            max_batch_size = 1

        # Group inputs by token length
        if max_batch_size > 1:
            tokd_inputs_order = sorted(range(len(tokd_inputs)), key=lambda i: tokd_inputs[i]["input_ids"].shape[-1])
            tokd_inputs = [tokd_inputs[i] for i in tokd_inputs_order]
        else:
            tokd_inputs_order = range(len(tokd_inputs))
        tokd_input_batches = []
        for i in range(0, len(tokd_inputs), max_batch_size):
            batch = self.tokenizer.pad(tokd_inputs[i:i+max_batch_size], padding=True, padding_side="left").to(self.model.device, non_blocking=True)
            tokd_input_batches.append(batch)
        torch.cuda.synchronize()

        outputs = []
        with torch.no_grad():
            bar = tqdm(total=len(tokd_inputs), desc="Generating text", disable=not use_tqdm)
            for tokd_input_batch in tokd_input_batches:
                self.logits_processor.reset(n_gram)
                output = self.model.generate(
                    **tokd_input_batch,
                    logits_processor=logits_processor,
                    pad_token_id=self.tokenizer.eos_token_id,
                    tokenizer=self.tokenizer,
                    **kwargs
                    )
                output = output[:,tokd_input_batch["input_ids"].shape[-1]:]
                if discard_incomplete:
                    output[output[:, -1] != self.tokenizer.eos_token_id] = self.tokenizer.eos_token_id
                output = output.to("cpu", non_blocking=True)
                outputs.append(output)
                bar.update(tokd_input_batch["input_ids"].shape[0])
        torch.cuda.synchronize()
        outputs = [j for i in outputs for j in i]  # Flatten the list of outputs
        
        # Restore original ordering
        if max_batch_size > 1:
            reordered_outputs = [None] * len(outputs)
            num_return_sequences = len(outputs) // len(tokd_inputs)
            for i, idx in enumerate(tokd_inputs_order):
                reordered_outputs[idx * num_return_sequences:(idx + 1) * num_return_sequences] = outputs[i * num_return_sequences:(i + 1) * num_return_sequences]
            outputs = reordered_outputs

        return_dict = {}

        if return_scores:
            cumulative_token_count = self.get_cumulative_token_count(self.id, outputs, n_gram = n_gram, return_dense=False)
            cumulative_token_count = vstack([i[0] for i in cumulative_token_count], format="csr")
            q_score, _, _ = self.watermarking_fn.q(cumulative_token_count, k_p = [self.k_p], use_tqdm=False)
            return_dict["q_score"] = q_score

        if return_tokens:
            return_dict["tokens"] = outputs

        if return_text:
            decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_output = [i.strip() for i in decoded_output]
            return_dict["text"] = decoded_output

        if is_single:
            for k, v in return_dict.items():
                return_dict[k] = v[0]

        if return_text and len(return_dict) == 1:
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

        if k_p is None:
            k_p = self.k_p

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