from transformers import LogitsProcessor
import numpy as np
import torch
from Watermark.WatermarkingFn import WatermarkingFn
from Watermark.Permute import Permute
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

    def get_permutation(self, prev_tok, id = None, cache = True):
        if id is None:
            id = self.id
        return self.permute.get_permutation(prev_tok, id, cache)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.skip_watermark:
            return scores

        if self.init_token_count is None:
            self.init_token_count = input_ids.shape[1]

        # Insufficient tokens generated for n-gram
        if self.init_token_count + self.n_gram - 1 > input_ids.shape[1]:
            return scores

        prev_tokens = input_ids[:,-self.n_gram+1:].cpu()
        permutations = [self.get_permutation(prev_tokens[i,:]) for i in range(prev_tokens.shape[0])]
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

        self.watermarking_fn = watermarkingFnClass(id = id, k_p = k_p, N = self.N, kappa = kappa)
        self.phi = self.watermarking_fn.phi
        self.q = self.watermarking_fn.q

        self.logits_processor.set_phi(self.phi)

    def generate(self, prompt = None, n_gram = None, max_new_tokens = 1000, return_text=True, return_tokens=False, return_scores=False, do_sample=True, **kwargs):
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
            cumulative_token_count = self.get_cumulative_token_count(output, id = self.logits_processor.id)
            res = self.q(cumulative_token_count)
            q_score = res[:,self.k_p-1]
            return_dict["q_score"] = q_score

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

    def get_cumulative_token_count(self, tokens, n_gram = 2, id = None):
        if tokens.ndim == 1:
            tokens = tokens[None,:]
        window = n_gram - 1
        cumulative_token_count = np.zeros((tokens.shape[0], self.N), dtype=int)
        for i in range(tokens.shape[0]):
            for j in range(window, tokens.shape[1]):
                prev_token = tokens[i, j-window:j]
                t = tokens[i,j]
                if t >= self.tokenizer.vocab_size or t == self.tokenizer.eos_token_id:  # EOS token
                    break
                indices = self.logits_processor.get_permutation(prev_token, id = id)
                x = indices[t]
                cumulative_token_count[i, x] += 1
        return cumulative_token_count
    
    def verify(self, text, id = None):
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        cumulative_token_count = self.get_cumulative_token_count(tokens, id = id)
        res = self.q(cumulative_token_count)
        return res