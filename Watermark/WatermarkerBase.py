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

    def get_permutation(self, prev_tok, id = None, cache = False):
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
        self.n_gram = n_gram
        self.kappa = kappa

        self.N = len(self.tokenizer)
        self.logits_processor = PerturbationProcessor(N = self.N, id = id)

        self.watermarking_fn = watermarkingFnClass(id = id, k_p = k_p, N = self.N, kappa = kappa)
        self.phi = self.watermarking_fn.phi
        self.q = self.watermarking_fn.q

        self.logits_processor.set_phi(self.phi)

    def generate(self, prompt = None, n_gram = None):
        if n_gram is None:
            n_gram = self.n_gram
        tokd_input = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            self.logits_processor.reset(n_gram)
            output = self.model.generate(
                **tokd_input, 
                max_new_tokens=1000, 
                do_sample=True, 
                logits_processor=([self.logits_processor] if self.kappa > 0 else None)
                )
        output = output[:,tokd_input["input_ids"].shape[-1]:]
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

        return decoded_output

    def get_cumulative_token_count(self, tokens, n_gram = 2, id = None):
        if tokens.ndim == 1:
            tokens = tokens[None,:]
        window = n_gram - 1
        cumulative_token_count = np.zeros((tokens.shape[0], self.N), dtype=int)
        for i in range(tokens.shape[0]):
            for j in range(window, tokens.shape[1]):
                prev_token = tokens[i, j-window:j]
                t = tokens[i,j]
                indices = self.logits_processor.get_permutation(prev_token, id = id)
                x = indices[t]
                cumulative_token_count[i, x] += 1
        return cumulative_token_count
    
    def verify(self, text, id = None):
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        cumulative_token_count = self.get_cumulative_token_count(tokens, id = id)
        res = self.q(cumulative_token_count)
        return res