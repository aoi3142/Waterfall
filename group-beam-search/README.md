---

library_name: transformers
tags:
  - custom_generate
---

## Description

[Diverse beam search](https://hf.co/papers/1610.02424) is a variant of beam search that produces more diverse output candidates to choose from. This strategy measures the dissimilarity of sequences and a penalty is applied if sequences are too similar. To avoid high computation costs, the number of beams is divided into groups.

Enable diverse beam search with the `num_beams`, `num_beam_groups` and `diversity_penalty` parameters (the `num_beams` parameter should be divisible by `num_beam_groups`).

This implementation matches the `group_beam_search` functionality present in `transformers<4.56.0`.

---

## Base model

* [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

---

## Model compatibility

* Decoder-only transformer models

---

## Additional Arguments

* **`num_beams`** (*int*, optional, defaults to `1`):
  Number of beams for beam search. If not greater than `num_beam_groups`, will be set to `num_beam_groups`.

* **`num_beam_groups`** (*int*, optional, defaults to `1`):
  Number of groups to divide `num_beams` into for beam search.

* **`diversity_penalty`** (*float*, optional, defaults to `0.0`):
  Diversity penalty applied to beams.

* **`early_stopping`** (*bool* or *str*, optional, defaults to `False`):
  Whether to stop beam search when at least `num_beams` complete candidates are finished per batch or not. If not `False`, it should be an integer greater than 1 indicating the minimum number of beams required to be finished per batch.
  
* **`max_length`** (*int*, optional, defaults to `20`):
  The maximum length of the generated sequence.

* **`num_return_sequences`** (*int*, optional, defaults to `1`):
  The number of sequences to return.

* **`repetition_penalty`** (*float*, optional, defaults to `None`):
  Helps reduce repetition. A value of `1.2` is recommended.

---

## Output Type changes

* The `generate` method output remains the same as default `transformers` generation,
  but logits are post-processed using the DoLa contrastive scoring before token selection.

---

## Example usage


```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, infer_device

device = infer_device()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).to(device)
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=6, num_beam_groups=3, diversity_penalty=1.0, do_sample=False, custom_generate="transformers-community/group-beam-search", trust_remote_code=True)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```