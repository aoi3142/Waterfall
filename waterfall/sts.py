from typing import List

import torch
from sentence_transformers import SentenceTransformer


def STS_scorer_batch(
    original_texts: List[str],
    test_texts: List[List[str]],
    sts_model: SentenceTransformer,
) -> torch.Tensor:
    assert len(original_texts) == len(test_texts), "original_texts and test_texts must have the same length"
    assert all(len(test_texts[0]) == len(sublist) for sublist in test_texts[1:]), "All sublists in test_texts must have the same length"

    all_text = original_texts + [text for sublist in test_texts for text in sublist]
    with torch.no_grad():
        embeddings = sts_model.encode(
            all_text,
            convert_to_tensor=True,
            prompt_name="STS" if "STS" in sts_model.prompts else None,
        )
    original_embeddings = embeddings[:len(original_texts)]
    test_embeddings = embeddings[len(original_texts):].reshape(len(test_texts), -1, embeddings.shape[1])
    num_candidates = test_embeddings.shape[1]
    original_embeddings = original_embeddings.repeat_interleave(num_candidates, dim=0)
    test_embeddings = test_embeddings.flatten(start_dim=0, end_dim=1)
    sts_scores = sts_model.similarity(original_embeddings, test_embeddings).diagonal().reshape(len(original_texts), num_candidates)
    return sts_scores.cpu()


def STS_scorer(
    original_text: str,
    test_texts: str | List[str],
    sts_model: SentenceTransformer,
) -> float | torch.Tensor:
    cos_sim = STS_scorer_batch(
        original_texts=[original_text],
        test_texts=[[test_texts] if isinstance(test_texts, str) else test_texts],
        sts_model=sts_model,
    )[0]
    if isinstance(test_texts, str):
        cos_sim = cos_sim.item()
    return cos_sim
