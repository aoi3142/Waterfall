import argparse
import logging
import os
import gc
import torch
import numpy as np
from typing import List, Literal, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from sentence_transformers import SentenceTransformer

from waterfall.WatermarkingFnFourier import WatermarkingFnFourier
from waterfall.WatermarkingFnSquare import WatermarkingFnSquare
from waterfall.WatermarkerBase import Watermarker

PROMPT = (
    "Paraphrase the user provided text while preserving semantic similarity. "
    "Do not include any other sentences in the response, such as explanations of the paraphrasing. "
    "Do not summarize."
)
PRE_PARAPHRASED = "Here is a paraphrased version of the text while preserving the semantic similarity:\n\n"

waterfall_cached_watermarking_model: PreTrainedModel | None = None  # Global variable to cache the watermarking model

def detect_gpu() -> str:
    """
    Use torch to detect if MPS, CUDA, or neither (default CPU)
    are available.

    Returns:
        String for the torch device available.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def del_cached_model():
    global waterfall_cached_watermarking_model
    if isinstance(waterfall_cached_watermarking_model, PreTrainedModel):
        device = waterfall_cached_watermarking_model.device.type
        waterfall_cached_watermarking_model = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

def watermark_texts(
    T_os: List[str],
    id: Optional[int] = None,
    k_p: int = 1,
    kappa: float = 2.0,
    model_path: Optional[str] = "meta-llama/Llama-3.1-8B-Instruct",
    sts_model_path: Optional[str] = "sentence-transformers/all-mpnet-base-v2",
    watermark_fn: Literal["fourier", "square"] = "fourier",
    watermarker: Optional[Watermarker] = None,
    sts_model: Optional[SentenceTransformer] = None,
    device: str = detect_gpu(),
    STS_scale: float = 2.0,
    use_tqdm: bool = False,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    num_beam_groups: int = 4,
    beams_per_group: int = 2,
    diversity_penalty: float = 0.5,
    stop_at_double_newline: bool = True,    # if True, will stop generation at the first double newline. Prevent repeated paraphrasing of the same text.
    **kwargs,
) -> List[str]:
    if watermark_fn == 'fourier':
        watermarkingFnClass = WatermarkingFnFourier
    elif watermark_fn == 'square':
        watermarkingFnClass = WatermarkingFnSquare
    else:
        raise ValueError("Invalid watermarking function")

    # Check if watermarker/model/tokenizer are loaded
    if watermarker is None:
        assert model_path is not None, "model_path must be provided if watermarker is not passed"
        assert id is not None, "id must be provided if watermarker is not passed"
        if isinstance(waterfall_cached_watermarking_model, PreTrainedModel) and waterfall_cached_watermarking_model.name_or_path != model_path:
            del_cached_model()

        if waterfall_cached_watermarking_model is None:
            model = model_path
        else:
            model = waterfall_cached_watermarking_model

        watermarker = Watermarker(model=model, id=id, kappa=kappa, k_p=k_p, watermarkingFnClass=watermarkingFnClass)
    else:
        device = watermarker.model.device.type
        if id is not None:
            watermarker.set_id(id)
        else:
            id = watermarker.id
    waterfall_cached_watermarking_model = watermarker.model

    # Check if sts model is loaded
    if sts_model is None:
        assert sts_model_path is not None, "sts_model_path must be provided if sts_model is not passed"
        sts_model = SentenceTransformer(sts_model_path, device=device)

    # Replace all \n\n in source text if stop_at_double_newline is True
    # Models tend to generate \n\n before endlessly repeating itself, so we want to stop the model from doing that
    if stop_at_double_newline:
        for i in range(len(T_os)):
            if "\n\n" in T_os[i]:
                logging.warning(f"Text idx {i} contains \\n\\n and stop_at_double_newline is set to True, replacing all \\n\\n in text.")
                T_os[i] = T_os[i].replace("\n\n", " ")  # replace double newlines with space

    # Add system prompt and prefill, and format into appropriate chat format
    formatted_T_os = watermarker.format_prompt(
        T_os,
        system_prompt=PROMPT,
        assistant_prefill=PRE_PARAPHRASED,
    )

    if max_new_tokens is None:
        max_input_len = max(len(p) for p in formatted_T_os)
        max_new_tokens = max_input_len

    if do_sample:
        assert (do_sample and temperature is not None and top_p is not None and num_beam_groups == 1 and beams_per_group == 1), \
           "do_sample=True requires temperature, top_p, num_beam_groups=1 and beams_per_group=1"
    else:   # Using beam search
        assert (not do_sample and num_beam_groups >= 1 and beams_per_group >= 1), \
           "do_sample=False requires num_beam_groups>=1 and beams_per_group>=1"

    eos_token_id = watermarker.tokenizer.eos_token_id
    # add "\n\n" tokens to eos_token_id list
    if stop_at_double_newline:
        eos_token_id = [eos_token_id]
        # llama tokenizer's .vocab() has weird symbols and doesn't work with GenerationConfig's stop_strings, so we have to brute force check all tokens
        for token_id,string in enumerate(watermarker.tokenizer.batch_decode(torch.arange(watermarker.tokenizer.vocab_size).unsqueeze(1))):
            if "\n\n" in string:
                eos_token_id.append(token_id)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_beam_groups=num_beam_groups,
        num_beams=num_beam_groups * beams_per_group,
        diversity_penalty=diversity_penalty,
        eos_token_id=eos_token_id,
        num_return_sequences=num_beam_groups * beams_per_group,
        **kwargs
    )

    watermarked = watermarker.generate(
        prompts=formatted_T_os,
        return_text=True,
        return_scores=True,
        use_tqdm=use_tqdm,
        generation_config=generation_config,
        use_model_defaults=False,
    )
    T_ws = watermarked["text"]
    # Reshape T_ws to Queries X Beams
    num_beams = num_beam_groups * beams_per_group
    T_ws = [T_ws[i * num_beams:(i + 1) * num_beams] for i in range(len(T_os))]

    # Select best paraphrasing based on q_score and semantic similarity
    sts_scores = STS_scorer_batch(T_os, T_ws, sts_model)
    selection_scores = sts_scores * STS_scale + torch.from_numpy(watermarked["q_score"]).reshape(-1, num_beams)
    selections = torch.argmax(selection_scores, dim = -1)

    T_ws = [T_w[selection] for T_w, selection in zip(T_ws, selections)]

    return T_ws

def verify_texts(texts: List[str], id: int,
                     watermarker: Optional[Watermarker] = None,
                     k_p: Optional[int] = None,
                     model_path: Optional[str] = "meta-llama/Llama-3.1-8B-Instruct",
                     return_extracted_k_p: bool = False
                     ) -> np.ndarray | Tuple[np.ndarray,np.ndarray]:
    """Returns the q_score and extracted k_p"""

    if watermarker is None:
        assert model_path is not None, "model_path must be provided if watermarker is not passed"
        watermarker = Watermarker(tokenizer=model_path)

    verify_results = watermarker.verify(texts, id=[id], k_p=k_p, return_extracted_k_p=return_extracted_k_p)  # results are [text x id x k_p]

    if not return_extracted_k_p:
        return verify_results[:,0,0]

    q_score = verify_results["q_score"]
    k_p_extracted = verify_results["k_p_extracted"]

    return q_score[:,0,0], k_p_extracted[:, 0]

def STS_scorer_batch(
    original_texts: List[str],
    test_texts: List[List[str]],
    sts_model: SentenceTransformer
) -> torch.Tensor:

    assert len(original_texts) == len(test_texts), "original_texts and test_texts must have the same length"
    assert all(len(test_texts[0]) == len(sublist) for sublist in test_texts[1:]), "All sublists in test_texts must have the same length"

    all_text = original_texts + [text for sublist in test_texts for text in sublist]
    embeddings = sts_model.encode(all_text, convert_to_tensor=True, normalize_embeddings=True)
    original_embeddings = embeddings[:len(original_texts)]
    test_embeddings = embeddings[len(original_texts):].reshape(len(test_texts), -1, embeddings.shape[1])
    cos_sim = torch.einsum('ik,ijk->ij', original_embeddings, test_embeddings).cpu()
    return cos_sim

def STS_scorer(
    original_text: str,
    test_texts: str | List[str],
    sts_model: SentenceTransformer
) -> float | torch.Tensor:
    cos_sim = STS_scorer_batch(
        original_texts=[original_text],
        test_texts=[[test_texts] if isinstance(test_texts, str) else test_texts],
        sts_model=sts_model
    )[0]
    if isinstance(test_texts, str):
        cos_sim = cos_sim.item()
    return cos_sim

def pretty_print(
        T_o: str, T_w: str,
        sts_score: float,
        T_o_q_score: float, T_w_q_score: float,
        k_p: int, T_w_k_p: int,
        ) -> None:
    print(f"\nOriginal text T_o:\n\n{T_o}\n")
    print(f"\nWatermarked text T_w:\n\n{T_w}\n")

    # Original text
    print(f"Verification score of T_o: \033[93m{T_o_q_score:.4f}\033[0m")

    # Watermarked text
    print(f"Verification score of T_w: \033[92m{T_w_q_score:.4f}\033[0m\n")

    print(f"STS score of T_w         : \033[94m{sts_score:.4f}\033[0m\n")

    # Extract from watermarked text
    print(f"Watermarking k_p         : \033[95m{k_p}\033[0m")
    print(f"Extracted k_p from T_w   : \033[96m{T_w_k_p}\033[0m\n")

def main():
    parser = argparse.ArgumentParser(description='generate text watermarked with a key')
    parser.add_argument('--id',default=42,type=int,
        help='id: unique ID')
    parser.add_argument('--kappa',default=2.,type=float,
        help='kappa: watermarking strength')
    parser.add_argument('--k_p', default=1, type=int,
        help="k_p: Perturbation key")
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct', type=str,
        help="watermarking model")
    parser.add_argument('--sts_model', default='sentence-transformers/all-mpnet-base-v2', type=str,
        help="STS model")
    parser.add_argument('--T_o', default=None, type=str,
        help="original_text")
    parser.add_argument('--watermark_fn', default='fourier', type=str,
        help="watermarking function, can be 'fourier' or 'square'")
    parser.add_argument('--device', default=detect_gpu(), type=str,
        help="device to use for generation")
    parser.add_argument('--num_beam_groups', default=4, type=int,
        help="number of beam groups for generation")
    parser.add_argument('--beams_per_group', default=2, type=int,
        help="number of beams per group for generation")
    parser.add_argument('--diversity_penalty', default=0.5, type=float,
        help="diversity penalty for group beam search")
    parser.add_argument('--STS_scale', default=2.0, type=float,
        help="scale factor for trade-off between STS and q score. Higher means more emphasis on STS.")

    args = parser.parse_args()

    if args.watermark_fn == 'fourier':
        watermarkingFnClass = WatermarkingFnFourier
    elif args.watermark_fn == 'square':
        watermarkingFnClass = WatermarkingFnSquare
    else:
        # Add any other self-defined watermarking functions here
        raise ValueError("Invalid watermarking function")

    id = args.id
    kappa = args.kappa
    k_p = args.k_p
    model_name_or_path = args.model
    sts_model_name = args.sts_model
    T_o = args.T_o
    device = args.device
    num_beam_groups = args.num_beam_groups
    beams_per_group = args.beams_per_group
    diversity_penalty = args.diversity_penalty
    STS_scale = args.STS_scale

    if args.T_o is None:
        T_o = "Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation."
    T_os = [T_o]    # Replace with your own list of texts to watermark

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        )

    watermarker = Watermarker(tokenizer=tokenizer, model=model, id=id, kappa=kappa, k_p=k_p, watermarkingFnClass=watermarkingFnClass)

    sts_model = SentenceTransformer(sts_model_name, device=device)

    T_ws = watermark_texts(
        T_os,
        id=id, k_p=k_p, kappa=kappa,
        watermarker=watermarker,
        sts_model=sts_model,
        beams_per_group=beams_per_group,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        STS_scale=STS_scale,
        use_tqdm=True,
        )

    # watermarker = Watermarker(tokenizer=tokenizer, model=None, id=id, k_p=k_p, watermarkingFnClass=watermarkingFnClass)   # If only verifying the watermark, do not need to instantiate the model
    q_scores, extracted_k_ps = verify_texts(T_os + T_ws, id, watermarker, k_p=k_p, return_extracted_k_p=True)

    for i in range(len(T_os)):
        # Handle the case where this is being run
        # in an IDE or something else without terminal size
        try:
            column_size = os.get_terminal_size().columns
        except OSError:
            column_size = 80

        print("=" * column_size)

        sts_score = STS_scorer(T_os[i], T_ws[i], sts_model)
        pretty_print(
            T_os[i], T_ws[i],
            sts_score,
            q_scores[i], q_scores[i + len(T_os)],
            k_p, extracted_k_ps[i + len(T_os)],
        )

if __name__ == "__main__":
    main()