import argparse
import logging
import os
import torch
from typing import List, Literal, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from waterfall.WatermarkingFnFourier import WatermarkingFnFourier
from waterfall.WatermarkingFnSquare import WatermarkingFnSquare
from waterfall.WatermarkerBase import Watermarker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT = (
    "Paraphrase the user provided text while preserving semantic similarity. "
    "Do not include any other sentences in the response, such as explanations of the paraphrasing. "
    "Do not summarize."
)
PRE_PARAPHRASED = "Here is a paraphrased version of the text while preserving the semantic similarity:\n\n"

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

def watermark(
    T_o: str,
    watermarker: Watermarker,
    sts_model: SentenceTransformer,
    num_beam_groups: int = 4,
    beams_per_group: int = 2,
    STS_scale:float = 2.0,
    diversity_penalty: float = 0.5,
    max_new_tokens: Optional[int] = None,
) -> str:
    paraphrasing_prompt = watermarker.tokenizer.apply_chat_template(
        [
            {"role":"system", "content":PROMPT},
            {"role":"user", "content":T_o},
        ], tokenize=False, add_generation_prompt = True) + PRE_PARAPHRASED

    watermarked = watermarker.generate(
        paraphrasing_prompt,
        return_scores = True,
        max_new_tokens = int(len(paraphrasing_prompt) * 1.5) if max_new_tokens is None else max_new_tokens,
        do_sample = False, temperature=None, top_p=None,
        num_beams = num_beam_groups * beams_per_group,
        num_beam_groups = num_beam_groups,
        num_return_sequences = num_beam_groups * beams_per_group,
        diversity_penalty = diversity_penalty,
        )

    # Select best paraphrasing based on q_score and semantic similarity
    sts_scores = STS_scorer(T_o, watermarked["text"], sts_model)
    selection_score = sts_scores * STS_scale + torch.from_numpy(watermarked["q_score"])
    selection = torch.argmax(selection_score)

    T_w = watermarked["text"][selection]

    return T_w

def verify_texts(texts: List[str], id: int, 
                     watermarker: Optional[Watermarker] = None, 
                     k_p: Optional[int] = None, 
                     model_path: Optional[str] = "meta-llama/Llama-3.1-8B-Instruct"
                     ) -> Tuple[float,float]:
    """Returns the q_score and extracted k_p"""

    if watermarker is None:
        assert model_path is not None, "model_path must be provided if watermarker is not passed"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        watermarker = Watermarker(tokenizer=tokenizer)
    
    if k_p is None:
        k_p = watermarker.k_p

    verify_results = watermarker.verify(texts, id=[id], k_p=[k_p], return_extracted_k_p=True)  # results are [text x id x k_p]
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

def watermark_texts(
    T_os: List[str],
    id: Optional[int] = None,
    k_p: int = 1,
    kappa: float = 2.0,
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype: torch.dtype = torch.bfloat16,
    sts_model_path: str = "sentence-transformers/all-mpnet-base-v2",
    watermark_fn: Literal["fourier", "square"] = "fourier",
    watermarker: Optional[Watermarker] = None,
    sts_model: Optional[SentenceTransformer] = None,
    device: str = detect_gpu(),
    num_beam_groups: int = 4,
    beams_per_group: int = 2,
    diversity_penalty: float = 0.5,
    STS_scale:float = 2.0,
    use_tqdm: bool = False,
) -> List[str]:
    if watermark_fn == 'fourier':
        watermarkingFnClass = WatermarkingFnFourier
    elif watermark_fn == 'square':
        watermarkingFnClass = WatermarkingFnSquare
    else:
        raise ValueError("Invalid watermarking function")

    if watermarker is None:
        assert model_path is not None, "model_path must be provided if watermarker is not passed"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        watermarker = Watermarker(tokenizer=tokenizer, model=model, id=id, kappa=kappa, k_p=k_p, watermarkingFnClass=watermarkingFnClass)
    else:
        tokenizer = watermarker.tokenizer
        device = watermarker.model.device
        id = watermarker.id

    if id is None:
        raise Exception("ID or Watermarker class must be passed to watermark_texts.")

    if sts_model is None:
        assert sts_model_path is not None, "sts_model_path must be provided if sts_model is not passed"
        sts_model = SentenceTransformer(sts_model_path, device=device)

    T_ws = []

    for T_o in tqdm(T_os, desc="Watermarking texts",  disable=not use_tqdm):
        T_w = watermark(
            T_o,
            watermarker = watermarker,
            sts_model = sts_model,
            num_beam_groups = num_beam_groups,
            beams_per_group = beams_per_group,
            diversity_penalty = diversity_penalty,
            STS_scale = STS_scale,
            )
        T_ws.append(T_w)

    return T_ws

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
        T_os, id, k_p, kappa,
        watermarker=watermarker, sts_model=sts_model,
        beams_per_group=beams_per_group,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        STS_scale=STS_scale,
        use_tqdm=True
        )

    # watermarker = Watermarker(tokenizer=tokenizer, model=None, id=id, k_p=k_p, watermarkingFnClass=watermarkingFnClass)   # If only verifying the watermark, do not need to instantiate the model
    q_scores, extracted_k_ps = verify_texts(T_os + T_ws, id, watermarker, k_p=k_p)

    for i in range(len(T_os)):
        # Handle the case where this is being run
        # in an IDE or something else without terminal size
        try:
            column_size = os.get_terminal_size().columns
        except OSError as ose:
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