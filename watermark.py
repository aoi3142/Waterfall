import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from Watermark.WatermarkingFnFourier import WatermarkingFnFourier
from Watermark.WatermarkingFnSquare import WatermarkingFnSquare
from Watermark.WatermarkerBase import Watermarker
from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt = (
    "Paraphrase the user provided text while preserving semantic similarity. "
    "Do not include any other sentences in the response, such as explanations of the paraphrasing. "
    "Do not summarize."
)
pre_paraphrased = "Here is a paraphrased version of the text while preserving the semantic similarity:\n\n"

def watermark_and_evaluate(T_o, id, k_p, num_beam_groups=4, beams_per_group=2, STS_scale=2.0):
    print(f"\nOriginal text T_o:\n\n{T_o}\n")

    # Generate watermarked text
    paraphrasing_prompt = tokenizer.apply_chat_template(
        [
            {"role":"system", "content":prompt},
            {"role":"user", "content":T_o},
        ], tokenize=False, add_generation_prompt = True) + f"{pre_paraphrased}\n\n"
    watermarked = watermarker.generate(
        paraphrasing_prompt, 
        return_scores = True,
        max_new_tokens = int(len(paraphrasing_prompt) * 1.5),
        do_sample = False, temperature=None, top_p=None,
        num_beams = num_beam_groups * beams_per_group, 
        num_beam_groups = num_beam_groups, 
        num_return_sequences = num_beam_groups * beams_per_group, 
        diversity_penalty = 0.5,
        )
    
    # Select best paraphrasing based on q_score and semantic similarity
    sts_scores = sts_model.encode([T_o, *watermarked["text"]], convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(sts_scores[0], sts_scores[1:], dim=1).cpu()
    selection_score = cos_sim * STS_scale + watermarked["q_score"]
    selection = torch.argmax(selection_score)

    T_w = watermarked["text"][selection]

    print(f"\nWatermarked text T_w:\n\n{T_w}\n")

    # Verify on original and watermarked text
    verify_results = watermarker.verify([T_o, T_w], id=[id], k_p=[k_p], return_extracted_k_p=True)  # results are [text x id x k_p]
    q_score = verify_results["q_score"]
    k_p_extracted = verify_results["k_p_extracted"]

    # Original text
    print(f"Verification score of T_o: \033[93m{q_score[0,0,0]:.4f}\033[0m")

    # Watermarked text
    print(f"Verification score of T_w: \033[92m{q_score[1,0,0]:.4f}\033[0m\n")

    print(f"STS score of T_w         : \033[94m{cos_sim[selection].item():.4f}\033[0m\n")

    # Extract from watermarked text
    print(f"Watermarking k_p         : \033[95m{k_p}\033[0m")
    print(f"Extracted k_p from T_w   : \033[96m{k_p_extracted[1,0]}\033[0m\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate text watermarked with a key')
    parser.add_argument('--id',default=42,type=int,
            help='id: unique ID')
    parser.add_argument('--kappa',default=2,type=float,
            help='kappa: watermarking strength')
    parser.add_argument('--k_p', default=1, type=int,
            help="k_p: Perturbation key")
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct', type=str,
            help="model")
    parser.add_argument('--T_o', default='Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation.',
            type=str, help="original_text")
    parser.add_argument('--watermark_fn', default='fourier', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_beam_groups', default=4, type=int)
    parser.add_argument('--beams_per_group', default=2, type=int)
    parser.add_argument('--STS_scale', default=2, type=float, help="Scale factor for trade-off between STS and q score. Higher means more emphasis on STS.")

    args = parser.parse_args()

    id = args.id
    kappa = args.kappa
    k_p = args.k_p
    model_name_or_path = args.model
    T_o = args.T_o
    
    if args.watermark_fn == 'fourier':
        watermarkingFnClass = WatermarkingFnFourier
    elif args.watermark_fn == 'square':
        watermarkingFnClass = WatermarkingFnSquare
    else:
        raise ValueError("Invalid watermarking function")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        )

    watermarker = Watermarker(model, tokenizer, id, kappa, k_p, watermarkingFnClass=watermarkingFnClass)

    sts_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=args.device)

    watermark_and_evaluate(T_o, id, k_p, args.num_beam_groups, args.beams_per_group, args.STS_scale)