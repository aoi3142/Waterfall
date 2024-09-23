import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from Watermark.WatermarkingFnFourier import WatermarkingFnFourier
from Watermark.WatermarkingFnSquare import WatermarkingFnSquare
from Watermark.WatermarkerBase import Watermarker
from sentence_transformers import SentenceTransformer

prompt = (
    "Paraphrase the user provided text while preserving semantic similarity. "
    "Do not include any other sentences in the response, such as explanations of the paraphrasing. "
    "Do not summarize."
)
pre_paraphrased = "Here is a paraphrased version of the text while preserving the semantic similarity:\n\n"

def watermark_and_evaluate(T_o):
    print(f"\nOriginal text T_o:\n\n{T_o}\n")

    # Generate watermarked text
    paraphrasing_prompt = tokenizer.apply_chat_template(
        [
            {"role":"system", "content":prompt},
            {"role":"user", "content":T_o},
        ], tokenize=False, add_generation_prompt = True) + f"{pre_paraphrased}\n\n"
    watermarked = watermarker.generate(
        paraphrasing_prompt, 
        return_scores=True,
        max_new_tokens=int(len(paraphrasing_prompt) * 1.5),
        do_sample=False, temperature=None, top_p=None,
        num_beams=8, num_beam_groups=4, num_return_sequences=8, diversity_penalty = 0.5
        )
    
    # Select best paraphrasing based on q_score and semantic similarity
    sts_scores = sts_model.encode([T_o, *watermarked["text"]], convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(sts_scores[0], sts_scores[1:], dim=1).cpu()
    selection_score = cos_sim + watermarked["q_score"]
    selection = torch.argmax(selection_score)

    watermarked = watermarked["text"][selection]

    print(f"\nWatermarked text T_w:\n\n{watermarked}\n")

    # Verify on original text
    res = watermarker.verify(T_o)[0]
    q = res[k_p-1]
    print(f"Verification score of T_o: \033[93m{q:.4f}\033[0m")

    # Verify on watermarked text
    res = watermarker.verify(watermarked)[0]
    q = res[k_p-1]
    print(f"Verification score of T_w: \033[92m{q:.4f}\033[0m")

    print(f"\nSTS score of T_w         : \033[94m{cos_sim[selection].item():.4f}\033[0m")

    # Extract from watermarked text
    print(f"\nWatermarking k_p         : \033[95m{k_p}\033[0m")
    print(  f"Extracted k_p from T_w   : \033[96m{np.argmax(res)+1}\033[0m\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate text watermarked with a key')
    parser.add_argument('--id',default=42,type=int,
            help='id: unique ID')
    parser.add_argument('--kappa',default=4,type=float,
            help='kappa: watermarking strength')
    parser.add_argument('--k_p', default=1, type=int,
            help="k_p: Perturbation key")
    parser.add_argument('--model', default='meta-llama/Llama-2-13b-chat-hf', type=str,
            help="model")
    parser.add_argument('--T_o', default='Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation.',
            type=str, help="original_text")
    parser.add_argument('--watermark_fn', default='fourier', type=str)
    parser.add_argument('--device', default='cuda', type=str)

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
        torch_dtype=torch.float16,
        ).to(args.device)

    watermarker = Watermarker(model, tokenizer, id, kappa, k_p, watermarkingFnClass=watermarkingFnClass)

    sts_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=args.device)

    watermark_and_evaluate(T_o)