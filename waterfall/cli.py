import argparse
import os

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from waterfall.WatermarkerBase import Watermarker, model_from_pretrained_kwargs
from waterfall.WatermarkingFnFourier import WatermarkingFnFourier
from waterfall.WatermarkingFnSquare import WatermarkingFnSquare
from waterfall.sts import STS_scorer
from waterfall.watermark import detect_gpu, verify_texts, watermark_texts


def pretty_print(
    T_o: str,
    T_w: str,
    sts_score: float,
    T_o_q_score: float,
    T_w_q_score: float,
    k_p: int,
    T_w_k_p: int,
) -> None:
    print(f"\nOriginal text T_o:\n\n{T_o}\n")
    print(f"\nWatermarked text T_w:\n\n{T_w}\n")

    print(f"Verification score of T_o: \033[93m{T_o_q_score:.4f}\033[0m")
    print(f"Verification score of T_w: \033[92m{T_w_q_score:.4f}\033[0m\n")
    print(f"STS score of T_w         : \033[94m{sts_score:.4f}\033[0m\n")
    print(f"Watermarking k_p         : \033[95m{k_p}\033[0m")
    print(f"Extracted k_p from T_w   : \033[96m{T_w_k_p}\033[0m\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="generate text watermarked with a key")
    parser.add_argument("--id", default=42, type=int, help="id: unique ID")
    parser.add_argument("--kappa", default=2.0, type=float, help="kappa: watermarking strength")
    parser.add_argument("--k_p", default=1, type=int, help="k_p: Perturbation key")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", type=str, help="watermarking model")
    parser.add_argument("--sts_model", default="google/embeddinggemma-300m", type=str, help="STS model")
    parser.add_argument("--T_o", default=None, type=str, help="original_text")
    parser.add_argument("--watermark_fn", default="fourier", type=str, help="watermarking function, can be 'fourier' or 'square'")
    parser.add_argument("--device", default=detect_gpu(), type=str, help="device to use for generation")
    parser.add_argument("--num_beam_groups", default=4, type=int, help="number of beam groups for generation")
    parser.add_argument("--beams_per_group", default=2, type=int, help="number of beams per group for generation")
    parser.add_argument("--diversity_penalty", default=0.5, type=float, help="diversity penalty for group beam search")
    parser.add_argument(
        "--STS_scale",
        default=2.0,
        type=float,
        help="scale factor for trade-off between STS and q score. Higher means more emphasis on STS.",
    )

    args = parser.parse_args()

    if args.watermark_fn == "fourier":
        watermarkingFnClass = WatermarkingFnFourier
    elif args.watermark_fn == "square":
        watermarkingFnClass = WatermarkingFnSquare
    else:
        raise ValueError("Invalid watermarking function")

    T_o = args.T_o
    if T_o is None:
        T_o = (
            "Protecting intellectual property (IP) of text such as articles and code is increasingly important, "
            "especially as sophisticated attacks become possible, such as paraphrasing by large language models "
            "(LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing "
            "text watermarking methods are not robust enough against such attacks nor scalable to millions of users "
            "for practical implementation."
        )
    T_os = [T_o]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device,
        **model_from_pretrained_kwargs,
    )

    watermarker = Watermarker(
        tokenizer=tokenizer,
        model=model,
        id=args.id,
        kappa=args.kappa,
        k_p=args.k_p,
        watermarkingFnClass=watermarkingFnClass,
    )
    sts_model = SentenceTransformer(args.sts_model, device=args.device)

    T_ws = watermark_texts(
        T_os,
        id=args.id,
        k_p=args.k_p,
        kappa=args.kappa,
        watermarker=watermarker,
        sts_model=sts_model,
        beams_per_group=args.beams_per_group,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty,
        STS_scale=args.STS_scale,
        use_tqdm=True,
    )

    q_scores, extracted_k_ps = verify_texts(T_os + T_ws, args.id, watermarker, k_p=args.k_p, return_extracted_k_p=True)

    for i in range(len(T_os)):
        try:
            column_size = os.get_terminal_size().columns
        except OSError:
            column_size = 80

        print("=" * column_size)
        sts_score = STS_scorer(T_os[i], T_ws[i], sts_model)
        pretty_print(
            T_os[i],
            T_ws[i],
            sts_score,
            q_scores[i],
            q_scores[i + len(T_os)],
            args.k_p,
            extracted_k_ps[i + len(T_os)],
        )


if __name__ == "__main__":
    main()
