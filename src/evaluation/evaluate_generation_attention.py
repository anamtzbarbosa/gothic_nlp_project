import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from evaluation.generate_attention import load_tokenizer, load_model, generate
from evaluation.evaluate_generation import (
    load_eval_samples,
    compute_all_metrics,
    compute_generated_perplexity,
    save_metrics_and_examples,
)


def collect_from_eval_samples(model, tokenizer, eval_samples, seq_length, temperature, top_p, device):
    generated_texts = []
    reference_texts = []
    prompt_texts = []

    for sample in eval_samples:
        prompt_ids = sample["prompt_ids"]
        reference_text = sample["reference_text"]
        max_new_tokens = seq_length - len(prompt_ids)
        if max_new_tokens <= 0:
            max_new_tokens = seq_length

        prompt_text = sample.get("prompt_text", tokenizer.decode(prompt_ids))
        full_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            seq_length=seq_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        generated_text = full_text[len(prompt_text):]
        generated_texts.append(generated_text)
        reference_texts.append(reference_text)
        prompt_texts.append(prompt_text)

    return generated_texts, reference_texts, prompt_texts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/gothic_tokenizer.json")
    parser.add_argument("--eval-samples", default="data/eval_samples.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", default="results/generation_metrics_attention")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(args.tokenizer)
    model, cfg = load_model(args.checkpoint, device)
    seq_length = cfg["seq_length"]

    print(f"Loaded: {cfg['num_layers']}-layer Attention LSTM | K={cfg['window_size_k']}")
    print(f"Sequence length: {seq_length}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")

    eval_samples = load_eval_samples(args.eval_samples)
    print(f"Loaded {len(eval_samples)} fixed eval samples from {args.eval_samples}")

    generated_texts, reference_texts, prompt_texts = collect_from_eval_samples(
        model=model,
        tokenizer=tokenizer,
        eval_samples=eval_samples,
        seq_length=seq_length,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    metrics = compute_all_metrics(
        generated_texts, reference_texts,
        model=model, tokenizer=tokenizer, seq_length=seq_length, device=device,
    )

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    txt_path, json_path = save_metrics_and_examples(
        metrics=metrics,
        generated_texts=generated_texts,
        reference_texts=reference_texts,
        prompt_texts=prompt_texts,
        output_prefix=args.output,
    )

    print(f"\nSaved txt results to: {txt_path}")
    print(f"Saved json results to: {json_path}")


if __name__ == "__main__":
    main()
