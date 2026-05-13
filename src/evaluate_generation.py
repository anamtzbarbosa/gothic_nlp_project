import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import torch

from dataset import get_dataloaders
from generate import (
    get_device,
    load_bpe_tokenizer,
    load_model_from_checkpoint,
    sample_next_token,
)


def tokenize_words(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def make_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_ngram_overlap(generated_text, reference_text, n):
    generated_tokens = tokenize_words(generated_text)
    reference_tokens = tokenize_words(reference_text)

    generated_ngrams = make_ngrams(generated_tokens, n)
    reference_ngrams = set(make_ngrams(reference_tokens, n))

    if not generated_ngrams:
        return 0.0

    matches = 0

    for ngram in generated_ngrams:
        if ngram in reference_ngrams:
            matches += 1

    return matches / len(generated_ngrams)


def compute_repetition_rate(text, n=3):
    tokens = tokenize_words(text)
    ngrams = make_ngrams(tokens, n)

    if not ngrams:
        return 0.0

    unique_ngrams = set(ngrams)
    return 1.0 - len(unique_ngrams) / len(ngrams)


def load_dictionary_words():
    dictionary_paths = [
        "/usr/share/dict/words",
        "/usr/dict/words",
    ]

    for path in dictionary_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                words = set()

                for line in f:
                    word = line.strip().lower()

                    if word.isalpha():
                        words.add(word)

                return words

    return None


def compute_spelling_accuracy(text, dictionary_words):
    words = tokenize_words(text)

    if not words:
        return 0.0

    if dictionary_words is None:
        return None

    correct_words = 0

    for word in words:
        if word in dictionary_words:
            correct_words += 1

    return correct_words / len(words)


def compute_corpus_bleu(generated_texts, reference_texts, max_n=4):
    generated_length = 0
    reference_length = 0
    precisions = []

    for n in range(1, max_n + 1):
        total_matches = 0
        total_generated_ngrams = 0

        for generated_text, reference_text in zip(generated_texts, reference_texts):
            generated_tokens = tokenize_words(generated_text)
            reference_tokens = tokenize_words(reference_text)

            if n == 1:
                generated_length += len(generated_tokens)
                reference_length += len(reference_tokens)

            generated_ngrams = Counter(make_ngrams(generated_tokens, n))
            reference_ngrams = Counter(make_ngrams(reference_tokens, n))

            total_generated_ngrams += sum(generated_ngrams.values())

            for ngram, count in generated_ngrams.items():
                total_matches += min(count, reference_ngrams.get(ngram, 0))

        precision = (total_matches + 1e-9) / (total_generated_ngrams + 1e-9)
        precisions.append(precision)

    if generated_length == 0:
        return 0.0

    if generated_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - reference_length / generated_length)

    average_log_precision = sum(math.log(p) for p in precisions) / max_n
    bleu = brevity_penalty * math.exp(average_log_precision)

    return bleu


def generate_continuation(
    model,
    prompt_ids,
    seq_length,
    max_new_tokens,
    temperature,
    top_p,
    device,
):
    generated_ids = list(prompt_ids)

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[-seq_length:]

            x = torch.tensor(
                context_ids,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            output = model(x)
            logits = output[0]

            next_token_logits = logits[:, -1, :]

            next_token = sample_next_token(
                logits=next_token_logits,
                temperature=temperature,
                top_p=top_p,
            )

            generated_ids.append(next_token.item())

    return generated_ids


def collect_generation_pairs(
    model,
    tokenizer,
    test_loader,
    seq_length,
    num_samples,
    prompt_length,
    temperature,
    top_p,
    device,
):
    generated_texts = []
    reference_texts = []

    max_new_tokens = seq_length - prompt_length

    if max_new_tokens <= 0:
        raise ValueError("prompt_length must be smaller than seq_length.")

    for x, _ in test_loader:
        for i in range(x.size(0)):
            if len(generated_texts) >= num_samples:
                return generated_texts, reference_texts

            full_sequence = x[i].tolist()

            prompt_ids = full_sequence[:prompt_length]
            reference_ids = full_sequence[prompt_length:]

            generated_ids = generate_continuation(
                model=model,
                prompt_ids=prompt_ids,
                seq_length=seq_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )

            generated_continuation_ids = generated_ids[prompt_length:]

            generated_text = tokenizer.decode(generated_continuation_ids)
            reference_text = tokenizer.decode(reference_ids)

            generated_texts.append(generated_text)
            reference_texts.append(reference_text)

    return generated_texts, reference_texts


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def compute_all_metrics(generated_texts, reference_texts):
    dictionary_words = load_dictionary_words()

    bleu = compute_corpus_bleu(generated_texts, reference_texts)

    bigram_overlaps = []
    trigram_overlaps = []
    repetition_rates = []
    spelling_scores = []

    for generated_text, reference_text in zip(generated_texts, reference_texts):
        bigram_overlaps.append(
            compute_ngram_overlap(generated_text, reference_text, n=2)
        )

        trigram_overlaps.append(
            compute_ngram_overlap(generated_text, reference_text, n=3)
        )

        repetition_rates.append(
            compute_repetition_rate(generated_text, n=3)
        )

        spelling_score = compute_spelling_accuracy(generated_text, dictionary_words)

        if spelling_score is not None:
            spelling_scores.append(spelling_score)

    metrics = {
        "num_samples": len(generated_texts),
        "bleu": bleu,
        "bigram_overlap": average(bigram_overlaps),
        "trigram_overlap": average(trigram_overlaps),
        "trigram_repetition_rate": average(repetition_rates),
        "spelling_accuracy": average(spelling_scores) if spelling_scores else None,
    }

    return metrics


def save_metrics_and_examples(metrics, generated_texts, reference_texts, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    txt_path = output_prefix.with_suffix(".txt")
    json_path = output_prefix.with_suffix(".json")

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Generation Evaluation Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Metrics\n")
        f.write("-" * 80 + "\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nGenerated Examples\n")
        f.write("-" * 80 + "\n")

        for i in range(min(5, len(generated_texts))):
            f.write(f"\nExample {i + 1}\n")
            f.write("Generated:\n")
            f.write(generated_texts[i] + "\n\n")
            f.write("Reference:\n")
            f.write(reference_texts[i] + "\n")
            f.write("-" * 80 + "\n")

    result_data = {
        "metrics": metrics,
        "generated_examples": generated_texts[:10],
        "reference_examples": reference_texts[:10],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return txt_path, json_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="checkpoints/best_lstm.pt")
    parser.add_argument("--tokenizer", default="data/gothic_tokenizer.json")
    parser.add_argument("--tokenized-data", default="data/corpus_tokenized.pkl")

    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--prompt-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--output", default="results/generation_metrics")

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = load_bpe_tokenizer(args.tokenizer)
    model, model_config = load_model_from_checkpoint(args.checkpoint, device)

    seq_length = model_config["seq_length"]

    print(f"Loaded model: {model_config['model_name']}")
    print(f"Sequence length: {seq_length}")
    print(f"Prompt length: {args.prompt_length}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Samples: {args.num_samples}")

    _, _, test_loader = get_dataloaders(
        path=args.tokenized_data,
        seq_length=seq_length,
        batch_size=args.batch_size,
    )

    generated_texts, reference_texts = collect_generation_pairs(
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        seq_length=seq_length,
        num_samples=args.num_samples,
        prompt_length=args.prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    metrics = compute_all_metrics(generated_texts, reference_texts)

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    txt_path, json_path = save_metrics_and_examples(
        metrics=metrics,
        generated_texts=generated_texts,
        reference_texts=reference_texts,
        output_prefix=args.output,
    )

    print(f"\nSaved txt results to: {txt_path}")
    print(f"Saved json results to: {json_path}")


if __name__ == "__main__":
    main()