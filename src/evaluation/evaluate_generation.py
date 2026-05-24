import argparse
import json
import math
import os
import re
from pathlib import Path
from bert_score import score as bert_score_compute

import torch

from evaluation.generate import (
    get_device,
    load_bpe_tokenizer,
    load_model_from_checkpoint,
    sample_next_token,
)


def tokenize_words(text):
    return re.findall(r"[a-z]+(?:['-][a-z]+)*", text.lower())


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


def compute_ttr(text):
    words = tokenize_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_generated_perplexity(model, tokenizer, generated_texts, seq_length, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for text in generated_texts:
            token_ids = tokenizer.encode(text)
            if len(token_ids) < 2:
                continue
            for start in range(0, len(token_ids) - 1, seq_length):
                chunk = token_ids[start: start + seq_length + 1]
                if len(chunk) < 2:
                    continue
                x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(chunk[1:],  dtype=torch.long, device=device)
                output = model(x)
                logits = output[0].squeeze(0)
                total_loss += criterion(logits, y).item()
                total_tokens += len(y)

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    try:
        return math.exp(avg_loss)
    except OverflowError:
        return float("inf")


def compute_distinct_n(generated_texts, n):
    all_ngrams = []
    for text in generated_texts:
        tokens = tokenize_words(text)
        all_ngrams.extend(make_ngrams(tokens, n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


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


def _modified_precision(generated_tokens, reference_tokens, n):
    from collections import Counter
    gen_ngrams = Counter(make_ngrams(generated_tokens, n))
    ref_ngrams = Counter(make_ngrams(reference_tokens, n))
    matches = sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in gen_ngrams.items())
    total = sum(gen_ngrams.values())
    return matches, total


def _corpus_bleu(generated_texts, reference_texts, max_n, smooth=False):
    gen_len, ref_len = 0, 0
    precisions = []

    for n in range(1, max_n + 1):
        total_matches, total_generated = 0, 0
        for gen, ref in zip(generated_texts, reference_texts):
            g_tok = tokenize_words(gen)
            r_tok = tokenize_words(ref)
            if n == 1:
                gen_len += len(g_tok)
                ref_len += len(r_tok)
            m, t = _modified_precision(g_tok, r_tok, n)
            total_matches += m
            total_generated += t

        if smooth:
            precision = (total_matches + 1) / (total_generated + 1)
        else:
            precision = (total_matches + 1e-9) / (total_generated + 1e-9)
        precisions.append(precision)

    if gen_len == 0:
        return 0.0

    bp = 1.0 if gen_len > ref_len else math.exp(1 - ref_len / gen_len)
    log_avg = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_avg)


def compute_bleu_scores(generated_texts, reference_texts):
    bleu1 = _corpus_bleu(generated_texts, reference_texts, max_n=1)
    bleu2 = _corpus_bleu(generated_texts, reference_texts, max_n=2)
    bleu3 = _corpus_bleu(generated_texts, reference_texts, max_n=3)
    bleu4 = _corpus_bleu(generated_texts, reference_texts, max_n=4)
    bleu4_smoothed = _corpus_bleu(generated_texts, reference_texts, max_n=4, smooth=True)
    return bleu1, bleu2, bleu3, bleu4, bleu4_smoothed


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

            full_text = tokenizer.decode(full_sequence)
            approx_boundary = len(tokenizer.decode(full_sequence[:prompt_length]))
            split_pos = full_text.find(' ', approx_boundary)
            if split_pos == -1:
                split_pos = approx_boundary

            prompt_text = full_text[:split_pos]
            reference_text = full_text[split_pos:].lstrip()
            prompt_ids = tokenizer.encode(prompt_text)

            generated_ids = generate_continuation(
                model=model,
                prompt_ids=prompt_ids,
                seq_length=seq_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )

            generated_text = tokenizer.decode(generated_ids[len(prompt_ids):])

            generated_texts.append(generated_text)
            reference_texts.append(reference_text)

    return generated_texts, reference_texts


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def compute_all_metrics(generated_texts, reference_texts,
                        model=None, tokenizer=None, seq_length=None, device=None):
    dictionary_words = load_dictionary_words()

    bleu1, bleu2, bleu3, bleu4, bleu4_smoothed = compute_bleu_scores(generated_texts, reference_texts)

    P, R, F1 = bert_score_compute(
        generated_texts,
        reference_texts,
        model_type="roberta-large",
        rescale_with_baseline=False,
    )

    mean_bert_score_f1 = F1.mean().item()

    bigram_overlaps = []
    trigram_overlaps = []
    repetition_rates = []
    spelling_scores = []
    ttr_scores = []

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

        ttr_scores.append(compute_ttr(generated_text))

    gen_ppl = None
    if model is not None and tokenizer is not None and seq_length is not None and device is not None:
        gen_ppl = compute_generated_perplexity(model, tokenizer, generated_texts, seq_length, device)

    metrics = {
        "num_samples": len(generated_texts),
        "bleu_1": bleu1,
        "bleu_2": bleu2,
        "bleu_3": bleu3,
        "bleu_4": bleu4,
        "bleu_4_smoothed": bleu4_smoothed,
        "bert_score_f1": mean_bert_score_f1,
        "bigram_overlap": average(bigram_overlaps),
        "trigram_overlap": average(trigram_overlaps),
        "distinct_1": compute_distinct_n(generated_texts, 1),
        "distinct_2": compute_distinct_n(generated_texts, 2),
        "trigram_repetition_rate": average(repetition_rates),
        "spelling_accuracy": average(spelling_scores) if spelling_scores else None,
        "ttr": average(ttr_scores),
        "generated_perplexity": gen_ppl,
    }

    return metrics


def save_metrics_and_examples(metrics, generated_texts, reference_texts, output_prefix, prompt_texts=None):
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
            if prompt_texts is not None:
                f.write("Prompt:\n")
                f.write(prompt_texts[i] + "\n\n")
            f.write("Generated:\n")
            f.write(generated_texts[i] + "\n\n")
            f.write("Reference:\n")
            f.write(reference_texts[i] + "\n")
            f.write("-" * 80 + "\n")

    result_data = {
        "metrics": metrics,
        "examples": [
            {
                "prompt": (prompt_texts[i] if prompt_texts else None),
                "generated": generated_texts[i],
                "reference": reference_texts[i],
            }
            for i in range(min(10, len(generated_texts)))
        ],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return txt_path, json_path


def load_eval_samples(eval_samples_path):
    with open(eval_samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


def collect_from_eval_samples(model, tokenizer, eval_samples, seq_length, temperature, top_p, device):
    generated_texts = []
    reference_texts = []
    prompt_texts = []

    for sample in eval_samples:
        prompt_ids = sample["prompt_ids"]
        reference_text = sample["reference_text"]
        prompt_text = sample.get("prompt_text", tokenizer.decode(prompt_ids))
        max_new_tokens = seq_length - len(prompt_ids)
        if max_new_tokens <= 0:
            max_new_tokens = seq_length

        generated_ids = generate_continuation(
            model=model,
            prompt_ids=prompt_ids,
            seq_length=seq_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )

        generated_text = tokenizer.decode(generated_ids[len(prompt_ids):])
        generated_texts.append(generated_text)
        reference_texts.append(reference_text)
        prompt_texts.append(prompt_text)

    return generated_texts, reference_texts, prompt_texts


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="checkpoints/best_lstm.pt")
    parser.add_argument("--tokenizer", default="data/gothic_tokenizer.json")
    parser.add_argument("--eval-samples", default="data/eval_samples.json")

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