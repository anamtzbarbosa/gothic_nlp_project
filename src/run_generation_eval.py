"""
Evaluates all grid search checkpoints on the same fixed test samples.

All models are tested on the exact same 20 (prompt, reference) pairs so
scores are directly comparable. Samples are evenly spaced across the test
set to avoid the clustering-on-one-passage problem from the grid search.

Results are saved to results/generation_eval/ (separate from grid search files).
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

import pickle

try:
    from langdetect import detect as _langdetect
    def _is_english(text):
        try:
            return _langdetect(text) == "en"
        except Exception:
            return True
except ImportError:
    def _is_english(text):
        return True

from dataset import GothicDataset, split_tokens_by_chunks
from evaluate_generation import compute_all_metrics, generate_continuation
from generate import load_bpe_tokenizer, load_model_from_checkpoint

CHECKPOINT_DIR = "checkpoints/grid_search"
OUTPUT_DIR = "results/generation_eval"
TOKENIZER_PATH = "data/gothic_tokenizer.json"
TOKENIZED_PATH = "data/corpus_tokenized.pkl"

NUM_SAMPLES = 20
PROMPT_LENGTH = 30
SEQ_LENGTH = 100
TEMPERATURE = 0.5
TOP_P = 0.9


def get_fixed_samples(tokenizer):
    """
    Returns the same NUM_SAMPLES (prompt_ids, reference_text) pairs for every
    model. Indices are evenly spaced across the full test set so we never
    cluster at the start of the set. Splitting is done at word boundaries to
    avoid partial-word fragments at the start of reference texts.
    """
    with open(TOKENIZED_PATH, "rb") as f:
        tokens = pickle.load(f)

    _, _, test_tokens = split_tokens_by_chunks(tokens)
    dataset = GothicDataset(test_tokens, SEQ_LENGTH)
    n = len(dataset)

    # Center each sample within its segment so index 0 (start of test set) is avoided
    indices = [(2 * i + 1) * n // (2 * NUM_SAMPLES) for i in range(NUM_SAMPLES)]

    samples = []
    candidate = 0
    while len(samples) < NUM_SAMPLES and candidate < n:
        idx = indices[candidate] if candidate < len(indices) else candidate
        candidate += 1

        x, _ = dataset[idx]
        full_sequence = x.tolist()

        full_text = tokenizer.decode(full_sequence)
        if not _is_english(full_text):
            continue

        approx_boundary = len(tokenizer.decode(full_sequence[:PROMPT_LENGTH]))
        split_pos = full_text.find(' ', approx_boundary)
        if split_pos == -1:
            split_pos = approx_boundary

        prompt_text = full_text[:split_pos]
        reference_text = full_text[split_pos:].lstrip()
        prompt_ids = tokenizer.encode(prompt_text)

        samples.append((prompt_ids, reference_text))

    return samples


def evaluate_model(model, tokenizer, samples, device):
    max_new_tokens = SEQ_LENGTH - PROMPT_LENGTH
    generated_texts = []
    reference_texts = []

    for prompt_ids, reference_text in samples:
        generated_ids = generate_continuation(
            model=model,
            prompt_ids=prompt_ids,
            seq_length=SEQ_LENGTH,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            device=device,
        )
        # use len(prompt_ids) not PROMPT_LENGTH since re-encoding may shift the count
        generated_texts.append(tokenizer.decode(generated_ids[len(prompt_ids):]))
        reference_texts.append(reference_text)

    return generated_texts, reference_texts


def save_model_results(run_name, metrics, generated_texts, reference_texts):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_path = os.path.join(OUTPUT_DIR, f"{run_name}.txt")
    json_path = os.path.join(OUTPUT_DIR, f"{run_name}.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Generation Evaluation: {run_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Metrics\n")
        f.write("-" * 80 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\nExamples\n")
        f.write("-" * 80 + "\n")
        for i in range(len(generated_texts)):
            f.write(f"\nExample {i + 1}\n")
            f.write("Generated:\n")
            f.write(generated_texts[i] + "\n\n")
            f.write("Reference:\n")
            f.write(reference_texts[i] + "\n")
            f.write("-" * 80 + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "metrics": metrics,
                "generated_texts": generated_texts,
                "reference_texts": reference_texts,
            },
            f,
            indent=2,
        )


def save_summary(all_results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ranked = sorted(all_results, key=lambda r: r["bleu"], reverse=True)

    txt_path = os.path.join(OUTPUT_DIR, "summary.txt")
    json_path = os.path.join(OUTPUT_DIR, "summary.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Generation Evaluation Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Samples per model: {NUM_SAMPLES} (same fixed samples for all models)\n")
        f.write(f"Prompt length: {PROMPT_LENGTH} tokens\n")
        f.write(f"Generated length: {SEQ_LENGTH - PROMPT_LENGTH} tokens\n")
        f.write("\nSorted by BLEU (descending)\n")
        f.write("-" * 80 + "\n")
        for r in ranked:
            f.write(
                f"{r['run_name']} | "
                f"bleu={r['bleu']:.6f} | "
                f"bert_score_f1={r['bert_score_f1']:.6f} | "
                f"bigram={r['bigram_overlap']:.4f} | "
                f"trigram={r['trigram_overlap']:.4f} | "
                f"distinct1={r['distinct_1']:.4f} | "
                f"distinct2={r['distinct_2']:.4f} | "
                f"spell={r['spelling_accuracy']:.4f}\n"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    print(f"\nSaved summary to {txt_path}")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = load_bpe_tokenizer(TOKENIZER_PATH)

    print(f"Extracting {NUM_SAMPLES} fixed test samples...")
    samples = get_fixed_samples(tokenizer)
    print("Reference texts (first 60 chars each):")
    for i, (_, ref_text) in enumerate(samples):
        print(f"  [{i}] {ref_text[:60].replace(chr(10), ' ')}")

    checkpoints = sorted(
        f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")
    )
    print(f"\nFound {len(checkpoints)} checkpoints\n")

    all_results = []

    for i, ckpt_file in enumerate(checkpoints, start=1):
        run_name = ckpt_file[:-3]
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)

        print(f"[{i}/{len(checkpoints)}] {run_name}")

        model, _ = load_model_from_checkpoint(ckpt_path, device)

        generated_texts, reference_texts = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            device=device,
        )

        metrics = compute_all_metrics(generated_texts, reference_texts)

        save_model_results(run_name, metrics, generated_texts, reference_texts)

        all_results.append(
            {
                "run_name": run_name,
                **metrics,
            }
        )

        print(
            f"  bleu={metrics['bleu']:.6f} | "
            f"bert score={metrics.get('bert_score_f1')} | "
            f"bigram={metrics['bigram_overlap']:.4f} | "
            f"trigram={metrics['trigram_overlap']:.4f} | "
            f"distinct1={metrics['distinct_1']:.4f} | "
            f"distinct2={metrics['distinct_2']:.4f} | "
            f"spell={metrics['spelling_accuracy']:.4f}"
        )

    save_summary(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
