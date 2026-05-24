"""Run evaluation N times per model and report averaged metrics."""
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


MODELS = [
    # (label, checkpoint, eval_script_flag, temp, top_p)
    ("rnn",       "checkpoints/final_rnn_lstm/final_rnn_b64_s200_h256_d0p3_lr0p0005.pt",    "base", 0.5, 0.85),
    ("rnn",       "checkpoints/final_rnn_lstm/final_rnn_b64_s200_h256_d0p3_lr0p0005.pt",    "base", 0.7, 0.90),
    ("lstm_l1",   "checkpoints/final_rnn_lstm/final_lstm_l1_b64_s100_h256_d0p3_lr0p0005.pt","base", 0.5, 0.85),
    ("lstm_l1",   "checkpoints/final_rnn_lstm/final_lstm_l1_b64_s100_h256_d0p3_lr0p0005.pt","base", 0.7, 0.90),
    ("lstm_l2",   "checkpoints/final_rnn_lstm/final_lstm_l2_b32_s100_h256_d0p3_lr0p001.pt", "base", 0.5, 0.85),
    ("lstm_l2",   "checkpoints/final_rnn_lstm/final_lstm_l2_b32_s100_h256_d0p3_lr0p001.pt", "base", 0.7, 0.90),
    ("lstm_l3",   "checkpoints/final_rnn_lstm/final_lstm_l3_b32_s100_h256_d0p2_lr0p001.pt", "base", 0.5, 0.85),
    ("lstm_l3",   "checkpoints/final_rnn_lstm/final_lstm_l3_b32_s100_h256_d0p2_lr0p001.pt", "base", 0.7, 0.90),
    ("attn_l1",   "checkpoints/attention_lstm_new/multihead_1layer_K20.pt",                  "attn", 0.5, 0.85),
    ("attn_l1",   "checkpoints/attention_lstm_new/multihead_1layer_K20.pt",                  "attn", 0.7, 0.90),
    ("attn_l2",   "checkpoints/attention_lstm_new/multihead_2layer_K20.pt",                  "attn", 0.5, 0.85),
    ("attn_l2",   "checkpoints/attention_lstm_new/multihead_2layer_K20.pt",                  "attn", 0.7, 0.90),
    ("attn_l3",   "checkpoints/attention_lstm_new/multihead_3layer_K20.pt",                  "attn", 0.5, 0.85),
    ("attn_l3",   "checkpoints/attention_lstm_new/multihead_3layer_K20.pt",                  "attn", 0.7, 0.90),
]

N_RUNS = 3
EVAL_SAMPLES = "data/eval_samples.json"
TOKENIZER = "data/gothic_tokenizer.json"
OUT_DIR = Path("results/averaged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_KEYS = [
    "bleu_1", "bleu_2", "bleu_3", "bleu_4", "bleu_4_smoothed",
    "bert_score_f1", "bigram_overlap", "trigram_overlap",
    "distinct_1", "distinct_2", "trigram_repetition_rate",
    "spelling_accuracy", "ttr", "generated_perplexity",
]


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def run_one(label, checkpoint, script_type, temp, top_p, run_idx):
    t_tag = str(temp).replace(".", "p")
    out_prefix = str(OUT_DIR / f"{label}_t{t_tag}_run{run_idx}")
    script = (
        "src/evaluation/evaluate_generation_attention.py"
        if script_type == "attn"
        else "src/evaluation/evaluate_generation.py"
    )
    cmd = [
        sys.executable, script,
        "--checkpoint", checkpoint,
        "--tokenizer", TOKENIZER,
        "--eval-samples", EVAL_SAMPLES,
        "--temperature", str(temp),
        "--top-p", str(top_p),
        "--output", out_prefix,
    ]
    print(f"  [{label} t={temp} run={run_idx}] running...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None
    json_path = out_prefix + ".json"
    with open(json_path) as f:
        data = json.load(f)
    return data["metrics"]


def main():
    all_results = {}  # key -> list of metric dicts

    total = len(MODELS) * N_RUNS
    done = 0

    for (label, checkpoint, script_type, temp, top_p) in MODELS:
        key = f"{label}_t{str(temp).replace('.','p')}"
        runs = []
        for run_idx in range(1, N_RUNS + 1):
            done += 1
            print(f"[{done}/{total}]", flush=True)
            m = run_one(label, checkpoint, script_type, temp, top_p, run_idx)
            if m is not None:
                runs.append(m)
        all_results[key] = runs

    print("\n\n=== AVERAGED RESULTS ===\n")
    summary = {}
    for key, runs in all_results.items():
        if not runs:
            continue
        avg_metrics = {}
        for k in NUMERIC_KEYS:
            vals = [r.get(k) for r in runs]
            avg_metrics[k] = avg(vals)
        summary[key] = avg_metrics

        print(f"{key}  (n={len(runs)} runs)")
        for k, v in avg_metrics.items():
            if v is not None:
                print(f"  {k}: {v:.6f}")
        print()

    out_json = OUT_DIR / "averaged_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved averaged results to {out_json}")


if __name__ == "__main__":
    main()
