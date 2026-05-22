import csv
import json
import os
import pickle
import time
from dataclasses import asdict

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import GothicDataset
from train import (
    TrainConfig,
    set_seed,
    get_device,
    build_model,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_best_checkpoint,
)

RESULT_DIR = "results/grid_search_v3_attention_lstm"
CHECKPOINT_DIR = "checkpoints/grid_search_v3_attention_lstm"

FIXED = {
    "num_epochs": 1,
    "embed_dim": 128,
    "vocab_size": 5000,
    "dropout": 0.3,          # Locked to 0.3
    "learning_rate": 1e-3,   # Locked to 1e-3 (Adam default)
}


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def get_dataloaders_from_splits(seq_length, batch_size):
    loaders = []
    for name in ["train", "val", "test"]:
        with open(f"data/{name}_tokens.pkl", "rb") as f:
            tokens = pickle.load(f)
        ds = GothicDataset(tokens, seq_length)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=(name == "train")))
    return loaders[0], loaders[1], loaders[2]


def print_and_save(message, path):
    print(message)
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def make_run_name(params):
    parts = ["attn"]
    short = {"batch_size": "B", "seq_length": "S", "hidden_dim": "H",
             "num_layers": "L", "window_size_k": "K"}
    for key, val in params.items():
        parts.append(f"{short.get(key, key)}{str(val).replace('.', 'p')}")
    return "_".join(parts)


def create_grid():
    runs = []
    # Simplified grid: 2 * 2 * 2 * 2 * 2 = 32 runs instead of 192
    for batch_size in [32, 64]:
        for seq_length in [100, 200]:
            for hidden_dim in [128, 256]:
                for num_layers in [1, 2]: # Removed 3-layer
                    for k in [20, 40]:
                        params = {
                            "batch_size": batch_size,
                            "seq_length": seq_length,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "window_size_k": k,
                        }
                        run_name = make_run_name(params)
                        config = TrainConfig(
                            model_name="cross_attention_lstm",
                            checkpoint_dir=CHECKPOINT_DIR,
                            checkpoint_name=f"{run_name}.pt",
                            **params,
                            **FIXED,
                        )
                        runs.append({"name": run_name, "config": config})
    return runs


def run_one_experiment(run, device):
    run_name = run["name"]
    config = run["config"]

    set_seed(config.seed)
    make_dir(RESULT_DIR)
    log_file = os.path.join(RESULT_DIR, f"{run_name}.log")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Grid Search V3 (Optimized) Attention LSTM: {run_name}\n")
        f.write("=" * 80 + "\n")

    print_and_save(f"run={run_name}", log_file)

    train_loader, val_loader, test_loader = get_dataloaders_from_splits(
        seq_length=config.seq_length,
        batch_size=config.batch_size,
    )

    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    # Weight decay included for stability
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    started = time.time()

    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, config, log_file)
    val_loss, val_ppl = evaluate(model, val_loader, loss_fn, device)

    save_checkpoint(model, optimizer, config, val_loss, epoch=1)
    load_best_checkpoint(model, config, device)
    test_loss, test_ppl = evaluate(model, test_loader, loss_fn, device)

    result = {
        "run_name": run_name,
        "model_name": "cross_attention_lstm",
        "batch_size": config.batch_size,
        "seq_length": config.seq_length,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "window_size_k": config.window_size_k,
        "val_loss": val_loss,
        "test_ppl": test_ppl,
        "total_time_seconds": time.time() - started,
    }

    print_and_save(f"val_loss={val_loss:.4f} test_ppl={test_ppl:.2f}", log_file)
    return result


def save_results(results):
    make_dir(RESULT_DIR)
    # Simplified columns for the new grid
    columns = ["run_name", "model_name", "batch_size", "seq_length", "hidden_dim",
               "num_layers", "window_size_k", "val_loss", "test_ppl", "total_time_seconds"]

    with open(os.path.join(RESULT_DIR, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    ranked = sorted(results, key=lambda r: r["val_loss"])
    with open(os.path.join(RESULT_DIR, "summary.txt"), "w") as f:
        f.write("Grid Search V3 — Optimized\n")
        f.write("-" * 80 + "\n")
        for r in ranked:
            f.write(f"{r['run_name']} | val={r['val_loss']:.4f} | test_ppl={r['test_ppl']:.2f}\n")

    print(f"Saved results to {RESULT_DIR}/")

def main():
    device = get_device()
    runs = create_grid()

    print(f"Using device: {device}")
    print(f"Optimized search: 32 total runs.")
    print(f"Fixed: dropout=0.3, lr=1e-3, embed_dim=128, vocab=5000\n")

    results = []
    for i, run in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] {run['name']}")
        result = run_one_experiment(run, device)
        results.append(result)
        save_results(results)

if __name__ == "__main__":
    main()