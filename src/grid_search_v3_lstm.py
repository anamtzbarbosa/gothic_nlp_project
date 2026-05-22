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

RESULT_DIR = "results/grid_search_v3_lstm"
CHECKPOINT_DIR = "checkpoints/grid_search_v3_lstm"

FIXED = {
    "num_epochs": 1,
    "embed_dim": 128,
    "vocab_size": 5000,
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
    parts = ["lstm"]
    short = {"batch_size": "B", "seq_length": "S", "hidden_dim": "H",
             "num_layers": "L", "dropout": "D", "learning_rate": "LR"}
    for key, val in params.items():
        parts.append(f"{short.get(key, key)}{str(val).replace('.', 'p')}")
    return "_".join(parts)


def create_grid():
    runs = []
    for batch_size in [32, 64]:
        for seq_length in [100, 200]:
            for hidden_dim in [128, 256]:
                for lr in [5e-4, 1e-3]:
                    for num_layers in [1, 2, 3]:
                        for dropout in [0.2, 0.3]:
                            params = {
                                "batch_size": batch_size,
                                "seq_length": seq_length,
                                "hidden_dim": hidden_dim,
                                "learning_rate": lr,
                                "num_layers": num_layers,
                                "dropout": dropout,
                            }
                            run_name = make_run_name(params)
                            config = TrainConfig(
                                model_name="lstm",
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
        f.write(f"Grid Search V3 LSTM: {run_name}\n")
        f.write("=" * 80 + "\n")

    print_and_save(f"run={run_name}", log_file)

    train_loader, val_loader, test_loader = get_dataloaders_from_splits(
        seq_length=config.seq_length,
        batch_size=config.batch_size,
    )

    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    started = time.time()

    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, config, log_file)
    val_loss, val_ppl = evaluate(model, val_loader, loss_fn, device)

    save_checkpoint(model, optimizer, config, val_loss, epoch=1)
    load_best_checkpoint(model, config, device)
    test_loss, test_ppl = evaluate(model, test_loader, loss_fn, device)

    result = {
        "run_name": run_name,
        "model_name": "lstm",
        "batch_size": config.batch_size,
        "seq_length": config.seq_length,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "total_time_seconds": time.time() - started,
    }

    print_and_save(f"val_loss={val_loss:.4f} test_ppl={test_ppl:.2f}", log_file)
    return result


def save_results(results):
    make_dir(RESULT_DIR)
    columns = ["run_name", "model_name", "batch_size", "seq_length", "hidden_dim",
               "num_layers", "dropout", "learning_rate",
               "val_loss", "val_ppl", "test_loss", "test_ppl", "total_time_seconds"]

    with open(os.path.join(RESULT_DIR, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(RESULT_DIR, "all_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({col: r.get(col) for col in columns})

    ranked = sorted(results, key=lambda r: r["val_loss"])
    with open(os.path.join(RESULT_DIR, "summary.txt"), "w") as f:
        f.write("Grid Search V3 — LSTM\n")
        f.write("=" * 80 + "\n")
        f.write("Searching: batch_size, seq_length, hidden_dim, num_layers, dropout, learning_rate\n")
        f.write("Fixed: embed_dim=128, vocab_size=5000\n")
        f.write("-" * 80 + "\n")
        for r in ranked:
            f.write(f"{r['run_name']} | val={r['val_loss']:.4f} | test_ppl={r['test_ppl']:.2f}\n")

    print(f"Saved results to {RESULT_DIR}/")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-length", type=int, choices=[100, 200], default=None,
                        help="Fix seq_length to 100 or 200. If not set, runs both.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    runs = create_grid()

    if args.seq_length is not None:
        runs = [r for r in runs if r["config"].seq_length == args.seq_length]
        global RESULT_DIR
        RESULT_DIR = f"results/grid_search_v3_lstm_s{args.seq_length}"

    print(f"Using device: {device}")
    print(f"Total runs: {len(runs)}")
    print(f"Seq length: {args.seq_length if args.seq_length else '[100, 200]'}")
    print(f"Searching: batch_size=[32,64], hidden_dim=[128,256]")
    print(f"           num_layers=[1,2,3], dropout=[0.2,0.3], learning_rate=[5e-4,1e-3]")
    print(f"Fixed: embed_dim=128, vocab_size=5000\n")

    results = []
    for i, run in enumerate(runs, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(runs)}] {run['name']}")
        print("=" * 80)
        result = run_one_experiment(run, device)
        results.append(result)
        save_results(results)

    save_results(results)


if __name__ == "__main__":
    main()
