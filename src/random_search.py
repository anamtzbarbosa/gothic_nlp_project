import csv
import json
import os
import random
import time
from dataclasses import asdict

import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
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

RESULT_DIR = "results/random_search"
CHECKPOINT_DIR = "checkpoints/random_search"
NUM_RUNS = 60
SEARCH_SEED = 42

FIXED = {
    "num_epochs": 1,
    "embed_dim": 128,
    "vocab_size": 5000,
}

PARAM_SPACE = {
    "model_name": ["rnn", "lstm"],
    "batch_size": [32, 64, 128],
    "seq_length": [100, 200],
    "hidden_dim": [128, 256],
    "learning_rate": [5e-4, 1e-3],
    "num_layers": [1, 2, 3],       # ignored for RNN (fixed at 1)
    "dropout": [0.0, 0.2, 0.3],   # 0.0 only valid for RNN
}


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def print_and_save(message, path):
    print(message)
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def sample_config(rng):
    model_name = rng.choice(PARAM_SPACE["model_name"])
    batch_size = rng.choice(PARAM_SPACE["batch_size"])
    seq_length = rng.choice(PARAM_SPACE["seq_length"])
    hidden_dim = rng.choice(PARAM_SPACE["hidden_dim"])
    lr = rng.choice(PARAM_SPACE["learning_rate"])

    if model_name == "rnn":
        num_layers = 1
        dropout = rng.choice([0.0, 0.2, 0.3])
    else:
        num_layers = rng.choice(PARAM_SPACE["num_layers"])
        dropout = rng.choice([0.2, 0.3])

    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "hidden_dim": hidden_dim,
        "learning_rate": lr,
        "num_layers": num_layers,
        "dropout": dropout,
    }


def make_run_name(params):
    m = params["model_name"]
    b = params["batch_size"]
    s = params["seq_length"]
    h = params["hidden_dim"]
    l = params["num_layers"]
    d = str(params["dropout"]).replace(".", "p")
    lr = str(params["learning_rate"]).replace(".", "p")
    return f"{m}_B{b}_S{s}_H{h}_L{l}_D{d}_LR{lr}"


def get_log_file(run_name):
    make_dir(RESULT_DIR)
    return os.path.join(RESULT_DIR, f"{run_name}.log")


def run_one_experiment(run_name, params, device):
    config = TrainConfig(
        model_name=params["model_name"],
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_name=f"{run_name}.pt",
        seq_length=params["seq_length"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        **FIXED,
    )

    set_seed(config.seed)
    log_file = get_log_file(run_name)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Random Search Run: {run_name}\n")
        f.write("=" * 80 + "\n")

    print_and_save(f"run={run_name}", log_file)
    print_and_save(f"params={params}", log_file)

    loaders = get_dataloaders(
        path=config.tokenized_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
    )
    train_loader, val_loader, test_loader = loaders

    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    started = time.time()

    train_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        device=device,
        config=config,
        log_path=log_file,
    )

    val_loss, val_ppl = evaluate(model=model, data_loader=val_loader, criterion=loss_fn, device=device)

    save_checkpoint(model=model, optimizer=optimizer, config=config, val_loss=val_loss, epoch=1)
    load_best_checkpoint(model, config, device)

    test_loss, test_ppl = evaluate(model=model, data_loader=test_loader, criterion=loss_fn, device=device)

    result = {
        "run_name": run_name,
        **params,
        "train_loss": train_loss,
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

    json_path = os.path.join(RESULT_DIR, "all_results.json")
    csv_path = os.path.join(RESULT_DIR, "all_results.csv")
    txt_path = os.path.join(RESULT_DIR, "summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    columns = ["run_name", "model_name", "batch_size", "seq_length", "hidden_dim",
               "num_layers", "dropout", "learning_rate",
               "train_loss", "val_loss", "val_ppl", "test_loss", "test_ppl", "total_time_seconds"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({col: r.get(col) for col in columns})

    ranked = sorted(results, key=lambda r: r["val_loss"])
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Random Search Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total runs: {len(results)}\n")
        f.write("Sorted by validation loss\n")
        f.write("-" * 80 + "\n")
        for r in ranked:
            f.write(
                f"{r['run_name']} | "
                f"val={r['val_loss']:.4f} | "
                f"test_ppl={r['test_ppl']:.2f}\n"
            )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


def main():
    device = get_device()
    rng = random.Random(SEARCH_SEED)

    print(f"Using device: {device}")
    print(f"Random search: {NUM_RUNS} runs (seed={SEARCH_SEED})")
    print(f"Parameter space: {PARAM_SPACE}\n")

    sampled = []
    seen = set()
    while len(sampled) < NUM_RUNS:
        params = sample_config(rng)
        run_name = make_run_name(params)
        if run_name not in seen:
            seen.add(run_name)
            sampled.append((run_name, params))

    results = []
    for i, (run_name, params) in enumerate(sampled, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{NUM_RUNS}] {run_name}")
        print("=" * 80)

        result = run_one_experiment(run_name, params, device)
        results.append(result)
        save_results(results)

    save_results(results)


if __name__ == "__main__":
    main()
