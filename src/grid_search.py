import csv
import json
import os
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


RESULT_DIR = "results/grid_search"
CHECKPOINT_DIR = "checkpoints/grid_search"


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def print_and_save(message, path):
    print(message)

    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def make_run_name(model_name, params):
    parts = [model_name]

    for key, value in params.items():
        if key in ["model_name", "checkpoint_name"]:
            continue

        short_key = {
            "num_layers": "L",
            "hidden_dim": "H",
            "embed_dim": "E",
            "dropout": "D",
            "learning_rate": "LR",
            "batch_size": "B",
            "seq_length": "S",
            "window_size_k": "K",
        }.get(key, key)

        parts.append(f"{short_key}{value}")

    name = "_".join(str(p).replace(".", "p") for p in parts)
    return name


def add_run(runs, model_name, params):
    run_name = make_run_name(model_name, params)

    config = TrainConfig(
        model_name=model_name,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_name=f"{run_name}.pt",
        **params,
    )

    runs.append(
        {
            "name": run_name,
            "config": config,
        }
    )


def create_grid():
    runs = []

    common = {
        "num_epochs": 1,
        "seq_length": 100,
        "batch_size": 64,
        "embed_dim": 128,
        "vocab_size": 5000,
    }

    for hidden_dim in [128, 256]:
        for lr in [1e-3, 5e-4]:
            add_run(
                runs,
                model_name="rnn",
                params={
                    **common,
                    "hidden_dim": hidden_dim,
                    "learning_rate": lr,
                    "dropout": 0.0,
                    "num_layers": 1,
                },
            )

    for layers in [1, 2]:
        for hidden_dim in [128, 256]:
            for dropout in [0.2, 0.3]:
                for lr in [1e-3, 5e-4]:
                    add_run(
                        runs,
                        model_name="lstm",
                        params={
                            **common,
                            "num_layers": layers,
                            "hidden_dim": hidden_dim,
                            "dropout": dropout,
                            "learning_rate": lr,
                        },
                    )

    for k in [5, 10, 20]:
        for hidden_dim in [128, 256]:
            for dropout in [0.2, 0.3]:
                for lr in [1e-3, 5e-4]:
                    add_run(
                        runs,
                        model_name="attention_lstm",
                        params={
                            **common,
                            "num_layers": 2,
                            "hidden_dim": hidden_dim,
                            "dropout": dropout,
                            "learning_rate": lr,
                            "window_size_k": k,
                        },
                    )

    return runs


def get_log_file(run_name):
    make_dir(RESULT_DIR)
    return os.path.join(RESULT_DIR, f"{run_name}.log")


def save_json(path, data):
    make_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_training_loop(model, loaders, config, device, log_file):
    train_loader, val_loader, _ = loaders

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    best_epoch = None
    last_train_loss = None
    last_val_ppl = None

    for epoch in range(1, config.num_epochs + 1):
        epoch_started = time.time()

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            device=device,
            config=config,
            log_path=log_file,
        )

        val_loss, val_ppl = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=loss_fn,
            device=device,
        )

        last_train_loss = train_loss
        last_val_ppl = val_ppl

        print_and_save(
            f"epoch={epoch}/{config.num_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_ppl={val_ppl:.2f} "
            f"time={time.time() - epoch_started:.1f}s",
            log_file,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            saved_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                val_loss=val_loss,
                epoch=epoch,
            )

            print_and_save(f"saved_best={saved_path}", log_file)

    return {
        "best_epoch": best_epoch,
        "last_train_loss": last_train_loss,
        "best_val_loss": best_val_loss,
        "last_val_ppl": last_val_ppl,
    }


def run_test_metrics(model, test_loader, device):
    loss_fn = nn.CrossEntropyLoss()

    test_loss, test_ppl = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=loss_fn,
        device=device,
    )

    return {
        "test_loss": test_loss,
        "test_ppl": test_ppl,
    }


def run_one_experiment(run, device):
    run_name = run["name"]
    config = run["config"]

    set_seed(config.seed)

    log_file = get_log_file(run_name)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Grid Search Run: {run_name}\n")
        f.write("=" * 80 + "\n")

    print_and_save(f"run={run_name}", log_file)
    print_and_save(f"config={config}", log_file)

    loaders = get_dataloaders(
        path=config.tokenized_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
    )

    model = build_model(config).to(device)

    started = time.time()

    train_result = run_training_loop(
        model=model,
        loaders=loaders,
        config=config,
        device=device,
        log_file=log_file,
    )

    load_best_checkpoint(model, config, device)

    _, _, test_loader = loaders

    test_result = run_test_metrics(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    final_result = {
        "run_name": run_name,
        "model_name": config.model_name,
        "checkpoint_name": config.checkpoint_name,
        "num_layers": config.num_layers,
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "seq_length": config.seq_length,
        "window_size_k": config.window_size_k if config.model_name == "attention_lstm" else None,
        **train_result,
        **test_result,
        "total_time_seconds": time.time() - started,
        "config": asdict(config),
    }

    print_and_save("\nfinal_result:", log_file)
    print_and_save(json.dumps(final_result, indent=2), log_file)

    return final_result


def save_results_table(results):
    make_dir(RESULT_DIR)

    json_path = os.path.join(RESULT_DIR, "all_results.json")
    csv_path = os.path.join(RESULT_DIR, "all_results.csv")
    txt_path = os.path.join(RESULT_DIR, "summary.txt")

    save_json(json_path, results)

    columns = [
        "run_name",
        "model_name",
        "num_layers",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "batch_size",
        "seq_length",
        "window_size_k",
        "best_epoch",
        "last_train_loss",
        "best_val_loss",
        "last_val_ppl",
        "test_loss",
        "test_ppl",
        "total_time_seconds",
        "checkpoint_name",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for result in results:
            writer.writerow({column: result.get(column) for column in columns})

    ranked = sorted(results, key=lambda item: item["best_val_loss"])

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Grid Search Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Sorted by best validation loss\n")
        f.write("-" * 80 + "\n")

        for result in ranked:
            f.write(
                f"{result['run_name']} | "
                f"model={result['model_name']} | "
                f"val={result['best_val_loss']:.4f} | "
                f"test={result['test_loss']:.4f} | "
                f"ppl={result['test_ppl']:.2f}\n"
            )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


def main():
    device = get_device()
    runs = create_grid()
    results = []

    print(f"Using device: {device}")
    print(f"Number of runs: {len(runs)}")

    for i, run in enumerate(runs, start=1):
        print("\n" + "=" * 80)
        print(f"Grid run {i}/{len(runs)}: {run['name']}")
        print("=" * 80)

        result = run_one_experiment(run, device)
        results.append(result)

        save_results_table(results)

    save_results_table(results)


if __name__ == "__main__":
    main()