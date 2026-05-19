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


def build_runs():
    shared = {
        "num_epochs": 1,
        "batch_size": 64,
        "seq_length": 100,
        "learning_rate": 1e-3,
        "run_generation_eval": False,
    }

    runs = [
        {
            "label": "vanilla_rnn",
            "config": TrainConfig(
                model_name="rnn",
                checkpoint_name="best_rnn.pt",
                **shared,
            ),
        },
        {
            "label": "lstm_1_layer",
            "config": TrainConfig(
                model_name="lstm",
                checkpoint_name="best_lstm_1layer.pt",
                num_layers=1,
                **shared,
            ),
        },
        {
            "label": "lstm_2_layer",
            "config": TrainConfig(
                model_name="lstm",
                checkpoint_name="best_lstm_2layer.pt",
                num_layers=2,
                **shared,
            ),
        },
        {
            "label": "attention_lstm_k10",
            "config": TrainConfig(
                model_name="attention_lstm",
                checkpoint_name="best_attention_lstm_k10.pt",
                num_layers=2,
                window_size_k=10,
                **shared,
            ),
        },
    ]

    return runs


def make_log_file(config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint_stem = config.checkpoint_name.replace(".pt", "")
    return os.path.join(config.checkpoint_dir, f"{checkpoint_stem}_comparison_log.txt")


def write_log(text, log_file):
    print(text)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def prepare_log_file(log_file, title):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * 80 + "\n")


def train_and_test_one_run(run, device):
    label = run["label"]
    config = run["config"]

    set_seed(config.seed)

    log_file = make_log_file(config)
    prepare_log_file(log_file, f"Training log for {label}")

    write_log(f"Run label: {label}", log_file)
    write_log(f"Model type: {config.model_name}", log_file)
    write_log(f"Checkpoint file: {config.checkpoint_name}", log_file)
    write_log(f"Full config: {config}", log_file)

    train_loader, val_loader, test_loader = get_dataloaders(
        path=config.tokenized_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
    )

    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    best_epoch = None
    best_val_loss = float("inf")
    last_train_loss = None
    last_val_ppl = None

    run_start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()

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

        epoch_seconds = time.time() - epoch_start_time

        write_log(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.2f} | "
            f"time={epoch_seconds:.1f}s",
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

            write_log(f"Saved new best checkpoint: {saved_path}", log_file)

    load_best_checkpoint(model, config, device)

    test_loss, test_ppl = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=loss_fn,
        device=device,
    )

    total_seconds = time.time() - run_start_time

    result = {
        "label": label,
        "model_name": config.model_name,
        "checkpoint_name": config.checkpoint_name,
        "num_layers": config.num_layers,
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "window_size_k": config.window_size_k if config.model_name == "attention_lstm" else None,
        "best_epoch": best_epoch,
        "last_train_loss": last_train_loss,
        "best_val_loss": best_val_loss,
        "last_val_ppl": last_val_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "total_time_seconds": total_seconds,
        "config": asdict(config),
    }

    write_log("\nFinal result:", log_file)
    write_log(json.dumps(result, indent=2), log_file)

    return result


def save_comparison_report(results, output_json="results/model_comparison.json"):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    output_txt = output_json.replace(".json", ".txt")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Model Comparison Summary\n")
        f.write("=" * 80 + "\n\n")

        header = (
            "label | model | layers | K | best_val_loss | "
            "test_loss | test_ppl | time_seconds\n"
        )
        f.write(header)
        f.write("-" * 80 + "\n")

        for item in results:
            f.write(
                f"{item['label']} | "
                f"{item['model_name']} | "
                f"{item['num_layers']} | "
                f"{item['window_size_k']} | "
                f"{item['best_val_loss']:.4f} | "
                f"{item['test_loss']:.4f} | "
                f"{item['test_ppl']:.2f} | "
                f"{item['total_time_seconds']:.1f}\n"
            )

    print(f"\nSaved JSON summary to: {output_json}")
    print(f"Saved text summary to: {output_txt}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    runs = build_runs()
    all_results = []

    for run in runs:
        print("\n" + "=" * 80)
        print(f"Starting run: {run['label']}")
        print("=" * 80)

        result = train_and_test_one_run(run, device)
        all_results.append(result)

    save_comparison_report(all_results)


if __name__ == "__main__":
    main()