import json
import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train import (
    TrainConfig,
    set_seed,
    get_device,
    build_model,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_best_checkpoint,
    log,
)
from dataset import get_dataloaders, save_splits
from plot_results import plot_training_curves, plot_model_comparison

CHECKPOINT_DIR = "checkpoints/lstm_weight_decay"
RESULTS_DIR = "results/lstm_weight_decay"
EARLY_STOPPING_PATIENCE = 2
WEIGHT_DECAY = 1e-4




def lstm_runs(num_epochs=4):
    common = {
        "num_epochs": num_epochs,
        "batch_size": 64,
        "embed_dim": 128,
        "vocab_size": 5000,
        "run_generation_eval": False,
        "generation_num_samples": 50,
        "generation_prompt_length": 30,
        "generation_temperature": 0.8,
        "generation_top_p": 0.9,
        "checkpoint_dir": CHECKPOINT_DIR,
    }

    return [
        # {
        #     "label": "lstm_wd_l1_s200_h128_d0p3_lr0p0005",
        #     "config": TrainConfig(
        #         model_name="lstm",
        #         checkpoint_name="lstm_wd_l1_s200_h128_d0p3_lr0p0005.pt",
        #         seq_length=200,
        #         num_layers=1,
        #         hidden_dim=128,
        #         dropout=0.3,
        #         learning_rate=5e-4,
        #         generation_output_path=f"{RESULTS_DIR}/lstm_wd_l1_generation",
        #         **common,
        #     ),
        # },
        # {
        #     "label": "lstm_wd_l2_s200_h128_d0p3_lr0p001",
        #     "config": TrainConfig(
        #         model_name="lstm",
        #         checkpoint_name="lstm_wd_l2_s200_h128_d0p3_lr0p001.pt",
        #         seq_length=200,
        #         num_layers=2,
        #         hidden_dim=128,
        #         dropout=0.3,
        #         learning_rate=1e-3,
        #         generation_output_path=f"{RESULTS_DIR}/lstm_wd_l2_generation",
        #         **common,
        #     ),
        # },
        {
            "label": "lstm_wd_l3_s100_h256_d0p3_lr0p001",
            "config": TrainConfig(
                model_name="lstm",
                checkpoint_name="lstm_wd_l3_s100_h256_d0p3_lr0p001.pt",
                seq_length=100,
                num_layers=3,
                hidden_dim=256,
                dropout=0.3,
                learning_rate=1e-3,
                generation_output_path=f"{RESULTS_DIR}/lstm_wd_l3_generation",
                **common,
            ),
        },
    ]


def make_log_path(label):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{label}_training_log.txt")


def train_one_run(run, device):
    label = run["label"]
    config = run["config"]

    set_seed(config.seed)

    log_path = make_log_path(label)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"LSTM Weight Decay Run: {label}\n")
        f.write(f"weight_decay={WEIGHT_DECAY}, scheduler=ReduceLROnPlateau(factor=0.5, patience=1)\n")
        f.write("=" * 80 + "\n")

    log(f"Run: {label}", log_path)
    log(f"Config: {config}", log_path)
    log(f"Weight decay: {WEIGHT_DECAY}", log_path)

    train_loader, val_loader, test_loader = get_dataloaders(
        path=config.tokenized_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
    )

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            log_path=log_path,
        )

        val_loss, val_ppl = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "lr": current_lr,
            "epoch_time": epoch_time,
        })

        log(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"LR: {current_lr:.6f}"
            + (f" → {new_lr:.6f} (reduced)" if new_lr < current_lr else "") +
            f" | Time: {epoch_time:.1f}s",
            log_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                val_loss=val_loss,
                epoch=epoch,
            )
            log(f"Saved best checkpoint to {checkpoint_path}", log_path)
        else:
            epochs_without_improvement += 1
            log(f"No improvement for {epochs_without_improvement} epoch(s)", log_path)
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                log(f"Early stopping triggered at epoch {epoch}", log_path)
                break

    log("Loading best checkpoint before final test...", log_path)
    best_checkpoint = load_best_checkpoint(model, config, device)

    test_loss, test_ppl = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    result = {
        "label": label,
        "model_name": config.model_name,
        "checkpoint_name": config.checkpoint_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_checkpoint["val_loss"],
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "total_time_seconds": time.time() - start_time,
        "history": history,
        "config": asdict(config),
    }

    log("\nFinal test result:", log_path)
    log(json.dumps(result, indent=2), log_path)

    print(f"\nSaving plots for {label}...")
    plot_training_curves(result["history"], label)

    return result


def save_summary(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(RESULTS_DIR, "results.json")
    txt_path = os.path.join(RESULTS_DIR, "summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("LSTM Weight Decay Training Summary\n")
        f.write(f"weight_decay={WEIGHT_DECAY}, scheduler=ReduceLROnPlateau(factor=0.5, patience=1)\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(
                f"{r['label']} | "
                f"best_epoch={r['best_epoch']} | "
                f"val_loss={r['best_val_loss']:.4f} | "
                f"test_loss={r['test_loss']:.4f} | "
                f"test_ppl={r['test_ppl']:.2f}\n"
            )

    print(f"\nSaved summary to: {json_path}")
    print(f"Saved summary to: {txt_path}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    runs = lstm_runs(num_epochs=10)
    results = []

    print(f"\nTraining {len(runs)} LSTM models with AdamW (weight_decay={WEIGHT_DECAY}) + ReduceLROnPlateau")
    print(f"Early stopping patience={EARLY_STOPPING_PATIENCE}, max epochs={runs[0]['config'].num_epochs}")
    print("Splitting and saving data from data/corpus_tokenized.pkl (seed=42)...")
    save_splits()


    for i, run in enumerate(runs, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(runs)}] Starting run: {run['label']}")
        print("=" * 80)

        result = train_one_run(run, device)
        results.append(result)
        save_summary(results)

    save_summary(results)

    print("\nSaving comparison plots...")
    plot_model_comparison(results)


if __name__ == "__main__":
    main()
