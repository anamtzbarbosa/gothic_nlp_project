import json
import os
import time
from dataclasses import asdict

import torch.nn as nn
from torch.optim import AdamW

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
from evaluation.plot_results import plot_training_curves, plot_model_comparison


# Best configs from grid_search_v3 (chosen by best val PPL per model/layer variant)
#
# RNN:    B64, S200, H256, LR=5e-4, D=0.3         → val PPL 124.65
# LSTM L1: B64, S100, H256, LR=5e-4, D=0.3        → val PPL  97.97
# LSTM L2: B32, S100, H256, LR=1e-3, D=0.3        → val PPL  79.06
# LSTM L3: B32, S100, H256, LR=1e-3, D=0.2        → val PPL  75.34

FINAL_CHECKPOINT_DIR = "checkpoints/final_rnn_lstm"
FINAL_RESULTS_DIR = "results/final_rnn_lstm"
EARLY_STOPPING_PATIENCE = 2


def final_runs(num_epochs=5):
    return [
        {
            "label": "final_rnn_b64_s200_h256_d0p3_lr0p0005",
            "config": TrainConfig(
                model_name="rnn",
                checkpoint_name="final_rnn_b64_s200_h256_d0p3_lr0p0005.pt",
                seq_length=200,
                hidden_dim=256,
                num_layers=1,
                dropout=0.3,
                learning_rate=5e-4,
                batch_size=64,
                num_epochs=num_epochs,
                embed_dim=128,
                vocab_size=5000,
                run_generation_eval=False,
                generation_num_samples=50,
                generation_prompt_length=30,
                generation_temperature=0.8,
                generation_top_p=0.9,
                checkpoint_dir=FINAL_CHECKPOINT_DIR,
                generation_output_path=f"{FINAL_RESULTS_DIR}/final_rnn_b64_s200_h256_d0p3_lr0p0005_generation",
            ),
        },
        {
            "label": "final_lstm_l1_b64_s100_h256_d0p3_lr0p0005",
            "config": TrainConfig(
                model_name="lstm",
                checkpoint_name="final_lstm_l1_b64_s100_h256_d0p3_lr0p0005.pt",
                seq_length=100,
                hidden_dim=256,
                num_layers=1,
                dropout=0.3,
                learning_rate=5e-4,
                batch_size=64,
                num_epochs=num_epochs,
                embed_dim=128,
                vocab_size=5000,
                run_generation_eval=False,
                generation_num_samples=50,
                generation_prompt_length=30,
                generation_temperature=0.8,
                generation_top_p=0.9,
                checkpoint_dir=FINAL_CHECKPOINT_DIR,
                generation_output_path=f"{FINAL_RESULTS_DIR}/final_lstm_l1_b64_s100_h256_d0p3_lr0p0005_generation",
            ),
        },
        {
            "label": "final_lstm_l2_b32_s100_h256_d0p3_lr0p001",
            "config": TrainConfig(
                model_name="lstm",
                checkpoint_name="final_lstm_l2_b32_s100_h256_d0p3_lr0p001.pt",
                seq_length=100,
                hidden_dim=256,
                num_layers=2,
                dropout=0.3,
                learning_rate=1e-3,
                batch_size=32,
                num_epochs=num_epochs,
                embed_dim=128,
                vocab_size=5000,
                run_generation_eval=False,
                generation_num_samples=50,
                generation_prompt_length=30,
                generation_temperature=0.8,
                generation_top_p=0.9,
                checkpoint_dir=FINAL_CHECKPOINT_DIR,
                generation_output_path=f"{FINAL_RESULTS_DIR}/final_lstm_l2_b32_s100_h256_d0p3_lr0p001_generation",
            ),
        },
        {
            "label": "final_lstm_l3_b32_s100_h256_d0p2_lr0p001",
            "config": TrainConfig(
                model_name="lstm",
                checkpoint_name="final_lstm_l3_b32_s100_h256_d0p2_lr0p001.pt",
                seq_length=100,
                hidden_dim=256,
                num_layers=3,
                dropout=0.2,
                learning_rate=1e-3,
                batch_size=32,
                num_epochs=num_epochs,
                embed_dim=128,
                vocab_size=5000,
                run_generation_eval=False,
                generation_num_samples=50,
                generation_prompt_length=30,
                generation_temperature=0.8,
                generation_top_p=0.9,
                checkpoint_dir=FINAL_CHECKPOINT_DIR,
                generation_output_path=f"{FINAL_RESULTS_DIR}/final_lstm_l3_b32_s100_h256_d0p2_lr0p001_generation",
            ),
        },
    ]


def make_log_path(config, label):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    return os.path.join(config.checkpoint_dir, f"{label}_training_log.txt")


def train_one_final_model(run, device):
    label = run["label"]
    config = run["config"]

    set_seed(config.seed)

    log_path = make_log_path(config, label)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Final training run: {label}\n")
        f.write("=" * 80 + "\n")

    log(f"Run: {label}", log_path)
    log(f"Config: {config}", log_path)

    train_loader, val_loader, test_loader = get_dataloaders(
        path=config.tokenized_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
    )

    model = build_model(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

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

        epoch_time = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "epoch_time": epoch_time,
            }
        )

        log(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"Time: {epoch_time:.1f}s",
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


def save_final_summary(results):
    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(FINAL_RESULTS_DIR, "final_rnn_lstm_results.json")
    txt_path = os.path.join(FINAL_RESULTS_DIR, "final_rnn_lstm_summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Final RNN & LSTM Training Summary\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(
                f"{r['label']} | "
                f"model={r['model_name']} | "
                f"best_epoch={r['best_epoch']} | "
                f"val_loss={r['best_val_loss']:.4f} | "
                f"test_loss={r['test_loss']:.4f} | "
                f"test_ppl={r['test_ppl']:.2f}\n"
            )

    print(f"\nSaved final JSON summary to: {json_path}")
    print(f"Saved final text summary to: {txt_path}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    print("\nSaving train/val/test splits...")
    save_splits()

    runs = final_runs(num_epochs=5)
    results = []

    print(f"\nTraining {len(runs)} models (max 5 epochs, early stopping patience={EARLY_STOPPING_PATIENCE})")
    print("Models: RNN, LSTM-L1, LSTM-L2, LSTM-L3 (best config per variant from grid_search_v3)\n")

    for i, run in enumerate(runs, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(runs)}] Starting: {run['label']}")
        print("=" * 80)

        result = train_one_final_model(run, device)
        results.append(result)

        save_final_summary(results)

    save_final_summary(results)

    print("\nSaving comparison plots across all models...")
    plot_model_comparison(results)


if __name__ == "__main__":
    main()
