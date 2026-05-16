import os
import matplotlib.pyplot as plt

PLOT_DIR = "results/final_models/plots"

COLORS = {
    "final_rnn_h128_lr0p0005": "#e74c3c",
    "final_lstm_l1_h128_d0p3_lr0p0005": "#3498db",
    "final_lstm_l2_h256_d0p3_lr0p001": "#2ecc71",
    "final_attention_lstm_l2_h128_d0p3_lr0p001_k20": "#9b59b6",
}

NAMES = {
    "final_rnn_h128_lr0p0005": "RNN",
    "final_lstm_l1_h128_d0p3_lr0p0005": "LSTM (1-layer)",
    "final_lstm_l2_h256_d0p3_lr0p001": "LSTM (2-layer)",
    "final_attention_lstm_l2_h128_d0p3_lr0p001_k20": "Attention LSTM",
}


def plot_training_curves(history, label):
    os.makedirs(PLOT_DIR, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_ppls = [h["val_ppl"] for h in history]
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]

    name = NAMES.get(label, label)
    color = COLORS.get(label, "#333333")

    # train vs val loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", color=color, linestyle="--", marker="o")
    ax.plot(epochs, val_losses, label="Val loss", color=color, linestyle="-", marker="s")
    ax.axvline(x=best_epoch, color="gray", linestyle=":", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax.set_title(f"{name} — Train vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"{label}_loss.png"), dpi=150)
    plt.close(fig)

    # val perplexity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_ppls, label="Val perplexity", color=color, linestyle="-", marker="s")
    ax.set_title(f"{name} — Validation Perplexity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"{label}_perplexity.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved plots for {name}")


def plot_model_comparison(results):
    os.makedirs(PLOT_DIR, exist_ok=True)

    # val loss of all models
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        epochs = [h["epoch"] for h in r["history"]]
        val_losses = [h["val_loss"] for h in r["history"]]
        ax.plot(epochs, val_losses, label=NAMES.get(r["label"], r["label"]),
                color=COLORS.get(r["label"], "#333333"), marker="o")
    ax.set_title("Validation Loss — All Models")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "comparison_val_loss.png"), dpi=150)
    plt.close(fig)

    # test perplexity bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [NAMES.get(r["label"], r["label"]) for r in results]
    ppls = [r["test_ppl"] for r in results]
    colors = [COLORS.get(r["label"], "#333333") for r in results]
    bars = ax.bar(names, ppls, color=colors, width=0.5, edgecolor="white")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{ppl:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Test Perplexity — All Models")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "comparison_test_ppl.png"), dpi=150)
    plt.close(fig)

    print("  Saved comparison plots")
