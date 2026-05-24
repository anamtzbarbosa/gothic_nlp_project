"""
Generate all README figures from training logs and evaluation results.
Run from the project root:
    python src/evaluation/plot_readme_figures.py
"""
import json
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "legend.fontsize":    8,
    "legend.framealpha":  0.85,
    "figure.dpi":         150,
})

OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette (vivid, high-contrast, print-friendly) ───────────────────────────
PAL = {
    "rnn":    "#e63946",   # vivid red
    "lstm1":  "#2196f3",   # bright blue
    "lstm2":  "#4caf50",   # vivid green
    "lstm3":  "#9c27b0",   # deep purple
    "attn1":  "#00bcd4",   # cyan
    "attn2":  "#ff9800",   # amber
    "attn3":  "#795548",   # brown
    "init":   "#2196f3",   # blue  → initial attn config
    "reg":    "#e63946",   # red   → regularized attn config
    "h64":    "#b0bec5",   # blue-grey
    "h128":   "#29b6f6",   # light blue
    "h256":   "#e63946",   # vivid red
}

MODEL_COLORS = [PAL["rnn"], PAL["lstm1"], PAL["lstm2"], PAL["lstm3"],
                PAL["attn1"], PAL["attn2"], PAL["attn3"]]
MODEL_LABELS = ["RNN", "LSTM-1", "LSTM-2", "LSTM-3",
                "Attn-LSTM-1", "Attn-LSTM-2", "Attn-LSTM-3"]
MODEL_LABELS_WRAP = ["RNN", "LSTM\n1-layer", "LSTM\n2-layer", "LSTM\n3-layer",
                     "Attn-LSTM\n1-layer", "Attn-LSTM\n2-layer", "Attn-LSTM\n3-layer"]


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _mark_best(ax, epoch, train_vals, val_vals):
    """Place a large ✕ marker at the best epoch on both train and val curves."""
    idx = epoch - 1
    ax.plot(epoch, train_vals[idx], marker="X", ms=14, color="black",
            zorder=5, markeredgewidth=0.5, markeredgecolor="white",
            linestyle="none", label=f"best epoch ({epoch})")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Final RNN / LSTM: Train vs Val Loss
# ─────────────────────────────────────────────────────────────────────────────
def plot_rnn_lstm_curves():
    models = [
        ("RNN",        PAL["rnn"],   1,
         [4.8985, 4.6228, 4.5752], [4.8263, 4.8552, 4.8643]),
        ("LSTM 1-layer", PAL["lstm1"], 1,
         [4.6376, 4.2796, 4.1868], [4.5780, 4.5925, 4.6366]),
        ("LSTM 2-layer", PAL["lstm2"], 2,
         [4.5234, 4.2732, 4.2199, 4.1943], [4.3644, 4.3569, 4.3626, 4.3652]),
        ("LSTM 3-layer", PAL["lstm3"], 1,
         [4.3518, 4.0904, 4.0321], [4.3230, 4.3260, 4.3362]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    axes = axes.flatten()

    for ax, (label, color, best, train, val) in zip(axes, models):
        epochs = list(range(1, len(train) + 1))
        ax.plot(epochs, train, color=color, ls="--", marker="o", ms=5,
                lw=1.8, label="Train loss")
        ax.plot(epochs, val,   color=color, ls="-",  marker="s", ms=5,
                lw=1.8, label="Val loss")
        _mark_best(ax, best, train, val)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend()

    fig.suptitle("Final RNN / LSTM — Train vs Validation Loss",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig1_rnn_lstm_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Attention LSTM: Train vs Val Loss (initial config)
# ─────────────────────────────────────────────────────────────────────────────
def plot_attn_curves():
    models = [
        ("Attn-LSTM 1-layer", PAL["attn1"], 1,
         [4.014, 3.636, 3.527], [4.923, 5.127, 5.259]),
        ("Attn-LSTM 2-layer", PAL["attn2"], 1,
         [4.206, 3.885, 3.800], [4.392, 4.432, 4.464]),
        ("Attn-LSTM 3-layer  ★", PAL["attn3"], 1,
         [4.282, 3.957, 3.875], [4.331, 4.350, 4.369]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, (label, color, best, train, val) in zip(axes, models):
        epochs = list(range(1, len(train) + 1))
        ax.plot(epochs, train, color=color, ls="--", marker="o", ms=5,
                lw=1.8, label="Train loss")
        ax.plot(epochs, val,   color=color, ls="-",  marker="s", ms=5,
                lw=1.8, label="Val loss")
        _mark_best(ax, best, train, val)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend()

    fig.suptitle("Attention LSTM (LR=1e-3, dropout=0.3) — Train vs Validation Loss",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig2_attn_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Attention LSTM: initial vs regularized config per layer count
# ─────────────────────────────────────────────────────────────────────────────
def plot_attn_configs():
    layers = [
        ("1-layer",
         ([4.014, 3.636, 3.527], [4.923, 5.127, 5.259], 1),
         ([4.174, 3.747, 3.622], [4.795, 4.940, 5.062], 1)),
        ("2-layer",
         ([4.206, 3.885, 3.800], [4.392, 4.432, 4.464], 1),
         ([4.447, 4.067, 3.970, 3.913], [4.382, 4.363, 4.373, 4.395], 2)),
        ("3-layer",
         ([4.282, 3.957, 3.875], [4.331, 4.350, 4.369], 1),
         ([4.568, 4.162, 4.067, 4.009], [4.392, 4.373, 4.389, 4.409], 2)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    for ax, (title, (tr1, vl1, b1), (tr2, vl2, b2)) in zip(axes, layers):
        e1 = list(range(1, len(tr1) + 1))
        e2 = list(range(1, len(tr2) + 1))
        ax.plot(e1, tr1, color=PAL["init"],  ls="--", marker="o", ms=4, lw=1.6,
                label="LR=1e-3, drop=0.3 — train")
        ax.plot(e1, vl1, color=PAL["init"],  ls="-",  marker="s", ms=4, lw=1.6,
                label="LR=1e-3, drop=0.3 — val")
        ax.plot(e2, tr2, color=PAL["reg"],   ls="--", marker="^", ms=4, lw=1.6,
                label="LR=5e-4, drop=0.4 — train")
        ax.plot(e2, vl2, color=PAL["reg"],   ls="-",  marker="D", ms=4, lw=1.6,
                label="LR=5e-4, drop=0.4 — val")
        # mark best epochs with ✕
        ax.plot(b1, tr1[b1-1], marker="X", ms=13, color=PAL["init"],
                zorder=5, markeredgewidth=1, markeredgecolor="white", linestyle="none")
        ax.plot(b2, tr2[b2-1], marker="X", ms=13, color=PAL["reg"],
                zorder=5, markeredgewidth=1, markeredgecolor="white", linestyle="none")
        ax.set_title(f"Attn-LSTM {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7)

    fig.suptitle(
        "Attention LSTM — initial config (LR=1e-3, drop=0.3) vs "
        "regularized config (LR=5e-4, drop=0.4)",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig3_attn_configs.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Hidden dimension: val PPL comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_hidden_dim():
    labels = ["RNN", "LSTM\n1-layer", "LSTM\n2-layer", "LSTM\n3-layer"]
    h64  = [194.67, 143.08, 140.32, 132.59]
    h128 = [141.87, 109.73,  99.01,  92.23]
    h256 = [124.65,  97.97,  79.06,  75.34]

    x = np.arange(len(labels))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w, h64,  w, label="hidden=64  (1-epoch probe)",
                color=PAL["h64"],  edgecolor="white", lw=0.6)
    b2 = ax.bar(x,     h128, w, label="hidden=128",
                color=PAL["h128"], edgecolor="white", lw=0.6)
    b3 = ax.bar(x + w, h256, w, label="hidden=256",
                color=PAL["h256"], edgecolor="white", lw=0.6)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=7.5,
                    color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Validation Perplexity (lower is better)")
    ax.set_title("Effect of Hidden Dimension on Validation Perplexity")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig4_hidden_dim.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Val + Test PPL: all final models
# ─────────────────────────────────────────────────────────────────────────────
def plot_val_test_ppl():
    entries = [
        ("RNN",                       124.65, 136.89, PAL["rnn"]),
        ("LSTM\n1-layer",              97.97, 108.07, PAL["lstm1"]),
        ("LSTM\n2-layer",              79.06,  87.10, PAL["lstm2"]),
        ("LSTM\n3-layer",              75.34,  84.78, PAL["lstm3"]),
        ("Attn-LSTM\n1-layer",        120.90, 138.91, PAL["attn1"]),
        ("Attn-LSTM\n2-layer",         78.51,  89.01, PAL["attn2"]),
        ("Attn-LSTM\n3-layer",         76.02,  86.00, PAL["attn3"]),
        ("Attn-LSTM\n3-layer\nseq=200",75.90,  85.33, "#c0936c"),
    ]

    names    = [e[0] for e in entries]
    val_ppls = [e[1] for e in entries]
    tst_ppls = [e[2] for e in entries]
    colors   = [e[3] for e in entries]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bv = ax.bar(x - w/2, val_ppls, w, color=colors, alpha=0.45,
                edgecolor="white", lw=0.6, label="Val PPL")
    bt = ax.bar(x + w/2, tst_ppls, w, color=colors, alpha=0.95,
                edgecolor="white", lw=0.6, label="Test PPL")

    for bar, v in zip(bv, val_ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(bt, tst_ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Val and Test Perplexity — All Final Models")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig5_val_test_ppl.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Evaluation: quality metrics (T=0.5)
# ─────────────────────────────────────────────────────────────────────────────
def plot_eval_quality():
    # t0p5 values from JSON files
    data = {
        "BLEU-1": [0.1213, 0.1850, 0.1847, 0.1881, 0.1754, 0.1779, 0.1844],
        "BLEU-4s (smoothed)": [0.0023, 0.0029, 0.0045, 0.0087, 0.0027, 0.0052, 0.0037],
        "BERTScore F1": [0.8100, 0.8186, 0.8228, 0.8227, 0.8181, 0.8237, 0.8241],
        "Bigram Overlap": [0.0079, 0.0112, 0.0134, 0.0156, 0.0098, 0.0122, 0.0128],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    x = np.arange(len(MODEL_LABELS))
    w = 0.6

    for ax, (metric, vals) in zip(axes, data.items()):
        bars = ax.bar(x, vals, w, color=MODEL_COLORS, edgecolor="white", lw=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xlim(-0.6, len(MODEL_LABELS) - 0.4)

    # shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODEL_COLORS]
    fig.legend(handles, MODEL_LABELS, loc="lower center", ncol=7,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Generation Quality Metrics — All Models (T=0.5, top-p=0.85)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "fig6_eval_quality.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Evaluation: diversity & fluency metrics (T=0.5 vs T=0.7)
# ─────────────────────────────────────────────────────────────────────────────
def plot_eval_diversity():
    t5 = {
        "Distinct-2":        [0.4922, 0.6435, 0.5948, 0.5868, 0.6795, 0.6209, 0.5924],
        "TTR":               [0.6433, 0.7704, 0.7176, 0.7236, 0.7745, 0.7186, 0.7162],
        "Spelling Accuracy": [0.9229, 0.9209, 0.9366, 0.9378, 0.9155, 0.9231, 0.9335],
        "Generated PPL":     [20.44,  18.30,  17.28,  14.76,  11.61,  12.20,  11.96],
    }
    t7 = {
        "Distinct-2":        [0.6462, 0.7661, 0.7436, 0.7336, 0.7630, 0.7341, 0.7375],
        "TTR":               [0.6876, 0.8098, 0.7842, 0.7809, 0.8009, 0.7717, 0.7654],
        "Spelling Accuracy": [0.9024, 0.9195, 0.9238, 0.9217, 0.9196, 0.9169, 0.9242],
        "Generated PPL":     [36.34,  30.19,  28.19,  23.02,  19.28,  19.44,  19.29],
    }

    metrics = list(t5.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    x = np.arange(len(MODEL_LABELS))
    w = 0.35

    for ax, metric in zip(axes, metrics):
        v5 = t5[metric]
        v7 = t7[metric]
        b5 = ax.bar(x - w/2, v5, w, color=MODEL_COLORS, alpha=0.55,
                    edgecolor="white", lw=0.5, label="T=0.5")
        b7 = ax.bar(x + w/2, v7, w, color=MODEL_COLORS, alpha=0.95,
                    edgecolor="white", lw=0.5, label="T=0.7")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xlim(-0.6, len(MODEL_LABELS) - 0.4)

        # light vs dark legend for temperature
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#888", alpha=0.5, label="T=0.5"),
                            Patch(facecolor="#888", alpha=0.95, label="T=0.7")],
                  fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODEL_COLORS]
    fig.legend(handles, MODEL_LABELS, loc="lower center", ncol=7,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Diversity & Fluency Metrics — T=0.5 vs T=0.7",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "fig7_eval_diversity.png")


if __name__ == "__main__":
    print("Generating training curve plots...")
    plot_rnn_lstm_curves()
    plot_attn_curves()
    plot_attn_configs()

    print("Generating analysis plots...")
    plot_hidden_dim()
    plot_val_test_ppl()

    print("Generating evaluation metric plots...")
    plot_eval_quality()
    plot_eval_diversity()

    print(f"\nAll figures saved to {OUT_DIR}/")
