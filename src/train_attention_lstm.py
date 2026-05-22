import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GothicDataset
from train import get_device, set_seed, evaluate


# ── Model ─────────────────────────────────────────────────────────────────────

class CrossAttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.3, window_size_k=20, num_heads=4, ff_multiplier=4):
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )

        self.window_size_k = window_size_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_multiplier * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def _make_causal_window_mask(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        distance = positions[None, :] - positions[:, None]
        future_mask = distance > 0
        if self.window_size_k is not None and self.window_size_k > 0:
            mask = future_mask | (distance < -self.window_size_k)
        else:
            mask = future_mask
        return mask

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)

        seq_len = lstm_out.shape[1]
        attn_mask = self._make_causal_window_mask(seq_len, lstm_out.device)

        attn_out, attn_weights = self.self_attention(
            query=lstm_out, key=lstm_out, value=lstm_out,
            attn_mask=attn_mask, need_weights=True, average_attn_weights=True,
        )

        x = self.norm1(lstm_out + self.dropout(attn_out))
        x = self.norm2(x + self.feed_forward(x))
        logits = self.fc(x)

        return logits, hidden, attn_weights


# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_SIZE    = 5000
EMBED_DIM     = 128
HIDDEN_DIM    = 256
NUM_LAYERS    = 3
DROPOUT       = 0.3
WINDOW_SIZE_K = 20
NUM_HEADS     = 4

SEQ_LENGTH  = 100
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS  = 4
PATIENCE    = 2
GRAD_CLIP   = 1.0


CHECKPOINT_PATH = "checkpoints/attention_lstm/best.pt"
RESULT_DIR      = "results/attention_lstm_new"


# ── Data ──────────────────────────────────────────────────────────────────────

def get_loaders():
    loaders = []
    for name in ["train", "val", "test"]:
        with open(f"data/{name}_tokens.pkl", "rb") as f:
            tokens = pickle.load(f)
        ds = GothicDataset(tokens, SEQ_LENGTH)
        loaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(name == "train")))
    return loaders


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False, unit="batch")
    for i, (x, y) in enumerate(pbar, start=1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / i:.4f}")
    return total_loss / len(loader)


CONFIGS = [
    {"num_layers": 2, "window_size_k": 20, "label": "multihead_attn_lstm_2layer_K20", "checkpoint": "checkpoints/attention_lstm_new/multihead_2layer_K20.pt"},
    {"num_layers": 2, "window_size_k": 40, "label": "multihead_attn_lstm_2layer_K40", "checkpoint": "checkpoints/attention_lstm_new/multihead_2layer_K40.pt"},
    {"num_layers": 3, "window_size_k": 20, "label": "multihead_attn_lstm_3layer_K20", "checkpoint": "checkpoints/attention_lstm_new/multihead_3layer_K20.pt"},
    {"num_layers": 3, "window_size_k": 40, "label": "multihead_attn_lstm_3layer_K40", "checkpoint": "checkpoints/attention_lstm_new/multihead_3layer_K40.pt"},
]


def train_model(num_layers, window_size_k, label, checkpoint_path, train_loader, val_loader, test_loader, device):
    import json

    print(f"\n{'='*80}")
    print(f"Training: {label} (num_layers={num_layers}, window_size_k={window_size_k})")
    print(f"{'='*80}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    model = CrossAttentionLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=num_layers,
        dropout=DROPOUT,
        window_size_k=window_size_k,
        num_heads=NUM_HEADS,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        started = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        val_loss, val_ppl = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - started

        print(f"Epoch {epoch}/{NUM_EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f} | ppl={val_ppl:.2f} | {elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ppl": val_ppl})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_config = {
                "model_name": "attention_lstm",
                "vocab_size": VOCAB_SIZE,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": num_layers,
                "dropout": DROPOUT,
                "window_size_k": window_size_k,
                "seq_length": SEQ_LENGTH,
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"  → checkpoint saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  → no improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_ppl = evaluate(model, test_loader, loss_fn, device)
    print(f"\nBest epoch: {ckpt['epoch']} | Test loss={test_loss:.4f} | Test PPL={test_ppl:.2f}")

    result_path = os.path.join(RESULT_DIR, f"{label}_results.json")
    with open(result_path, "w") as f:
        json.dump({"label": label, "num_layers": num_layers, "history": history,
                   "test_loss": test_loss, "test_ppl": test_ppl}, f, indent=2)
    print(f"Results saved to {result_path}")


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_loaders()

    for cfg in CONFIGS:
        train_model(
            num_layers=cfg["num_layers"],
            window_size_k=cfg["window_size_k"],
            label=cfg["label"],
            checkpoint_path=cfg["checkpoint"],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )

    print("\nAll models done!")


if __name__ == "__main__":
    main()
