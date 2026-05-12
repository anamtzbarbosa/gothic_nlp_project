import math
import os
import time
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from models import VanillaRNN, DeepLSTM, CrossAttentionLSTM


@dataclass
class TrainConfig:
    # Paths
    tokenized_path: str = "data/corpus_tokenized.pkl"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best_lstm.pt"

    # Model choice: "rnn", "lstm", or "attention_lstm"
    model_name: str = "lstm"

    # Data
    vocab_size: int = 5000
    seq_length: int = 100
    batch_size: int = 64

    # Model hyperparameters
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    window_size_k: int = 5

    # Training hyperparameters
    learning_rate: float = 1e-3
    num_epochs: int = 10
    grad_clip: float = 1.0
    seed: int = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model(config: TrainConfig):
    if config.model_name == "rnn":
        return VanillaRNN(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
        )

    if config.model_name == "lstm":
        return DeepLSTM(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    if config.model_name == "attention_lstm":
        return CrossAttentionLSTM(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            window_size_k=config.window_size_k,
        )

    raise ValueError(f"Unknown model name: {config.model_name}")


def compute_loss(logits, targets, criterion):
    batch_size, seq_len, vocab_size = logits.shape

    logits = logits.reshape(batch_size * seq_len, vocab_size)
    targets = targets.reshape(batch_size * seq_len)

    return criterion(logits, targets)


def train_one_epoch(model, train_loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0.0
    log_every = 100

    for batch_idx, (x, y) in enumerate(train_loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)
        logits = output[0]

        loss = compute_loss(logits, y, criterion)
        loss.backward()

        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_every == 0:
            avg_loss_so_far = total_loss / batch_idx
            print(
                f"  Batch {batch_idx}/{len(train_loader)} | "
                f"Avg Train Loss: {avg_loss_so_far:.4f}"
            )

    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            logits = output[0]

            loss = compute_loss(logits, y, criterion)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    return avg_loss, perplexity


def save_checkpoint(model, optimizer, config, val_loss, epoch):
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)

    torch.save(
        {
            "epoch": epoch,
            "model_name": config.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "val_loss": val_loss,
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_best_checkpoint(model, config, device):
    checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


def main():
    config = TrainConfig(
        model_name="lstm",
        checkpoint_name="best_lstm.pt",
        num_epochs=1,
        batch_size=64,
        seq_length=100,
        learning_rate=1e-3,
    )

    set_seed(config.seed)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Training model: {config.model_name}")
    print(f"Config: {config}")

    train_loader, val_loader, test_loader = get_dataloaders(
        path=config.tokenized_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
    )

    model = build_model(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, config.num_epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
        )

        val_loss, val_ppl = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - start_time

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "epoch_time": epoch_time,
            }
        )

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                val_loss=val_loss,
                epoch=epoch,
            )
            print(f"Saved best model to {checkpoint_path}")

    print("Loading best checkpoint before final test...")
    best_checkpoint = load_best_checkpoint(model, config, device)

    test_loss, test_ppl = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nTraining finished.")
    print(f"Best validation loss: {best_checkpoint['val_loss']:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Perplexity: {test_ppl:.2f}")

    print("\nTraining history:")
    for row in history:
        print(row)


if __name__ == "__main__":
    main()