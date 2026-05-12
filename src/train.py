import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from models import VanillaRNN, DeepLSTM, CrossAttentionLSTM


@dataclass
class TrainConfig:
    # Paths
    tokenized_path: str = "data/corpus_tokenized.pkl"
    checkpoint_path: str = "best_model.pt"

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

    for x, y in train_loader:
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
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def save_checkpoint(model, optimizer, config, val_loss):
    torch.save(
        {
            "model_name": config.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": config.vocab_size,
            "seq_length": config.seq_length,
            "embed_dim": config.embed_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "window_size_k": config.window_size_k,
            "val_loss": val_loss,
        },
        config.checkpoint_path,
    )


def main():
    config = TrainConfig(
        model_name="lstm",
        checkpoint_path="best_lstm.pt",
        num_epochs=10,
        batch_size=64,
        seq_length=100,
        learning_rate=1e-3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training model: {config.model_name}")

    train_loader, val_loader, test_loader = get_dataloaders(
        path=config.tokenized_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
    )

    model = build_model(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")

    for epoch in range(1, config.num_epochs + 1):
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

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, config, val_loss)
            print(f"Saved best model to {config.checkpoint_path}")

    test_loss, test_ppl = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Perplexity: {test_ppl:.2f}")


if __name__ == "__main__":
    main()