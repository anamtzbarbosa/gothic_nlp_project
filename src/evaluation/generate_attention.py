import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import GothicBPE


class MultiHeadAttentionLSTM(nn.Module):
    """v1 architecture: uses nn.MultiheadAttention (train_attention_lstm.py)."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.3, window_size_k=20, num_heads=4, ff_multiplier=4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.window_size_k = window_size_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
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
        attn_out, _ = self.self_attention(
            query=lstm_out, key=lstm_out, value=lstm_out,
            attn_mask=attn_mask, need_weights=False,
        )
        x = self.norm1(lstm_out + self.dropout(attn_out))
        x = self.norm2(x + self.feed_forward(x))
        return self.fc(x), hidden


class _CustomMultiHeadAttention(nn.Module):
    """Custom W_q/W_k/W_v attention used in v2 (train_attention_lstm_v2.py)."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, H = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B, T, H)
        return self.out_proj(context), weights


class MultiHeadAttentionLSTMV2(nn.Module):
    """v2 architecture: uses custom W_q/W_k/W_v attention (train_attention_lstm_v2.py)."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.4, window_size_k=20, num_heads=4, ff_multiplier=4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.window_size_k = window_size_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.self_attention = _CustomMultiHeadAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
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
        attn_out, _ = self.self_attention(lstm_out, attn_mask=attn_mask)
        x = self.norm1(lstm_out + self.dropout(attn_out))
        x = self.norm2(x + self.feed_forward(x))
        return self.fc(x), hidden


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(path):
    tokenizer = GothicBPE()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tokenizer.merges = {
        tuple(map(int, k.split(","))): v for k, v in data["merges"].items()
    }
    tokenizer.vocab = tokenizer.build_vocab()
    return tokenizer


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    state = ckpt["model_state_dict"]

    # Auto-detect architecture: v1 uses nn.MultiheadAttention (in_proj_weight),
    # v2 uses custom W_q/W_k/W_v projection matrices.
    is_v2 = "self_attention.W_q.weight" in state
    cls = MultiHeadAttentionLSTMV2 if is_v2 else MultiHeadAttentionLSTM

    model = cls(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        window_size_k=cfg["window_size_k"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    cfg["_arch"] = "v2" if is_v2 else "v1"
    return model, cfg


def generate(model, tokenizer, prompt, seq_length, max_new_tokens, temperature, top_p, device, deterministic=False):
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("Prompt produced no tokens.")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = token_ids[-seq_length:]
            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            next_logits = logits[0, -1, :]

            if deterministic:
                next_token = torch.argmax(next_logits).item()
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = (cumulative - sorted_probs) >= top_p
                    remove[0] = False
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
                    probs = probs / probs.sum()
                next_token = torch.multinomial(probs, 1).item()

            token_ids.append(next_token)

    return tokenizer.decode(token_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/gothic_tokenizer.json")
    parser.add_argument("--prompt", default="The castle was dark and silent")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--deterministic", action="store_true", default=False)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    tokenizer = load_tokenizer(args.tokenizer)
    model, cfg = load_model(args.checkpoint, device)
    print(f"Loaded: {cfg['num_layers']}-layer attention LSTM | K={cfg['window_size_k']} | val_loss recorded in checkpoint")
    print(f"Prompt: {args.prompt}")
    print("-" * 80)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        seq_length=cfg["seq_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        deterministic=args.deterministic,
    )
    print(text)


if __name__ == "__main__":
    main()
