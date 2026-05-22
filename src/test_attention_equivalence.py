import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_HEADS = 4


#  Original model (nn.MultiheadAttention) 
class SelfAttentionLSTMOriginal(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.3, window_size_k=20, num_heads=4, ff_multiplier=4):
        super().__init__()
        assert hidden_dim % num_heads == 0

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


# Custom model (from-scratch multi-head self-attention)

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

        return self.out_proj(context), weights


class SelfAttentionLSTMCustom(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.3, window_size_k=20, num_heads=4, ff_multiplier=4):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.window_size_k = window_size_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.self_attention = CustomMultiHeadAttention(hidden_dim, num_heads, dropout)

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

        attn_out, attn_weights = self.self_attention(lstm_out, attn_mask=attn_mask)

        x = self.norm1(lstm_out + self.dropout(attn_out))
        x = self.norm2(x + self.feed_forward(x))
        logits = self.fc(x)

        return logits, hidden, attn_weights


# Weight conversion: nn.MultiheadAttention  CustomMultiHeadAttention 

def convert_state_dict(old_state_dict, hidden_dim):
    new_sd = {}
    H = hidden_dim

    for key, value in old_state_dict.items():
        if key == "self_attention.in_proj_weight":
            new_sd["self_attention.W_q.weight"] = value[:H].clone()
            new_sd["self_attention.W_k.weight"] = value[H:2 * H].clone()
            new_sd["self_attention.W_v.weight"] = value[2 * H:].clone()
        elif key == "self_attention.in_proj_bias":
            new_sd["self_attention.W_q.bias"] = value[:H].clone()
            new_sd["self_attention.W_k.bias"] = value[H:2 * H].clone()
            new_sd["self_attention.W_v.bias"] = value[2 * H:].clone()
        else:
            new_sd[key] = value

    return new_sd


#  Equivalence test 

def run_equivalence_test(checkpoint_path, device):
    print(f"\n{'─' * 60}")
    print(f"Checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    print(f"Config: num_layers={cfg['num_layers']}, window_size_k={cfg['window_size_k']}, "
          f"hidden_dim={cfg['hidden_dim']}")

    original = SelfAttentionLSTMOriginal(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        window_size_k=cfg["window_size_k"],
        num_heads=NUM_HEADS,
    ).to(device)
    original.load_state_dict(ckpt["model_state_dict"])
    original.eval()

    custom = SelfAttentionLSTMCustom(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        window_size_k=cfg["window_size_k"],
        num_heads=NUM_HEADS,
    ).to(device)
    custom.load_state_dict(convert_state_dict(ckpt["model_state_dict"], cfg["hidden_dim"]))
    custom.eval()

    torch.manual_seed(0)
    x = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_length"]), device=device)

    with torch.no_grad():
        orig_logits, _, _ = original(x)
        custom_logits, _, _ = custom(x)

    max_diff = (orig_logits - custom_logits).abs().max().item()
    mean_diff = (orig_logits - custom_logits).abs().mean().item()
    passed = torch.allclose(orig_logits, custom_logits, atol=1e-5)

    print(f"Max logit diff:  {max_diff:.2e}")
    print(f"Mean logit diff: {mean_diff:.2e}")
    print(f"Result: {'PASSED ' if passed else 'FAILED '}")

    return passed


def main():
    device = torch.device("cpu")

    checkpoints = [
        "checkpoints/attention_lstm_new/multihead_1layer_K20.pt",
        "checkpoints/attention_lstm_new/multihead_1layer_K40.pt",
        "checkpoints/attention_lstm_new/multihead_2layer_K20.pt",
        "checkpoints/attention_lstm_new/multihead_2layer_K40.pt",
        "checkpoints/attention_lstm_new/multihead_3layer_K20.pt",
        "checkpoints/attention_lstm_new/multihead_3layer_K40.pt",
    ]

    results = []
    for ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping (not found): {ckpt_path}")
            continue
        results.append(run_equivalence_test(ckpt_path, device))

    print(f"\n{'─' * 60}")
    print(f"{'All tests PASSED ' if all(results) else 'Some tests FAILED '} ({sum(results)}/{len(results)})")


if __name__ == "__main__":
    main()
