# Gothic NLP Project

Language models trained on Gothic literature using RNN, LSTM, and Attention-LSTM architectures with BPE tokenization (~3.51M tokens, vocab size 5000).

---

## Source Code (`src/`)

| File | Description |
|------|-------------|
| `models.py` | `VanillaRNN`, `DeepLSTM`, `SelfAttentionLSTM` definitions |
| `train.py` | Core training loop, evaluation, checkpointing |
| `train_final_rnn_lstm.py` | Final RNN and LSTM runs (best configs from grid search v3) |
| `train_final_models.py` | Alternative final training script |
| `train_attention_lstm.py` | Multi-head Attention LSTM — v1 (LR=1e-3, dropout=0.3, 4 epochs) |
| `train_attention_lstm_v2.py` | Multi-head Attention LSTM — v2 (LR=5e-4, dropout=0.4, warmup, 5 epochs) |
| `test_attention_equivalence.py` | Forward equivalence test: custom vs `nn.MultiheadAttention` |
| `dataset.py` | Dataset class and train/val/test splits (80/10/10) |
| `tokenizer.py` | BPE tokenizer (train + encode/decode) |
| `grid_search/` | Grid search scripts for RNN, LSTM, Attention LSTM (v3) |
| `evaluation/generate.py` | Text generation for RNN and LSTM checkpoints |
| `evaluation/generate_attention.py` | Text generation for Attention LSTM checkpoints |
| `evaluation/evaluate_generation.py` | BLEU, n-gram overlap, distinct-n, spelling metrics |
| `evaluation/plot_results.py` | Training curves and model comparison plots |

### Custom Multi-Head Attention

Both `train_attention_lstm.py` (v1) and `train_attention_lstm_v2.py` (v2) implement `CustomMultiHeadAttention` **from scratch** — separate `W_q`, `W_k`, `W_v` linear layers, scaled dot-product attention, causal + local window masking. Neither uses `nn.MultiheadAttention`. Forward equivalence against the PyTorch library implementation was verified via `test_attention_equivalence.py` (all checkpoints passed, max logit diff < 1e-5).

**v1 vs v2:**

- **v1** (`train_attention_lstm.py`): LR=1e-3, dropout=0.3, 4 epochs max, no warmup. Trained on 1/2/3 layers with K=20 and K=40. Results are available (see table below).
- **v2** (`train_attention_lstm_v2.py`): LR=5e-4, dropout=0.4, 5 epochs max, linear warmup over 200 steps, K=20 only (1/2/3 layers). **Not yet trained** — prepared to investigate whether lower LR and warmup reduce overfitting in the multi-head attention block.

---

## Checkpoints (`checkpoints/`)

| Folder | Contents |
| ------ | -------- |
| `grid_search_v3_rnn/` | Best checkpoint per RNN grid search run (48 runs) |
| `grid_search_v3_lstm/` | Best checkpoint per LSTM grid search run (48 runs) |
| `final_rnn_lstm/` | Final trained RNN and LSTM models (see configs below) |
| `attention_lstm_new/` | Multi-head Attention LSTM v1 — 1/2/3 layers, K=20 and K=40 (trained) |
| `attention_lstm_v2/` | Multi-head Attention LSTM v2 — best val epoch, 1/2/3 layers, K=20 (not yet trained) |
| `attention_lstm_v2_last_epoch/` | Multi-head Attention LSTM v2 — last epoch checkpoint with epoch number in filename (not yet trained) |

### Final RNN/LSTM configs (`final_rnn_lstm/`)

Best configs chosen per model type and layer count from grid search v3 (by val PPL):

| Model | Batch | Seq | Hidden | LR | Dropout | Val PPL |
| ----- | ----- | --- | ------ | --- | ------- | ------- |
| RNN | 64 | 200 | 256 | 5e-4 | 0.3 | 124.65 |
| LSTM 1-layer | 64 | 100 | 256 | 5e-4 | 0.3 | 97.97 |
| LSTM 2-layer | 32 | 100 | 256 | 1e-3 | 0.3 | 79.06 |
| LSTM 3-layer | 32 | 100 | 256 | 1e-3 | 0.2 | 75.34 |

All final runs use AdamW with weight decay = 1e-4, early stopping patience = 2.

### Multi-Head Attention LSTM results (`attention_lstm_new/`)

Custom multi-head attention (4 heads, K=window size), LR=1e-3, dropout=0.3, 4 epochs max, AdamW weight decay=1e-4:

| Model | Layers | K | Best Val PPL | Test PPL |
| ----- | ------ | -- | ------------ | -------- |
| Multi-head Attention LSTM | 1 | 20 | 137.35 | 158.67 |
| Multi-head Attention LSTM | 1 | 40 | 135.97 | 157.01 |
| Multi-head Attention LSTM | 2 | 20 | 80.80 | 91.46 |
| Multi-head Attention LSTM | 2 | 40 | 81.60 | 92.40 |
| Multi-head Attention LSTM | 3 | 20 | 76.02 | 86.00 |
| Multi-head Attention LSTM | 3 | 40 | 76.53 | 86.40 |

K=20 consistently outperforms K=40. Best overall: 3-layer K=20 (val PPL 76.02).

---

## Results (`results/`)

| Folder | Contents |
| ------ | -------- |
| `grid_search_v3_rnn/` | RNN grid search results |
| `grid_search_v3_lstm_s100/` | LSTM grid search results (seq_length=100) |
| `attention_lstm_new/` | Attention LSTM v1 results |
| `attention_lstm_v2/` | Attention LSTM v2 results |
| `final_rnn_lstm/` | Final RNN/LSTM training results and plots |

---

## Generate Text

**RNN / LSTM** (run from project root):

```bash
PYTHONPATH=src python src/evaluation/generate.py \
  --checkpoint checkpoints/final_rnn_lstm/final_lstm_l3_b32_s100_h256_d0p2_lr0p001.pt \
  --tokenizer data/gothic_tokenizer.json \
  --prompt "The castle was dark and silent" \
  --max-new-tokens 300 --temperature 0.5 --top-p 0.85
```

**Attention LSTM** (run from `src/`):

```bash
python evaluation/generate_attention.py \
  --checkpoint ../checkpoints/attention_lstm_new/multihead_3layer_K20.pt \
  --tokenizer ../data/gothic_tokenizer.json \
  --prompt "The castle was dark and silent" \
  --max-new-tokens 300 --temperature 0.5 --top-p 0.85
```
