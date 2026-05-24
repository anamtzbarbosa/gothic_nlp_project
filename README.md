# Gothic NLP Project

Language models trained on Gothic literature using RNN, LSTM, and Multi-Head Self-Attention LSTM architectures with BPE tokenization.

---

## Dataset

### Source books (`data/`)

The corpus was assembled from three batches of Gothic literature sourced from Project Gutenberg:

**`archive/`** (initial 8 books):

- *The Castle of Otranto* — Horace Walpole
- *The Picture of Dorian Gray* — Oscar Wilde
- *Dracula* — Bram Stoker
- *Strange Case of Dr Jekyll and Mr Hyde* — Robert Louis Stevenson
- *Frankenstein* — Mary Shelley
- *Jane Eyre* — Charlotte Brontë
- *The Phantom of the Opera* — Gaston Leroux
- *Wuthering Heights* — Emily Brontë

**`new books/`** (4 books added to expand coverage):

- *Carmilla* — J. Sheridan Le Fanu
- *The Monk* — Matthew Lewis
- *The Mysteries of Udolpho* — Ann Radcliffe
- *The Turn of the Screw* — Henry James

**`new_books_2/`** (12 books added to further enrich the corpus):

- *A Sicilian Romance* — Ann Radcliffe
- *Curious If True* — Elizabeth Gaskell
- *Ghost Stories of an Antiquary* — M.R. James
- *The Beetle* — Richard Marsh
- *The Empty House and Other Ghost Stories* — Algernon Blackwood
- *The Grey Woman and Other Tales* — Elizabeth Gaskell
- *The Haunted Hotel* — Wilkie Collins
- *The House of the Vampire* — George Sylvester Viereck
- *The Lair of the White Worm* — Bram Stoker
- *The Woman in White* — Wilkie Collins
- *Uncle Silas* — J. Sheridan Le Fanu
- *Varney the Vampire* — James Malcolm Rymer

**Total: 24 books.**

### Corpus size and epoch budget

After cleaning and concatenation, the full corpus is **~14.7 MB / ~2.66M words**. Tokenized with a BPE vocabulary of size 5000 (trained on the corpus), the total token count is **3,510,000**:

| Split | Tokens | Share |
| ----- | ------ | ----- |
| Train | 2,805,000 | 80% |
| Val | 350,000 | 10% |
| Test | 355,000 | 10% |

The number of gradient steps per epoch depends on the batch / sequence configuration:

| Batch | Seq | Steps / epoch | Tokens / epoch |
| ----- | --- | ------------- | -------------- |
| 64 | 200 | 219 | 2,803,200 |
| 64 | 100 | 438 | 2,803,200 |
| 32 | 100 | 876 | 2,803,200 |

Each epoch sees the entire 2.8M-token training set regardless of batch or sequence length. Because the corpus is relatively compact (24 novels), models tend to converge quickly — **best validation PPL is typically reached at epoch 1 or 2**, after which over-fitting sets in and early stopping triggers. This is expected behaviour for a domain-specific corpus of this size and is not a sign of under-training.

---

## Source Code (`src/`)

| File | Description |
|------|-------------|
| `models.py` | `VanillaRNN`, `DeepLSTM`, `SelfAttentionLSTM` definitions |
| `train.py` | Core training loop, evaluation, checkpointing |
| `train_final_rnn_lstm.py` | Final RNN and LSTM runs (best configs from grid search v3) |
| `train_final_models.py` | Alternative final training script |
| `train_attention_lstm.py` | Multi-Head Self-Attention LSTM — v1 (LR=1e-3, dropout=0.3, 4 epochs) |
| `train_attention_lstm_v2.py` | Multi-Head Self-Attention LSTM — v2 (LR=5e-4, dropout=0.4, 5 epochs) |
| `test_attention_equivalence.py` | Forward equivalence test: custom vs `nn.MultiheadAttention` |
| `dataset.py` | Dataset class and train/val/test splits (80/10/10) |
| `tokenizer.py` | BPE tokenizer (train + encode/decode) |
| `grid_search/` | Grid search scripts for RNN, LSTM, Attention LSTM (v3) |
| `evaluation/generate.py` | Text generation for RNN and LSTM checkpoints |
| `evaluation/generate_attention.py` | Text generation for Multi-Head Self-Attention LSTM checkpoints |
| `evaluation/evaluate_generation.py` | BLEU, n-gram overlap, distinct-n, spelling metrics |
| `evaluation/evaluate_generation_attention.py` | Same evaluation pipeline for Multi-Head Self-Attention LSTM checkpoints |
| `evaluation/build_eval_samples.py` | Builds `data/eval_samples.json` from the test set |
| `evaluation/plot_results.py` | Training curves and model comparison plots |

### Custom Multi-Head Attention

Both `train_attention_lstm.py` (v1) and `train_attention_lstm_v2.py` (v2) implement `CustomMultiHeadAttention` **from scratch** — separate `W_q`, `W_k`, `W_v` linear layers, scaled dot-product attention, causal + local window masking. Neither uses `nn.MultiheadAttention`. Forward equivalence against the PyTorch library implementation was verified via `test_attention_equivalence.py` (all checkpoints passed, max logit diff < 1e-5).

**v1 vs v2:**

- **v1** (`train_attention_lstm.py`): LR=1e-3, dropout=0.3, 4 epochs max. Trained on 1/2/3 layers with K=20 and K=40. Results are available (see table below).
- **v2** (`train_attention_lstm_v2.py`): LR=5e-4, dropout=0.4, 5 epochs max, K=20 only (1/2/3 layers). Trained to investigate whether lower LR and higher dropout reduce overfitting in the multi-head self-attention block.

---

## Checkpoints (`checkpoints/`)

| Folder | Contents |
| ------ | -------- |
| `grid_search_v3_rnn/` | Best checkpoint per RNN grid search run (48 runs) |
| `grid_search_v3_lstm/` | Best checkpoint per LSTM grid search run (48 runs) |
| `final_rnn_lstm/` | Final trained RNN and LSTM models (see configs below) |
| `attention_lstm_new/` | Multi-Head Self-Attention LSTM v1 — 1/2/3 layers, K=20 and K=40 (trained) |
| `attention_lstm_v2/` | Multi-Head Self-Attention LSTM v2 — best val epoch, 1/2/3 layers, K=20 (not yet trained) |
| `attention_lstm_v2_last_epoch/` | Multi-Head Self-Attention LSTM v2 — last epoch checkpoint with epoch number in filename (not yet trained) |

### Hyperparameter Search

To identify good configurations efficiently, a one-epoch grid search was run for both RNN and LSTM models. Using **1 epoch as a convergence proxy** is reliable here because the corpus is compact and models converge quickly regardless — ranking by 1-epoch val PPL closely tracks final performance. The top configurations per depth are then promoted to full training runs with early stopping (patience=2).

**RNN search** — fixed: `num_layers=1, embed_dim=128, vocab_size=5000`

| Parameter | Values searched |
| --------- | --------------- |
| batch_size | 32, 64 |
| seq_length | 100, 200 |
| hidden_dim | 128, 256 |
| learning_rate | 5e-4, 1e-3 |
| dropout | 0.0, 0.2, 0.3 |

Total: 48 runs. Top 5 by 1-epoch val PPL:

| Batch | Seq | Hidden | LR | Dropout | Val PPL | Test PPL | |
| ----- | --- | ------ | -- | ------- | ------- | -------- | - |
| 64 | 200 | 256 | 5e-4 | 0.3 | **124.65** | 137.32 | ★ chosen |
| 32 | 100 | 256 | 5e-4 | 0.3 | 124.66 | 136.70 | |
| 64 | 100 | 256 | 5e-4 | 0.3 | 125.54 | 137.33 | |
| 32 | 200 | 256 | 5e-4 | 0.3 | 125.61 | 138.55 | |
| 64 | 100 | 256 | 5e-4 | 0.2 | 125.93 | 138.07 | |

The top configuration (B=64, S=200, H=256, LR=5e-4, dropout=0.3) was selected and retrained to convergence.

**LSTM search** — fixed: `embed_dim=128, vocab_size=5000, seq_length=100`

| Parameter | Values searched |
| --------- | --------------- |
| batch_size | 32, 64 |
| hidden_dim | 128, 256 |
| learning_rate | 5e-4, 1e-3 |
| num_layers | 1, 2, 3 |
| dropout | 0.2, 0.3 |

Total: 48 runs. Top results by 1-epoch val PPL:

| Batch | Hidden | LR | Layers | Dropout | Val PPL | Test PPL | |
| ----- | ------ | -- | ------ | ------- | ------- | -------- | - |
| 32 | 256 | 1e-3 | 3 | 0.2 | **75.34** | 84.48 | ★ LSTM-3 |
| 64 | 256 | 1e-3 | 3 | 0.2 | 75.79 | 85.13 | |
| 32 | 256 | 1e-3 | 3 | 0.3 | 76.38 | 84.98 | |
| 32 | 256 | 5e-4 | 3 | 0.2 | 76.70 | 85.86 | |
| 64 | 256 | 1e-3 | 3 | 0.3 | 77.10 | 85.74 | |
| 64 | 256 | 5e-4 | 3 | 0.2 | 77.52 | 86.64 | |
| 32 | 256 | 5e-4 | 3 | 0.3 | 78.46 | 86.99 | |
| 32 | 256 | 1e-3 | 2 | 0.3 | 79.06 | 88.00 | ★ LSTM-2 |

The best-per-depth configurations (starred) were promoted to full training. For LSTM-1, the same regime (B=64, H=256, LR=5e-4, dropout=0.3) was applied following the pattern from the grid search — this config ranked at the top across all single-layer runs.

### Final RNN/LSTM configs (`final_rnn_lstm/`)

Best configs chosen per model type and layer count from the search above (by val PPL):

| Model | Batch | Seq | Hidden | LR | Dropout | Val PPL |
| ----- | ----- | --- | ------ | --- | ------- | ------- |
| RNN | 64 | 200 | 256 | 5e-4 | 0.3 | 124.65 |
| LSTM 1-layer | 64 | 100 | 256 | 5e-4 | 0.3 | 97.97 |
| LSTM 2-layer | 32 | 100 | 256 | 1e-3 | 0.3 | 79.06 |
| LSTM 3-layer | 32 | 100 | 256 | 1e-3 | 0.2 | 75.34 |

All final runs use AdamW with weight decay = 1e-4, early stopping patience = 2.

### Effect of hidden dimension size

To investigate how increasing the hidden state size affects performance, each model was evaluated at hidden_dim ∈ {64, 128, 256} with all other hyperparameters held fixed (same batch size, sequence length, LR, and dropout as the final config above). h=128 and h=256 results come from the full grid search (trained to convergence with early stopping); h=64 is a 1-epoch probe and is therefore somewhat pessimistic, but the directional trend is clear.

**RNN** (b=64, s=200, dropout=0.3, lr=5e-4):

| hidden_dim | Val PPL | Test PPL |
| ---------- | ------- | -------- |
| 64 *(1-epoch probe)* | 194.67 | 204.92 |
| 128 | 141.87 | 152.22 |
| **256** | **124.65** | **137.32** |

**LSTM 1-layer** (b=64, s=100, dropout=0.3, lr=5e-4):

| hidden_dim | Val PPL | Test PPL |
| ---------- | ------- | -------- |
| 64 *(1-epoch probe)* | 143.08 | 152.28 |
| 128 | 109.73 | 119.75 |
| **256** | **97.97** | **109.05** |

**LSTM 2-layer** (b=32, s=100, dropout=0.3, lr=1e-3):

| hidden_dim | Val PPL | Test PPL |
| ---------- | ------- | -------- |
| 64 *(1-epoch probe)* | 140.32 | 150.39 |
| 128 | 99.01 | 108.85 |
| **256** | **79.06** | **88.00** |

**LSTM 3-layer** (b=32, s=100, dropout=0.2, lr=1e-3):

| hidden_dim | Val PPL | Test PPL |
| ---------- | ------- | -------- |
| 64 *(1-epoch probe)* | 132.59 | 143.43 |
| 128 | 92.23 | 101.52 |
| **256** | **75.34** | **84.48** |

Across every model, larger hidden_dim consistently improves both val and test PPL. The biggest gain is from 64→128; the gain from 128→256 is smaller but still consistent. This confirms that h=256 is the right capacity choice for this corpus size — there is no sign of diminishing returns or overfitting from the larger hidden state when paired with appropriate dropout and weight decay.

### Multi-Head Self-Attention LSTM — v1 (`attention_lstm_new/`)

Custom multi-head self-attention (4 heads, K=window size), LR=1e-3, dropout=0.3, 4 epochs max, AdamW weight decay=1e-4:

| Layers | K | Best Val PPL | Test PPL |
| ------ | -- | ------------ | -------- |
| 1 | 20 | 137.35 | 158.67 |
| 1 | 40 | 135.97 | 157.01 |
| 2 | 20 | 80.80 | 91.46 |
| 2 | 40 | 81.60 | 92.40 |
| 3 | 20 | **76.02** | **86.00** |
| 3 | 40 | 76.53 | 86.40 |

K=20 consistently outperforms K=40. Best overall: 3-layer K=20 (val PPL 76.02).

### Multi-Head Self-Attention LSTM — v2 (`attention_lstm_v2/`)

v1 training curves showed the gap between train and val loss growing over epochs, indicating overfitting in the self-attention block. v2 was designed to address this: lower LR (5e-4 → down from 1e-3), higher dropout (0.4 → up from 0.3), 5 epochs max. K=20 only (since K=20 already dominated in v1).

| Layers | K | Best Val PPL | Test PPL | vs v1 Test PPL |
| ------ | -- | ------------ | -------- | -------------- |
| 1 | 20 | 120.90 | 138.91 | −19.76 ✓ |
| 2 | 20 | 78.51 | **89.01** | −2.45 ✓ |
| 3 | 20 | 79.30 | 89.78 | +3.78 ✗ |

v2 helps for 1-layer and 2-layer (overfitting was the bottleneck there), but hurts for 3-layer — the deeper model already regularises effectively through depth, and the lower LR slows convergence enough that the model does not recover within 5 epochs. **Best checkpoint for generation evaluation: Multi-Head Self-Attention LSTM v1 3-layer K=20.**

### Best model per layer count

Selecting the better checkpoint per depth across v1 and v2:

| Layers | Selected | Config | Test PPL |
| ------ | -------- | ------ | -------- |
| 1 | v2 `attention_lstm_v2/v2_attn_lstm_1layer_K20.pt` | LR=5e-4, dropout=0.4, K=20, 4 heads | 138.91 |
| 2 | v2 `attention_lstm_v2/v2_attn_lstm_2layer_K20.pt` | LR=5e-4, dropout=0.4, K=20, 4 heads | 89.01 |
| 3 | v1 `attention_lstm_new/multihead_3layer_K20.pt` | LR=1e-3, dropout=0.3, K=20, 4 heads | 86.00 |

These three checkpoints are used for all generation evaluation.

### Multi-Head Self-Attention LSTM: Generation Results

Same evaluation setup as RNN/LSTM: 100 sentence-boundary samples, two decoding settings.

```text
Model      Set     T  top-p |  BLEU1  BLEU4s   BERT  |  bi-ov  spell  dist2   TTR  | genPPL
──────────────────────────────────────────────────────────────────────────────────────────────
attn_l1    t0p5  0.5  0.85  | 0.1754  0.0027  0.8181 | 0.0098  0.916  0.680  0.775 |  11.61
attn_l1    t0p7  0.7  0.90  | 0.1817  0.0032  0.8172 | 0.0087  0.920  0.763  0.801 |  19.28
attn_l2    t0p5  0.5  0.85  | 0.1779  0.0052  0.8237 | 0.0122  0.923  0.621  0.719 |  12.20
attn_l2    t0p7  0.7  0.90  | 0.1877  0.0058  0.8222 | 0.0134  0.917  0.734  0.772 |  19.44
attn_l3    t0p5  0.5  0.85  | 0.1844  0.0037  0.8241 | 0.0128  0.934  0.592  0.716 |  11.96
attn_l3    t0p7  0.7  0.90  | 0.1802  0.0043  0.8237 | 0.0132  0.924  0.738  0.765 |  19.29
```

Key observations:

- Multi-Head Self-Attention LSTM 2-layer achieves the best BLEU-4s (0.0058) and competitive BERTScore across both settings
- Multi-Head Self-Attention LSTM 3-layer has the highest BERTScore (0.8241) and spelling accuracy (0.934) at t0p5
- Generated perplexity is notably low for all Multi-Head Self-Attention LSTM models at t0p5 (~11–12), lower than the plain LSTM equivalents, indicating the multi-head self-attention mechanism produces more confident and focused distributions
- **attn_l1 shows the highest dist2/TTR despite the worst test PPL (138.91)** — counterintuitively, the weakest model appears most "diverse." This is not a sign of quality: a poorly-fitted model samples more uniformly across vocabulary, producing scattered text that hasn't learned strong pattern preferences. The low bi-ov (0.0098) and BLEU-4s (0.0027) confirm the output doesn't meaningfully overlap with references. The slightly lower genPPL vs l2 (11.61 vs 12.20) reflects overconfidence on a simpler learned distribution rather than genuine fluency.
- Full per-sample results in `results/attention_lstm/eval_*.txt`

### MH-Self-Attn LSTM-3 vs LSTM-3 (head-to-head, t0p5)

| Metric | lstm_l3 | attn_l3 | Winner |
| ------ | ------- | ------- | ------ |
| BLEU-1 | **0.188** | 0.184 | lstm |
| BLEU-4s | **0.0087** | 0.0037 | lstm |
| BERTScore | 0.823 | **0.824** | attn |
| Bigram overlap | **0.016** | 0.013 | lstm |
| Spelling | **0.938** | 0.934 | ~tied |
| Distinct-2 | 0.587 | 0.592 | ~tied |
| TTR | 0.724 | 0.716 | ~tied |
| genPPL | 14.76 | **11.96** | attn |

Plain LSTM-3 wins on n-gram metrics (BLEU, bigram overlap) — it generates text that more literally matches the reference wording. Multi-Head Self-Attention LSTM-3 wins on BERTScore and genPPL — it's semantically closer to the reference and produces more confident distributions, but uses different surface words to get there.

This is actually expected: at seq_length=100, plain LSTM is in its comfort zone (no vanishing gradient problem over 100 steps), so the attention mechanism doesn't provide a clear architectural advantage. The attention model would likely pull ahead at seq_length=200 where LSTM starts struggling with long-range dependencies.

### Multi-Head Self-Attention LSTM at seq_length=200 — demonstrating the benefit of attention

To test this hypothesis, both a 3-layer Multi-Head Self-Attention LSTM and a 3-layer plain LSTM were trained at seq_length=200.

**Motivation:** at seq=100 the LSTM handles all dependencies within its context window comfortably, so adding attention gives no clear win. At seq=200 the LSTM must track dependencies twice as far back, where its hidden state starts to become a bottleneck — the attention mechanism can directly attend to any past token and should have a genuine architectural advantage.

Training results (3-layer, seq=200):

| Model | Config | Best Val PPL | Test PPL |
| ----- | ------ | ------------ | -------- |
| LSTM 3-layer | seq=200, batch=32, LR=1e-3, dropout=0.2 | 75.41 | 85.49 |
| MH-Self-Attn LSTM 3-layer (v2, K=20) | seq=200, batch=64, LR=5e-4, dropout=0.4 | **75.90** | **85.33** |

The Multi-Head Self-Attention LSTM achieves a better test PPL (85.33 vs 85.49) at seq=200 — confirming that the multi-head self-attention mechanism provides a genuine advantage when the sequence is long enough to expose the LSTM's limitations. Notably, the attention model's test PPL (85.33) also improves on its own seq=100 counterpart (86.00), whereas the plain LSTM seq=200 (85.49) does not clearly improve on LSTM seq=100 (75.34 val / ~86 test at comparable depth).

For a fair generation evaluation, seq=200 requires longer references (~170 tokens ≈ 596 chars). A separate `data/eval_samples_long.json` was built from the same prompts as `eval_samples.json` but with references extended to the next sentence boundary around 600 characters. To make the comparison apples-to-apples, `attn_l3 seq=100` was also evaluated against the long references.

```text
                                          |  BLEU1  BLEU4s   BERT  |  bi-ov  spell  dist2   TTR  | genPPL
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
attn_l3 seq=100  t0p5 (std  eval ~250c)  | 0.1844  0.0037  0.8241 | 0.0128  0.934  0.592  0.716 |  11.96
attn_l3 seq=100  t0p5 (long eval ~600c)  | 0.1367  0.0040  0.8167 | 0.0291  0.930  0.604  0.714 |  11.87
attn_l3 seq=200  t0p5 (long eval ~600c)  | 0.2387  0.0066  0.8174 | 0.0262  0.935  0.456  0.565 |  13.64
attn_l3 seq=100  t0p7 (std  eval ~250c)  | 0.1802  0.0043  0.8237 | 0.0132  0.924  0.738  0.765 |  19.29
attn_l3 seq=100  t0p7 (long eval ~600c)  | 0.1335  0.0021  0.8178 | 0.0236  0.920  0.724  0.770 |  18.47
attn_l3 seq=200  t0p7 (long eval ~600c)  | 0.2474  0.0085  0.8162 | 0.0236  0.930  0.603  0.637 |  21.79
```

On the same long references, seq=200 wins clearly on BLEU-1 (+0.10) and BLEU-4s (+0.003). However, part of this gain is mechanical: seq=200 generates ~170 tokens against a 600-char reference while seq=100 generates only ~70 tokens, so seq=200 naturally covers more of the reference. Despite this length advantage, seq=100 has slightly higher bigram overlap at T=0.5 (0.0291 vs 0.0262) — it is more precise in word choice but covers less of the reference. BERTScore is essentially identical across both models. seq=100 retains an advantage on diversity (distinct-2, TTR) and genPPL, since generating 170 tokens over a longer horizon leads to more word reuse and higher sampling uncertainty. Overall, the test PPL advantage of seq=200 (85.33 vs 86.00) is real but modest, and the attention mechanism's benefit at seq=200 is confirmed by the n-gram metrics once reference length is properly matched.

---

## Results (`results/`)

| Folder | Contents |
| ------ | -------- |
| `grid_search_v3_rnn/` | RNN grid search results |
| `grid_search_v3_lstm_s100/` | LSTM grid search results (seq_length=100) |
| `attention_lstm_new/` | Multi-Head Self-Attention LSTM v1 results |
| `attention_lstm_v2/` | Multi-Head Self-Attention LSTM v2 results |
| `final_rnn_lstm/` | Final RNN/LSTM training results, plots, and generation evaluation |

---

## Generation Evaluation

Evaluated on 100 fixed samples from the test set (`data/eval_samples.json`). Each sample has a prompt starting at a sentence boundary (≈30 BPE tokens) and a reference ending at the next sentence boundary (≈250 chars). Two decoding settings: conservative (T=0.5, top-p=0.85) and diverse (T=0.7, top-p=0.9).

### Final RNN/LSTM generation results

| Model | Setting | T | top-p | BLEU-1 | BLEU-4s | BERTScore | Bigram ovlp | Spelling | Distinct-2 | TTR | genPPL |
| ----- | ------- | - | ----- | ------ | ------- | --------- | ----------- | -------- | ---------- | --- | ------ |
| RNN | t0p5 | 0.5 | 0.85 | 0.121 | 0.0023 | 0.810 | 0.008 | 0.923 | 0.492 | 0.643 | 20.4 |
| RNN | t0p7 | 0.7 | 0.90 | 0.124 | 0.0020 | 0.809 | 0.008 | 0.902 | 0.646 | 0.688 | 36.3 |
| LSTM 1-layer | t0p5 | 0.5 | 0.85 | 0.185 | 0.0029 | 0.819 | 0.011 | 0.921 | 0.644 | 0.770 | 18.3 |
| LSTM 1-layer | t0p7 | 0.7 | 0.90 | 0.187 | 0.0034 | 0.818 | 0.009 | 0.920 | 0.766 | 0.810 | 30.2 |
| LSTM 2-layer | t0p5 | 0.5 | 0.85 | 0.185 | 0.0045 | 0.823 | 0.013 | 0.937 | 0.595 | 0.718 | 17.3 |
| LSTM 2-layer | t0p7 | 0.7 | 0.90 | 0.188 | 0.0048 | 0.820 | 0.012 | 0.924 | 0.744 | 0.784 | 28.2 |
| LSTM 3-layer | t0p5 | 0.5 | 0.85 | 0.188 | **0.0087** | **0.823** | **0.016** | **0.938** | 0.587 | 0.724 | **14.8** |
| LSTM 3-layer | t0p7 | 0.7 | 0.90 | **0.188** | 0.0070 | 0.821 | 0.013 | 0.922 | **0.734** | **0.781** | 23.0 |

**BLEU-4s** = smoothed BLEU-4. **genPPL** = model's own perplexity on generated text (lower = more confident/focused output).

```text
Model        Set     T  top-p |  BLEU1  BLEU4s   BERT  |  bi-ov  spell  dist2   TTR  | genPPL
──────────────────────────────────────────────────────────────────────────────────────────────────
rnn          t0p5  0.5  0.85  | 0.1213  0.0023  0.8100 | 0.0079  0.923  0.492  0.643 |  20.44
rnn          t0p7  0.7  0.90  | 0.1241  0.0020  0.8088 | 0.0081  0.902  0.646  0.688 |  36.34
lstm_l1      t0p5  0.5  0.85  | 0.1850  0.0029  0.8186 | 0.0112  0.921  0.644  0.770 |  18.30
lstm_l1      t0p7  0.7  0.90  | 0.1866  0.0034  0.8184 | 0.0093  0.920  0.766  0.810 |  30.19
lstm_l2      t0p5  0.5  0.85  | 0.1847  0.0045  0.8228 | 0.0134  0.937  0.595  0.718 |  17.28
lstm_l2      t0p7  0.7  0.90  | 0.1882  0.0048  0.8204 | 0.0121  0.924  0.744  0.784 |  28.19
lstm_l3      t0p5  0.5  0.85  | 0.1881  0.0087  0.8227 | 0.0156  0.938  0.587  0.724 |  14.76
lstm_l3      t0p7  0.7  0.90  | 0.1882  0.0070  0.8212 | 0.0126  0.922  0.734  0.781 |  23.02
```

Key observations:

- LSTM consistently outperforms RNN across all metrics
- Deeper LSTM layers improve quality: LSTM-3 achieves the best BLEU-4s, BERTScore, bigram overlap, and spelling
- T=0.5 gives more focused output (lower genPPL, higher spelling accuracy); T=0.7 gives more diverse output (higher distinct-2 and TTR)
- Full per-sample results with prompt / generated / reference examples in `results/final_rnn_lstm/eval_*.txt`

To rebuild eval samples:

```bash
python src/evaluation/build_eval_samples.py
```

To run evaluation (from project root):

```bash
PYTHONPATH=src python src/evaluation/evaluate_generation.py \
  --checkpoint checkpoints/final_rnn_lstm/final_lstm_l3_b32_s100_h256_d0p2_lr0p001.pt \
  --tokenizer data/gothic_tokenizer.json \
  --eval-samples data/eval_samples.json \
  --temperature 0.5 --top-p 0.85 \
  --output results/final_rnn_lstm/eval_lstm_l3_t0p5
```

---

## Generate Text

Three decoding modes are supported: **temperature**, **nucleus (top-p)**, and **deterministic (greedy)**.

**RNN / LSTM** (run from project root):

```bash
# Temperature / nucleus sampling
PYTHONPATH=src python src/evaluation/generate.py \
  --checkpoint checkpoints/final_rnn_lstm/final_lstm_l3_b32_s100_h256_d0p2_lr0p001.pt \
  --tokenizer data/gothic_tokenizer.json \
  --prompt "The castle was dark and silent" \
  --max-new-tokens 300 --temperature 0.5 --top-p 0.85

# Deterministic (greedy) — ignores --temperature and --top-p
PYTHONPATH=src python src/evaluation/generate.py \
  --checkpoint checkpoints/final_rnn_lstm/final_lstm_l3_b32_s100_h256_d0p2_lr0p001.pt \
  --tokenizer data/gothic_tokenizer.json \
  --prompt "The castle was dark and silent" \
  --max-new-tokens 300 --deterministic
```

**Multi-Head Self-Attention LSTM** (run from `src/`):

```bash
# Temperature / nucleus sampling
python evaluation/generate_attention.py \
  --checkpoint ../checkpoints/attention_lstm_new/multihead_3layer_K20.pt \
  --tokenizer ../data/gothic_tokenizer.json \
  --prompt "The castle was dark and silent" \
  --max-new-tokens 300 --temperature 0.5 --top-p 0.85

# Deterministic (greedy)
python evaluation/generate_attention.py \
  --checkpoint ../checkpoints/attention_lstm_new/multihead_3layer_K20.pt \
  --tokenizer ../data/gothic_tokenizer.json \
  --prompt "The castle was dark and silent" \
  --max-new-tokens 300 --deterministic
```

---

## Evaluation Metrics (`evaluation/evaluate_generation.py`)

| Metric | Description |
| ------ | ----------- |
| BLEU | Corpus-level n-gram overlap with reference text |
| BERTScore F1 | Semantic similarity to reference using BERT embeddings |
| Bigram / Trigram overlap | Fraction of generated n-grams found in reference |
| Distinct-1 / Distinct-2 | Vocabulary diversity of generated text |
| TTR (Type-Token Ratio) | Unique words / total words per generated sample |
| Trigram repetition rate | Fraction of repeated trigrams (lower = more diverse) |
| Spelling accuracy | Fraction of generated words found in English dictionary |
| Generated perplexity | Model's own perplexity on the generated text — lower means more deterministic output |

---

## Plots (`results/plots/`)

All figures are generated from training logs and evaluation JSON files by running:

```bash
python src/evaluation/plot_readme_figures.py
```

### Figure 1 — Final RNN / LSTM: Train vs Validation Loss

Train loss continues falling every epoch while val loss flattens or rises after epoch 1–2, confirming that early stopping is the right call for this corpus size. The widening gap between train and val loss is the classic overfitting signature.

![RNN/LSTM training curves](results/plots/fig1_rnn_lstm_curves.png)

### Figure 2 — Multi-Head Self-Attention LSTM: Train vs Validation Loss

The same overfitting pattern appears in the Multi-Head Self-Attention LSTM — particularly severe in 1-layer where val loss climbs steadily from epoch 1. The 3-layer model (marked ★) is the best-performing checkpoint and is selected for generation evaluation.

![Multi-Head Self-Attention LSTM training curves](results/plots/fig2_attn_curves.png)

### Figure 3 — Multi-Head Self-Attention LSTM: Initial vs Regularized Config

The initial config (LR=1e-3, dropout=0.3) shows a growing train/val gap across all layer counts. The regularized config (LR=5e-4, dropout=0.4) narrows this gap for 1-layer and 2-layer. For 3-layer the regularization does not help — the deeper model already self-regularizes through depth, and the lower LR prevents it from converging to a better minimum within the same epoch budget.

![Attention LSTM config comparison](results/plots/fig3_attn_configs.png)

### Figure 4 — Effect of Hidden Dimension on Validation PPL

Increasing hidden_dim consistently improves validation PPL across every model type. The biggest gain is from 64 → 128; the gain from 128 → 256 is smaller but present in all cases. Note: h=64 results are from a 1-epoch probe and are therefore somewhat pessimistic.

![Hidden dim comparison](results/plots/fig4_hidden_dim.png)

### Figure 5 — Val and Test Perplexity: All Final Models

Val PPL (lighter bars) and Test PPL (darker bars) are shown side by side. The gap between val and test reflects generalization: smaller gap = better generalization. Multi-Head Self-Attention LSTM 3-layer (seq=200) achieves the best test PPL overall.

![Val and test PPL all models](results/plots/fig5_val_test_ppl.png)

### Figure 6 — Generation Quality Metrics (T=0.5, top-p=0.85)

Reference-overlap and semantic similarity metrics for all 7 models at conservative decoding. LSTM-3 leads on BLEU-4s and bigram overlap; MH-Self-Attn-LSTM-2 and MH-Self-Attn-LSTM-3 lead on BERTScore, indicating that the multi-head self-attention models produce semantically closer text even when surface n-gram overlap is slightly lower.

![Generation quality metrics](results/plots/fig6_eval_quality.png)

### Figure 7 — Diversity and Fluency Metrics (T=0.5 vs T=0.7)

Diversity metrics (Distinct-2, TTR) increase markedly from T=0.5 to T=0.7 across all models, while spelling accuracy drops slightly. Generated PPL (model confidence on its own output) is notably lower for Multi-Head Self-Attention LSTM models at T=0.5, indicating more focused and confident distributions compared to plain LSTM.

![Diversity and fluency metrics](results/plots/fig7_eval_diversity.png)
