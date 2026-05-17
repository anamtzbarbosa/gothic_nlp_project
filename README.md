# Gothic NLP Project

Language model trained on Gothic literature using RNN, LSTM, and Attention-LSTM architectures with BPE tokenization.

---

## Project Structure

```
gothic_nlp_project/
├── data/                        # Corpus and tokenizer files
├── src/                         # All source code
├── checkpoints/                 # Saved model weights
├── results/                     # All outputs and evaluation results
```

---

## Data (`data/`)

| File | Description |
|------|-------------|
| `corpus_clean.txt` | Cleaned raw text (~763K words from Gothic novels) |
| `gothic_tokenizer.json` | Trained BPE tokenizer (vocab size 5000) |
| `corpus_tokenized.pkl` | Full corpus encoded as BPE token IDs (~984K tokens) |
| `eval_samples.json` | 20 fixed test samples used for generation evaluation |

---

## Source Code (`src/`)

| File | What it does |
|------|--------------|
| `tokenizer.py` | BPE tokenizer implementation (train + encode/decode) |
| `data_utils.py` | Text cleaning and preprocessing |
| `dataset.py` | Dataset class and train/val/test split (80/10/10) |
| `models.py` | Model definitions: VanillaRNN, DeepLSTM, CrossAttentionLSTM |
| `train.py` | Training loop, evaluation, checkpointing |
| `generate.py` | Text generation with temperature and nucleus sampling |
| `grid_search.py` | Hyperparameter grid search across all model types |
| `train_final_models.py` | Train the 4 best models found from grid search |
| `run_generation_eval.py` | Evaluate all grid search checkpoints on fixed test samples |
| `evaluate_generation.py` | Generation metrics: BLEU, bigram/trigram overlap, distinct-n, spelling |
| `plot_results.py` | Training curve and model comparison plots |
| `compare_models.py` | Generate sample text from all 4 final models side by side |

---

## Checkpoints (`checkpoints/`)

```
checkpoints/
├── final_models/       # Best checkpoints for the 4 final models
└── grid_search/        # Best checkpoint per grid search run (44 total)
```

The 4 final models:

| File | Model |
|------|-------|
| `final_rnn_h128_lr0p0005.pt` | RNN baseline |
| `final_lstm_l1_h128_d0p3_lr0p0005.pt` | LSTM 1-layer |
| `final_lstm_l2_h256_d0p3_lr0p001.pt` | LSTM 2-layer |
| `final_attention_lstm_l2_h128_d0p3_lr0p001_k20.pt` | Attention LSTM (2-layer, K=20) |

---

## Results (`results/`)

```
results/
├── grid_search/          # Grid search outputs
│   ├── summary.txt       # All 44 runs ranked by val loss (start here)
│   ├── all_results.json  # Full results in JSON
│   ├── all_results.csv   # Full results in CSV
│   └── *.log             # Per-run training logs
│
├── final_models/         # Final training outputs
│   ├── final_training_summary.txt   # Best val loss + test PPL for each model
│   ├── final_training_results.json  # Full training history
│   └── plots/            # Training curves and model comparison plots
│
└── generation_eval/      # Generation evaluation on fixed test samples
    ├── summary.txt        # All models ranked by BLEU (start here)
    ├── summary.json
    └── *.txt / *.json     # Per-model generated examples + metrics
```

---

## Grid Search

We ran **44 runs** in total across 4 model types (RNN, LSTM 1-layer, LSTM 2-layer, Attention LSTM), each trained for 1 epoch. The following hyperparameters were searched:

| Parameter | Values searched |
|-----------|----------------|
| `hidden_dim` | 128, 256 |
| `num_layers` | 1, 2 |
| `learning_rate` | 5e-4, 1e-3 |
| `dropout` | 0.2, 0.3 |
| `window_size_k` | 5, 10, 20 (Attention LSTM only) |

Fixed across all runs: `embed_dim=128`, `seq_length=100`, `batch_size=64`, `vocab_size=5000`.

The best config per model type was selected by **validation loss** and used to train the final models for 4 epochs. See `results/grid_search/summary.txt` for the full ranking.

---

## How to Run

```bash
# 1. Train final models (saves checkpoints + plots)
python src/train_final_models.py

# 2. Run generation evaluation on final models
python src/run_generation_eval.py

# 3. Compare generated text across all 4 models side by side
python src/compare_models.py
```

### Generate text from a specific model with your own prompt

```bash
python src/generate.py \
  --checkpoint checkpoints/final_models/final_attention_lstm_l2_h128_d0p3_lr0p001_k20.pt \
  --tokenizer data/gothic_tokenizer.json \
  --prompt "The old castle stood in darkness" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-p 0.9
```

Swap `--checkpoint` to try a different model:

| Model | Checkpoint |
| ----- | ---------- |
| RNN | `checkpoints/final_models/final_rnn_h128_lr0p0005.pt` |
| LSTM 1-layer | `checkpoints/final_models/final_lstm_l1_h128_d0p3_lr0p0005.pt` |
| LSTM 2-layer | `checkpoints/final_models/final_lstm_l2_h256_d0p3_lr0p001.pt` |
| Attention LSTM | `checkpoints/final_models/final_attention_lstm_l2_h128_d0p3_lr0p001_k20.pt` |

- `--temperature` controls randomness: lower (e.g. 0.5) = more predictable, higher (e.g. 1.2) = more creative
- `--top-p` controls nucleus sampling: 0.9 is a good default, set to 1.0 to disable it

---

## Metrics

- **Perplexity** — primary metric. Lower is better. Random baseline ~5000; our models range 200–350.
- **BLEU** — corpus-level n-gram overlap with reference text.
- **Bigram/Trigram overlap** — fraction of generated n-grams that appear in the reference.
- **Distinct-1 / Distinct-2** — diversity of generated text (unique unigrams/bigrams). Higher is better.
- **Spelling accuracy** — fraction of generated words found in the English dictionary.
