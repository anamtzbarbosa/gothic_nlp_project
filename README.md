# Gothic NLP Project

Language model trained on Gothic literature using RNN, LSTM, and Attention-LSTM architectures with BPE tokenization.

---

## Project Structure

```
gothic_nlp_project/
├── data/                        # Corpus, tokenizer, and split files
├── src/                         # All source code
│   ├── evaluation/              # Generation, evaluation, and plotting
│   ├── grid_search/             # Grid search scripts (normal + resume)
│   └── ...                      # Core training and model files
├── checkpoints/                 # Saved model weights
├── results/                     # All outputs and evaluation results
```

---

## Data (`data/`)

| File | Description |
|------|-------------|
| `corpus_clean.txt` | Cleaned raw text (~3.51M tokens, expanded Gothic corpus) |
| `gothic_tokenizer.json` | Trained BPE tokenizer (vocab size 5000) |
| `corpus_tokenized.pkl` | Full corpus encoded as BPE token IDs |
| `train_tokens.pkl` | Training split (80%) |
| `val_tokens.pkl` | Validation split (10%) |
| `test_tokens.pkl` | Test split (10%) |
| `eval_samples.json` | 20 fixed test samples used for generation evaluation |

---

## Source Code (`src/`)

### Core files

| File | What it does |
|------|--------------|
| `tokenizer.py` | BPE tokenizer implementation (train + encode/decode) |
| `data_utils.py` | Text cleaning and preprocessing |
| `dataset.py` | Dataset class, train/val/test split (80/10/10), save splits |
| `models.py` | Model definitions: VanillaRNN, DeepLSTM, CrossAttentionLSTM |
| `train.py` | Training loop, evaluation, checkpointing |
| `train_final_models.py` | Train the best models found from grid search |
| `train_attention_lstm.py` | Train multi-head Attention LSTM variants (1/2/3 layers, K=20/40) |

### `evaluation/`

| File | What it does |
|------|--------------|
| `generate.py` | Text generation with temperature and nucleus sampling |
| `evaluate_generation.py` | Generation metrics: BLEU, bigram/trigram overlap, distinct-n, spelling |
| `run_generation_eval.py` | Evaluate all checkpoints on fixed test samples |
| `plot_results.py` | Training curve and model comparison plots |

### `grid_search/`

| File | What it does |
|------|--------------|
| `grid_search_v3_rnn.py` | Grid search for RNN (48 runs) |
| `grid_search_v3_lstm.py` | Grid search for LSTM (48 runs, `--seq-length` arg) |
| `grid_search_v3_attention_lstm.py` | Grid search for Attention LSTM (48 runs, `--seq-length`, `--batch-size` args) |
| `grid_search_v3_rnn_resume.py` | Resume interrupted RNN grid search |
| `grid_search_v3_lstm_resume.py` | Resume interrupted LSTM grid search |
| `grid_search_v3_attention_lstm_resume.py` | Resume interrupted Attention LSTM grid search |

---

## Checkpoints (`checkpoints/`)

```
checkpoints/
├── final_models/             # Best checkpoints for the final trained models
├── attention_lstm_new/       # Multi-head Attention LSTM variants (1/2/3 layers, K=20/40)
├── grid_search_v3_rnn/       # Best checkpoint per RNN grid search run
├── grid_search_v3_lstm/      # Best checkpoint per LSTM grid search run
└── grid_search_v3_attention_lstm/  # Best checkpoint per Attention LSTM grid search run
```

---

## Results (`results/`)

```
results/
├── grid_search_v2/                        # Old grid search (reference only)
├── grid_search_v3_rnn/                    # RNN grid search results
├── grid_search_v3_lstm_s100/              # LSTM grid search (seq_length=100)
├── grid_search_v3_attention_lstm_s100_b64/ # Attention LSTM grid search (S=100, B=64)
├── attention_lstm_new/                    # Multi-head Attention LSTM training results
├── final_models/                          # Final training outputs
│   ├── final_training_summary.txt
│   ├── final_training_results.json
│   └── plots/
└── generation_eval/                       # Generation evaluation on fixed test samples
    ├── summary.txt
    ├── summary.json
    └── *.txt / *.json
```

---

## Grid Search

We ran separate per-model grid searches on the expanded corpus (~3.51M tokens), each trained for 1 epoch:

| Model | Runs | Batch Size | Seq Length | Other |
| --- | --- | --- | --- | --- |
| RNN | 48 | 32, 64 | 100, 200 | dropout: 0.0, 0.2, 0.3 |
| LSTM | 48 | 32, 64 | 100 | layers: 1, 2, 3; dropout: 0.2, 0.3 |
| Attention LSTM | 48 | 64 | 100 | layers: 1, 2, 3; K: 20, 40 |

All models searched: `hidden_dim` ∈ {128, 256}, `learning_rate` ∈ {5e-4, 1e-3}.

Resume scripts automatically detect completed runs from `all_results.json` and skip them.

---

## How to Run

```bash
# Grid search (run each in a separate notebook/GPU)
python src/grid_search/grid_search_v3_rnn.py
python src/grid_search/grid_search_v3_lstm.py --seq-length 100
python src/grid_search/grid_search_v3_attention_lstm.py --seq-length 100 --batch-size 64

# Resume an interrupted grid search
python src/grid_search/grid_search_v3_lstm_resume.py --seq-length 100

# Train multi-head Attention LSTM variants (2-layer and 3-layer, K=20 and K=40)
python src/train_attention_lstm.py

# Train final models
python src/train_final_models.py

# Run generation evaluation
python src/evaluation/run_generation_eval.py
```

### Generate text from a specific model

```bash
python src/evaluation/generate.py \
  --checkpoint checkpoints/final_models/best.pt \
  --tokenizer data/gothic_tokenizer.json \
  --prompt "The old castle stood in darkness" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-p 0.9
```

- `--temperature`: lower (e.g. 0.5) = more predictable, higher (e.g. 1.2) = more creative
- `--top-p`: nucleus sampling threshold; 0.9 is a good default, 1.0 disables it

---

## Metrics

- **Perplexity** — primary metric. Lower is better.
- **BLEU** — corpus-level n-gram overlap with reference text.
- **Bigram/Trigram overlap** — fraction of generated n-grams that appear in the reference.
- **Distinct-1 / Distinct-2** — diversity of generated text. Higher is better.
- **Spelling accuracy** — fraction of generated words found in the English dictionary.
