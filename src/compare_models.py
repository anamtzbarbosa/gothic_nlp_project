import sys
import torch

sys.path.insert(0, ".")
from generate import load_bpe_tokenizer, load_model_from_checkpoint, generate_text, GenerateConfig

TOKENIZER_PATH = "data/gothic_tokenizer.json"
CHECKPOINT_DIR = "checkpoints/final_models"

MODELS = [
    ("RNN",            "final_rnn_h128_lr0p0005.pt"),
    ("LSTM (1-layer)", "final_lstm_l1_h128_d0p3_lr0p0005.pt"),
    ("LSTM (2-layer)", "final_lstm_l2_h256_d0p3_lr0p001.pt"),
    ("Attention LSTM", "final_attention_lstm_l2_h128_d0p3_lr0p001_k20.pt"),
]

PROMPT = "The old castle stood in darkness, its towers"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_P = 0.9


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    tokenizer = load_bpe_tokenizer(TOKENIZER_PATH)

    print(f"\nPrompt: \"{PROMPT}\"")
    print(f"Temperature: {TEMPERATURE} | Top-p: {TOP_P} | Max new tokens: {MAX_NEW_TOKENS}")
    print("=" * 80)

    for name, ckpt_file in MODELS:
        ckpt_path = f"{CHECKPOINT_DIR}/{ckpt_file}"
        model, model_config = load_model_from_checkpoint(ckpt_path, device)

        config = GenerateConfig(
            prompt=PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        text = generate_text(model, tokenizer, config, model_config, device)

        print(f"\n{'─' * 80}")
        print(f"  {name}")
        print(f"{'─' * 80}")
        print(text)

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
