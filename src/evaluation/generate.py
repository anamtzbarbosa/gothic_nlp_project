import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from models import VanillaRNN, DeepLSTM, SelfAttentionLSTM
from tokenizer import GothicBPE


@dataclass
class GenerateConfig:
    checkpoint_path: str = "checkpoints/best_lstm.pt"
    tokenizer_path: str = "data/gothic_tokenizer.json"
    prompt: str = "The castle was"
    max_new_tokens: int = 300
    temperature: float = 1.0
    top_p: float | None = None
    deterministic: bool = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_bpe_tokenizer(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")

    tokenizer = GothicBPE()

    with path.open("r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    tokenizer.merges = {
        tuple(map(int, pair.split(","))): merge_id
        for pair, merge_id in tokenizer_data["merges"].items()
    }

    tokenizer.vocab = tokenizer.build_vocab()
    return tokenizer


def build_model(model_config):
    model_name = model_config["model_name"]

    common_args = {
        "vocab_size": model_config["vocab_size"],
        "embed_dim": model_config["embed_dim"],
        "hidden_dim": model_config["hidden_dim"],
    }

    if model_name == "rnn":
        return VanillaRNN(**common_args, dropout=model_config.get("dropout", 0.0))

    if model_name == "lstm":
        return DeepLSTM(
            **common_args,
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )

    if model_name == "attention_lstm":
        return SelfAttentionLSTM(
            **common_args,
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            window_size_k=model_config["window_size_k"],
        )

    raise ValueError(f"Unknown model name: {model_name}")


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint["config"]
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, model_config


def apply_nucleus_sampling(probs, top_p):
    if top_p is None or top_p == 1.0:
        return probs

    if not (0.0 < top_p < 1.0):
        raise ValueError("top_p must be in (0.0, 1.0]")

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = (cumulative_probs - sorted_probs) >= top_p
    remove_mask[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    return filtered_probs


def sample_next_token(logits, temperature, top_p, deterministic=False):
    if deterministic:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if temperature <= 0.0:
        raise ValueError("temperature must be greater than 0.")

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    probs = apply_nucleus_sampling(probs, top_p)

    return torch.multinomial(probs, num_samples=1)


def generate_text(model, tokenizer, config, model_config, device):
    token_ids = tokenizer.encode(config.prompt)

    if not token_ids:
        raise ValueError("Prompt produced no tokens.")

    seq_length = model_config["seq_length"]

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            context = token_ids[-seq_length:]

            x = torch.tensor(
                context,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            output = model(x)
            logits = output[0]

            next_token_logits = logits[:, -1, :]
            next_token = sample_next_token(
                logits=next_token_logits,
                temperature=config.temperature,
                top_p=config.top_p,
                deterministic=config.deterministic,
            )

            token_ids.append(next_token.item())

    return tokenizer.decode(token_ids)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Gothic text from a trained model.")

    parser.add_argument("--checkpoint", default="checkpoints/best_lstm.pt")
    parser.add_argument("--tokenizer", default="data/gothic_tokenizer.json")
    parser.add_argument("--prompt", default="The castle was")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--deterministic", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    config = GenerateConfig(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        deterministic=args.deterministic,
    )

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = load_bpe_tokenizer(config.tokenizer_path)
    model, model_config = load_model_from_checkpoint(config.checkpoint_path, device)

    print(f"Loaded model: {model_config['model_name']}")
    print(f"Prompt: {config.prompt}")
    print(f"Temperature: {config.temperature}")
    print(f"Top-p: {config.top_p}")
    print("-" * 80)

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        config=config,
        model_config=model_config,
        device=device,
    )

    print(generated_text)


if __name__ == "__main__":
    main()