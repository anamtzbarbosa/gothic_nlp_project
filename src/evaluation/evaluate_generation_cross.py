"""  Please take note THIS FILE ONLY WORKS WITH THE VERSION OF MODELS.PY THAT IS IN THE RESOLVE-CONFLICTS BRANCH
"""
import argparse
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import CrossAttentionLSTM
from evaluation.generate_attention import load_tokenizer
from evaluation.evaluate_generation import (
    load_eval_samples,
    compute_all_metrics,
    save_metrics_and_examples,
)

def load_cross_model(checkpoint_path, device):
    """Loads the custom Cached Cross-Attention model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint['config']
    
    model = CrossAttentionLSTM(
        vocab_size=cfg['vocab_size'],
        embed_dim=cfg['embed_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        window_size_k=cfg['window_size_k']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, cfg

def generate_cross(model, tokenizer, prompt, seq_length, max_new_tokens, temperature=0.7, top_p=0.9, device="cpu"):
    """Generates text while maintaining the detached cached_memory state."""
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    hidden = None
    memory = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = input_ids[:, -seq_length:]
            
            logits, hidden, _, memory = model(curr_input, hidden, memory)
            
            next_token_logits = logits[:, -1, :]
            
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            else:
                next_token_logits = next_token_logits / temperature
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
    return tokenizer.decode(input_ids[0].tolist())

class CrossAttentionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, hidden=None):
        logits, h, _, _ = self.model(x, hidden, memory=None)
        return logits, h

def collect_from_eval_samples(model, tokenizer, eval_samples, seq_length, temperature, top_p, device):
    generated_texts, reference_texts, prompt_texts = [], [], []

    for sample in eval_samples:
        prompt_ids = sample["prompt_ids"]
        reference_text = sample["reference_text"]
        max_new_tokens = seq_length - len(prompt_ids)
        if max_new_tokens <= 0:
            max_new_tokens = seq_length

        prompt_text = sample.get("prompt_text", tokenizer.decode(prompt_ids))
        
        full_text = generate_cross(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            seq_length=seq_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        
        generated_text = full_text[len(prompt_text):]
        generated_texts.append(generated_text)
        reference_texts.append(reference_text)
        prompt_texts.append(prompt_text)

    return generated_texts, reference_texts, prompt_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/gothic_tokenizer.json")
    parser.add_argument("--eval-samples", default="data/eval_samples.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", default="results/generation_metrics_cross")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(args.tokenizer)
    model, cfg = load_cross_model(args.checkpoint, device)
    seq_length = cfg["seq_length"]

    print(f"Loaded: {cfg['num_layers']}-layer CROSS Attention LSTM | K={cfg['window_size_k']}")
    eval_samples = load_eval_samples(args.eval_samples)

    print("Generating text samples...")
    generated_texts, reference_texts, prompt_texts = collect_from_eval_samples(
        model=model, tokenizer=tokenizer, eval_samples=eval_samples,
        seq_length=seq_length, temperature=args.temperature, top_p=args.top_p, device=device,
    )

    print("Computing metrics...")
    wrapped_model = CrossAttentionWrapper(model)
    metrics = compute_all_metrics(
        generated_texts, reference_texts,
        model=wrapped_model, tokenizer=tokenizer, seq_length=seq_length, device=device,
    )

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    txt_path, json_path = save_metrics_and_examples(
        metrics=metrics, generated_texts=generated_texts,
        reference_texts=reference_texts, prompt_texts=prompt_texts,
        output_prefix=args.output,
    )
    print(f"\nSaved results to: {txt_path}")

if __name__ == "__main__":
    main()

