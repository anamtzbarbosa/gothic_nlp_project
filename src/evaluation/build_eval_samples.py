"""
Build eval_samples.json from the test set.

Each sample:
  - prompt starts at a sentence boundary, ends at a word boundary (~30 BPE tokens)
  - reference starts right after the prompt, ends at the next sentence boundary (~250 chars)

Run from the project root:
    python src/evaluation/build_eval_samples.py
    python src/evaluation/build_eval_samples.py --output data/eval_samples.json --num-samples 100
"""
import argparse
import json
import pickle
import re
from pathlib import Path

SENT_RE     = re.compile(r'[.!?]["\']?\s+[A-Z]')       # sentence-start boundary
SENT_END_RE = re.compile(r'[.!?]["\']?(?=\s|$)')        # sentence-end boundary

DEFAULT_NUM_SAMPLES   = 100
DEFAULT_OVERSAMPLE    = 500
DEFAULT_PROMPT_TOKENS = 30
DEFAULT_TARGET_CHARS  = 250
DEFAULT_MIN_CHARS     = 80
DEFAULT_MAX_CHARS     = 450


def load_tokenizer(path):
    with open(path) as f:
        data = json.load(f)

    merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
    vocab  = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    def _merge(ids, pair, idx):
        out = []; i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(idx); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    def encode(text):
        tokens = list(text.encode("utf-8"))
        for pair, new_id in merges.items():
            tokens = _merge(tokens, pair, new_id)
            if len(tokens) < 2:
                break
        return tokens

    def decode(ids):
        return b"".join(vocab[idx] for idx in ids).decode("utf-8", errors="replace")

    return encode, decode


def build_samples(
    test_tokens,
    encode,
    decode,
    num_samples=DEFAULT_NUM_SAMPLES,
    oversample=DEFAULT_OVERSAMPLE,
    prompt_tokens=DEFAULT_PROMPT_TOKENS,
    target_chars=DEFAULT_TARGET_CHARS,
    min_chars=DEFAULT_MIN_CHARS,
    max_chars=DEFAULT_MAX_CHARS,
):
    n    = len(test_tokens)
    step = n // oversample

    samples          = []
    skipped_no_start = 0
    skipped_no_end   = 0

    for i in range(oversample):
        if len(samples) >= num_samples:
            break

        pos         = i * step
        window_text = decode(test_tokens[pos: pos + 400])

        # Find sentence-start boundary
        m = SENT_RE.search(window_text)
        if m is None:
            skipped_no_start += 1
            continue

        sent_text = window_text[m.end() - 1:]   # starts at the capital letter

        # Encode the sentence text, take first prompt_tokens tokens
        sent_ids = encode(sent_text)
        if len(sent_ids) < prompt_tokens + 20:
            skipped_no_start += 1
            continue

        prompt_text_raw = decode(sent_ids[:prompt_tokens])

        # Trim to last word boundary so prompt never cuts mid-word
        last_space  = prompt_text_raw.rfind(' ')
        prompt_text = prompt_text_raw[:last_space] if last_space > 10 else prompt_text_raw
        prompt_ids  = encode(prompt_text)

        # Reference starts exactly where prompt_text ends in sent_text
        ref_window = sent_text[len(prompt_text): len(prompt_text) + 600]

        # Find sentence-end boundary closest to target_chars
        best_end  = None
        best_dist = float('inf')
        for em in SENT_END_RE.finditer(ref_window[min_chars:max_chars]):
            end_pos = min_chars + em.end()
            dist    = abs(end_pos - target_chars)
            if dist < best_dist:
                best_dist = dist
                best_end  = end_pos

        if best_end is None:
            skipped_no_end += 1
            continue

        reference_text = ref_window[:best_end]
        if not reference_text.strip():
            skipped_no_end += 1
            continue

        samples.append({
            "prompt_ids":     prompt_ids,
            "prompt_text":    prompt_text,
            "reference_text": reference_text,
        })

    print(f"Collected: {len(samples)} | Skipped no-start: {skipped_no_start} | Skipped no-end: {skipped_no_end}")
    return samples


def extend_samples(base_path, full_text, target_chars, min_chars, max_chars):
    """Extend references in an existing eval_samples.json to a longer sentence boundary.
    Prompts are kept identical; only reference_text is extended.
    """
    with open(base_path) as f:
        base = json.load(f)["samples"]

    new_samples = []
    fallbacks = 0

    for s in base:
        prompt_text = s["prompt_text"]
        orig_ref    = s["reference_text"]

        pos = full_text.find(prompt_text + orig_ref[:30])
        if pos == -1:
            pos = full_text.find(prompt_text)
        if pos == -1:
            new_samples.append(s)
            fallbacks += 1
            continue

        ref_start  = pos + len(prompt_text)
        ref_window = full_text[ref_start: ref_start + max_chars + 300]

        best_end  = None
        best_dist = float("inf")
        for em in SENT_END_RE.finditer(ref_window[min_chars:max_chars]):
            end_pos = min_chars + em.end()
            dist    = abs(end_pos - target_chars)
            if dist < best_dist:
                best_dist = dist
                best_end  = end_pos

        if best_end is None:
            best_end = min(max_chars, len(ref_window))
            fallbacks += 1

        new_samples.append({
            "prompt_ids":     s["prompt_ids"],
            "prompt_text":    prompt_text,
            "reference_text": ref_window[:best_end],
        })

    print(f"Extended {len(new_samples)} samples | fallbacks: {fallbacks}")
    return new_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-tokens",   default="data/test_tokens.pkl")
    parser.add_argument("--tokenizer",     default="data/gothic_tokenizer.json")
    parser.add_argument("--output",        default="data/eval_samples.json")
    parser.add_argument("--num-samples",   type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--oversample",    type=int, default=DEFAULT_OVERSAMPLE)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--target-chars",  type=int, default=DEFAULT_TARGET_CHARS)
    parser.add_argument("--min-chars",     type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-chars",     type=int, default=DEFAULT_MAX_CHARS)
    # Extend mode: keep prompts from an existing file, only lengthen references
    parser.add_argument("--extend-from",   default=None,
                        help="Path to existing eval_samples.json whose prompts to reuse with longer references")
    return parser.parse_args()


def main():
    args = parse_args()

    encode, decode = load_tokenizer(args.tokenizer)

    with open(args.test_tokens, "rb") as f:
        test_tokens = pickle.load(f)
    print(f"Test tokens: {len(test_tokens):,}")

    if args.extend_from:
        full_text = decode(test_tokens)
        samples = extend_samples(
            base_path    = args.extend_from,
            full_text    = full_text,
            target_chars = args.target_chars,
            min_chars    = args.min_chars,
            max_chars    = args.max_chars,
        )
    else:
        samples = build_samples(
            test_tokens  = test_tokens,
            encode       = encode,
            decode       = decode,
            num_samples  = args.num_samples,
            oversample   = args.oversample,
            prompt_tokens= args.prompt_tokens,
            target_chars = args.target_chars,
            min_chars    = args.min_chars,
            max_chars    = args.max_chars,
        )

    ref_lens  = [len(s["reference_text"]) for s in samples]
    prom_lens = [len(s["prompt_ids"])     for s in samples]
    print(f"Prompt tokens  — min:{min(prom_lens)}  max:{max(prom_lens)}  avg:{sum(prom_lens)/len(prom_lens):.1f}")
    print(f"Reference chars — min:{min(ref_lens)}  max:{max(ref_lens)}  avg:{sum(ref_lens)/len(ref_lens):.0f}")

    out = {"num_samples": len(samples), "samples": samples}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
