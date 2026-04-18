#!/usr/bin/env python3
"""Generate synthetic prompt-length bucket datasets for the prompt length ablation.

Each bucket produces a ShareGPT-compatible JSON file with:
  - 20 unique prompts sharing a common long prefix (simulating system prompt reuse)
  - Each prompt repeated 10 times (interleaved) = 200 total entries
  - Prefix length scaled to hit the target token range

Output format (compatible with experiments/collect_metrics.py --dataset):
  [{"conversations": [{"from": "human", "value": "..."}]}, ...]

Usage:
    python3 scripts/prepare_length_buckets.py \
        --output-dir /path/to/data/prompt_length
"""

import argparse
import json
import random
from pathlib import Path


BUCKETS = {
    "short":   {"n_prefix_words": 100,  "label": "64–256 tokens"},
    "medium":  {"n_prefix_words": 320,  "label": "256–512 tokens"},
    "long":    {"n_prefix_words": 700,  "label": "512–1024 tokens"},
    "xlarge":  {"n_prefix_words": 1300, "label": "1024–1800 tokens"},
}

N_UNIQUE = 20
N_REPEATS = 10

# Word pool for prefix generation. Each prompt gets a DIFFERENT random sequence
# drawn from this pool (via a per-prompt seed), so every prompt has a distinct
# block-hash chain → 20 prompts truly compete for cache space.
_WORD_POOL = (
    "the quick brown fox jumps over lazy dog cat sat mat sun shone brightly "
    "clear blue sky green hills river flows swiftly through valley ancient "
    "stones rest silently beneath moss covered trees birds singing morning "
    "dew glitters gently upon petals roses bloom wildly garden paths winding "
    "deep forest mountain peaks stretch toward horizon distant clouds drift "
    "across endless meadows children laugh together houses stand firm along "
    "busy streets markets open early vendors selling fresh bread cheese wine "
    "books stories ancient wisdom modern dreams future uncertain paths brave "
    "walking forward searching truth quiet evenings fireplace warm memories "
    "shared family friends gathered around wooden table simple meals laughter"
).split()

QUESTIONS = [
    "What is the main argument presented?",
    "Summarize the key points briefly.",
    "What conclusions can be drawn?",
    "List the most important details.",
    "What evidence supports this claim?",
    "How does this relate to common knowledge?",
    "What are the implications of this information?",
    "Identify any assumptions made here.",
    "What would be a counterargument?",
    "How would you rate the quality of this content?",
    "What additional context is needed?",
    "Does this information seem accurate?",
    "What is the intended audience?",
    "How could this be improved?",
    "What is missing from this analysis?",
    "Describe the tone of this passage.",
    "What action should be taken based on this?",
    "Who would benefit from this information?",
    "What risks are associated with this?",
    "Provide a one-sentence summary.",
]


def build_prefix(n_words: int, seed: int) -> str:
    """Generate a prefix of n_words using a per-prompt seed.

    Different seeds → different word sequences → different block hashes →
    prompts truly compete for cache space instead of sharing one prefix chain.
    """
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORD_POOL) for _ in range(n_words))


def generate_bucket(n_prefix_words: int):
    # 20 DISTINCT prefixes (one per unique prompt) using different seeds.
    # Each prefix has the same target length but different content.
    unique_prompts = [
        f"{build_prefix(n_prefix_words, seed=1000 + i)}\n\nQuery: {QUESTIONS[i]}"
        for i in range(N_UNIQUE)
    ]
    # Repeat each prompt N_REPEATS times, then interleave (not block-repeat)
    # Interleaving tests scheduling fairness across different requests
    entries = unique_prompts * N_REPEATS
    random.shuffle(entries)
    return [
        {"conversations": [{"from": "human", "value": p}]}
        for p in entries
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompt-length bucket datasets")
    parser.add_argument(
        "--output-dir", type=str,
        default="/ocean/projects/cis250265p/xli45/opensource/data/prompt_length",
        help="Output directory for bucket JSON files")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for bucket_name, cfg in BUCKETS.items():
        entries = generate_bucket(cfg["n_prefix_words"])
        out_path = out / f"bucket_{bucket_name}.json"
        with open(out_path, "w") as f:
            json.dump(entries, f, ensure_ascii=False)

        # Rough token estimate: n_prefix_words * 1.3 ≈ tokens
        sample_len = len(entries[0]["conversations"][0]["value"].split())
        est_tokens = int(sample_len * 1.3)
        print(
            f"  {bucket_name:8s}: {len(entries):3d} entries, "
            f"~{sample_len} words/prompt (~{est_tokens} tokens)  → {out_path}"
        )

    print(f"\nAll buckets written to: {out}")
    print(f"Each bucket: {N_UNIQUE} unique prompts × {N_REPEATS} repeats = "
          f"{N_UNIQUE * N_REPEATS} total requests")


if __name__ == "__main__":
    main()
