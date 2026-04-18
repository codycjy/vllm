#!/usr/bin/env python3
"""Generate synthetic prompt-length bucket datasets for the prompt length ablation.

Two popularity distributions:
  - "uniform" (v1): 20 unique prompts × 10 repeats each = 200 requests.
      Every prompt is equally popular. Eviction's reuse_count signal is flat;
      prefix_match scheduling has no cache-hit discrimination across requests.
      Useful as a control / methodology comparison.
  - "skewed" (v2, default): 4 hot × 30 + 16 cold × 5 = 200 requests.
      Mirrors proposal §4.1 "distinguish high-reuse prefixes from one-off
      prompts". Hot prompts simulate popular system prompts; cold prompts
      simulate one-off queries. Exercises both adaptive eviction's reuse
      signal and prefix_match's cache-hit discrimination.

Each bucket produces a ShareGPT-compatible JSON file at:
    <output-dir>/bucket_<name>.json

Output format (compatible with experiments/collect_metrics.py --dataset):
    [{"conversations": [{"from": "human", "value": "..."}]}, ...]

Usage:
    python3 scripts/prepare_length_buckets.py                           # skewed (default)
    python3 scripts/prepare_length_buckets.py --popularity uniform      # v1 control
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
# Popularity configs: tuples of (hot_count, hot_repeats, cold_repeats)
# Both must produce the same total (N_UNIQUE × 10 = 200) for apples-to-apples
# comparison with equal workload size.
_POPULARITY = {
    # uniform: every prompt appears 10 times
    "uniform": (N_UNIQUE, 10, 10),
    # skewed: 4 hot × 30 + 16 cold × 5 = 120 + 80 = 200
    "skewed":  (4, 30, 5),
}

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


def build_prefix(n_words, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORD_POOL) for _ in range(n_words))


def build_repeat_counts(popularity):
    """Return a list of N_UNIQUE repeat counts (first hot_count are hot)."""
    hot_count, hot_reps, cold_reps = _POPULARITY[popularity]
    return [hot_reps] * hot_count + [cold_reps] * (N_UNIQUE - hot_count)


def generate_bucket(n_prefix_words, popularity):
    unique_prompts = [
        f"{build_prefix(n_prefix_words, seed=1000 + i)}\n\nQuery: {QUESTIONS[i]}"
        for i in range(N_UNIQUE)
    ]
    repeats = build_repeat_counts(popularity)

    entries = []
    for prompt, rep in zip(unique_prompts, repeats):
        entries.extend([prompt] * rep)
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
        "--popularity", choices=sorted(_POPULARITY.keys()), default="skewed",
        help="Prompt popularity distribution (default: skewed)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hot_count, hot_reps, cold_reps = _POPULARITY[args.popularity]
    total = hot_count * hot_reps + (N_UNIQUE - hot_count) * cold_reps

    print(f"Popularity: {args.popularity}")
    print(f"  {hot_count} hot prompts × {hot_reps} repeats = "
          f"{hot_count * hot_reps} hot requests")
    print(f"  {N_UNIQUE - hot_count} cold prompts × {cold_reps} repeats = "
          f"{(N_UNIQUE - hot_count) * cold_reps} cold requests")
    print(f"  Total: {total} requests per bucket\n")

    for bucket_name, cfg in BUCKETS.items():
        entries = generate_bucket(cfg["n_prefix_words"], args.popularity)
        out_path = out / f"bucket_{bucket_name}.json"
        with open(out_path, "w") as f:
            json.dump(entries, f, ensure_ascii=False)

        sample_len = len(entries[0]["conversations"][0]["value"].split())
        est_tokens = int(sample_len * 1.3)
        print(
            f"  {bucket_name:8s}: {len(entries):3d} entries, "
            f"~{sample_len} words/prompt (~{est_tokens} tokens)  → {out_path}"
        )

    print(f"\nAll buckets written to: {out}")


if __name__ == "__main__":
    main()
