#!/usr/bin/env python3
"""Convert MMLU dataset (HuggingFace parquet) to vLLM custom JSONL format.

Each MMLU question becomes an independent prompt with no shared prefix,
making it ideal as a low-reuse / negative control workload.

Output format:
  {"prompt": "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:", "output_tokens": 10, "subject": "..."}
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


CHOICES_LABELS = ["A", "B", "C", "D"]

SKIP_DIRS = {"all", "auxiliary_train", ".huggingface"}


def format_question(row) -> str:
    """Format a single MMLU question as a prompt string."""
    question = row["question"]
    choices = row["choices"]
    lines = [f"Question: {question}"]
    for label, choice in zip(CHOICES_LABELS, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_subject(subject_dir: Path, split: str = "test") -> pd.DataFrame:
    """Load a single subject's parquet file."""
    pattern = f"{split}-*.parquet"
    files = list(subject_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[0])


def main():
    parser = argparse.ArgumentParser(description="Convert MMLU to vLLM JSONL")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="MMLU dataset directory (HuggingFace download)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "validation", "dev"],
                        help="Which split to use")
    parser.add_argument("--subjects", type=str, default="all",
                        help="Comma-separated subjects or 'all'")
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="Max prompts to output (0 = all)")
    parser.add_argument("--output-tokens", type=int, default=10,
                        help="Expected output tokens (short for MCQ)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Discover subjects
    if args.subjects == "all":
        subjects = sorted([
            d.name for d in input_dir.iterdir()
            if d.is_dir() and d.name not in SKIP_DIRS
        ])
    else:
        subjects = [s.strip() for s in args.subjects.split(",")]

    records = []
    subject_counts = {}

    for subject in subjects:
        subject_dir = input_dir / subject
        if not subject_dir.exists():
            print(f"  Warning: subject '{subject}' not found, skipping")
            continue

        df = load_subject(subject_dir, args.split)
        if df.empty:
            continue

        count = 0
        for _, row in df.iterrows():
            prompt = format_question(row)
            records.append({
                "prompt": prompt,
                "output_tokens": args.output_tokens,
                "subject": subject,
            })
            count += 1
        subject_counts[subject] = count

    # Shuffle and optionally limit
    import random
    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.max_prompts > 0:
        records = records[:args.max_prompts]

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Converted {len(records)} MMLU questions → {output_path}")
    print(f"  Subjects: {len(subject_counts)}")
    print(f"  Split: {args.split}")
    if args.max_prompts > 0:
        print(f"  (Limited to {args.max_prompts} prompts)")


if __name__ == "__main__":
    main()
