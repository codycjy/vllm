#!/usr/bin/env python3
"""
Generate a multi-turn replay dataset from ShareGPT for prefix cache testing.

Each conversation contributes N-1 requests (turn 2..N), where turn K's prompt
is the full conversation history up to human turn K. This guarantees that
turn K shares a growing prefix with turn K-1, exercising prefix caching.

Optionally, prepend the same system prompt to every request. This creates a
global shared prefix across conversations and is useful for testing
shared-prefix-aware KV cache eviction/protection.

Requests are written in conversation order (use --disable-shuffle when serving).

Output JSONL format:
    {"prompt": "...", "output_tokens": N, "conv_id": "...", "turn": K}

Usage:
    python3 scripts/gen_multiturn_dataset.py \
        --input  data/ShareGPT_V3_unfiltered_cleaned_split.json \
        --output data/sharegpt_multiturn.jsonl \
        --min-turns 4 \
        --max-convs 300 \
        --max-prompt-chars 6000 \
        --max-output-tokens 512

    python3 scripts/gen_multiturn_dataset.py \
        --input  data/ShareGPT_V3_unfiltered_cleaned_split.json \
        --output data/sharegpt_multiturn_shared_system.jsonl \
        --use-default-common-system-prompt
"""

import argparse
import json
import random
from pathlib import Path

DEFAULT_COMMON_SYSTEM_PROMPT = """
You are a shared system assistant used for multi-turn LLM serving experiments.
Follow the user's instructions carefully, preserve the conversation context
across turns, answer with concise but complete reasoning, and avoid inventing
facts. Keep your response style stable across requests so that this system
message forms a reusable prefix for prefix-cache evaluation. Treat all
following dialogue as part of the same chat transcript format with Human and
Assistant roles.
""".strip()

DEFAULT_SYSTEM_PROMPT_TEMPLATE = "[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\n"


def build_history_prompt(turns: list[dict], up_to_human_idx: int) -> tuple[str, str]:
    """
    Build the prompt (history up to and including human turn `up_to_human_idx`)
    and the expected output (the corresponding gpt turn).

    turns: flat list of {"from": "human"|"gpt", "value": "..."}
    up_to_human_idx: index of the target human turn (0-based among human turns)

    Returns (prompt_text, output_text) where:
      prompt_text = "Human: H0\n\nAssistant: G0\n\nHuman: H1\n\n..."
      output_text = corresponding gpt response
    """
    human_seen = -1
    prompt_parts = []
    output_text = ""
    target_turn_idx = -1

    for idx, turn in enumerate(turns):
        role = turn["from"]
        val = turn["value"].strip()
        if role == "human":
            human_seen += 1
            prompt_parts.append(f"Human: {val}")
            if human_seen == up_to_human_idx:
                # next gpt turn is our expected output
                target_turn_idx = idx
                break
        elif role == "gpt" and human_seen < up_to_human_idx:
            prompt_parts.append(f"Assistant: {val}")

    if target_turn_idx >= 0:
        for turn in turns[target_turn_idx + 1 :]:
            if turn["from"] == "gpt":
                output_text = turn["value"].strip()
                break

    prompt_text = "\n\n".join(prompt_parts)
    return prompt_text, output_text


def load_common_system_prompt(args: argparse.Namespace) -> str | None:
    sources = [
        args.use_default_common_system_prompt,
        args.common_system_prompt is not None,
        args.common_system_prompt_file is not None,
    ]
    if sum(bool(source) for source in sources) > 1:
        raise SystemExit(
            "Use only one of --use-default-common-system-prompt, "
            "--common-system-prompt, or --common-system-prompt-file"
        )

    if args.use_default_common_system_prompt:
        return DEFAULT_COMMON_SYSTEM_PROMPT
    if args.common_system_prompt is not None:
        return args.common_system_prompt.strip()
    if args.common_system_prompt_file is not None:
        return Path(args.common_system_prompt_file).read_text().strip()
    return None


def render_common_system_prefix(system_prompt: str | None, template: str) -> str:
    if not system_prompt:
        return ""
    if "{system_prompt}" not in template:
        raise SystemExit("--system-prompt-template must contain {system_prompt}")
    return template.format(system_prompt=system_prompt.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    parser.add_argument("--output", default="data/sharegpt_multiturn.jsonl")
    parser.add_argument(
        "--min-turns",
        type=int,
        default=4,
        help="Min conversation turns (human+gpt combined) to include",
    )
    parser.add_argument("--max-convs", type=int, default=300, help="Max conversations")
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=6000,
        help="Skip requests where prompt exceeds this char length",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Cap output_tokens at this value",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-default-common-system-prompt",
        action="store_true",
        help="Prepend a built-in shared system prompt to every request",
    )
    parser.add_argument(
        "--common-system-prompt",
        default=None,
        help="Custom shared system prompt text to prepend to every request",
    )
    parser.add_argument(
        "--common-system-prompt-file",
        default=None,
        help="Path to a file containing the shared system prompt",
    )
    parser.add_argument(
        "--system-prompt-template",
        default=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        help="Template used to wrap the shared prompt; must contain {system_prompt}",
    )
    parser.add_argument(
        "--system-prompt-id",
        default="common_system_v1",
        help="Metadata id written to each record when a shared system prompt is used",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    common_system_prompt = load_common_system_prompt(args)
    common_system_prefix = render_common_system_prefix(
        common_system_prompt, args.system_prompt_template
    )

    with open(args.input) as f:
        data = json.load(f)

    # Filter: need at least min-turns and alternating human/gpt
    valid = []
    for conv in data:
        turns = conv["conversations"]
        if len(turns) < args.min_turns:
            continue
        # must start with human and have at least 2 human turns
        human_turns = [t for t in turns if t["from"] == "human"]
        gpt_turns = [t for t in turns if t["from"] == "gpt"]
        if len(human_turns) < 2 or len(gpt_turns) < 2:
            continue
        valid.append(conv)

    random.shuffle(valid)
    selected = valid[: args.max_convs]
    print(f"Total valid conversations (>={args.min_turns} turns): {len(valid)}")
    print(f"Selected: {len(selected)}")

    records = []
    skipped_len = 0

    for conv in selected:
        turns = conv["conversations"]
        conv_id = conv["id"]

        # Collect human turn indices
        human_indices = [i for i, t in enumerate(turns) if t["from"] == "human"]

        # Generate requests for turn 2..N (0-indexed: human_turn_idx 1..)
        for h_idx in range(1, len(human_indices)):
            prompt, output = build_history_prompt(turns, h_idx)
            if common_system_prefix:
                prompt = common_system_prefix + prompt

            if len(prompt) > args.max_prompt_chars:
                skipped_len += 1
                continue

            # Estimate output tokens (~4 chars per token, min 1)
            output_tokens = max(1, min(len(output) // 4, args.max_output_tokens))

            rec = {
                "prompt": prompt,
                "output_tokens": output_tokens,
                "conv_id": conv_id,
                "turn": h_idx + 1,  # 1-based human turn number
            }
            if common_system_prefix:
                rec["prefix_group"] = args.system_prompt_id
                rec["has_common_system_prompt"] = True
            records.append(rec)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(records)} requests to {args.output}  "
        f"(skipped {skipped_len} too-long prompts)"
    )
    if common_system_prefix:
        print(
            f"Common system prompt: id={args.system_prompt_id}  "
            f"chars={len(common_system_prefix)}"
        )

    # Stats
    turns_per_conv = {}
    for rec in records:
        turns_per_conv.setdefault(rec["conv_id"], 0)
        turns_per_conv[rec["conv_id"]] += 1
    avg_turns = (
        sum(turns_per_conv.values()) / len(turns_per_conv) if turns_per_conv else 0
    )

    output_lens = [rec["output_tokens"] for rec in records]
    print(f"Avg follow-up turns per conversation: {avg_turns:.1f}")
    print(
        "Output tokens — "
        f"min: {min(output_lens)}, "
        f"median: {sorted(output_lens)[len(output_lens) // 2]}, "
        f"max: {max(output_lens)}"
    )

    # Show first 2 records for sanity check
    print("\n--- Sample record (turn 2) ---")
    for rec in records[:1]:
        print(
            f"conv_id={rec['conv_id']}  "
            f"turn={rec['turn']}  "
            f"output_tokens={rec['output_tokens']}"
        )
        print(f"prompt preview: {rec['prompt'][:300]}...")


if __name__ == "__main__":
    main()
