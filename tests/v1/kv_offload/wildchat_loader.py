# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
WildChat dataset loader for KV cache eviction policy benchmarking.

Loads multi-turn conversations from allenai/WildChat and expands them
into incremental prompt sequences that naturally exhibit prefix sharing,
simulating real chat application access patterns.
"""

from __future__ import annotations

import logging
import random
from typing import Literal

logger = logging.getLogger(__name__)

TurnCategory = Literal["single", "short_multi", "long_multi"]

TURN_CATEGORIES: dict[TurnCategory, tuple[int, int]] = {
    "single": (1, 1),
    "short_multi": (2, 5),
    "long_multi": (6, 120),
}

SCALE_SIZES = {
    "small": 50,
    "medium": 500,
    "large": 5000,
}


def load_wildchat(
    scale: str = "small",
    language: str = "English",
    seed: int = 42,
    stratified: bool = True,
    max_chars: int = 0,
) -> list[list[str]]:
    """
    Load and sample conversations from WildChat dataset.

    Args:
        scale: Sampling scale - "small" (50), "medium" (500), "large" (5000)
        language: Filter by language (default: English)
        seed: Random seed for reproducibility
        stratified: Whether to stratify by turn count category
        max_chars: Max character length per prompt (0 = no limit).
                   Use model_max_tokens * 3 as a safe estimate.

    Returns:
        List of conversations, where each conversation is a list of
        incrementally expanding prompts (prefix-sharing sequence).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for WildChat loading. "
            "Install with: pip install datasets"
        )

    num_conversations = SCALE_SIZES.get(scale)
    if num_conversations is None:
        raise ValueError(
            f"Unknown scale: {scale}. Choose from {list(SCALE_SIZES.keys())}"
        )

    logger.info("Loading WildChat dataset from Hugging Face...")
    ds = load_dataset("allenai/WildChat", split="train")

    # Filter: non-toxic, target language, at least 1 turn
    ds = ds.filter(
        lambda row: (
            not row["toxic"]
            and row["language"] == language
            and row["turn"] >= 1
        ),
        num_proc=4,
    )

    logger.info("Filtered dataset size: %d conversations", len(ds))

    if stratified:
        conversations = _stratified_sample(ds, num_conversations, seed)
    else:
        conversations = _random_sample(ds, num_conversations, seed)

    return [_expand_conversation(conv, max_chars=max_chars) for conv in conversations]


def _random_sample(ds, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in indices]


def _stratified_sample(ds, n: int, seed: int) -> list[dict]:
    """Sample proportionally from each turn category."""
    rng = random.Random(seed)

    buckets: dict[TurnCategory, list[int]] = {
        cat: [] for cat in TURN_CATEGORIES
    }

    for i in range(len(ds)):
        turn = ds[i]["turn"]
        for cat, (lo, hi) in TURN_CATEGORIES.items():
            if lo <= turn <= hi:
                buckets[cat].append(i)
                break

    # Proportional allocation
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        return []

    result: list[dict] = []
    for cat, indices in buckets.items():
        if not indices:
            continue
        cat_n = max(1, round(n * len(indices) / total))
        cat_n = min(cat_n, len(indices))
        sampled = rng.sample(indices, cat_n)
        result.extend(ds[i] for i in sampled)

    # Trim or pad to exact n
    rng.shuffle(result)
    return result[:n]


def _expand_conversation(
    row: dict,
    max_chars: int = 0,
) -> list[str]:
    """
    Expand a multi-turn conversation into incremental prompt sequences.

    A 3-turn conversation becomes:
        prompt_1 = "user1"
        prompt_2 = "user1\\nassistant1\\nuser2"
        prompt_3 = "user1\\nassistant1\\nuser2\\nassistant2\\nuser3"

    Args:
        row: A WildChat conversation row.
        max_chars: Maximum character length per prompt. 0 means no limit.
                   Roughly 1 token ≈ 4 chars, so for a 4096-token model
                   use max_chars=12000 to leave room for generation.
    """
    messages = row["conversation"]
    prompts = []
    accumulated = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if accumulated:
            accumulated += "\n"
        accumulated += f"{role}: {content}"

        # Stop expanding if we'd exceed the context limit
        if max_chars > 0 and len(accumulated) > max_chars:
            break

        # Emit a prompt after each user message
        if role == "user":
            prompts.append(accumulated)

    return prompts if prompts else [messages[0]["content"][:max_chars or None]]


def flatten_to_prompts(conversations: list[list[str]]) -> list[str]:
    """
    Flatten expanded conversations into a single prompt list.

    This interleaves conversations to simulate concurrent users,
    which creates more realistic cache pressure patterns.
    """
    # Round-robin across conversations
    result = []
    max_turns = max((len(c) for c in conversations), default=0)

    for turn_idx in range(max_turns):
        for conv in conversations:
            if turn_idx < len(conv):
                result.append(conv[turn_idx])

    return result


def get_wildchat_prompts(
    scale: str = "small",
    language: str = "English",
    seed: int = 42,
    interleave: bool = True,
    max_model_len: int = 0,
) -> list[str]:
    """
    High-level API: get a flat list of prompts from WildChat.

    Args:
        scale: "small" (50), "medium" (500), "large" (5000)
        language: Filter language
        seed: Random seed
        interleave: If True, interleave conversations (realistic);
                    if False, sequential (one conversation at a time)
        max_model_len: Model's max context length in tokens. If > 0,
                       prompts are truncated to fit (leaving room for output).

    Returns:
        Flat list of prompts ready for benchmarking
    """
    # Reserve ~10% for generation tokens, convert tokens to chars (~4 chars/token)
    max_chars = int(max_model_len * 0.9 * 4) if max_model_len > 0 else 0

    conversations = load_wildchat(
        scale=scale, language=language, seed=seed, max_chars=max_chars,
    )

    if interleave:
        return flatten_to_prompts(conversations)
    else:
        return [p for conv in conversations for p in conv]
