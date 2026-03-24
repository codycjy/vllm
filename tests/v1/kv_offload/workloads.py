# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared workload definitions and utilities for eviction policy benchmarking.

This module provides:
- WORKLOADS: predefined benchmark workload configurations
- EvictionMetrics: dataclass for tracking eviction performance
- prepare_prompts_from_workload: convert workload dict to prompt list
- load_workload_from_file: load custom workload from JSON
"""

import json
from dataclasses import dataclass

# Predefined workload definitions
WORKLOADS = {
    "chat_application": {
        "description": "Simulates a chat application with shared system prompts",
        "system_prompt": """You are a helpful AI assistant. You provide accurate,
concise, and friendly responses. Always be respectful and professional.""",
        "user_queries": [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "What are the health benefits of exercise?",
            "How does photosynthesis work?",
            "Recommend a good book for learning AI.",
            "What is the difference between RAM and ROM?",
            "Explain the water cycle.",
        ],
        "expected_sharing": "high",
    },
    "code_completion": {
        "description": "Code completion with shared repository context",
        "repo_context": """# File: utils.py
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}

    def process(self, data):
        # Process data according to config
        pass
""",
        "completion_requests": [
            "def validate_input(data):",
            "def load_config(path):",
            "def save_results(results, filename):",
            "class ConfigLoader:",
            "def parse_arguments():",
        ],
        "expected_sharing": "medium",
    },
    "few_shot_learning": {
        "description": "Few-shot learning with shared examples",
        "examples": """Q: Translate to French: Hello
A: Bonjour

Q: Translate to French: Goodbye
A: Au revoir

Q: Translate to French: Thank you
A: Merci
""",
        "queries": [
            "Q: Translate to French: Good morning",
            "Q: Translate to French: Please",
            "Q: Translate to French: How are you",
            "Q: Translate to French: I love you",
        ],
        "expected_sharing": "high",
    },
    "mixed_workload": {
        "description": "Mixed queries with varying sharing patterns",
        "queries": [
            "What is machine learning?",
            "Explain neural networks.",
            "What is machine learning?",
            "How does backpropagation work?",
            "What is machine learning?",
            "Explain convolutional neural networks.",
            "What are transformers in AI?",
            "What is machine learning?",
        ],
        "expected_sharing": "variable",
    },
}


@dataclass
class EvictionMetrics:
    """Metrics for evaluating eviction policy performance."""

    policy_name: str
    total_requests: int
    cache_hits: int
    cache_misses: int
    eviction_count: int
    avg_latency_ms: float

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.policy_name}:\n"
            f"  Requests: {self.total_requests}\n"
            f"  Hit Rate: {self.hit_rate:.1%}\n"
            f"  Hits: {self.cache_hits}, Misses: {self.cache_misses}\n"
            f"  Evictions: {self.eviction_count}\n"
            f"  Avg Latency: {self.avg_latency_ms:.2f}ms"
        )


def load_workload_from_file(file_path: str) -> dict:
    """
    Load a custom workload from an external JSON file.

    File format:
    {
        "description": "Workload description",
        "shared_prefix": "Common prefix for all requests",
        "queries": ["query1", "query2", ...],
        "expected_sharing": "high|medium|low"
    }
    """
    with open(file_path, "r") as f:
        return json.load(f)


def load_prompts(
    workload_name: str,
    wildchat_scale: str = "small",
    max_model_len: int = 0,
    seed: int = 42,
) -> list[str]:
    """Load prompts based on workload name.

    Args:
        workload_name: A key in WORKLOADS, ``"wildchat"``, or a path to a
            custom JSON workload file.
        wildchat_scale: Scale for WildChat workload (small/medium/large).
        max_model_len: Model's max context length (used for WildChat truncation).
        seed: Random seed for WildChat sampling.

    Returns:
        List of prompt strings ready for benchmarking.
    """
    if workload_name == "wildchat":
        from tests.v1.kv_offload.wildchat_loader import get_wildchat_prompts

        prompts = get_wildchat_prompts(
            scale=wildchat_scale,
            interleave=True,
            max_model_len=max_model_len,
            seed=seed,
        )
    elif workload_name in WORKLOADS:
        workload = WORKLOADS[workload_name]
        prompts = prepare_prompts_from_workload(workload)
    else:
        # Treat as path to custom JSON workload file
        workload = load_workload_from_file(workload_name)
        prompts = prepare_prompts_from_workload(workload)

    if not prompts:
        raise ValueError(
            f"No prompts generated from workload '{workload_name}'"
        )
    return prompts


def prepare_prompts_from_workload(workload: dict) -> list[str]:
    """Convert workload definition to list of full prompts."""
    prompts = []
    description = workload.get("description", "").lower()

    if "chat" in description and "system_prompt" in workload:
        system_prompt = workload["system_prompt"]
        for query in workload["user_queries"]:
            prompts.append(f"{system_prompt}\n\nUser: {query}\nAssistant:")

    elif "code" in description and "repo_context" in workload:
        repo_context = workload["repo_context"]
        for request in workload["completion_requests"]:
            prompts.append(f"{repo_context}\n\n{request}")

    elif "few" in description and "examples" in workload:
        examples = workload["examples"]
        for query in workload["queries"]:
            prompts.append(f"{examples}\n\n{query}\nA:")

    elif "mixed" in description:
        prompts = workload.get("queries", [])

    else:
        # Generic format
        shared_prefix = workload.get("shared_prefix", "")
        queries = workload.get("queries", [])
        prompts = [f"{shared_prefix}{q}" if shared_prefix else q for q in queries]

    return prompts
