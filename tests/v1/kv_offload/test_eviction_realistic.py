# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Realistic eviction policy testing with actual prompts and workloads.

This module tests eviction strategies using realistic scenarios:
- Common system prompts (chat applications)
- Code completion contexts
- Few-shot learning examples
- Mixed workload patterns

Tests can be extended to read from external files for custom workloads.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from vllm import LLM, SamplingParams
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.platforms import current_platform

# Test workload definitions
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
        "expected_sharing": "high",  # System prompt shared across all queries
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
        "expected_sharing": "medium",  # Repo context shared, completions unique
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
        "expected_sharing": "high",  # Examples shared across all queries
    },
    "mixed_workload": {
        "description": "Mixed queries with varying sharing patterns",
        "queries": [
            "What is machine learning?",
            "Explain neural networks.",
            "What is machine learning?",  # Duplicate - full sharing
            "How does backpropagation work?",
            "What is machine learning?",  # Duplicate again
            "Explain convolutional neural networks.",
            "What are transformers in AI?",
            "What is machine learning?",  # Popular query - high frequency
        ],
        "expected_sharing": "variable",  # Some queries repeat, others unique
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


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires CUDA for realistic testing"
)
@pytest.mark.parametrize("workload_name", ["chat_application", "few_shot_learning"])
@pytest.mark.parametrize("eviction_policy", ["lru", "arc", "lfu", "lru-2"])
def test_eviction_policy_on_workload(workload_name: str, eviction_policy: str):
    """
    Test eviction policy performance on realistic workloads.

    This test measures hit rate and eviction behavior for different
    eviction strategies on predefined workloads.
    """
    workload = WORKLOADS[workload_name]
    prompts = prepare_prompts_from_workload(workload)

    assert len(prompts) > 0, (
        f"No prompts generated from workload {workload_name}: "
        f"{workload.get('description')}"
    )

    # Configure LLM with eviction policy
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 1 << 30,  # 1GB CPU cache
            "block_size": 16,
            "eviction_policy": eviction_policy,
        },
    )

    llm = LLM(
        model="facebook/opt-125m",  # Small model for fast testing
        gpu_memory_utilization=0.3,
        kv_transfer_config=kv_transfer_config,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    # Run workload
    import time

    latencies = []
    for prompt in prompts:
        start = time.perf_counter()
        llm.generate(prompt, sampling_params, use_tqdm=False)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

    avg_latency = sum(latencies) / len(latencies)

    # Print results (informational only, no assertions)
    speedup = latencies[0] / latencies[-1] if latencies[-1] > 0 else 1.0

    print(f"\n{eviction_policy.upper()} on {workload_name}:")
    print(f"  Workload: {workload['description']}")
    print(f"  Expected Sharing: {workload['expected_sharing']}")
    print(f"  Avg Latency: {avg_latency:.2f}ms")
    print(f"  First Request: {latencies[0]:.2f}ms")
    print(f"  Last Request: {latencies[-1]:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Informational notes
    expected_sharing = workload["expected_sharing"]
    if expected_sharing == "high":
        print(f"  Note: High-sharing workload - expect good caching performance")


def test_compare_eviction_policies_on_chat_workload():
    """
    Compare all three eviction policies on a chat application workload.

    This test demonstrates the performance difference between strategies
    in a realistic high-sharing scenario.
    """
    if not current_platform.is_cuda():
        pytest.skip("Requires CUDA")

    workload = WORKLOADS["chat_application"]
    prompts = prepare_prompts_from_workload(workload)

    assert len(prompts) > 0, f"No prompts generated from workload: {workload.get('description')}"

    results = {}

    for policy in ["lru", "arc", "lfu", "lru-2"]:
        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": 1 << 30,
                "block_size": 16,
                "eviction_policy": policy,
            },
        )

        llm = LLM(
            model="facebook/opt-125m",
            gpu_memory_utilization=0.3,
            kv_transfer_config=kv_transfer_config,
            enable_prefix_caching=True,
        )

        sampling_params = SamplingParams(temperature=0, max_tokens=10)

        import time

        latencies = []
        for prompt in prompts:
            start = time.perf_counter()
            llm.generate(prompt, sampling_params, use_tqdm=False)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        assert len(latencies) > 0, f"No latencies recorded for policy {policy}"
        avg_latency = sum(latencies) / len(latencies)
        speedup = latencies[0] / latencies[-1]

        results[policy] = {"avg_latency": avg_latency, "speedup": speedup}

        del llm  # Clean up

    # Print comparison
    print("\n" + "=" * 60)
    print("EVICTION POLICY COMPARISON - Chat Application")
    print("=" * 60)
    for policy, metrics in results.items():
        print(
            f"{policy.upper():6s}: Avg={metrics['avg_latency']:.2f}ms, "
            f"Speedup={metrics['speedup']:.2f}x"
        )
    print("=" * 60)

    # Analyze results (informational, not assertion)
    best_speedup = max(results.items(), key=lambda x: x[1]['speedup'])
    best_latency = min(results.items(), key=lambda x: x[1]['avg_latency'])

    print(f"\nBest Speedup: {best_speedup[0].upper()} ({best_speedup[1]['speedup']:.2f}x)")
    print(f"Best Latency: {best_latency[0].upper()} ({best_latency[1]['avg_latency']:.2f}ms)")
    print("\nNote: Results may vary based on hardware, model, and workload characteristics.")


def test_load_custom_workload_from_file(tmp_path):
    """
    Test loading custom workload from external file.

    This demonstrates how to extend tests with custom workloads.
    """
    # Create a sample workload file
    custom_workload = {
        "description": "Custom RAG application",
        "shared_prefix": "Context: The quick brown fox jumps over the lazy dog. ",
        "queries": [
            "What animal jumps?",
            "What color is the fox?",
            "Is the dog active or lazy?",
        ],
        "expected_sharing": "high",
    }

    workload_file = tmp_path / "custom_workload.json"
    with open(workload_file, "w") as f:
        json.dump(custom_workload, f)

    # Load and verify
    loaded = load_workload_from_file(workload_file)
    assert loaded["description"] == "Custom RAG application"
    assert len(loaded["queries"]) == 3

    # Prepare prompts
    prompts = prepare_prompts_from_workload(loaded)
    assert len(prompts) == 3
    assert all(p.startswith("Context: The quick brown fox") for p in prompts)


def test_workload_definitions_are_valid():
    """Verify all predefined workloads are valid."""
    for name, workload in WORKLOADS.items():
        prompts = prepare_prompts_from_workload(workload)
        assert len(prompts) > 0, f"Workload {name} generated no prompts"
        assert all(isinstance(p, str) for p in prompts), (
            f"Workload {name} generated non-string prompts"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
