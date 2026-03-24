# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lightweight validation tests for eviction policy workload definitions.

GPU-based benchmarks have been moved to run_eviction_benchmark.py,
a standalone CLI script that can be invoked directly:

    python tests/v1/kv_offload/run_eviction_benchmark.py --help
"""

import json

from tests.v1.kv_offload.workloads import (
    WORKLOADS,
    load_workload_from_file,
    prepare_prompts_from_workload,
)


def test_workload_definitions_are_valid():
    """Verify all predefined workloads are valid."""
    for name, workload in WORKLOADS.items():
        prompts = prepare_prompts_from_workload(workload)
        assert len(prompts) > 0, f"Workload {name} generated no prompts"
        assert all(isinstance(p, str) for p in prompts), (
            f"Workload {name} generated non-string prompts"
        )


def test_load_custom_workload_from_file(tmp_path):
    """
    Test loading custom workload from external file.

    This demonstrates how to extend tests with custom workloads.
    """
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
