# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone benchmark script for KV cache eviction policies.

Measures latency, throughput, and speedup for different eviction strategies
on predefined or custom workloads, including WildChat conversations.

Usage examples:
    # Show help
    python tests/v1/kv_offload/run_eviction_benchmark.py --help

    # Single policy on a workload
    python tests/v1/kv_offload/run_eviction_benchmark.py \
        --workload chat_application --eviction-policy lru

    # Compare all policies
    python tests/v1/kv_offload/run_eviction_benchmark.py \
        --workload chat_application --compare

    # WildChat workload
    python tests/v1/kv_offload/run_eviction_benchmark.py \
        --workload wildchat --wildchat-scale small \
        --model facebook/opt-1.3b --max-model-len 2048

    # Save results to JSON
    python tests/v1/kv_offload/run_eviction_benchmark.py \
        --compare --output results.json
"""

import json
import os
import sys
import time

# Ensure the repo root is on sys.path so `tests.*` imports work
# when the script is invoked directly (python tests/v1/kv_offload/...).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser

from tests.v1.kv_offload.workloads import (
    WORKLOADS,
    load_workload_from_file,
    prepare_prompts_from_workload,
)

ALL_POLICIES = ["lru", "arc", "lfu", "lru-2"]


def parse_human_bytes(s: str) -> int:
    """Parse human-readable byte strings like '1G', '512M', '256K'."""
    s = s.strip().upper()
    suffixes = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if s[-1] in suffixes:
        return int(float(s[:-1]) * suffixes[s[-1]])
    return int(s)


def _load_prompts(args) -> list[str]:
    """Load prompts based on workload argument."""
    workload_name = args.workload

    if workload_name == "wildchat":
        from tests.v1.kv_offload.wildchat_loader import get_wildchat_prompts

        prompts = get_wildchat_prompts(
            scale=args.wildchat_scale,
            interleave=True,
            max_model_len=args.max_model_len,
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


def run_single_policy(args, policy: str, prompts: list[str]) -> dict:
    """Run benchmark for a single eviction policy. Returns metrics dict."""
    cpu_cache_bytes = parse_human_bytes(args.cpu_cache_bytes)

    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": cpu_cache_bytes,
            "block_size": 16,
            "eviction_policy": policy,
        },
    )

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_transfer_config=kv_transfer_config,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    batch_size = args.batch_size
    batch_latencies = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        start = time.perf_counter()
        llm.generate(batch, sampling_params, use_tqdm=False)
        latency = (time.perf_counter() - start) * 1000  # ms
        batch_latencies.append(latency)

    total_time_ms = sum(batch_latencies)
    avg_batch_latency_ms = total_time_ms / len(batch_latencies)
    throughput = len(prompts) / (total_time_ms / 1000)  # prompts/sec
    speedup = (
        batch_latencies[0] / batch_latencies[-1]
        if batch_latencies[-1] > 0
        else 1.0
    )

    del llm  # free GPU memory

    return {
        "policy": policy,
        "workload": args.workload,
        "model": args.model,
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "cpu_cache_bytes": cpu_cache_bytes,
        "total_time_ms": round(total_time_ms, 2),
        "avg_batch_latency_ms": round(avg_batch_latency_ms, 2),
        "throughput_prompts_per_sec": round(throughput, 2),
        "first_batch_ms": round(batch_latencies[0], 2),
        "last_batch_ms": round(batch_latencies[-1], 2),
        "speedup": round(speedup, 2),
    }


def _print_result(result: dict) -> None:
    """Print a single policy result."""
    print(f"\n{result['policy'].upper()} on {result['workload']}:")
    print(f"  Model: {result['model']}")
    print(f"  Prompts: {result['num_prompts']}, Batch size: {result['batch_size']}")
    print(f"  CPU cache: {result['cpu_cache_bytes'] / (1 << 30):.1f}GB")
    print(f"  Total time: {result['total_time_ms']:.0f}ms")
    print(f"  Avg batch latency: {result['avg_batch_latency_ms']:.2f}ms")
    print(f"  Throughput: {result['throughput_prompts_per_sec']:.1f} prompts/sec")
    print(f"  First batch: {result['first_batch_ms']:.2f}ms")
    print(f"  Last batch: {result['last_batch_ms']:.2f}ms")
    print(f"  Speedup: {result['speedup']:.2f}x")


def _print_comparison_table(results: list[dict]) -> None:
    """Print a comparison table across policies."""
    print("\n" + "=" * 70)
    print(f"EVICTION POLICY COMPARISON — {results[0]['workload']}")
    print(f"  Model: {results[0]['model']}")
    print(f"  Prompts: {results[0]['num_prompts']}, "
          f"Batch size: {results[0]['batch_size']}")
    print("=" * 70)
    print(f"{'Policy':>8s}  {'Total(ms)':>10s}  {'AvgBatch(ms)':>12s}  "
          f"{'Throughput':>12s}  {'Speedup':>8s}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['policy'].upper():>8s}  "
            f"{r['total_time_ms']:>10.0f}  "
            f"{r['avg_batch_latency_ms']:>12.2f}  "
            f"{r['throughput_prompts_per_sec']:>10.1f} p/s  "
            f"{r['speedup']:>7.2f}x"
        )
    print("=" * 70)

    best_tp = max(results, key=lambda r: r["throughput_prompts_per_sec"])
    best_sp = max(results, key=lambda r: r["speedup"])
    print(f"\nBest Throughput: {best_tp['policy'].upper()} "
          f"({best_tp['throughput_prompts_per_sec']:.1f} prompts/sec)")
    print(f"Best Speedup:   {best_sp['policy'].upper()} "
          f"({best_sp['speedup']:.2f}x)")


def main(args) -> None:
    prompts = _load_prompts(args)
    policies = ALL_POLICIES if args.compare else args.eviction_policy

    print(f"Workload: {args.workload} ({len(prompts)} prompts)")
    print(f"Policies: {', '.join(policies)}")

    results = []
    for policy in policies:
        result = run_single_policy(args, policy, prompts)
        results.append(result)
        _print_result(result)

    if len(results) > 1:
        _print_comparison_table(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark KV cache eviction policies on realistic workloads."
    )
    workload_names = list(WORKLOADS.keys()) + ["wildchat"]
    parser.add_argument(
        "--workload",
        type=str,
        default="chat_application",
        help=("Workload name (%s) or path to custom JSON file."
              % ", ".join(workload_names)),
    )
    parser.add_argument(
        "--eviction-policy",
        type=str,
        nargs="+",
        default=ALL_POLICIES,
        choices=ALL_POLICIES,
        help="Eviction policy/policies to benchmark.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name or path.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--cpu-cache-bytes",
        type=str,
        default="1G",
        help="CPU cache size (e.g. 512M, 1G, 4G).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for prompt submission.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Max generation tokens per request.",
    )
    parser.add_argument(
        "--wildchat-scale",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Scale for WildChat workload.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all policies and print comparison table.",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()
