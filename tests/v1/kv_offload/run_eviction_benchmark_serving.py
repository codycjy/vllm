# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Serving-mode benchmark for KV cache eviction policies.

Starts a vllm serve subprocess and sends concurrent async HTTP requests
to measure per-request TTFT, latency, and throughput under realistic load.

Usage examples:
    # Show help
    python tests/v1/kv_offload/run_eviction_benchmark_serving.py --help

    # Single policy
    python tests/v1/kv_offload/run_eviction_benchmark_serving.py \
        --workload chat_application --eviction-policy lru --concurrency 16

    # Compare all policies on WildChat
    python tests/v1/kv_offload/run_eviction_benchmark_serving.py \
        --workload wildchat --wildchat-scale small --compare

    # With rate limiting
    python tests/v1/kv_offload/run_eviction_benchmark_serving.py \
        --workload wildchat --request-rate 10 --compare

    # Save results to JSON
    python tests/v1/kv_offload/run_eviction_benchmark_serving.py \
        --compare --output serving_results.json
"""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time

# Ensure the repo root is on sys.path so `tests.*` imports work
# when the script is invoked directly (python tests/v1/kv_offload/...).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.v1.kv_offload.workloads import WORKLOADS, load_prompts
from vllm.utils.argparse_utils import FlexibleArgumentParser

ALL_POLICIES = ["lru", "arc", "lfu", "lru-2"]

SERVER_STARTUP_TIMEOUT = 300  # seconds
HEALTH_POLL_INTERVAL = 2  # seconds


def parse_human_bytes(s: str) -> int:
    """Parse human-readable byte strings like '1G', '512M', '256K'."""
    s = s.strip().upper()
    suffixes = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if s[-1] in suffixes:
        return int(float(s[:-1]) * suffixes[s[-1]])
    return int(s)


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_server(args, policy: str, port: int) -> subprocess.Popen:
    """Launch a vllm serve subprocess and wait until it is healthy."""
    cache_bytes = parse_human_bytes(args.kv_cache_size)
    kv_config = json.dumps({
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": cache_bytes,
            "block_size": args.block_size,
            "eviction_policy": policy,
        },
    })

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(port),
        "--host", args.host,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--kv-transfer-config", kv_config,
        "--enable-prefix-caching",
    ]

    print(f"  Starting server on {args.host}:{port} with {policy} policy...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    # Poll health endpoint until ready
    import urllib.request
    import urllib.error

    health_url = f"http://{args.host}:{port}/health"
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            # Process exited prematurely — read output for diagnostics
            output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(
                f"Server exited with code {proc.returncode} before becoming healthy.\n"
                f"Output:\n{output[-2000:]}"
            )
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"  Server healthy (took "
                          f"{SERVER_STARTUP_TIMEOUT - (deadline - time.monotonic()):.0f}s)")
                    return proc
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)

    stop_server(proc)
    raise TimeoutError(
        f"Server did not become healthy within {SERVER_STARTUP_TIMEOUT}s"
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server process group."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
    except (ProcessLookupError, OSError):
        pass
    print("  Server stopped.")


async def send_request(
    session,
    sem: asyncio.Semaphore,
    url: str,
    prompt: str,
    max_tokens: int,
    model: str,
) -> dict:
    """Send a single streaming completion request and measure timing."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    async with sem:
        t_start = time.perf_counter()
        ttft = None
        success = False
        error_msg = None

        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return {
                        "latency_ms": (time.perf_counter() - t_start) * 1000,
                        "ttft_ms": None,
                        "success": False,
                        "error": f"HTTP {resp.status}: {body[:200]}",
                    }

                async for chunk in resp.content:
                    if ttft is None:
                        ttft = (time.perf_counter() - t_start) * 1000

                latency = (time.perf_counter() - t_start) * 1000
                success = True
        except Exception as e:
            latency = (time.perf_counter() - t_start) * 1000
            error_msg = str(e)

    return {
        "latency_ms": latency,
        "ttft_ms": ttft,
        "success": success,
        "error": error_msg,
    }


async def run_benchmark(
    host: str,
    port: int,
    prompts: list[str],
    concurrency: int,
    request_rate: float,
    max_tokens: int,
    model: str,
) -> list[dict]:
    """Send all prompts concurrently and collect per-request metrics."""
    import aiohttp

    url = f"http://{host}:{port}/v1/completions"
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for prompt in prompts:
            if request_rate != float("inf") and tasks:
                await asyncio.sleep(1.0 / request_rate)
            tasks.append(
                asyncio.create_task(
                    send_request(session, sem, url, prompt, max_tokens, model)
                )
            )

        results = await asyncio.gather(*tasks)

    return list(results)


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100.0) * (len(sorted_v) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def run_single_policy(args, policy: str, prompts: list[str]) -> dict:
    """Orchestrate server lifecycle + benchmark for one policy."""
    port = args.port if args.port else _find_free_port()

    proc = start_server(args, policy, port)
    try:
        print(f"  Sending {len(prompts)} requests (concurrency={args.concurrency}, "
              f"rate={args.request_rate})...")

        t0 = time.perf_counter()
        request_results = asyncio.run(
            run_benchmark(
                host=args.host,
                port=port,
                prompts=prompts,
                concurrency=args.concurrency,
                request_rate=float(args.request_rate),
                max_tokens=args.max_tokens,
                model=args.model,
            )
        )
        total_time = time.perf_counter() - t0
    finally:
        stop_server(proc)

    # Aggregate metrics
    successes = [r for r in request_results if r["success"]]
    latencies = [r["latency_ms"] for r in successes]
    ttfts = [r["ttft_ms"] for r in successes if r["ttft_ms"] is not None]

    num_success = len(successes)
    success_rate = num_success / len(request_results) if request_results else 0.0

    result = {
        "policy": policy,
        "workload": args.workload,
        "model": args.model,
        "num_prompts": len(prompts),
        "concurrency": args.concurrency,
        "request_rate": args.request_rate,
        "total_time_s": round(total_time, 2),
        "throughput_prompts_per_sec": round(num_success / total_time, 2) if total_time > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p50_latency_ms": round(_percentile(latencies, 50), 1),
        "p99_latency_ms": round(_percentile(latencies, 99), 1),
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else 0,
        "p50_ttft_ms": round(_percentile(ttfts, 50), 1),
        "p99_ttft_ms": round(_percentile(ttfts, 99), 1),
        "success_rate": round(success_rate, 4),
        "num_failures": len(request_results) - num_success,
    }
    return result


def _print_result(result: dict) -> None:
    """Print a single policy result."""
    print(f"\n{result['policy'].upper()} on {result['workload']}:")
    print(f"  Model: {result['model']}")
    print(f"  Prompts: {result['num_prompts']}, "
          f"Concurrency: {result['concurrency']}")
    print(f"  Total time: {result['total_time_s']:.1f}s")
    print(f"  Throughput: {result['throughput_prompts_per_sec']:.1f} prompts/sec")
    print(f"  Avg latency: {result['avg_latency_ms']:.1f}ms, "
          f"P99: {result['p99_latency_ms']:.1f}ms")
    print(f"  Avg TTFT: {result['avg_ttft_ms']:.1f}ms, "
          f"P99: {result['p99_ttft_ms']:.1f}ms")
    print(f"  Success rate: {result['success_rate']:.1%} "
          f"({result['num_failures']} failures)")


def _print_comparison_table(results: list[dict]) -> None:
    """Print a comparison table across policies."""
    print("\n" + "=" * 85)
    print(f"SERVING BENCHMARK — {results[0]['workload']}")
    print(f"  Model: {results[0]['model']}, "
          f"Prompts: {results[0]['num_prompts']}, "
          f"Concurrency: {results[0]['concurrency']}")
    print("=" * 85)
    print(f"{'Policy':>8s}  {'Total(s)':>8s}  {'Throughput':>12s}  "
          f"{'Avg Lat(ms)':>11s}  {'P99 Lat(ms)':>11s}  "
          f"{'Avg TTFT':>10s}  {'Success':>8s}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['policy'].upper():>8s}  "
            f"{r['total_time_s']:>8.1f}  "
            f"{r['throughput_prompts_per_sec']:>10.1f} p/s  "
            f"{r['avg_latency_ms']:>11.1f}  "
            f"{r['p99_latency_ms']:>11.1f}  "
            f"{r['avg_ttft_ms']:>8.1f}ms  "
            f"{r['success_rate']:>7.1%}"
        )
    print("=" * 85)

    best = max(results, key=lambda r: r["throughput_prompts_per_sec"])
    fastest = min(results, key=lambda r: r["avg_ttft_ms"])
    print(f"\nBest Throughput: {best['policy'].upper()} "
          f"({best['throughput_prompts_per_sec']:.1f} prompts/sec)")
    print(f"Fastest TTFT:   {fastest['policy'].upper()} "
          f"({fastest['avg_ttft_ms']:.1f}ms avg)")


def main(args) -> None:
    prompts = load_prompts(
        workload_name=args.workload,
        wildchat_scale=args.wildchat_scale,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    policies = ALL_POLICIES if args.compare else args.eviction_policy

    print(f"Workload: {args.workload} ({len(prompts)} prompts)")
    print(f"Policies: {', '.join(policies)}")
    print(f"Concurrency: {args.concurrency}, Rate: {args.request_rate}")

    results = []
    for policy in policies:
        print(f"\n--- {policy.upper()} ---")
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
        description="Serving-mode benchmark for KV cache eviction policies."
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
        "--kv-cache-size",
        type=str,
        default="1G",
        help="CPU KV cache size (e.g. 512M, 1G, 4G).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV cache block size (default: 16).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Max generation tokens per request.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=str,
        default="inf",
        help="Requests per second ('inf' for all-at-once).",
    )
    parser.add_argument(
        "--wildchat-scale",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Scale for WildChat workload.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for WildChat sampling.",
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
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Server port (0 = auto-select free port).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host.",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()
