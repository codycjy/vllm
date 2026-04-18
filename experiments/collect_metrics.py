#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect per-request metrics from a running vLLM server.

Usage:
    # Start vLLM server (in another terminal / job):
    vllm serve Qwen/Qwen3-8B --eviction-policy adaptive \
        --enable-prefix-caching --port 8000

    # Run benchmark and collect metrics:
    python experiments/collect_metrics.py \
        --server http://localhost:8000 \
        --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-requests 100 \
        --output results/adaptive_sharegpt.jsonl

Each line in the output JSONL contains:
    {
        "request_id": "req-0",
        "prompt_len": 128,
        "output_len": 64,
        "e2e_latency_ms": 450.2,
        "ttft_ms": 120.3,          # time to first token
        "tpot_ms": 5.1,            # time per output token
        "prompt_tokens": 128,
        "cached_tokens": 96,       # from x-vllm-cached-tokens header
        "cache_hit_rate": 0.75,    # cached_tokens / prompt_tokens
        "timestamp": "2026-03-29T10:30:00",
    }

Prometheus metrics (eviction rate, cache utilization, hit rate) are
scraped at start and end of the run and included in a summary file.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


def scrape_prometheus(server: str) -> dict[str, float]:
    """Scrape vLLM Prometheus metrics endpoint."""
    url = f"{server}/metrics"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"Warning: could not scrape metrics: {e}")
        return {}

    metrics = {}
    for line in resp.text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0].split("{")[0]  # strip labels
            try:
                metrics[name] = float(parts[-1])
            except ValueError:
                pass
    return metrics


def load_sharegpt(path: str, num_requests: int) -> list[str]:
    """Load prompts from a ShareGPT-format JSON file."""
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for item in data:
        convs = item.get("conversations", [])
        if convs and convs[0].get("from") == "human":
            prompts.append(convs[0]["value"])
        if len(prompts) >= num_requests:
            break
    return prompts


def load_synthetic(num_requests: int, prompt_len: int) -> list[str]:
    """Generate synthetic prompts with shared prefix."""
    shared_prefix = "Summarize the following document:\n" + "word " * prompt_len
    prompts = []
    for i in range(num_requests):
        # Vary the tail slightly so each request is unique
        prompts.append(shared_prefix + f"\nVariation {i}: please summarize.")
    return prompts


def send_request(
    server: str,
    prompt: str,
    max_tokens: int = 32,
) -> dict:
    """Send a completion request and collect timing metrics."""
    url = f"{server}/v1/completions"
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    t0 = time.monotonic()
    # Use streaming to measure time to first token
    resp = requests.post(url, json=payload, timeout=120, stream=False)
    t_end = time.monotonic()

    resp.raise_for_status()
    result = resp.json()

    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    e2e_ms = (t_end - t0) * 1000

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "e2e_latency_ms": round(e2e_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-request metrics from vLLM server")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="vLLM server URL")
    parser.add_argument("--dataset", default=None,
                        help="Path to ShareGPT JSON file")
    parser.add_argument("--num-requests", type=int, default=50,
                        help="Number of requests to send")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max output tokens per request")
    parser.add_argument("--prompt-len", type=int, default=200,
                        help="Synthetic prompt length (if no dataset)")
    parser.add_argument("--output", default="results/metrics.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent in-flight requests. "
                             "Use >1 to test prefix_match scheduling "
                             "(scheduler needs competing requests to choose from)")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompts
    if args.dataset:
        prompts = load_sharegpt(args.dataset, args.num_requests)
        print(f"Loaded {len(prompts)} prompts from {args.dataset}")
    else:
        prompts = load_synthetic(args.num_requests, args.prompt_len)
        print(f"Generated {len(prompts)} synthetic prompts "
              f"(prefix length ~{args.prompt_len})")

    # Scrape metrics before run
    metrics_before = scrape_prometheus(args.server)

    # Send requests and collect per-request metrics
    concurrency = max(1, args.concurrency)
    if concurrency > 1:
        print(f"Sending with concurrency={concurrency} "
              f"(enables prefix_match scheduling differentiation)")

    results = []
    completed = [0]

    def _send(idx_prompt):
        idx, prompt = idx_prompt
        result = send_request(args.server, prompt, args.max_tokens)
        result["request_id"] = f"req-{idx}"
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_send, (i, p)): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed[0] += 1
                if completed[0] % 10 == 0:
                    print(f"  [{completed[0]}/{len(prompts)}] "
                          f"latency={result['e2e_latency_ms']:.0f}ms "
                          f"prompt={result['prompt_tokens']} "
                          f"gen={result['completion_tokens']}")
            except Exception as e:
                print(f"  [req-{idx}] FAILED: {e}")

    # Scrape metrics after run
    metrics_after = scrape_prometheus(args.server)

    # Write per-request JSONL
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} request records to {output_path}")

    # Write summary
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "num_requests": len(results),
        "total_time_ms": sum(r["e2e_latency_ms"] for r in results),
        "avg_latency_ms": (
            sum(r["e2e_latency_ms"] for r in results) / len(results)
            if results else 0
        ),
        "avg_prompt_tokens": (
            sum(r["prompt_tokens"] for r in results) / len(results)
            if results else 0
        ),
    }

    # Extract key Prometheus deltas
    for key in [
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:kv_cache_evictions_total",
    ]:
        before = metrics_before.get(key, 0)
        after = metrics_after.get(key, 0)
        summary[key + "_delta"] = after - before

    for key in [
        "vllm:kv_cache_usage_perc",
        "vllm:prefix_cache_utilization",
    ]:
        summary[key] = metrics_after.get(key, 0)

    # Compute hit rate from Prometheus
    queries_delta = summary.get(
        "vllm:prefix_cache_queries_total_delta", 0)
    hits_delta = summary.get("vllm:prefix_cache_hits_total_delta", 0)
    summary["prefix_cache_hit_rate"] = (
        hits_delta / queries_delta if queries_delta > 0 else 0
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {summary_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Requests: {summary['num_requests']}")
    print(f"Avg latency: {summary['avg_latency_ms']:.1f} ms")
    print(f"Cache hit rate: {summary['prefix_cache_hit_rate']:.1%}")
    print(f"Evictions: "
          f"{summary.get('vllm:kv_cache_evictions_total_delta', 'N/A')}")
    print(f"Cache utilization: "
          f"{summary.get('vllm:prefix_cache_utilization', 'N/A')}")


if __name__ == "__main__":
    main()
