#!/usr/bin/env python3
"""Smoke test: verify adaptive eviction strategy works end-to-end.

Sends requests with shared prefixes to create cache pressure, then
checks that the eviction policy is actually affecting behavior.

Usage:
    python experiments/smoke_test.py \
        --server http://localhost:8000 \
        --tag lru          # or "adaptive"
        --log-dir experiments/results

Output structure:
    experiments/results/
      2026-03-29_smoke_lru/
        requests.jsonl       # per-request latency + token counts
        prometheus_before.json
        prometheus_after.json
        summary.json         # aggregated metrics + pass/fail checks
        test.log             # human-readable log
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests as http_requests


def setup_logger(log_dir: Path) -> logging.Logger:
    log = logging.getLogger("smoke_test")
    log.setLevel(logging.DEBUG)
    # File handler
    fh = logging.FileHandler(log_dir / "test.log")
    fh.setLevel(logging.DEBUG)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def scrape_prometheus(server: str) -> dict[str, float]:
    try:
        resp = http_requests.get(f"{server}/metrics", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return {"_error": str(e)}
    metrics = {}
    for line in resp.text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0].split("{")[0]
            try:
                metrics[name] = float(parts[-1])
            except ValueError:
                pass
    return metrics


def send_completion(server: str, prompt: str, max_tokens: int = 16,
                    model: str = "qwen3-8b") -> dict:
    t0 = time.monotonic()
    resp = http_requests.post(
        f"{server}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    result = resp.json()
    usage = result.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "e2e_latency_ms": round(latency_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke test for eviction")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--tag", required=True,
                        help="Run tag, e.g. 'lru' or 'adaptive'")
    parser.add_argument("--log-dir", default="experiments/results")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--num-unique-prefixes", type=int, default=5,
                        help="Number of unique shared prefixes")
    parser.add_argument("--repeats-per-prefix", type=int, default=4,
                        help="How many times to reuse each prefix")
    parser.add_argument("--prefix-len", type=int, default=300,
                        help="Approximate prefix length in tokens (words)")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    # Create log directory
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(args.log_dir) / f"{date_str}_smoke_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(run_dir)
    log.info(f"Smoke test: tag={args.tag}, server={args.server}")
    log.info(f"Log directory: {run_dir}")

    # ── Step 1: Health check ──
    log.info("Step 1: Health check...")
    try:
        resp = http_requests.get(f"{args.server}/health", timeout=5)
        resp.raise_for_status()
        log.info("  Server is healthy")
    except Exception as e:
        log.error(f"  Server health check failed: {e}")
        sys.exit(1)

    # ── Step 2: Scrape prometheus before ──
    log.info("Step 2: Scraping Prometheus metrics (before)...")
    prom_before = scrape_prometheus(args.server)
    with open(run_dir / "prometheus_before.json", "w") as f:
        json.dump(prom_before, f, indent=2)
    log.info(f"  Scraped {len(prom_before)} metrics")

    # ── Step 3: Generate test prompts ──
    # Create N unique prefixes, each reused M times with different suffixes
    log.info(f"Step 3: Generating prompts ({args.num_unique_prefixes} prefixes"
             f" × {args.repeats_per_prefix} repeats = "
             f"{args.num_unique_prefixes * args.repeats_per_prefix} requests)")

    prefixes = []
    for i in range(args.num_unique_prefixes):
        # Each prefix is unique but long enough to span multiple KV blocks
        prefix = (f"You are assistant #{i}. "
                  f"Please analyze the following data carefully. "
                  + f"Data point {i}: " + f"value_{i} " * args.prefix_len
                  + "\n\n")
        prefixes.append(prefix)

    # Interleave: send prefix 0,1,2,...,N, then repeat
    # This maximizes cache reuse opportunity
    prompts = []
    for repeat in range(args.repeats_per_prefix):
        for i, prefix in enumerate(prefixes):
            suffix = f"Question (round {repeat}): Summarize the data above."
            prompts.append({
                "prompt": prefix + suffix,
                "prefix_id": i,
                "repeat": repeat,
            })

    # ── Step 4: Send requests ──
    log.info("Step 4: Sending requests...")
    results = []
    jsonl_path = run_dir / "requests.jsonl"
    with open(jsonl_path, "w") as f:
        for idx, item in enumerate(prompts):
            try:
                result = send_completion(
                    args.server, item["prompt"], args.max_tokens, args.model)
                result["request_id"] = idx
                result["prefix_id"] = item["prefix_id"]
                result["repeat"] = item["repeat"]
                result["timestamp"] = datetime.now(
                    timezone.utc).isoformat()
                results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()

                marker = "★" if item["repeat"] > 0 else " "
                log.info(
                    f"  [{idx+1:3d}/{len(prompts)}] {marker} "
                    f"prefix={item['prefix_id']} "
                    f"repeat={item['repeat']} "
                    f"latency={result['e2e_latency_ms']:7.0f}ms "
                    f"prompt_tok={result['prompt_tokens']}")
            except Exception as e:
                log.error(f"  [{idx+1}/{len(prompts)}] FAILED: {e}")

    # ── Step 5: Scrape prometheus after ──
    log.info("Step 5: Scraping Prometheus metrics (after)...")
    prom_after = scrape_prometheus(args.server)
    with open(run_dir / "prometheus_after.json", "w") as f:
        json.dump(prom_after, f, indent=2)

    # ── Step 6: Analyze ──
    log.info("Step 6: Analysis...")

    # Latency comparison: first request vs repeat requests per prefix
    first_latencies = [r["e2e_latency_ms"] for r in results
                       if r["repeat"] == 0]
    repeat_latencies = [r["e2e_latency_ms"] for r in results
                        if r["repeat"] > 0]

    avg_first = sum(first_latencies) / len(first_latencies) if first_latencies else 0
    avg_repeat = sum(repeat_latencies) / len(repeat_latencies) if repeat_latencies else 0

    # Prometheus deltas
    def prom_delta(key):
        return prom_after.get(key, 0) - prom_before.get(key, 0)

    cache_queries = prom_delta("vllm:prefix_cache_queries_total")
    cache_hits = prom_delta("vllm:prefix_cache_hits_total")
    evictions = prom_delta("vllm:kv_cache_evictions_total")
    hit_rate = cache_hits / cache_queries if cache_queries > 0 else 0

    cache_util = prom_after.get("vllm:prefix_cache_utilization", 0)
    kv_usage = prom_after.get("vllm:kv_cache_usage_perc", 0)

    summary = {
        "tag": args.tag,
        "num_requests": len(results),
        "num_unique_prefixes": args.num_unique_prefixes,
        "repeats_per_prefix": args.repeats_per_prefix,
        "avg_first_request_latency_ms": round(avg_first, 1),
        "avg_repeat_request_latency_ms": round(avg_repeat, 1),
        "speedup_ratio": round(avg_first / avg_repeat, 2) if avg_repeat > 0 else 0,
        "prefix_cache_queries": cache_queries,
        "prefix_cache_hits": cache_hits,
        "prefix_cache_hit_rate": round(hit_rate, 4),
        "evictions": evictions,
        "prefix_cache_utilization": round(cache_util, 4),
        "kv_cache_usage": round(kv_usage, 4),
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    log.info("")
    log.info("=" * 55)
    log.info(f"  Tag:                    {args.tag}")
    log.info(f"  Requests:               {len(results)}")
    log.info(f"  Avg first latency:      {avg_first:.0f} ms")
    log.info(f"  Avg repeat latency:     {avg_repeat:.0f} ms")
    log.info(f"  Speedup (first/repeat): {summary['speedup_ratio']:.2f}x")
    log.info(f"  Cache hit rate:         {hit_rate:.1%}")
    log.info(f"  Evictions:              {evictions}")
    log.info(f"  Cache utilization:      {cache_util:.1%}")
    log.info(f"  KV cache usage:         {kv_usage:.1%}")
    log.info("=" * 55)

    # Sanity checks
    checks = []
    if len(results) == len(prompts):
        checks.append(("All requests completed", True))
    else:
        checks.append(("All requests completed", False))

    if cache_queries > 0:
        checks.append(("Cache queries > 0", True))
    else:
        checks.append(("Cache queries > 0", False))

    if hit_rate > 0:
        checks.append(("Cache hit rate > 0", True))
    else:
        checks.append(("Cache hit rate > 0", False))

    if avg_repeat < avg_first or len(first_latencies) == 0:
        checks.append(("Repeat requests faster than first", True))
    else:
        checks.append(("Repeat requests faster than first", False))

    log.info("")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        log.info(f"  [{status}] {name}")

    summary["checks"] = {name: passed for name, passed in checks}
    summary["all_checks_passed"] = all_pass
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("")
    if all_pass:
        log.info("All checks passed!")
    else:
        log.warning("Some checks FAILED — see above")

    log.info(f"\nResults saved to: {run_dir}/")


if __name__ == "__main__":
    main()
