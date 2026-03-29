#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot comparison between LRU and adaptive eviction results.

Usage:
    python experiments/plot_comparison.py \
        --lru results/lru_sharegpt.summary.json \
        --adaptive results/adaptive_sharegpt.summary.json \
        --output results/comparison.png

    # Or compare per-request latency distributions:
    python experiments/plot_comparison.py \
        --lru-jsonl results/lru_sharegpt.jsonl \
        --adaptive-jsonl results/adaptive_sharegpt.jsonl \
        --output results/latency_dist.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: str) -> list[dict]:
    """Load per-request JSONL data."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_summary(path: str) -> dict:
    """Load summary JSON."""
    with open(path) as f:
        return json.load(f)


def plot_summary_comparison(lru_summary: dict, adaptive_summary: dict,
                            output: str):
    """Bar chart comparing key metrics between LRU and adaptive."""
    metrics = {
        "Avg Latency (ms)": ("avg_latency_ms", False),
        "Cache Hit Rate (%)": ("prefix_cache_hit_rate", True),
        "Evictions": ("vllm:kv_cache_evictions_total_delta", False),
        "Cache Util (%)": ("vllm:prefix_cache_utilization", True),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    colors = ["#2196F3", "#FF9800"]  # blue=LRU, orange=adaptive

    for ax, (label, (key, is_pct)) in zip(axes, metrics.items()):
        lru_val = lru_summary.get(key, 0)
        adp_val = adaptive_summary.get(key, 0)
        if is_pct:
            lru_val *= 100
            adp_val *= 100

        bars = ax.bar(["LRU", "Adaptive"], [lru_val, adp_val],
                      color=colors, width=0.5)
        ax.set_title(label, fontsize=11)
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_ylim(0, max(lru_val, adp_val, 1) * 1.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved summary comparison to {output}")


def plot_latency_distribution(lru_records: list[dict],
                              adaptive_records: list[dict],
                              output: str):
    """CDF and histogram of per-request latency."""
    lru_lat = [r["e2e_latency_ms"] for r in lru_records]
    adp_lat = [r["e2e_latency_ms"] for r in adaptive_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram
    bins = np.linspace(0, max(max(lru_lat), max(adp_lat)) * 1.1, 40)
    ax1.hist(lru_lat, bins=bins, alpha=0.6, label="LRU", color="#2196F3")
    ax1.hist(adp_lat, bins=bins, alpha=0.6, label="Adaptive", color="#FF9800")
    ax1.set_xlabel("E2E Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Latency Distribution")
    ax1.legend()

    # CDF
    for data, label, color in [
        (lru_lat, "LRU", "#2196F3"),
        (adp_lat, "Adaptive", "#FF9800"),
    ]:
        sorted_d = np.sort(data)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.plot(sorted_d, cdf, label=label, color=color, linewidth=2)

    ax2.set_xlabel("E2E Latency (ms)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Latency CDF")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add p50/p99 annotations
    for data, label in [(lru_lat, "LRU"), (adp_lat, "Adaptive")]:
        p50 = np.percentile(data, 50)
        p99 = np.percentile(data, 99)
        ax2.axvline(x=p99, linestyle="--", alpha=0.4)
        print(f"  {label}: p50={p50:.0f}ms  p99={p99:.0f}ms  "
              f"mean={np.mean(data):.0f}ms")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved latency comparison to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot LRU vs adaptive eviction comparison")
    parser.add_argument("--lru", help="LRU summary JSON path")
    parser.add_argument("--adaptive", help="Adaptive summary JSON path")
    parser.add_argument("--lru-jsonl", help="LRU per-request JSONL path")
    parser.add_argument("--adaptive-jsonl",
                        help="Adaptive per-request JSONL path")
    parser.add_argument("--output", default="results/comparison.png",
                        help="Output image path")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.lru and args.adaptive:
        lru_s = load_summary(args.lru)
        adp_s = load_summary(args.adaptive)
        plot_summary_comparison(lru_s, adp_s, args.output)

    if args.lru_jsonl and args.adaptive_jsonl:
        lru_r = load_jsonl(args.lru_jsonl)
        adp_r = load_jsonl(args.adaptive_jsonl)
        out = args.output.replace(".png", "_latency.png")
        plot_latency_distribution(lru_r, adp_r, out)


if __name__ == "__main__":
    main()
