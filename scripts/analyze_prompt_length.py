#!/usr/bin/env python3
"""Analyze and plot prompt-length ablation results.

Reads all summary.json files from experiments/results/prompt_length_ablation/
and produces a 4-panel comparison figure showing how each strategy's benefit
scales with prompt length.

Usage:
    python3 scripts/analyze_prompt_length.py \
        --results-dir experiments/results/prompt_length_ablation \
        --output experiments/results/prompt_length_ablation/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BUCKETS = ["short", "medium", "long", "xlarge"]
BUCKET_LABELS = ["Short\n(~64–256 tok)", "Medium\n(~256–512 tok)",
                 "Long\n(~512–1024 tok)", "XLarge\n(~1024–1800 tok)"]
CONFIGS = ["baseline", "eviction", "scheduling", "joint"]
CONFIG_LABELS = {
    "baseline":   "Baseline (LRU + FCFS)",
    "eviction":   "Eviction-only (Adaptive + FCFS)",
    "scheduling": "Scheduling-only (LRU + Prefix-match)",
    "joint":      "Joint (Adaptive + Prefix-match)",
}
COLORS = {
    "baseline":   "#555555",
    "eviction":   "#2196F3",
    "scheduling": "#FF9800",
    "joint":      "#4CAF50",
}
MARKERS = {"baseline": "o", "eviction": "s", "scheduling": "^", "joint": "D"}


def load_summary(results_dir: Path, bucket: str, config: str) -> dict | None:
    path = results_dir / f"{bucket}_{config}" / "metrics.summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_jsonl(results_dir: Path, bucket: str, config: str) -> list[dict]:
    path = results_dir / f"{bucket}_{config}" / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(len(arr) * p / 100)
    return arr[min(idx, len(arr) - 1)]


def gather_metrics(results_dir: Path) -> dict:
    data: dict[str, dict[str, dict]] = {c: {} for c in CONFIGS}

    for config in CONFIGS:
        for bucket in BUCKETS:
            summary = load_summary(results_dir, bucket, config)
            rows = load_jsonl(results_dir, bucket, config)
            if summary is None:
                continue

            latencies = [r["e2e_latency_ms"] for r in rows if "e2e_latency_ms" in r]
            data[config][bucket] = {
                "avg_latency_ms": summary.get("avg_latency_ms", 0),
                "p99_latency_ms": percentile(latencies, 99),
                "cache_hit_rate": summary.get("prefix_cache_hit_rate", 0),
                "evictions": summary.get("vllm:kv_cache_evictions_total_delta", 0),
                "cache_util": summary.get("vllm:prefix_cache_utilization", 0),
                "n": summary.get("num_requests", 0),
            }

    return data


def speedup_over_baseline(data: dict, metric: str, bucket: str,
                           lower_is_better: bool = True) -> float:
    base = data["baseline"].get(bucket, {}).get(metric)
    if not base or base == 0:
        return 1.0
    val = data.get("joint", {}).get(bucket, {}).get(metric)
    if val is None:
        return 1.0
    return (base / val) if lower_is_better else (val / base)


def plot(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(BUCKETS))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Prompt Length Ablation: Strategy Comparison\n"
                 "(opt-1.3b, V100, cache=256 blocks=4096 tokens)",
                 fontsize=13, fontweight="bold")

    panel_defs = [
        (axes[0, 0], "cache_hit_rate", "Prefix Cache Hit Rate", False, "Rate"),
        (axes[0, 1], "avg_latency_ms", "Avg E2E Latency (ms)", True, "ms"),
        (axes[1, 0], "evictions", "Eviction Count (Δ)", True, "blocks evicted"),
        (axes[1, 1], "p99_latency_ms", "P99 Latency (ms)", True, "ms"),
    ]

    for ax, metric, title, lower_better, ylabel in panel_defs:
        for config in CONFIGS:
            values = [
                data[config].get(b, {}).get(metric, None)
                for b in BUCKETS
            ]
            mask = [v is not None for v in values]
            xs = x[mask]
            ys = [v for v, m in zip(values, mask) if m]
            if not ys:
                continue
            ax.plot(xs, ys,
                    label=CONFIG_LABELS[config],
                    color=COLORS[config],
                    marker=MARKERS[config],
                    linewidth=2, markersize=7)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_LABELS, fontsize=8)
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(axis="y", alpha=0.3)
        if metric == "cache_hit_rate":
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

    plt.tight_layout()
    out_path = out_dir / "prompt_length_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # Second figure: speedup of Joint vs Baseline across buckets
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    speedups_latency = [
        speedup_over_baseline(data, "avg_latency_ms", b, lower_is_better=True)
        for b in BUCKETS
    ]
    speedups_hitrate = [
        speedup_over_baseline(data, "cache_hit_rate", b, lower_is_better=False)
        for b in BUCKETS
    ]
    ax2.plot(x, speedups_latency, "o-", color="#4CAF50", linewidth=2,
             markersize=8, label="Latency speedup (Joint vs Baseline)")
    ax2.plot(x, speedups_hitrate, "s--", color="#2196F3", linewidth=2,
             markersize=8, label="Cache hit rate ratio (Joint / Baseline)")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(BUCKET_LABELS, fontsize=9)
    ax2.set_ylabel("Ratio (>1 = improvement)", fontsize=10)
    ax2.set_title("Joint Strategy Benefit vs Prompt Length\n"
                  "(Speedup relative to LRU+FCFS baseline)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    out_path2 = out_dir / "speedup_vs_prompt_length.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")
    plt.close()


def print_table(data: dict) -> None:
    print("\n" + "=" * 80)
    print(f"{'':20s}" + "".join(f"{b:>15s}" for b in BUCKETS))
    print("=" * 80)
    for config in CONFIGS:
        print(f"\n[{config}]")
        for metric in ["cache_hit_rate", "avg_latency_ms", "evictions"]:
            values = [data[config].get(b, {}).get(metric, "N/A") for b in BUCKETS]
            fmt_vals = []
            for v in values:
                if isinstance(v, float):
                    fmt_vals.append(f"{v:>14.3f}")
                elif isinstance(v, int):
                    fmt_vals.append(f"{v:>14d}")
                else:
                    fmt_vals.append(f"{'N/A':>14s}")
            print(f"  {metric:20s}" + "".join(fmt_vals))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prompt-length ablation results")
    parser.add_argument(
        "--results-dir", type=str,
        default="experiments/results/prompt_length_ablation")
    parser.add_argument(
        "--output", type=str,
        default="experiments/results/prompt_length_ablation/plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output)

    data = gather_metrics(results_dir)

    available = sum(
        1 for c in CONFIGS for b in BUCKETS
        if b in data.get(c, {})
    )
    print(f"Loaded {available}/{len(CONFIGS) * len(BUCKETS)} result files "
          f"from {results_dir}")

    if available == 0:
        print("No results found. Run run_prompt_length_ablation.sh first.")
        return

    print_table(data)
    plot(data, out_dir)
    print(f"\nPlots saved to: {out_dir}")


if __name__ == "__main__":
    main()
