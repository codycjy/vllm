#!/usr/bin/env python3
"""Analyze full ablation results (Scheme A vs Scheme B).

Reads experiments/results/full_ablation/ and produces comparison plots.

Usage:
    python3 scripts/analyze_full_ablation.py
    python3 scripts/analyze_full_ablation.py --results-dir experiments/results/full_ablation
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

# Logical config name → on-disk directory suffix.
# Convention: {policy}_a{alpha}. α=1.0 is pure reuse count (Scheme A in the
# plan); α<1 blends recency (Scheme B). They're variants of one adaptive
# algorithm, not separate policies.
CONFIGS = ["baseline", "scheduling", "scheme_a", "scheme_b", "joint_a", "joint_b"]
DIR_SUFFIX = {
    "baseline":   "baseline",
    "scheduling": "sched_only",
    "scheme_a":   "adaptive_a1.0",
    "scheme_b":   "adaptive_a0.7",
    "joint_a":    "joint_a1.0",
    "joint_b":    "joint_a0.7",
}
CONFIG_LABELS = {
    "baseline":   "Baseline (LRU+FCFS)",
    "scheduling": "Scheduling only (LRU+PFX)",
    "scheme_a":   "Scheme A (adaptive α=1.0, FCFS)",
    "scheme_b":   "Scheme B (adaptive α<1, FCFS)",
    "joint_a":    "Joint A (adaptive α=1.0 + PFX)",
    "joint_b":    "Joint B (adaptive α<1 + PFX)",
}
COLORS = {
    "baseline":   "#777777",
    "scheduling": "#FF9800",
    "scheme_a":   "#2196F3",
    "scheme_b":   "#03A9F4",
    "joint_a":    "#4CAF50",
    "joint_b":    "#8BC34A",
}
MARKERS = {
    "baseline": "o", "scheduling": "^",
    "scheme_a": "s", "scheme_b": "D",
    "joint_a":  "P", "joint_b":  "*",
}
LINESTYLES = {
    "baseline": "-", "scheduling": "--",
    "scheme_a": "-", "scheme_b": "--",
    "joint_a":  "-", "joint_b":  "--",
}


def load_summary(results_dir: Path, bucket: str, config: str):
    suffix = DIR_SUFFIX.get(config, config)
    p = results_dir / f"{bucket}_{suffix}" / "metrics.summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_jsonl(results_dir: Path, bucket: str, config: str):
    suffix = DIR_SUFFIX.get(config, config)
    p = results_dir / f"{bucket}_{suffix}" / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def percentile(values, p):
    if not values:
        return 0.0
    arr = sorted(values)
    return arr[min(int(len(arr) * p / 100), len(arr) - 1)]


def gather(results_dir: Path):
    data = {}
    for config in CONFIGS:
        data[config] = {}
        for bucket in BUCKETS:
            summary = load_summary(results_dir, bucket, config)
            if summary is None:
                continue
            rows = load_jsonl(results_dir, bucket, config)
            lats = [r["e2e_latency_ms"] for r in rows if "e2e_latency_ms" in r]
            data[config][bucket] = {
                "avg_latency_ms":   summary.get("avg_latency_ms", 0),
                "p99_latency_ms":   percentile(lats, 99),
                "cache_hit_rate":   summary.get("prefix_cache_hit_rate", 0),
                "evictions":        summary.get("vllm:kv_cache_evictions_total_delta", 0),
            }
    return data


def print_table(data):
    print("\n" + "=" * 90)
    print(f"{'':22s}" + "".join(f"{b:>16s}" for b in BUCKETS))
    print("=" * 90)
    for config in CONFIGS:
        if not data[config]:
            continue
        print(f"\n[{config}]")
        for metric in ["cache_hit_rate", "avg_latency_ms", "p99_latency_ms", "evictions"]:
            vals = [data[config].get(b, {}).get(metric) for b in BUCKETS]
            row = []
            for v in vals:
                if v is None:
                    row.append(f"{'N/A':>15s}")
                elif isinstance(v, float):
                    row.append(f"{v:>15.3f}")
                else:
                    row.append(f"{v:>15d}")
            print(f"  {metric:22s}" + "".join(row))
    print("=" * 90)


def plot(data: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(BUCKETS))

    # ── Panel figure: 4 metrics ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Full Ablation: LRU vs Scheme A vs Scheme B\n"
                 "(Qwen3-8B, V100, cache=256 blocks=4096 tokens)",
                 fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "cache_hit_rate",  "Prefix Cache Hit Rate",  "Rate"),
        (axes[0, 1], "avg_latency_ms",  "Avg E2E Latency (ms)",   "ms"),
        (axes[1, 0], "evictions",       "Eviction Count (Δ)",     "blocks evicted"),
        (axes[1, 1], "p99_latency_ms",  "P99 Latency (ms)",       "ms"),
    ]
    for ax, metric, title, ylabel in panels:
        for config in CONFIGS:
            vals = [data[config].get(b, {}).get(metric) for b in BUCKETS]
            mask = [v is not None for v in vals]
            if not any(mask):
                continue
            xs = x[mask]
            ys = [v for v, m in zip(vals, mask) if m]
            ax.plot(xs, ys,
                    label=CONFIG_LABELS[config],
                    color=COLORS[config],
                    marker=MARKERS[config],
                    linestyle=LINESTYLES[config],
                    linewidth=2, markersize=7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_LABELS, fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", alpha=0.3)
        if metric == "cache_hit_rate":
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

    plt.tight_layout()
    p = out_dir / "full_ablation_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

    # ── Scheme A vs B: latency & hit-rate delta vs baseline ──────────────────
    fig2, (ax_lat, ax_hit) = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Scheme A vs B: improvement over LRU+FCFS baseline",
                  fontsize=12, fontweight="bold")

    compare_pairs = [
        ("scheme_a", "#2196F3", "Scheme A (α=1.0, FCFS)"),
        ("scheme_b", "#03A9F4", "Scheme B (α<1,   FCFS)"),
        ("joint_a",  "#4CAF50", "Joint A  (α=1.0, PFX)"),
        ("joint_b",  "#8BC34A", "Joint B  (α<1,   PFX)"),
    ]

    for config, color, label in compare_pairs:
        lat_ratios, hit_deltas = [], []
        for bucket in BUCKETS:
            base = data["baseline"].get(bucket, {})
            cfg  = data[config].get(bucket, {})
            base_lat = base.get("avg_latency_ms")
            cfg_lat  = cfg.get("avg_latency_ms")
            base_hit = base.get("cache_hit_rate")
            cfg_hit  = cfg.get("cache_hit_rate")
            lat_ratios.append(base_lat / cfg_lat if base_lat and cfg_lat else None)
            hit_deltas.append(cfg_hit - base_hit if cfg_hit is not None and base_hit is not None else None)

        mask = [v is not None for v in lat_ratios]
        xs = x[mask]

        ax_lat.plot(xs, [v for v, m in zip(lat_ratios, mask) if m],
                    "o-", color=color, linewidth=2, markersize=7, label=label)
        ax_hit.plot(xs, [v for v, m in zip(hit_deltas, mask) if m],
                    "s--", color=color, linewidth=2, markersize=7, label=label)

    for ax in (ax_lat, ax_hit):
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_LABELS, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax_lat.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax_lat.set_ylabel("Latency speedup vs baseline (>1 = better)", fontsize=9)
    ax_lat.set_title("Latency speedup", fontsize=11)

    ax_hit.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax_hit.set_ylabel("Cache hit rate delta vs baseline", fontsize=9)
    ax_hit.set_title("Cache hit rate gain", fontsize=11)
    ax_hit.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1%}"))

    plt.tight_layout()
    p2 = out_dir / "scheme_ab_vs_baseline.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Saved: {p2}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=
        "experiments/results/prompt_length_ablation")
    parser.add_argument("--output", default=None,
        help="Plot output dir (default: <results-dir>/plots)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output) if args.output else results_dir / "plots"

    data = gather(results_dir)
    available = sum(1 for c in CONFIGS for b in BUCKETS if b in data.get(c, {}))
    print(f"Loaded {available}/{len(CONFIGS) * len(BUCKETS)} result files from {results_dir}")

    if available == 0:
        print("No results found. Run scripts/run_prompt_length_ablation.sh first.")
        return

    print_table(data)
    plot(data, out_dir)
    print(f"\nPlots saved to: {out_dir}")


if __name__ == "__main__":
    main()
