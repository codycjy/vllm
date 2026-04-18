#!/usr/bin/env python3
"""Analyze alpha sensitivity sweep.

Reads results from experiments/results/prompt_length_ablation/:
  <bucket>_adaptive_a<alpha>/metrics.summary.json   (sweep + full-ablation runs)
  <bucket>_baseline/metrics.summary.json            (LRU baseline, reference)

Produces:
  plots/alpha_sweep.png  — hit-rate and latency vs alpha, per bucket
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BUCKETS_DEFAULT = ["short", "medium"]
BUCKET_COLORS = {
    "short":  "#2196F3",
    "medium": "#4CAF50",
    "long":   "#FF9800",
    "xlarge": "#F44336",
}


def load_summary(results_dir: Path, subdir: str):
    p = results_dir / subdir / "metrics.summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def collect(results_dir: Path, bucket: str):
    """Return sorted list of (alpha, summary) for all adaptive_a<x> dirs."""
    points = {}
    for p in sorted(results_dir.glob(f"{bucket}_adaptive_a*")):
        try:
            alpha = float(p.name.rsplit("_a", 1)[1])
        except (ValueError, IndexError):
            continue
        s = load_summary(results_dir, p.name)
        if s:
            points[alpha] = s
    return sorted(points.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default="experiments/results/prompt_length_ablation")
    ap.add_argument("--output", default=None,
                    help="Plot output dir (default: <results-dir>/plots)")
    ap.add_argument("--buckets", nargs="+", default=BUCKETS_DEFAULT)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output) if args.output else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_hit, ax_lat) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scheme B: Alpha Sensitivity Sweep\n"
                 "score = α·reuse_count + (1-α)·recency_rank",
                 fontsize=12, fontweight="bold")

    print(f"\n{'bucket':<10}{'α':>6}{'hit_rate':>12}{'avg_lat_ms':>14}"
          f"{'evictions':>12}{'vs baseline':>14}")
    print("-" * 70)

    for bucket in args.buckets:
        points = collect(results_dir, bucket)
        if not points:
            print(f"{bucket}: no data")
            continue

        baseline = load_summary(results_dir, f"{bucket}_baseline")
        base_hit = baseline.get("prefix_cache_hit_rate", 0) if baseline else 0
        base_lat = baseline.get("avg_latency_ms", 0) if baseline else 0

        alphas = [a for a, _ in points]
        hits   = [s.get("prefix_cache_hit_rate", 0) for _, s in points]
        lats   = [s.get("avg_latency_ms", 0) for _, s in points]
        evicts = [int(s.get("vllm:kv_cache_evictions_total_delta", 0))
                  for _, s in points]

        for a, h, l, e in zip(alphas, hits, lats, evicts):
            gain = (h - base_hit) * 100 if baseline else 0
            print(f"{bucket:<10}{a:>6.2f}{h:>11.2%}{l:>14.1f}{e:>12d}"
                  f"{gain:>+13.1f}pp")

        color = BUCKET_COLORS.get(bucket, "#888")

        ax_hit.plot(alphas, hits, "o-", color=color, linewidth=2,
                    markersize=8, label=f"{bucket}")
        if baseline:
            ax_hit.axhline(base_hit, color=color, linestyle=":", linewidth=1,
                           alpha=0.6,
                           label=f"{bucket} baseline ({base_hit:.1%})")

        ax_lat.plot(alphas, lats, "s-", color=color, linewidth=2,
                    markersize=8, label=f"{bucket}")
        if baseline:
            ax_lat.axhline(base_lat, color=color, linestyle=":", linewidth=1,
                           alpha=0.6,
                           label=f"{bucket} baseline ({base_lat:.0f} ms)")

    ax_hit.set_xlabel("α (1.0 = pure reuse count, 0.0 = pure recency / LRU)")
    ax_hit.set_ylabel("Prefix cache hit rate")
    ax_hit.set_title("Hit rate vs α")
    ax_hit.grid(alpha=0.3)
    ax_hit.legend(fontsize=9)
    ax_hit.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax_lat.set_xlabel("α")
    ax_lat.set_ylabel("Avg E2E latency (ms)")
    ax_lat.set_title("Latency vs α")
    ax_lat.grid(alpha=0.3)
    ax_lat.legend(fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "alpha_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
