#!/usr/bin/env python3
"""Analyze cache-size sweep on xlarge bucket.

Reads:
  experiments/results/cache_size_sweep/xlarge_c<cache>_<config>/
  experiments/results/prompt_length_ablation/xlarge_baseline/       (cache=256 reference)
  experiments/results/prompt_length_ablation/xlarge_adaptive_a0.7/  (cache=256 reference)

Produces plot showing hit-rate gap (adaptive - baseline) as function of
cache pressure — tests whether xlarge's collapse is explained by pressure.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 20 unique prompts × blocks-per-prompt.
# Rough block counts derived from prepare_length_buckets.py word counts:
#   short  ~100 words → ~8 blocks   → 160 blocks total
#   medium ~320 words → ~26 blocks  → 520 blocks total
#   long   ~700 words → ~57 blocks  → 1140 blocks total
#   xlarge ~1300 words → ~106 blocks → 2120 blocks total
BLOCKS_NEEDED_BY_BUCKET = {
    "short":   160,
    "medium":  520,
    "long":   1140,
    "xlarge": 2120,
}
CONFIGS = ["baseline", "adaptive_a0.7"]
CONFIG_COLORS = {"baseline": "#777777", "adaptive_a0.7": "#2196F3"}


def load_summary(path: Path):
    p = path / "metrics.summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def gather(sweep_dir: Path, ablation_dir: Path, bucket: str):
    """Return list of (cache, config, summary) points for one bucket."""
    points = []

    # Existing cache=256 results from prompt_length_ablation
    for config in CONFIGS:
        s = load_summary(ablation_dir / f"{bucket}_{config}")
        if s:
            points.append((256, config, s))

    # New sweep results
    for d in sorted(sweep_dir.glob(f"{bucket}_c*")):
        try:
            rest = d.name.split("_c", 1)[1]          # "1024_adaptive_a0.7"
            cache_str, config = rest.split("_", 1)
            cache = int(cache_str)
        except (ValueError, IndexError):
            continue
        if config not in CONFIGS:
            continue
        s = load_summary(d)
        if s:
            points.append((cache, config, s))

    return sorted(points, key=lambda x: (x[0], x[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir",
                    default="experiments/results/cache_size_sweep")
    ap.add_argument("--ablation-dir",
                    default="experiments/results/prompt_length_ablation")
    ap.add_argument("--bucket", default="xlarge",
                    choices=list(BLOCKS_NEEDED_BY_BUCKET.keys()))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    bucket = args.bucket
    blocks_needed = BLOCKS_NEEDED_BY_BUCKET[bucket]
    sweep_dir = Path(args.sweep_dir)
    ablation_dir = Path(args.ablation_dir)
    out_dir = Path(args.output) if args.output else sweep_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    points = gather(sweep_dir, ablation_dir, bucket)
    if not points:
        print(f"No results found for bucket={bucket}")
        return

    # Table
    print(f"\n=== bucket={bucket} (needs {blocks_needed} blocks total) ===")
    print(f"{'cache':>7}{'pressure':>11}{'config':>16}{'hit_rate':>11}"
          f"{'avg_lat':>10}{'evict':>8}")
    print("-" * 65)
    for cache, config, s in points:
        pressure = blocks_needed / cache
        print(f"{cache:>7}{pressure:>10.2f}x{config:>16}"
              f"{s.get('prefix_cache_hit_rate',0):>10.2%}"
              f"{s.get('avg_latency_ms',0):>10.1f}"
              f"{int(s.get('vllm:kv_cache_evictions_total_delta',0)):>8d}")

    # Organise per-config curves
    by_config = {c: [] for c in CONFIGS}
    for cache, config, s in points:
        by_config[config].append((cache, s))

    # Figure: hit rate + gap vs cache
    fig, (ax_hit, ax_gap) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Cache-size sweep on {bucket} bucket\n"
                 f"({bucket} needs ~{blocks_needed} blocks; "
                 f"pressure = needed / cache)",
                 fontsize=12, fontweight="bold")

    all_caches = sorted({c for c, _, _ in points})

    # LEFT: hit rate vs cache for each config
    for config in CONFIGS:
        data = by_config[config]
        if not data:
            continue
        caches = [c for c, _ in data]
        hits = [s.get("prefix_cache_hit_rate", 0) for _, s in data]
        ax_hit.plot(caches, hits, "o-", color=CONFIG_COLORS[config],
                    linewidth=2, markersize=8, label=config)

    ax_hit.set_xscale("log", base=2)
    ax_hit.set_xticks(all_caches)
    ax_hit.set_xticklabels([str(c) for c in all_caches])
    ax_hit.set_xlabel("Cache size (GPU blocks)")
    ax_hit.set_ylabel("Prefix cache hit rate")
    ax_hit.set_title("Hit rate vs cache size")
    ax_hit.grid(alpha=0.3)
    ax_hit.legend(fontsize=9)
    ax_hit.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_hit.axvline(blocks_needed, color="red", linestyle=":", alpha=0.5,
                   label=f"1x pressure ({blocks_needed} blocks)")

    # RIGHT: gap (adaptive - baseline) vs pressure
    caches_common = sorted(
        {c for c, s in by_config["baseline"]}
        & {c for c, s in by_config["adaptive_a0.7"]}
    )
    base_map = dict(by_config["baseline"])
    adp_map  = dict(by_config["adaptive_a0.7"])

    pressures = [blocks_needed / c for c in caches_common]
    gaps = [(adp_map[c].get("prefix_cache_hit_rate", 0)
             - base_map[c].get("prefix_cache_hit_rate", 0)) * 100
            for c in caches_common]

    ax_gap.plot(pressures, gaps, "D-", color="#4CAF50",
                linewidth=2, markersize=9)
    ax_gap.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax_gap.set_xscale("log")
    ax_gap.set_xlabel("Cache pressure (blocks_needed / cache_size)")
    ax_gap.set_ylabel("Hit-rate gap (adaptive − baseline), pp")
    ax_gap.set_title("Strategy gap closes under extreme pressure")
    ax_gap.grid(alpha=0.3)
    for c, p, g in zip(caches_common, pressures, gaps):
        ax_gap.annotate(f"c={c}\n{p:.1f}x",
                        xy=(p, g), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", fontsize=8)

    plt.tight_layout()
    out = out_dir / f"cache_size_sweep_{bucket}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
