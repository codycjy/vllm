#!/usr/bin/env python3
"""Generate the length-ablation headline plot.

Shows hit-rate gain vs baseline as a function of prompt length bucket,
decomposed into eviction-only, scheduling-only, and joint contributions,
plus end-to-end latency speedup vs baseline.

Answers: "How does the proposed system's performance improvement scale
with prompt length?"
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path("experiments/results/prompt_length_ablation")
BUCKETS = ["short", "medium", "long", "xlarge"]
BUCKET_LABELS = ["Short\n(~137 tok)", "Medium\n(~423 tok)",
                 "Long\n(~917 tok)", "XLarge\n(~1697 tok)"]


def load(path):
    p = RESULTS / path / "metrics.summary.json"
    return json.loads(p.read_text())


def hit(b, cfg):
    return load(f"{b}_{cfg}").get("prefix_cache_hit_rate", 0) * 100


def lat(b, cfg):
    return load(f"{b}_{cfg}").get("avg_latency_ms", 0)


def main():
    out_dir = RESULTS / "plots"

    # Data: improvements over baseline
    base_hit = [hit(b, "baseline") for b in BUCKETS]
    sched_gain   = [hit(b, "sched_only")     - h for b, h in zip(BUCKETS, base_hit)]
    evict_a_gain = [hit(b, "adaptive_a1.0")  - h for b, h in zip(BUCKETS, base_hit)]
    evict_b_gain = [hit(b, "adaptive_a0.7")  - h for b, h in zip(BUCKETS, base_hit)]
    joint_a_gain = [hit(b, "joint_a1.0")     - h for b, h in zip(BUCKETS, base_hit)]
    joint_b_gain = [hit(b, "joint_a0.7")     - h for b, h in zip(BUCKETS, base_hit)]

    # Latency speedup
    base_lat = [lat(b, "baseline") for b in BUCKETS]
    sched_sp   = [base_lat[i] / lat(b, "sched_only") for i, b in enumerate(BUCKETS)]
    evict_sp   = [base_lat[i] / lat(b, "adaptive_a0.7") for i, b in enumerate(BUCKETS)]
    joint_sp   = [base_lat[i] / lat(b, "joint_a0.7") for i, b in enumerate(BUCKETS)]

    # Figure
    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Prompt Length Ablation — performance improvement vs length\n"
                 "(Qwen3-8B, V100, skewed popularity workload, cache=256 blocks)",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(BUCKETS))
    width = 0.22

    # LEFT: hit rate gain
    ax_h.bar(x - 1.5*width, sched_gain,   width, label="Scheduling only",
             color="#FF9800")
    ax_h.bar(x - 0.5*width, evict_b_gain, width, label="Eviction only (α=0.7)",
             color="#2196F3")
    ax_h.bar(x + 0.5*width, joint_a_gain, width, label="Joint (α=1.0)",
             color="#4CAF50")
    ax_h.bar(x + 1.5*width, joint_b_gain, width, label="Joint (α=0.7)",
             color="#8BC34A")
    ax_h.axhline(0, color="gray", lw=0.8)
    ax_h.set_xticks(x)
    ax_h.set_xticklabels(BUCKET_LABELS, fontsize=9)
    ax_h.set_ylabel("Hit-rate gain vs baseline (pp)", fontsize=10)
    ax_h.set_title("Hit-rate improvement decomposition", fontsize=11)
    ax_h.legend(fontsize=8, loc="upper left")
    ax_h.grid(axis="y", alpha=0.3)
    # annotate joint_a values (the hero config)
    for xi, v in zip(x + 0.5*width, joint_a_gain):
        if v > 0.3:
            ax_h.annotate(f"+{v:.1f}pp", (xi, v), xytext=(0, 3),
                          textcoords="offset points",
                          ha="center", fontsize=8, color="#2E7D32")

    # RIGHT: latency speedup
    ax_l.plot(x, sched_sp,  "^-", color="#FF9800", linewidth=2, markersize=9,
              label="Scheduling only")
    ax_l.plot(x, evict_sp,  "s-", color="#2196F3", linewidth=2, markersize=9,
              label="Eviction only (α=0.7)")
    ax_l.plot(x, joint_sp,  "D-", color="#4CAF50", linewidth=2, markersize=9,
              label="Joint (α=0.7)")
    ax_l.axhline(1.0, color="gray", ls=":", lw=1)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(BUCKET_LABELS, fontsize=9)
    ax_l.set_ylabel("Latency speedup vs baseline (×)", fontsize=10)
    ax_l.set_title("Latency speedup", fontsize=11)
    ax_l.legend(fontsize=8, loc="upper left")
    ax_l.grid(axis="y", alpha=0.3)
    # annotate joint values
    for xi, v in zip(x, joint_sp):
        ax_l.annotate(f"{v:.2f}×", (xi, v), xytext=(0, 8),
                      textcoords="offset points",
                      ha="center", fontsize=8, color="#2E7D32")

    plt.tight_layout()
    out = out_dir / "length_vs_improvement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Print a summary table
    print("\n=== Length ablation summary ===")
    print(f"{'bucket':<8}{'base_hit':>10}{'sched':>10}{'evict':>10}"
          f"{'joint':>10}{'lat_speedup':>14}")
    for i, b in enumerate(BUCKETS):
        print(f"{b:<8}{base_hit[i]:>9.1f}%{sched_gain[i]:>+9.1f}pp"
              f"{evict_b_gain[i]:>+9.1f}pp{joint_a_gain[i]:>+9.1f}pp"
              f"{joint_sp[i]:>12.2f}x")


if __name__ == "__main__":
    main()
