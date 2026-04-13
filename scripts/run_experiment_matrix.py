#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Automated experiment matrix runner for KV cache eviction policy research.

Runs all combinations of eviction policy × scheduling policy × workload ×
cache budget, collecting per-request latency and Prometheus metrics, then
writes a summary CSV.

Usage:
    # Dry-run — print what would be run without executing:
    python scripts/run_experiment_matrix.py --dry-run

    # Quick sanity-check (1 workload × 1 budget × all 4 configs, 1 repeat):
    python scripts/run_experiment_matrix.py \\
        --workloads burst \\
        --budgets small \\
        --repeats 1

    # Full matrix (default: 4 configs × 3 workloads × 3 budgets × 3 repeats):
    python scripts/run_experiment_matrix.py \\
        --output-root experiments/results \\
        --model facebook/opt-1.3b \\
        --container /path/to/vllm.sif

    # Core paper matrix (skip ablations):
    python scripts/run_experiment_matrix.py --no-ablations

Options (see --help for full list):
    --model           Model to serve (default: facebook/opt-1.3b)
    --container       Singularity image path (optional)
    --src-dir         Source directory to copy into container overlay
    --output-root     Base directory for all results (default: experiments/results)
    --port            Starting port; each run gets port+run_idx to avoid conflicts
    --repeats         Number of repetitions per config (default: 3)
    --request-rate    Requests/second sent to server (default: 8)
    --max-model-len   (default: 2048)
    --gpu-util        GPU memory utilization (default: 0.4)
    --workloads       Comma-separated subset: sharegpt,burst,mmlu (default: all)
    --budgets         Comma-separated subset: small,medium,large (default: all)
    --configs         Comma-separated subset: baseline,eviction_only,scheduling_only,joint
    --dry-run         Print commands without running them
    --resume          Skip runs whose output directory already has a summary.json
    --no-ablations    Skip ablation configurations (run core matrix only)
    --csv             Path for summary CSV (default: <output-root>/matrix_summary.csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# ── Experiment matrix definition ──────────────────────────────────────────────

CONFIGS: dict[str, dict] = {
    "baseline": {
        "eviction": "lru",
        "scheduling": "fcfs",
        "label": "Baseline (LRU + FCFS)",
    },
    "eviction_only": {
        "eviction": "adaptive",
        "scheduling": "fcfs",
        "label": "Eviction-only (Adaptive + FCFS)",
    },
    "scheduling_only": {
        "eviction": "lru",
        "scheduling": "prefix_match",
        "label": "Scheduling-only (LRU + PrefixMatch)",
    },
    "joint": {
        "eviction": "adaptive",
        "scheduling": "prefix_match",
        "label": "Joint (Adaptive + PrefixMatch)",
    },
}

# GPU block counts mapping to logical pressure levels.
# 512 blocks × 16 tokens/block ≈ 8K tokens for opt-1.3b (block_size=16).
CACHE_BUDGETS: dict[str, int] = {
    "small":  512,    # ~8K tokens  — high eviction pressure
    "medium": 2048,   # ~32K tokens — moderate pressure
    "large":  8192,   # ~128K tokens — near-unlimited (baseline check)
}

# Workload definitions.  dataset-path is resolved at runtime from --data-root.
WORKLOADS: dict[str, dict] = {
    "sharegpt": {
        "dataset_name": "sharegpt",
        "filename":     "ShareGPT_V3_unfiltered_cleaned_split.json",
        "num_prompts":  200,
        "description":  "High-reuse multi-turn chat",
    },
    "burst": {
        "dataset_name": "custom",
        "filename":     "burst_synthetic.jsonl",
        "num_prompts":  500,
        "description":  "70% hot-prefix burst workload",
    },
    "mmlu": {
        "dataset_name": "custom",
        "filename":     "mmlu_vllm.jsonl",
        "num_prompts":  200,
        "description":  "Low-reuse QA (MMLU)",
    },
}


@dataclass
class RunSpec:
    run_id: str          # e.g. "2026-04-12_joint_burst_small_r1"
    config_name: str
    workload_name: str
    budget_name: str
    repeat: int
    eviction: str
    scheduling: str
    gpu_blocks: int
    dataset_name: str
    dataset_path: str
    num_prompts: int
    output_dir: str
    model: str
    port: int
    request_rate: str
    max_model_len: int
    gpu_util: float
    container: str
    src_dir: str


@dataclass
class RunResult:
    spec: RunSpec
    status: str = "pending"   # pending / success / failed / skipped
    error: str = ""
    # Metrics from summary.json
    throughput_rps: float | None = None
    throughput_tps: float | None = None
    ttft_mean_ms: float | None = None
    ttft_p99_ms: float | None = None
    e2e_latency_mean_ms: float | None = None
    e2e_latency_p99_ms: float | None = None
    cache_hit_rate: float | None = None
    evictions_total: float | None = None


# ── Build run list ─────────────────────────────────────────────────────────────

def build_run_specs(
    *,
    output_root: Path,
    data_root: Path,
    model: str,
    port_base: int,
    repeats: int,
    request_rate: str,
    max_model_len: int,
    gpu_util: float,
    container: str,
    src_dir: str,
    config_subset: list[str],
    workload_subset: list[str],
    budget_subset: list[str],
) -> list[RunSpec]:
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    specs: list[RunSpec] = []
    port = port_base

    for budget_name in budget_subset:
        gpu_blocks = CACHE_BUDGETS[budget_name]
        for workload_name in workload_subset:
            wl = WORKLOADS[workload_name]
            dataset_path = str(data_root / wl["filename"]) if data_root else ""
            for config_name in config_subset:
                cfg = CONFIGS[config_name]
                for repeat in range(1, repeats + 1):
                    run_id = (
                        f"{date_prefix}_{config_name}_{workload_name}"
                        f"_{budget_name}_r{repeat}"
                    )
                    specs.append(RunSpec(
                        run_id=run_id,
                        config_name=config_name,
                        workload_name=workload_name,
                        budget_name=budget_name,
                        repeat=repeat,
                        eviction=cfg["eviction"],
                        scheduling=cfg["scheduling"],
                        gpu_blocks=gpu_blocks,
                        dataset_name=wl["dataset_name"],
                        dataset_path=dataset_path,
                        num_prompts=wl["num_prompts"],
                        output_dir=str(output_root / run_id),
                        model=model,
                        port=port,
                        request_rate=request_rate,
                        max_model_len=max_model_len,
                        gpu_util=gpu_util,
                        container=container,
                        src_dir=src_dir,
                    ))
                    port += 1  # unique port per run (though runs are sequential)

    return specs


# ── Execute a single run ───────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
SINGLE_EXP_SCRIPT = SCRIPT_DIR / "run_single_experiment.sh"


def run_spec_to_cmd(spec: RunSpec) -> list[str]:
    cmd = [
        "bash", str(SINGLE_EXP_SCRIPT),
        "--model",        spec.model,
        "--eviction",     spec.eviction,
        "--scheduling",   spec.scheduling,
        "--gpu-blocks",   str(spec.gpu_blocks),
        "--dataset-name", spec.dataset_name,
        "--num-prompts",  str(spec.num_prompts),
        "--request-rate", spec.request_rate,
        "--output-dir",   spec.output_dir,
        "--port",         str(spec.port),
        "--max-model-len", str(spec.max_model_len),
        "--gpu-util",     str(spec.gpu_util),
    ]
    if spec.dataset_path:
        cmd += ["--dataset-path", spec.dataset_path]
    if spec.container:
        cmd += ["--container", spec.container]
    if spec.src_dir:
        cmd += ["--src-dir", spec.src_dir]
    return cmd


def execute_run(spec: RunSpec, *, dry_run: bool, resume: bool) -> RunResult:
    result = RunResult(spec=spec)
    summary_path = Path(spec.output_dir) / "summary.json"

    # Resume: skip if already completed
    if resume and summary_path.exists():
        print(f"  [SKIP] {spec.run_id} (summary.json exists)")
        result.status = "skipped"
        _load_summary_into(result, summary_path)
        return result

    cmd = run_spec_to_cmd(spec)
    if dry_run:
        print("  [DRY]", " ".join(cmd))
        result.status = "skipped"
        return result

    print(f"  [RUN ] {spec.run_id}")
    print("        " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        result.status = "success"
        _load_summary_into(result, summary_path)
    except subprocess.CalledProcessError as e:
        result.status = "failed"
        result.error = f"exit code {e.returncode}"
        print(f"  [FAIL] {spec.run_id}: {result.error}", file=sys.stderr)

    return result


def _load_summary_into(result: RunResult, path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)
    result.throughput_rps      = data.get("throughput_rps")
    result.throughput_tps      = data.get("throughput_tps")
    result.ttft_mean_ms        = data.get("ttft_mean_ms")
    result.ttft_p99_ms         = data.get("ttft_p99_ms")
    result.e2e_latency_mean_ms = data.get("e2e_latency_mean_ms")
    result.e2e_latency_p99_ms  = data.get("e2e_latency_p99_ms")
    result.cache_hit_rate      = data.get("cache_hit_rate")
    result.evictions_total     = data.get("evictions_total")


# ── CSV export ─────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "run_id", "config_name", "workload_name", "budget_name", "repeat",
    "eviction", "scheduling", "gpu_blocks",
    "status", "error",
    "throughput_rps", "throughput_tps",
    "ttft_mean_ms", "ttft_p99_ms",
    "e2e_latency_mean_ms", "e2e_latency_p99_ms",
    "cache_hit_rate", "evictions_total",
]


def write_csv(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {
                **{k: getattr(r.spec, k, "") for k in CSV_FIELDS},
                "status":               r.status,
                "error":                r.error,
                "throughput_rps":       r.throughput_rps,
                "throughput_tps":       r.throughput_tps,
                "ttft_mean_ms":         r.ttft_mean_ms,
                "ttft_p99_ms":          r.ttft_p99_ms,
                "e2e_latency_mean_ms":  r.e2e_latency_mean_ms,
                "e2e_latency_p99_ms":   r.e2e_latency_p99_ms,
                "cache_hit_rate":       r.cache_hit_rate,
                "evictions_total":      r.evictions_total,
            }
            writer.writerow(row)
    print(f"\nCSV written to {path}  ({len(results)} rows)")


# ── Progress print ─────────────────────────────────────────────────────────────

def print_progress(results: list[RunResult]) -> None:
    success = sum(1 for r in results if r.status == "success")
    failed  = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    total   = len(results)
    print(f"\n── Progress: {success}/{total} success, {failed} failed, {skipped} skipped")


def print_summary_table(results: list[RunResult]) -> None:
    done = [r for r in results if r.status == "success"]
    if not done:
        return
    print("\n── Results summary ──")
    header = f"{'Config':20s} {'Workload':10s} {'Budget':8s} {'HitRate':>8s} {'Evictions':>10s} {'Tput(r/s)':>10s} {'TTFT(ms)':>9s}"
    print(header)
    print("-" * len(header))
    for r in sorted(done, key=lambda x: (x.spec.workload_name, x.spec.budget_name, x.spec.config_name)):
        hr  = f"{r.cache_hit_rate:.1%}"  if r.cache_hit_rate  is not None else "N/A"
        ev  = f"{r.evictions_total:.0f}" if r.evictions_total is not None else "N/A"
        tpt = f"{r.throughput_rps:.2f}"  if r.throughput_rps  is not None else "N/A"
        ttft= f"{r.ttft_mean_ms:.1f}"    if r.ttft_mean_ms    is not None else "N/A"
        print(f"{r.spec.config_name:20s} {r.spec.workload_name:10s} {r.spec.budget_name:8s} "
              f"{hr:>8s} {ev:>10s} {tpt:>10s} {ttft:>9s}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model / infra
    p.add_argument("--model",         default="facebook/opt-1.3b")
    p.add_argument("--container",     default="", help="Singularity .sif path (optional)")
    p.add_argument("--src-dir",       default="", help="Source to copy into container overlay")
    p.add_argument("--data-root",     default="data",
                   help="Directory containing dataset files (default: data/)")
    p.add_argument("--output-root",   default="experiments/results")
    p.add_argument("--port",          type=int, default=8100,
                   help="Starting port (each run gets port+N; default: 8100)")
    # Benchmark params
    p.add_argument("--repeats",       type=int, default=3)
    p.add_argument("--request-rate",  default="8")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-util",      type=float, default=0.4)
    # Subset selection
    p.add_argument("--configs",   default=",".join(CONFIGS.keys()),
                   help="Comma-separated config names (default: all)")
    p.add_argument("--workloads", default=",".join(WORKLOADS.keys()),
                   help="Comma-separated workload names (default: all)")
    p.add_argument("--budgets",   default=",".join(CACHE_BUDGETS.keys()),
                   help="Comma-separated budget names (default: all)")
    # Run control
    p.add_argument("--dry-run",   action="store_true",
                   help="Print commands without executing")
    p.add_argument("--resume",    action="store_true",
                   help="Skip runs that already have summary.json")
    p.add_argument("--csv",       default="",
                   help="Override output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    data_root   = Path(args.data_root)
    csv_path    = Path(args.csv) if args.csv else output_root / "matrix_summary.csv"

    config_subset   = [c.strip() for c in args.configs.split(",")   if c.strip()]
    workload_subset = [w.strip() for w in args.workloads.split(",") if w.strip()]
    budget_subset   = [b.strip() for b in args.budgets.split(",")   if b.strip()]

    # Validate subsets
    for c in config_subset:
        if c not in CONFIGS:
            sys.exit(f"Unknown config '{c}'. Choose from: {list(CONFIGS)}")
    for w in workload_subset:
        if w not in WORKLOADS:
            sys.exit(f"Unknown workload '{w}'. Choose from: {list(WORKLOADS)}")
    for b in budget_subset:
        if b not in CACHE_BUDGETS:
            sys.exit(f"Unknown budget '{b}'. Choose from: {list(CACHE_BUDGETS)}")

    specs = build_run_specs(
        output_root=output_root,
        data_root=data_root,
        model=args.model,
        port_base=args.port,
        repeats=args.repeats,
        request_rate=args.request_rate,
        max_model_len=args.max_model_len,
        gpu_util=args.gpu_util,
        container=args.container,
        src_dir=args.src_dir,
        config_subset=config_subset,
        workload_subset=workload_subset,
        budget_subset=budget_subset,
    )

    total = len(specs)
    print(f"Experiment matrix: {len(config_subset)} configs × "
          f"{len(workload_subset)} workloads × "
          f"{len(budget_subset)} budgets × "
          f"{args.repeats} repeats = {total} runs")
    if args.dry_run:
        print("[DRY-RUN mode — commands will be printed but not executed]\n")

    results: list[RunResult] = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{total}] {spec.run_id}")
        result = execute_run(spec, dry_run=args.dry_run, resume=args.resume)
        results.append(result)
        # Write CSV incrementally so partial results survive a crash
        write_csv(results, csv_path)
        print_progress(results)

    print_summary_table(results)

    failed = [r for r in results if r.status == "failed"]
    if failed:
        print(f"\n{len(failed)} run(s) FAILED:")
        for r in failed:
            print(f"  {r.spec.run_id}: {r.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
