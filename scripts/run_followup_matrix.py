#!/usr/bin/env python3
"""
Follow-up experiment matrix — three targeted sub-experiments:

  Exp-A  max_wait sweep      scheduling_only × multiturn × wait={5,15,30,60}
                             → find optimal wait, explain burst p99 regression

  Exp-B  burst high-load     all 4 policies × burst × rate={16,24,32}
                             → quantify p99 degradation threshold

  Exp-C  confidence repeat   baseline + scheduling_only × multiturn medium × repeat=3
                             → statistical confidence on the +11% hit-rate finding

Run all:
    python3 scripts/run_followup_matrix.py

Run single sub-experiment:
    python3 scripts/run_followup_matrix.py --exp A
    python3 scripts/run_followup_matrix.py --exp B
    python3 scripts/run_followup_matrix.py --exp C
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR    = Path("/ocean/projects/cis260009p/jchen60/vllmc")
CONTAINER   = Path("/ocean/projects/cis260009p/jchen60/vllm.sif")
RESULTS_DIR = REPO_DIR / "experiments" / "results"

MODEL          = "Qwen/Qwen3-8B"
MAX_MODEL_LEN  = 4096
GPU_UTIL       = 0.9
SERVER_TIMEOUT = 120

CONFIGS = {
    "baseline":        {"eviction": "lru",      "scheduling": "fcfs"},
    "eviction_only":   {"eviction": "adaptive",  "scheduling": "fcfs"},
    "scheduling_only": {"eviction": "lru",       "scheduling": "prefix_match"},
    "joint":           {"eviction": "adaptive",  "scheduling": "prefix_match"},
}

WORKLOADS = {
    "multiturn": {
        "dataset_name": "custom",
        "dataset_path": str(REPO_DIR / "data" / "sharegpt_multiturn.jsonl"),
        "num_prompts": 500,
        "extra_args": ["--disable-shuffle"],
    },
    "burst": {
        "dataset_name": "custom",
        "dataset_path": str(REPO_DIR / "data" / "burst_synthetic.jsonl"),
        "num_prompts": 500,
    },
}

# ── Run spec ──────────────────────────────────────────────────────────────────
@dataclass
class RunSpec:
    config: str
    workload: str
    cache_budget: str       # e.g. "medium"
    gpu_blocks: int         # e.g. 1024
    request_rate: float
    repeat: int
    max_wait: float = 30.0  # --scheduling-max-wait (only effective for prefix_match)
    exp_tag: str = ""       # e.g. "A", "B", "C"


@dataclass
class RunResult:
    config: str
    workload: str
    cache_budget: str
    request_rate: float
    repeat: int
    max_wait: float
    exp_tag: str
    gpu_blocks: int
    num_prompts: int
    request_throughput: float = 0.0
    output_token_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    mean_itl_ms: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hit_rate: float = 0.0
    evictions_total: float = 0.0
    output_dir: str = ""
    status: str = "pending"


# ── Sub-experiment definitions ────────────────────────────────────────────────
def exp_a_runs() -> list[RunSpec]:
    """
    Exp-A: max_wait sweep
    scheduling_only × multiturn × medium+large cache × rate {8,16} × wait {5,15,30,60}
    = 2 caches × 2 rates × 4 waits = 16 runs
    Also include baseline (wait irrelevant) as anchor per cache/rate = +4 runs
    Total: 20 runs
    """
    runs = []
    for cache, blocks in [("medium", 1024), ("large", 2048)]:
        for rate in [8.0, 16.0]:
            # baseline anchor
            runs.append(RunSpec("baseline", "multiturn", cache, blocks, rate, 1,
                                max_wait=30.0, exp_tag="A"))
            # sweep wait
            for wait in [5.0, 15.0, 30.0, 60.0]:
                runs.append(RunSpec("scheduling_only", "multiturn", cache, blocks,
                                    rate, 1, max_wait=wait, exp_tag="A"))
    return runs


def exp_b_runs() -> list[RunSpec]:
    """
    Exp-B: burst high-load
    all 4 policies × burst × medium+large cache × rate {16,24,32}
    = 4 policies × 2 caches × 3 rates = 24 runs
    """
    runs = []
    for cfg in ["baseline", "eviction_only", "scheduling_only", "joint"]:
        for cache, blocks in [("medium", 1024), ("large", 2048)]:
            for rate in [16.0, 24.0, 32.0]:
                runs.append(RunSpec(cfg, "burst", cache, blocks, rate, 1,
                                    max_wait=30.0, exp_tag="B"))
    return runs


def exp_c_runs() -> list[RunSpec]:
    """
    Exp-C: confidence repeats
    baseline + scheduling_only × multiturn × medium × rate {8,16} × repeat {1,2,3}
    = 2 configs × 2 rates × 3 repeats = 12 runs
    """
    runs = []
    for cfg in ["baseline", "scheduling_only"]:
        for rate in [8.0, 16.0]:
            for rep in [1, 2, 3]:
                runs.append(RunSpec(cfg, "multiturn", "medium", 1024, rate, rep,
                                    max_wait=30.0, exp_tag="C"))
    return runs


# ── Server helpers ─────────────────────────────────────────────────────────────
def parse_prometheus(path: str) -> dict[str, float]:
    import re
    metrics: dict[str, float] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([\d.eE+\-]+)', line)
                if m:
                    name, val = m.group(1), float(m.group(2))
                    metrics[name] = metrics.get(name, 0.0) + val
    except FileNotFoundError:
        pass
    return metrics


def start_server(spec: RunSpec, log_path: str, port: int) -> subprocess.Popen:
    cfg = CONFIGS[spec.config]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        "singularity", "exec", "--nv", "--writable-tmpfs", str(CONTAINER),
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--enable-prefix-caching",
        "--eviction-policy", cfg["eviction"],
        "--scheduling-policy", cfg["scheduling"],
        "--scheduling-max-wait", str(spec.max_wait),
        "--num-gpu-blocks-override", str(spec.gpu_blocks),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_UTIL),
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_f, stderr=log_f,
                            env=env, start_new_session=True)


def wait_for_server(port: int, timeout: int = SERVER_TIMEOUT) -> bool:
    elapsed = 0
    while elapsed < timeout:
        r = subprocess.run(["curl", "-sf", f"http://localhost:{port}/health"],
                           capture_output=True, timeout=5)
        if r.returncode == 0:
            return True
        time.sleep(5)
        elapsed += 5
    return False


def _kill_tree(pid: int, sig: signal.Signals) -> None:
    try:
        r = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
        for child in r.stdout.split():
            _kill_tree(int(child), sig)
        os.kill(pid, sig)
    except (ProcessLookupError, ValueError):
        pass


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _kill_tree(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid, signal.SIGKILL)
        proc.wait()
    _wait_gpu_free()


def _wait_gpu_free(timeout: int = 120) -> None:
    for _ in range(timeout // 3):
        r = subprocess.run(["nvidia-smi", "--id=0",
                            "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader"],
                           capture_output=True, text=True)
        if not r.stdout.strip():
            print("  GPU0 freed")
            time.sleep(10)
            return
        time.sleep(3)
    print("  WARNING: GPU0 may still be occupied")


def collect_prometheus(out_path: str, port: int) -> None:
    with open(out_path, "w") as f:
        subprocess.run(["curl", "-sf", f"http://localhost:{port}/metrics"],
                       stdout=f, stderr=subprocess.DEVNULL)


def run_bench(spec: RunSpec, output_dir: str, port: int) -> bool:
    wl = WORKLOADS[spec.workload]
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--host", "localhost", "--port", str(port),
        "--model", MODEL,
        "--dataset-name", wl["dataset_name"],
        "--dataset-path", wl["dataset_path"],
        "--num-prompts", str(wl["num_prompts"]),
        "--request-rate", str(spec.request_rate),
        "--save-result",
        "--result-dir", output_dir,
        "--result-filename", "bench_result.json",
    ] + wl.get("extra_args", [])
    with open(os.path.join(output_dir, "bench.log"), "w") as lf:
        ret = subprocess.run(cmd, stdout=lf, stderr=lf)
    return ret.returncode == 0


# ── Single run ─────────────────────────────────────────────────────────────────
def run_one(spec: RunSpec, run_dir: Path) -> RunResult:
    wl     = WORKLOADS[spec.workload]
    wait_s = f"wait{int(spec.max_wait)}s"
    tag    = (f"{spec.config}_{spec.workload}_{spec.cache_budget}"
              f"_rate{int(spec.request_rate)}_{wait_s}_r{spec.repeat}")
    output_dir = str(run_dir / tag)
    port = random.randint(8300, 8899)

    result = RunResult(
        config=spec.config, workload=spec.workload,
        cache_budget=spec.cache_budget,
        request_rate=spec.request_rate, repeat=spec.repeat,
        max_wait=spec.max_wait, exp_tag=spec.exp_tag,
        gpu_blocks=spec.gpu_blocks, num_prompts=wl["num_prompts"],
        output_dir=output_dir,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*64}")
    print(f"[Exp-{spec.exp_tag}] {tag}")
    print(f"  eviction={CONFIGS[spec.config]['eviction']}  "
          f"scheduling={CONFIGS[spec.config]['scheduling']}  "
          f"max_wait={spec.max_wait}s  "
          f"blocks={spec.gpu_blocks}  rate={spec.request_rate}  port={port}")

    proc = start_server(spec, os.path.join(output_dir, "server.log"), port)
    print(f"  Server PID={proc.pid}, waiting...")

    try:
        if not wait_for_server(port):
            print("  ERROR: server did not start")
            result.status = "server_failed"
            return result
        print("  Server ready")

        collect_prometheus(os.path.join(output_dir, "prometheus_before.txt"), port)

        if not run_bench(spec, output_dir, port):
            print("  ERROR: benchmark failed")
            result.status = "bench_failed"
            return result

        collect_prometheus(os.path.join(output_dir, "prometheus_after.txt"), port)

        bench_path = os.path.join(output_dir, "bench_result.json")
        if os.path.exists(bench_path):
            with open(bench_path) as f:
                bench = json.load(f)
            result.request_throughput      = bench.get("request_throughput", 0)
            result.output_token_throughput = bench.get("output_throughput", 0)
            result.mean_ttft_ms            = bench.get("mean_ttft_ms", 0)
            result.p99_ttft_ms             = bench.get("p99_ttft_ms", 0)
            result.mean_tpot_ms            = bench.get("mean_tpot_ms", 0)
            result.mean_itl_ms             = bench.get("mean_itl_ms", 0)
            result.successful_requests     = bench.get("completed", 0)
            result.failed_requests         = bench.get("failed", 0)

        before  = parse_prometheus(os.path.join(output_dir, "prometheus_before.txt"))
        after   = parse_prometheus(os.path.join(output_dir, "prometheus_after.txt"))
        hits    = after.get("vllm:prefix_cache_hits_total", 0)    - before.get("vllm:prefix_cache_hits_total", 0)
        queries = after.get("vllm:prefix_cache_queries_total", 0) - before.get("vllm:prefix_cache_queries_total", 0)
        result.cache_hit_rate  = hits / queries if queries > 0 else 0.0
        result.evictions_total = (
            after.get("vllm:kv_cache_evictions_total", 0)
            - before.get("vllm:kv_cache_evictions_total", 0)
        )
        result.status = "success"
        print(f"  hit_rate={result.cache_hit_rate:.2%}  "
              f"p99_ttft={result.p99_ttft_ms/1000:.1f}s  "
              f"tput={result.output_token_throughput:.0f} tok/s")

    finally:
        stop_server(proc)
        print("  Server stopped")

    return result


# ── CSV ────────────────────────────────────────────────────────────────────────
def save_csv(results: list[RunResult], path: str) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        w.writerows(asdict(r) for r in results)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["A", "B", "C", "all"], default="all",
                        help="Which sub-experiment to run (default: all)")
    args = parser.parse_args()

    all_runs: list[RunSpec] = []
    if args.exp in ("A", "all"):
        all_runs += exp_a_runs()
    if args.exp in ("B", "all"):
        all_runs += exp_b_runs()
    if args.exp in ("C", "all"):
        all_runs += exp_c_runs()

    total = len(all_runs)
    label_a = f"Exp-A ({len(exp_a_runs())} runs): max_wait sweep — multiturn"
    label_b = f"Exp-B ({len(exp_b_runs())} runs): burst high-load rate{{16,24,32}}"
    label_c = f"Exp-C ({len(exp_c_runs())} runs): confidence repeats — multiturn medium"
    print(f"Follow-up matrix: {total} runs")
    if args.exp in ("A", "all"): print(f"  {label_a}")
    if args.exp in ("B", "all"): print(f"  {label_b}")
    if args.exp in ("C", "all"): print(f"  {label_c}")

    run_dir  = RESULTS_DIR / f"followup_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(run_dir / "summary.csv")
    results: list[RunResult] = []

    for i, spec in enumerate(all_runs, 1):
        print(f"\n[{i}/{total}]", end="")
        res = run_one(spec, run_dir)
        results.append(res)
        save_csv(results, csv_path)

    success = sum(1 for r in results if r.status == "success")
    failed  = sum(1 for r in results if r.status != "success")
    print(f"\nDone: {success} success, {failed} failed / {total} total")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
