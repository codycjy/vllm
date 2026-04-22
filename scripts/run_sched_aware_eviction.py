#!/usr/bin/env python3
"""
Scheduling-aware eviction experiment.

Hypothesis: when prefix_match scheduling is active, protecting pending-hit
blocks from eviction should increase cache hit rate, especially under memory
pressure (small/medium cache budgets).

Matrix:
  6 configs × 2 workloads × 3 cache budgets × 2 rates = 72 runs

Configs:
  baseline          lru      + fcfs          (control)
  sched_only_old    lru      + prefix_match  (scheduling only, no protection)
  sched_only_new    lru      + prefix_match  (scheduling + pending protection)
  joint_old         adaptive + prefix_match  (no protection)
  joint_new         adaptive + prefix_match  (pending protection)
  eviction_only     adaptive + fcfs          (eviction baseline)

The "old" vs "new" distinction is controlled by --enable-sched-aware-eviction
flag added to the server (True = new behavior, False/absent = old).

Workloads: multiturn, burst
Cache budgets: small (512), medium (1024), large (2048)
Rates: 8, 16 req/s
"""

from __future__ import annotations

import csv
import json
import os
import random
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# ── Cluster paths ──────────────────────────────────────────────────────────────
REPO_DIR    = Path("/ocean/projects/cis260009p/jchen60/vllmc")
CONTAINER   = Path("/ocean/projects/cis260009p/jchen60/vllm.sif")
RESULTS_DIR = REPO_DIR / "experiments" / "results"
DATA_DIR    = REPO_DIR / "data"

MODEL          = "Qwen/Qwen3-8B"
MAX_MODEL_LEN  = 4096
GPU_UTIL       = 0.9
SERVER_TIMEOUT = 120
NUM_PROMPTS    = 500

# ── Config matrix ──────────────────────────────────────────────────────────────
# sched_aware_eviction=True  → passes --enable-sched-aware-eviction to server
# "sched_aware" key is kept for CSV labelling only; the new code always
# enables pending-hit protection when scheduling=prefix_match, so we simply
# compare these results against the previously collected baseline data.
CONFIGS = {
    "baseline":       {"eviction": "lru",      "scheduling": "fcfs"},
    "eviction_only":  {"eviction": "adaptive",  "scheduling": "fcfs"},
    "sched_only":     {"eviction": "lru",       "scheduling": "prefix_match"},
    "joint":          {"eviction": "adaptive",  "scheduling": "prefix_match"},
}

WORKLOADS = {
    "multiturn": {
        "dataset_path": str(DATA_DIR / "sharegpt_multiturn.jsonl"),
        "extra_args": ["--disable-shuffle"],
    },
    "burst": {
        "dataset_path": str(DATA_DIR / "burst_synthetic.jsonl"),
        "extra_args": [],
    },
}

CACHE_BUDGETS = {"small": 512, "medium": 1024, "large": 2048}
REQUEST_RATES = [8.0, 16.0]


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class RunResult:
    config: str
    workload: str
    cache_budget: str
    request_rate: float
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


def start_server(cfg: dict, gpu_blocks: int, log_path: str, port: int) -> subprocess.Popen:
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
        "--num-gpu-blocks-override", str(gpu_blocks),
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


def run_bench(workload_name: str, output_dir: str, port: int,
              request_rate: float) -> bool:
    wl = WORKLOADS[workload_name]
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--host", "localhost", "--port", str(port),
        "--model", MODEL,
        "--dataset-name", "custom",
        "--dataset-path", wl["dataset_path"],
        "--num-prompts", str(NUM_PROMPTS),
        "--request-rate", str(request_rate),
        "--save-result",
        "--result-dir", output_dir,
        "--result-filename", "bench_result.json",
    ] + wl["extra_args"]
    with open(os.path.join(output_dir, "bench.log"), "w") as lf:
        ret = subprocess.run(cmd, stdout=lf, stderr=lf)
    return ret.returncode == 0


# ── Single run ─────────────────────────────────────────────────────────────────
def run_one(config_name: str, workload_name: str, budget_name: str,
            request_rate: float, run_dir: Path) -> RunResult:
    cfg        = CONFIGS[config_name]
    gpu_blocks = CACHE_BUDGETS[budget_name]
    tag        = f"{config_name}_{workload_name}_{budget_name}_rate{int(request_rate)}"
    output_dir = str(run_dir / tag)
    port       = random.randint(8300, 8899)

    result = RunResult(
        config=config_name, workload=workload_name,
        cache_budget=budget_name, request_rate=request_rate,
        gpu_blocks=gpu_blocks, num_prompts=NUM_PROMPTS,
        output_dir=output_dir,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*64}")
    print(f"RUN: {tag}")
    print(f"  eviction={cfg['eviction']}  scheduling={cfg['scheduling']}  "
          f"blocks={gpu_blocks}  "
          f"rate={request_rate}  port={port}")

    proc = start_server(cfg, gpu_blocks, os.path.join(output_dir, "server.log"), port)
    print(f"  Server PID={proc.pid}, waiting...")

    try:
        if not wait_for_server(port):
            print("  ERROR: server did not start")
            result.status = "server_failed"
            return result
        print("  Server ready")

        collect_prometheus(os.path.join(output_dir, "prometheus_before.txt"), port)

        if not run_bench(workload_name, output_dir, port, request_rate):
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
              f"tput={result.output_token_throughput:.0f} tok/s  "
              f"evictions={result.evictions_total:.0f}")

    finally:
        stop_server(proc)
        print("  Server stopped")

    return result


def save_csv(results: list[RunResult], path: str) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        w.writerows(asdict(r) for r in results)


def main() -> None:
    runs = [
        (cfg, wl, budget, rate)
        for cfg    in CONFIGS
        for wl     in WORKLOADS
        for budget in CACHE_BUDGETS
        for rate   in REQUEST_RATES
    ]
    total = len(runs)
    print(f"Scheduling-aware eviction matrix: {total} runs")
    print(f"  {len(CONFIGS)} configs × {len(WORKLOADS)} workloads × "
          f"{len(CACHE_BUDGETS)} budgets × {len(REQUEST_RATES)} rates")
    print("  Key comparison: sched_only vs sched_only_sa, joint vs joint_sa")

    run_dir  = RESULTS_DIR / f"sched_aware_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(run_dir / "summary.csv")
    results: list[RunResult] = []

    for i, (cfg, wl, budget, rate) in enumerate(runs, 1):
        print(f"\n[{i}/{total}]", end="")
        res = run_one(cfg, wl, budget, rate, run_dir)
        results.append(res)
        save_csv(results, csv_path)

    success = sum(1 for r in results if r.status == "success")
    failed  = sum(1 for r in results if r.status != "success")
    print(f"\nDone: {success} success, {failed} failed / {total} total")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
