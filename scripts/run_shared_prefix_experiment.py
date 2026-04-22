#!/usr/bin/env python3
"""
Run a focused shared-prefix KV cache experiment.

This is a smaller matrix than run_experiment_matrix.py. It is intended for
testing prefix-aware eviction on a workload where every multi-turn request
starts with the same system prompt.

Examples:
    python3 scripts/run_shared_prefix_experiment.py --dry-run

    python3 scripts/run_shared_prefix_experiment.py \
        --generate-dataset \
        --configs lru_fcfs adaptive_fcfs prefix_aware_fcfs \
        --cache-budgets tiny small \
        --request-rates 4 8 \
        --repeats 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_REPO_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(os.environ.get("VLLM_REPO_DIR", DEFAULT_REPO_DIR))
RESULTS_DIR = Path(
    os.environ.get("VLLM_RESULTS_DIR", REPO_DIR / "experiments" / "results")
)
CONTAINER = os.environ.get("VLLM_CONTAINER")

MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B")
MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
GPU_UTIL = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
SERVER_TIMEOUT = int(os.environ.get("VLLM_SERVER_TIMEOUT", "120"))

DEFAULT_SHARED_DATASET = REPO_DIR / "data" / "sharegpt_multiturn_shared_system.jsonl"
DEFAULT_PLAIN_DATASET = REPO_DIR / "data" / "sharegpt_multiturn.jsonl"
DEFAULT_RAW_SHAREGPT = REPO_DIR / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json"

CONFIGS = {
    "lru_fcfs": {"eviction": "lru", "scheduling": "fcfs"},
    "adaptive_fcfs": {"eviction": "adaptive", "scheduling": "fcfs"},
    "prefix_aware_fcfs": {"eviction": "prefix_aware", "scheduling": "fcfs"},
    "lru_prefix_match": {"eviction": "lru", "scheduling": "prefix_match"},
    "prefix_aware_prefix_match": {
        "eviction": "prefix_aware",
        "scheduling": "prefix_match",
    },
}

CACHE_BUDGETS = {
    "tiny": 256,
    "small": 512,
    "medium": 1024,
    "large": 1024,
}

DEFAULT_CONFIGS = ["lru_fcfs", "adaptive_fcfs", "prefix_aware_fcfs"]
DEFAULT_WORKLOADS = ["shared_system_multiturn"]
DEFAULT_REQUEST_RATES = [4.0, 8.0]


@dataclass
class RunResult:
    config: str
    workload: str
    cache_budget: str
    request_rate: float
    repeat: int
    gpu_blocks: int
    num_prompts: int
    request_throughput: float = 0.0
    output_token_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hit_rate: float = 0.0
    evictions_total: float = 0.0
    output_dir: str = ""
    status: str = "pending"


def build_workloads(
    shared_dataset_path: Path,
    plain_dataset_path: Path,
    num_prompts: int,
) -> dict[str, dict]:
    return {
        "shared_system_multiturn": {
            "dataset_name": "custom",
            "dataset_path": str(shared_dataset_path),
            "num_prompts": num_prompts,
            "extra_args": ["--disable-shuffle"],
        },
        "plain_multiturn": {
            "dataset_name": "custom",
            "dataset_path": str(plain_dataset_path),
            "num_prompts": num_prompts,
            "extra_args": ["--disable-shuffle"],
        },
    }


def parse_prometheus(path: Path) -> dict[str, float]:
    """Parse prometheus text format. Strips labels and sums duplicate names."""
    import re

    metrics: dict[str, float] = {}
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(
                    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([\d.eE+\-]+)",
                    line,
                )
                if match:
                    name, value = match.group(1), float(match.group(2))
                    metrics[name] = metrics.get(name, 0.0) + value
    except FileNotFoundError:
        pass
    return metrics


def generate_shared_dataset(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(args.repo_dir / "scripts" / "gen_multiturn_dataset.py"),
        "--input",
        str(args.raw_sharegpt_path),
        "--output",
        str(args.shared_dataset_path),
        "--use-default-common-system-prompt",
        "--system-prompt-id",
        args.system_prompt_id,
        "--max-convs",
        str(args.max_convs),
        "--max-prompt-chars",
        str(args.max_prompt_chars),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--seed",
        str(args.seed),
    ]
    if args.system_prompt_file:
        cmd.remove("--use-default-common-system-prompt")
        cmd.extend(["--common-system-prompt-file", str(args.system_prompt_file)])

    print("Generating shared-system multiturn dataset:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_server_cmd(
    cfg: dict,
    gpu_blocks: int,
    port: int,
    args: argparse.Namespace,
) -> list[str]:
    server_cmd = [
        args.python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        str(port),
        "--enable-prefix-caching",
        "--eviction-policy",
        cfg["eviction"],
        "--scheduling-policy",
        cfg["scheduling"],
        "--num-gpu-blocks-override",
        str(gpu_blocks),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--disable-log-requests",
    ]

    if args.serve_mode == "local":
        return server_cmd

    if not args.container:
        raise RuntimeError(
            "serve-mode=singularity requires --container or VLLM_CONTAINER"
        )
    return [
        "singularity",
        "exec",
        "--nv",
        "--writable-tmpfs",
        str(args.container),
        *server_cmd,
    ]


def start_server(
    cfg: dict,
    gpu_blocks: int,
    log_path: Path,
    port: int,
    args: argparse.Namespace,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    cmd = build_server_cmd(cfg, gpu_blocks, port, args)
    log_f = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=log_f,
        env=env,
        start_new_session=True,
    )
    proc._gpu_id = args.gpu_id  # type: ignore[attr-defined]
    return proc


def wait_for_server(port: int, timeout: int) -> bool:
    url = f"http://localhost:{port}/health"
    elapsed = 0
    while elapsed < timeout:
        result = subprocess.run(["curl", "-sf", url], capture_output=True, timeout=5)
        if result.returncode == 0:
            return True
        time.sleep(5)
        elapsed += 5
    return False


def kill_tree(pid: int, sig: signal.Signals) -> None:
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True
        )
        for child_pid in result.stdout.split():
            kill_tree(int(child_pid), sig)
        os.kill(pid, sig)
    except (ProcessLookupError, ValueError):
        pass


def wait_gpu_free(gpu_id: int, timeout: int = 120) -> None:
    for _ in range(timeout // 3):
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return
        if not result.stdout.strip():
            time.sleep(10)
            return
        time.sleep(3)
    print(f"  WARNING: GPU{gpu_id} may still be occupied after {timeout}s")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    kill_tree(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        kill_tree(proc.pid, signal.SIGKILL)
        proc.wait()
    wait_gpu_free(gpu_id=getattr(proc, "_gpu_id", 0), timeout=120)


def collect_prometheus(out_path: Path, port: int) -> None:
    with out_path.open("w") as f:
        subprocess.run(
            ["curl", "-sf", f"http://localhost:{port}/metrics"],
            stdout=f,
            stderr=subprocess.DEVNULL,
        )


def run_bench(
    workload: dict,
    output_dir: Path,
    port: int,
    request_rate: float,
    args: argparse.Namespace,
) -> bool:
    cmd = [
        args.bench_bin,
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        "localhost",
        "--port",
        str(port),
        "--model",
        args.model,
        "--dataset-name",
        workload["dataset_name"],
        "--dataset-path",
        workload["dataset_path"],
        "--num-prompts",
        str(workload["num_prompts"]),
        "--request-rate",
        str(request_rate),
        "--save-result",
        "--result-dir",
        str(output_dir),
        "--result-filename",
        "bench_result.json",
    ] + workload.get("extra_args", [])

    with (output_dir / "bench.log").open("w") as log_f:
        ret = subprocess.run(cmd, stdout=log_f, stderr=log_f)
    return ret.returncode == 0


def run_one(
    config_name: str,
    workload_name: str,
    budget_name: str,
    request_rate: float,
    repeat: int,
    workloads: dict[str, dict],
    run_dir: Path,
    args: argparse.Namespace,
) -> RunResult:
    cfg = CONFIGS[config_name]
    workload = workloads[workload_name]
    gpu_blocks = CACHE_BUDGETS[budget_name]
    rate_tag = f"rate{int(request_rate)}"
    tag = f"{config_name}_{workload_name}_{budget_name}_{rate_tag}_r{repeat}"
    output_dir = run_dir / tag

    result = RunResult(
        config=config_name,
        workload=workload_name,
        cache_budget=budget_name,
        request_rate=request_rate,
        repeat=repeat,
        gpu_blocks=gpu_blocks,
        num_prompts=workload["num_prompts"],
        output_dir=str(output_dir),
    )

    if args.dry_run:
        result.status = "dry_run"
        print(
            f"  [dry] {tag} eviction={cfg['eviction']} "
            f"scheduling={cfg['scheduling']} blocks={gpu_blocks} rate={request_rate}"
        )
        return result

    dataset_path = Path(workload["dataset_path"])
    if not dataset_path.exists():
        result.status = "failed"
        print(f"  ERROR: missing dataset {dataset_path}")
        return result

    port = args.port or random.randint(8300, 8899)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print(f"RUN: {tag}")
    print(
        f"  eviction={cfg['eviction']} scheduling={cfg['scheduling']} "
        f"gpu_blocks={gpu_blocks} rate={request_rate} port={port}"
    )
    print(f"  dataset={dataset_path}")
    print(f"  output={output_dir}")

    proc = start_server(cfg, gpu_blocks, output_dir / "server.log", port, args)
    print(f"  Server PID={proc.pid}, waiting...")

    try:
        if not wait_for_server(port, args.server_timeout):
            print("  ERROR: server did not start")
            result.status = "failed"
            return result
        print("  Server ready")

        before_path = output_dir / "prometheus_before.txt"
        after_path = output_dir / "prometheus_after.txt"
        collect_prometheus(before_path, port)

        if not run_bench(workload, output_dir, port, request_rate, args):
            print("  ERROR: benchmark failed")
            result.status = "failed"
            return result

        collect_prometheus(after_path, port)

        bench_path = output_dir / "bench_result.json"
        if bench_path.exists():
            with bench_path.open() as f:
                bench = json.load(f)
            result.request_throughput = bench.get("request_throughput", 0)
            result.output_token_throughput = bench.get("output_throughput", 0)
            result.mean_ttft_ms = bench.get("mean_ttft_ms", 0)
            result.p95_ttft_ms = bench.get("p95_ttft_ms", 0)
            result.p99_ttft_ms = bench.get("p99_ttft_ms", 0)
            result.successful_requests = bench.get("completed", 0)
            result.failed_requests = bench.get("failed", 0)

        before = parse_prometheus(before_path)
        after = parse_prometheus(after_path)
        hits = after.get("vllm:prefix_cache_hits_total", 0) - before.get(
            "vllm:prefix_cache_hits_total", 0
        )
        queries = after.get("vllm:prefix_cache_queries_total", 0) - before.get(
            "vllm:prefix_cache_queries_total", 0
        )
        result.cache_hit_rate = hits / queries if queries > 0 else 0.0
        result.evictions_total = after.get(
            "vllm:kv_cache_evictions_total", 0
        ) - before.get("vllm:kv_cache_evictions_total", 0)
        result.status = "success"
        print(
            f"  throughput={result.request_throughput:.2f} req/s "
            f"hit_rate={result.cache_hit_rate:.2%} "
            f"evictions={result.evictions_total:.0f}"
        )
    finally:
        stop_server(proc)
        print("  Server stopped")

    return result


def save_csv(results: list[RunResult], path: Path) -> None:
    if not results:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def validate_names(name: str, selected: list[str], available: dict) -> None:
    unknown = [item for item in selected if item not in available]
    if unknown:
        valid = ", ".join(sorted(available))
        raise SystemExit(f"Unknown {name}: {unknown}. Valid options: {valid}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=REPO_DIR)
    parser.add_argument("--container", type=Path, default=CONTAINER)
    parser.add_argument(
        "--serve-mode",
        choices=["singularity", "local"],
        default="singularity" if CONTAINER else "local",
    )
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--bench-bin", default="vllm")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=GPU_UTIL)
    parser.add_argument("--server-timeout", type=int, default=SERVER_TIMEOUT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)

    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--workloads", nargs="+", default=DEFAULT_WORKLOADS)
    parser.add_argument(
        "--cache-budgets",
        nargs="+",
        default=["tiny", "small"],
        dest="cache_budgets",
    )
    parser.add_argument(
        "--request-rates",
        nargs="+",
        type=float,
        default=DEFAULT_REQUEST_RATES,
        dest="request_rates",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--generate-dataset", action="store_true")
    parser.add_argument(
        "--shared-dataset-path", type=Path, default=DEFAULT_SHARED_DATASET
    )
    parser.add_argument(
        "--plain-dataset-path", type=Path, default=DEFAULT_PLAIN_DATASET
    )
    parser.add_argument("--raw-sharegpt-path", type=Path, default=DEFAULT_RAW_SHAREGPT)
    parser.add_argument("--max-convs", type=int, default=300)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--system-prompt-id", default="common_system_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.repo_dir = args.repo_dir.resolve()
    args.shared_dataset_path = args.shared_dataset_path.resolve()
    args.plain_dataset_path = args.plain_dataset_path.resolve()
    args.raw_sharegpt_path = args.raw_sharegpt_path.resolve()

    workloads = build_workloads(
        args.shared_dataset_path,
        args.plain_dataset_path,
        args.num_prompts,
    )
    validate_names("config", args.configs, CONFIGS)
    validate_names("workload", args.workloads, workloads)
    validate_names("cache budget", args.cache_budgets, CACHE_BUDGETS)

    if args.generate_dataset:
        generate_shared_dataset(args)

    run_dir = args.results_dir / f"shared_prefix_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "summary.csv"

    runs = [
        (config, workload, budget, rate, repeat)
        for config in args.configs
        for workload in args.workloads
        for budget in args.cache_budgets
        for rate in args.request_rates
        for repeat in range(1, args.repeats + 1)
    ]

    total = len(runs)
    print(
        f"Shared-prefix matrix: {total} runs "
        f"({len(args.configs)} configs x {len(args.workloads)} workloads x "
        f"{len(args.cache_budgets)} budgets x {len(args.request_rates)} rates x "
        f"{args.repeats} repeats)"
    )
    print(f"Results dir: {run_dir}")

    results: list[RunResult] = []
    for i, (config, workload, budget, rate, repeat) in enumerate(runs, 1):
        print(f"\n[{i}/{total}]", end="")
        result = run_one(
            config,
            workload,
            budget,
            rate,
            repeat,
            workloads,
            run_dir,
            args,
        )
        results.append(result)
        save_csv(results, csv_path)

    success = sum(1 for result in results if result.status == "success")
    failed = sum(1 for result in results if result.status == "failed")
    print(f"\nDone: {success} success, {failed} failed / {total} total")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
