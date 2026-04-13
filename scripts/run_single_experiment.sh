#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run one eviction-policy experiment: start vLLM server → benchmark → collect metrics → shutdown.
#
# Usage (all params via env or positional):
#   ./run_single_experiment.sh \
#       --model          facebook/opt-1.3b \
#       --eviction       adaptive \
#       --scheduling     prefix_match \
#       --gpu-blocks     512 \
#       --dataset-name   sharegpt \
#       --dataset-path   /data/ShareGPT_V3.json \
#       --num-prompts    200 \
#       --request-rate   8 \
#       --output-dir     experiments/results/2026-04-12_joint_sharegpt_small_r1 \
#       --port           8100
#
# Optional:
#   --container  /path/to/vllm.sif   (singularity image; omit for local venv)
#   --src-dir    /path/to/src        (copy src into container overlay; omit = skip)
#   --max-model-len 2048
#   --gpu-util   0.4
#
# Exit codes:
#   0  success
#   1  server failed to start
#   2  benchmark failed

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="facebook/opt-1.3b"
EVICTION="lru"
SCHEDULING="fcfs"
GPU_BLOCKS=""           # empty = no override
DATASET_NAME="sharegpt"
DATASET_PATH=""
NUM_PROMPTS=200
REQUEST_RATE="8"
OUTPUT_DIR="experiments/results/run_$(date +%Y%m%d_%H%M%S)"
PORT=8100
CONTAINER=""
SRC_DIR=""
MAX_MODEL_LEN=2048
GPU_UTIL=0.4
SERVER_STARTUP_TIMEOUT=120   # seconds

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";           shift 2 ;;
        --eviction)       EVICTION="$2";        shift 2 ;;
        --scheduling)     SCHEDULING="$2";      shift 2 ;;
        --gpu-blocks)     GPU_BLOCKS="$2";      shift 2 ;;
        --dataset-name)   DATASET_NAME="$2";    shift 2 ;;
        --dataset-path)   DATASET_PATH="$2";    shift 2 ;;
        --num-prompts)    NUM_PROMPTS="$2";     shift 2 ;;
        --request-rate)   REQUEST_RATE="$2";    shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";      shift 2 ;;
        --port)           PORT="$2";            shift 2 ;;
        --container)      CONTAINER="$2";       shift 2 ;;
        --src-dir)        SRC_DIR="$2";         shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2";   shift 2 ;;
        --gpu-util)       GPU_UTIL="$2";        shift 2 ;;
        --timeout)        SERVER_STARTUP_TIMEOUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
LOG_SERVER="$OUTPUT_DIR/server.log"
LOG_BENCH="$OUTPUT_DIR/bench.log"
SUMMARY="$OUTPUT_DIR/summary.json"

# ── Helpers ────────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local url="http://localhost:${PORT}/health"
    local elapsed=0
    log "Waiting for server at $url (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    while ! curl -sf "$url" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $SERVER_STARTUP_TIMEOUT ]]; then
            log "ERROR: server did not become ready within ${SERVER_STARTUP_TIMEOUT}s"
            return 1
        fi
    done
    log "Server ready after ${elapsed}s"
}

# ── Build vLLM server command ──────────────────────────────────────────────────
SERVER_ARGS=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --port "$PORT"
    --enable-prefix-caching
    --eviction-policy "$EVICTION"
    --scheduling-policy "$SCHEDULING"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --disable-log-requests
)
[[ -n "$GPU_BLOCKS" ]] && SERVER_ARGS+=(--num-gpu-blocks-override "$GPU_BLOCKS")

log "Config: eviction=$EVICTION scheduling=$SCHEDULING gpu_blocks=${GPU_BLOCKS:-auto} model=$MODEL"
log "Output: $OUTPUT_DIR"

# ── Start server ───────────────────────────────────────────────────────────────
if [[ -n "$CONTAINER" ]]; then
    CONTAINER_CMD=(singularity exec --nv --writable-tmpfs "$CONTAINER")
    # Optionally copy modified source into the container overlay
    if [[ -n "$SRC_DIR" ]]; then
        DST=$(singularity exec "$CONTAINER" python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "/tmp/vllm")
        log "Copying src $SRC_DIR → $DST inside container"
        "${CONTAINER_CMD[@]}" bash -c "cp -r ${SRC_DIR}/. ${DST}/"
    fi
    "${CONTAINER_CMD[@]}" "${SERVER_ARGS[@]}" >> "$LOG_SERVER" 2>&1 &
else
    "${SERVER_ARGS[@]}" >> "$LOG_SERVER" 2>&1 &
fi
SERVER_PID=$!
log "Server PID=$SERVER_PID, log: $LOG_SERVER"

# Ensure cleanup on exit
cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Shutting down server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── Wait for server ────────────────────────────────────────────────────────────
if ! wait_for_server; then
    log "Server startup failed. Last 20 lines of log:"
    tail -20 "$LOG_SERVER"
    exit 1
fi

# ── Collect baseline Prometheus snapshot ──────────────────────────────────────
curl -sf "http://localhost:${PORT}/metrics" > "$OUTPUT_DIR/prometheus_before.txt" 2>/dev/null || true

# ── Run benchmark ──────────────────────────────────────────────────────────────
BENCH_ARGS=(
    python3 -m vllm.benchmarks.serve
    --backend openai
    --base-url "http://localhost:${PORT}"
    --model "$MODEL"
    --dataset-name "$DATASET_NAME"
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$REQUEST_RATE"
    --save-result
    --result-dir "$OUTPUT_DIR"
    --result-filename "bench_result.json"
)
[[ -n "$DATASET_PATH" ]] && BENCH_ARGS+=(--dataset-path "$DATASET_PATH")

log "Running benchmark: $DATASET_NAME ($NUM_PROMPTS prompts @ ${REQUEST_RATE} req/s)"
if ! "${BENCH_ARGS[@]}" 2>&1 | tee "$LOG_BENCH"; then
    log "ERROR: benchmark failed"
    exit 2
fi

# ── Collect final Prometheus snapshot ─────────────────────────────────────────
curl -sf "http://localhost:${PORT}/metrics" > "$OUTPUT_DIR/prometheus_final.txt" 2>/dev/null || true

# ── Parse key Prometheus metrics into summary.json ────────────────────────────
python3 - <<PYEOF
import json, re, os

def parse_prometheus(path):
    metrics = {}
    if not os.path.exists(path):
        return metrics
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([\d.eE+\-]+)', line)
            if m:
                metrics[m.group(1)] = float(m.group(2))
    return metrics

before = parse_prometheus("$OUTPUT_DIR/prometheus_before.txt")
after  = parse_prometheus("$OUTPUT_DIR/prometheus_final.txt")

def delta(key):
    return after.get(key, 0) - before.get(key, 0)

# Read bench_result.json if it exists
bench = {}
bench_path = "$OUTPUT_DIR/bench_result.json"
if os.path.exists(bench_path):
    with open(bench_path) as f:
        bench = json.load(f)

summary = {
    "config": {
        "model":        "$MODEL",
        "eviction":     "$EVICTION",
        "scheduling":   "$SCHEDULING",
        "gpu_blocks":   "$GPU_BLOCKS" or None,
        "dataset":      "$DATASET_NAME",
        "num_prompts":  int("$NUM_PROMPTS"),
        "request_rate": "$REQUEST_RATE",
    },
    "throughput_rps":    bench.get("request_throughput"),
    "throughput_tps":    bench.get("output_throughput"),
    "ttft_mean_ms":      bench.get("mean_ttft_ms"),
    "ttft_p99_ms":       bench.get("p99_ttft_ms"),
    "tpot_mean_ms":      bench.get("mean_tpot_ms"),
    "itl_mean_ms":       bench.get("mean_itl_ms"),
    "e2e_latency_mean_ms": bench.get("mean_e2e_latency_ms"),
    "e2e_latency_p99_ms":  bench.get("p99_e2e_latency_ms"),
    # Prometheus deltas
    "cache_hit_rate":    after.get("vllm:gpu_prefix_cache_hit_rate"),
    "evictions_total":   delta("vllm:kv_cache_evictions_total"),
    "cache_queries":     delta("vllm:gpu_prefix_cache_queries_total"),
    "cache_hits":        delta("vllm:gpu_prefix_cache_hits_total"),
}

with open("$SUMMARY", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PYEOF

log "Done. Results in $OUTPUT_DIR"
