#!/usr/bin/env bash
# Cache-size sweep on one prompt-length bucket.
#
# Purpose: decouple cache pressure from algorithm effectiveness. By varying
# cache size on the same bucket, we observe whether the adaptive-vs-baseline
# gap opens or closes as pressure changes.
#
# Usage:
#   BUCKET=xlarge CACHE_SIZES="512 1024 2048" ./scripts/run_cache_size_sweep.sh
#   BUCKET=medium CACHE_SIZES="64 128 512"    ./scripts/run_cache_size_sweep.sh
#
# Defaults: BUCKET=xlarge, CACHE_SIZES="512 1024 2048".
# Note: c=256 is the default for all existing prompt_length_ablation runs —
# analyze_cache_size_sweep.py pulls that in automatically as a reference point.
#
# Output: experiments/results/cache_size_sweep/<bucket>_c{N}_{config}/

set -euo pipefail

CURRENT_SERVER_PID=""
SERVER_LOG=""
VLLM_PROC_PATTERN="vllm.entrypoints.openai.api_server"

cleanup() {
    local rc=$?
    echo "" >&2
    if pgrep -f "$VLLM_PROC_PATTERN" > /dev/null 2>&1; then
        echo "[cleanup] killing vllm server processes..." >&2
        pkill -TERM -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
        sleep 3
        pkill -KILL -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

# ── Paths ────────────────────────────────────────────────────────────────────
CONTAINER=/ocean/projects/cis250265p/xli45/opensource/containers/images/vllm.sif
VLLM_SRC=/ocean/projects/cis250265p/xli45/opensource/dev/vllm
MODELS=/ocean/projects/cis250265p/xli45/opensource/models
DATA=/ocean/projects/cis250265p/xli45/opensource/data/prompt_length
RESULTS=$VLLM_SRC/experiments/results/cache_size_sweep

# ── Config ───────────────────────────────────────────────────────────────────
SERVER_PORT=${SERVER_PORT:-8100}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-150}
MODEL=$MODELS/Qwen3-8B
NUM_REQUESTS=200
MAX_TOKENS=32
MAX_MODEL_LEN=4096
BUCKET=${BUCKET:-xlarge}

# Sweep axes
CACHE_SIZES_STR=${CACHE_SIZES:-"512 1024 2048"}
read -ra CACHE_SIZES_ARR <<< "$CACHE_SIZES_STR"
# 256 already exists in prompt_length_ablation/ for all buckets — analyze reads it.
declare -A EVICTION=(
    [baseline]=lru
    [adaptive_a0.7]=adaptive
)
declare -A ALPHA=(
    [baseline]=1.0              # unused for LRU
    [adaptive_a0.7]=0.7
)
CONFIGS=(baseline adaptive_a0.7)

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local port=$1 timeout=$2 elapsed=0
    echo -n "[$(date '+%H:%M:%S')] Waiting for server "
    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo ""; log "Server ready after ${elapsed}s"; return 0
        fi
        sleep 5; elapsed=$((elapsed + 5)); echo -n "."
    done
    echo ""; log "ERROR: server did not become ready — check $SERVER_LOG"
    return 1
}

kill_server() {
    local pid=$1
    log "Stopping server (wrapper PID $pid + vllm children)..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    pkill -TERM -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
    sleep 3
    if pgrep -f "$VLLM_PROC_PATTERN" > /dev/null 2>&1; then
        log "  still alive, sending KILL..."
        pkill -KILL -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
        sleep 1
    fi
}

start_server() {
    local eviction=$1 alpha=$2 cache=$3 port=$4
    SERVER_LOG="$RESULTS/server_c${cache}_${eviction}_a${alpha}.log"
    log "Starting server: eviction=$eviction alpha=$alpha cache=$cache" >&2
    log "Server log: $SERVER_LOG" >&2

    singularity exec --nv --writable-tmpfs "$CONTAINER" bash -c "
        cp -r $VLLM_SRC/vllm/* /usr/local/lib/python3.12/dist-packages/vllm/ 2>/dev/null
        cd /tmp
        CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL \
            --port $port \
            --enable-prefix-caching \
            --eviction-policy $eviction \
            --eviction-alpha $alpha \
            --scheduling-policy fcfs \
            --served-model-name default \
            --num-gpu-blocks-override $cache \
            --max-model-len $MAX_MODEL_LEN \
            --disable-log-requests \
            2>&1
    " > "$SERVER_LOG" 2>&1 &
    echo $!
}

run_collect() {
    local out_dir=$1 port=$2
    mkdir -p "$out_dir"
    singularity exec "$CONTAINER" python3 "$VLLM_SRC/experiments/collect_metrics.py" \
        --server "http://localhost:${port}" \
        --dataset "$DATA/bucket_${BUCKET}.json" \
        --num-requests "$NUM_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency 8 \
        --output "$out_dir/metrics.jsonl"
}

# ── Pre-flight ───────────────────────────────────────────────────────────────
log "=== Cache-size sweep on ${BUCKET} ==="
log "Cache sizes: ${CACHE_SIZES_ARR[*]}"
log "Configs: ${CONFIGS[*]}"
log "Results dir: $RESULTS"

[ -f "$CONTAINER" ] || { echo "ERROR: container not found"; exit 1; }
[ -f "$DATA/bucket_${BUCKET}.json" ] || { echo "ERROR: data missing"; exit 1; }
nvidia-smi -L > /dev/null 2>&1 || { echo "ERROR: No GPU detected"; exit 1; }
mkdir -p "$RESULTS"

if pgrep -f "$VLLM_PROC_PATTERN" > /dev/null 2>&1; then
    log "WARN: leftover vllm server found, killing..."
    pkill -TERM -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
    sleep 3
    pkill -KILL -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
    sleep 2
fi

# ── Main loop: server-per-(cache, config) ────────────────────────────────────
# cache is a server startup flag, so we need a fresh server for each (cache, config).
TOTAL=$(( ${#CACHE_SIZES_ARR[@]} * ${#CONFIGS[@]} ))
DONE=0

for cache in "${CACHE_SIZES_ARR[@]}"; do
    for config in "${CONFIGS[@]}"; do
        DONE=$((DONE + 1))
        out_dir="$RESULTS/${BUCKET}_c${cache}_${config}"

        if [ -f "$out_dir/metrics.summary.json" ]; then
            log "[$DONE/$TOTAL] SKIP: ${BUCKET}_c${cache}_${config} (already exists)"
            continue
        fi

        log "[$DONE/$TOTAL] START: cache=$cache config=$config"

        eviction="${EVICTION[$config]}"
        alpha="${ALPHA[$config]}"

        SERVER_PID=$(start_server "$eviction" "$alpha" "$cache" "$SERVER_PORT")
        CURRENT_SERVER_PID=$SERVER_PID
        log "Server PID: $SERVER_PID"

        if ! wait_for_server "$SERVER_PORT" "$SERVER_WAIT_TIMEOUT"; then
            kill_server "$SERVER_PID"; CURRENT_SERVER_PID=""
            log "WARN: server failed — skipping"
            continue
        fi

        if run_collect "$out_dir" "$SERVER_PORT"; then
            log "[$DONE/$TOTAL] DONE: cache=$cache config=$config"
        else
            log "WARN: collect_metrics failed"
        fi

        CURRENT_SERVER_PID=""
        kill_server "$SERVER_PID"
        sleep 5
        echo ""
    done
done

log "=== Cache-size sweep complete ==="
log "Results: $RESULTS"
echo ""
log "Analyze with: python3 $VLLM_SRC/scripts/analyze_cache_size_sweep.py"
