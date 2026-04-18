#!/usr/bin/env bash
# Alpha sensitivity sweep for Scheme B.
#
# Varies alpha ∈ {0.3, 0.5, 0.9} on (short, medium) buckets with adaptive+FCFS.
# Combined with existing α=0.7 (scheme_b) and α=1.0 (scheme_a / "eviction")
# results in experiments/results/prompt_length_ablation/, gives a 5-point
# curve: α ∈ {0.3, 0.5, 0.7, 0.9, 1.0}.
#
# Output dirs follow pattern: <bucket>_alpha_<value>  (e.g. medium_alpha_0.3)
#
# Usage:  ./scripts/run_alpha_sweep.sh
# Override buckets:  BUCKETS="medium" ./scripts/run_alpha_sweep.sh
# Override alphas:   ALPHAS="0.3 0.5" ./scripts/run_alpha_sweep.sh

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
RESULTS=$VLLM_SRC/experiments/results/prompt_length_ablation

# ── Config ───────────────────────────────────────────────────────────────────
SERVER_PORT=${SERVER_PORT:-8100}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-150}
MODEL=$MODELS/Qwen3-8B
NUM_REQUESTS=200
MAX_TOKENS=32
NUM_GPU_BLOCKS=256
MAX_MODEL_LEN=4096

# Sweep axes — override with env vars if needed
BUCKETS_STR=${BUCKETS:-"short medium"}
ALPHAS_STR=${ALPHAS:-"0.3 0.5 0.9"}
read -ra BUCKETS_ARR <<< "$BUCKETS_STR"
read -ra ALPHAS_ARR  <<< "$ALPHAS_STR"

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local port=$1 timeout=$2 elapsed=0
    echo -n "[$(date '+%H:%M:%S')] Waiting for server (model loading, ~60-90s) "
    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo ""; log "Server ready after ${elapsed}s"; return 0
        fi
        sleep 5; elapsed=$((elapsed + 5)); echo -n "."
    done
    echo ""; log "ERROR: server did not become ready in ${timeout}s — check $SERVER_LOG"
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
        log "  still alive after TERM, sending KILL..."
        pkill -KILL -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
        sleep 1
    fi
    if curl -sf "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
        log "  WARN: port $SERVER_PORT still serving — another server alive?"
    fi
}

start_server() {
    local alpha=$1 port=$2
    SERVER_LOG="$RESULTS/server_alpha_${alpha}.log"
    log "Starting server: eviction=adaptive alpha=$alpha scheduling=fcfs port=$port" >&2
    log "Server log: $SERVER_LOG" >&2

    singularity exec --nv --writable-tmpfs "$CONTAINER" bash -c "
        cp -r $VLLM_SRC/vllm/* /usr/local/lib/python3.12/dist-packages/vllm/ 2>/dev/null
        cd /tmp
        CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL \
            --port $port \
            --enable-prefix-caching \
            --eviction-policy adaptive \
            --eviction-alpha $alpha \
            --scheduling-policy fcfs \
            --served-model-name default \
            --num-gpu-blocks-override $NUM_GPU_BLOCKS \
            --max-model-len $MAX_MODEL_LEN \
            --disable-log-requests \
            2>&1
    " > "$SERVER_LOG" 2>&1 &
    echo $!
}

run_collect() {
    local bucket=$1 out_dir=$2 port=$3
    mkdir -p "$out_dir"
    singularity exec "$CONTAINER" python3 "$VLLM_SRC/experiments/collect_metrics.py" \
        --server "http://localhost:${port}" \
        --dataset "$DATA/bucket_${bucket}.json" \
        --num-requests "$NUM_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency 8 \
        --output "$out_dir/metrics.jsonl"
}

# ── Pre-flight ───────────────────────────────────────────────────────────────
log "=== Alpha Sweep ==="
log "Buckets: ${BUCKETS_ARR[*]}"
log "Alphas:  ${ALPHAS_ARR[*]}"
log "Results dir: $RESULTS"

[ -f "$CONTAINER" ] || { echo "ERROR: container not found"; exit 1; }
nvidia-smi -L > /dev/null 2>&1 || { echo "ERROR: No GPU detected"; exit 1; }
for b in "${BUCKETS_ARR[@]}"; do
    [ -f "$DATA/bucket_${b}.json" ] || { echo "ERROR: bucket data missing: $b"; exit 1; }
done
mkdir -p "$RESULTS"

if pgrep -f "$VLLM_PROC_PATTERN" > /dev/null 2>&1; then
    log "WARN: leftover vllm server found, killing before starting..."
    pkill -TERM -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
    sleep 3
    pkill -KILL -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
    sleep 2
fi

# ── Main loop: server-per-alpha (reuse across buckets) ───────────────────────
# Key optimization: alpha is a server-startup flag, so for each alpha we only
# start the server once, then run all buckets against it.
TOTAL=$(( ${#BUCKETS_ARR[@]} * ${#ALPHAS_ARR[@]} ))
DONE=0

for alpha in "${ALPHAS_ARR[@]}"; do
    # Check if all buckets for this alpha are already done → skip server start
    need_server=0
    for bucket in "${BUCKETS_ARR[@]}"; do
        out_dir="$RESULTS/${bucket}_adaptive_a${alpha}"
        [ -f "$out_dir/metrics.summary.json" ] || need_server=1
    done
    if [ "$need_server" -eq 0 ]; then
        for bucket in "${BUCKETS_ARR[@]}"; do
            DONE=$((DONE + 1))
            log "[$DONE/$TOTAL] SKIP: ${bucket}_adaptive_a${alpha} (already exists)"
        done
        continue
    fi

    log "--- Starting server for alpha=$alpha ---"
    SERVER_PID=$(start_server "$alpha" "$SERVER_PORT")
    CURRENT_SERVER_PID=$SERVER_PID
    log "Server PID: $SERVER_PID"

    if ! wait_for_server "$SERVER_PORT" "$SERVER_WAIT_TIMEOUT"; then
        kill_server "$SERVER_PID"; CURRENT_SERVER_PID=""
        log "WARN: server failed — skipping alpha=$alpha"
        DONE=$((DONE + ${#BUCKETS_ARR[@]}))
        continue
    fi

    for bucket in "${BUCKETS_ARR[@]}"; do
        DONE=$((DONE + 1))
        out_dir="$RESULTS/${bucket}_adaptive_a${alpha}"

        if [ -f "$out_dir/metrics.summary.json" ]; then
            log "[$DONE/$TOTAL] SKIP: ${bucket}_adaptive_a${alpha}"
            continue
        fi

        log "[$DONE/$TOTAL] RUN: bucket=$bucket alpha=$alpha"
        if run_collect "$bucket" "$out_dir" "$SERVER_PORT"; then
            log "[$DONE/$TOTAL] DONE: ${bucket}_adaptive_a${alpha}"
        else
            log "WARN: collect_metrics failed for ${bucket}_adaptive_a${alpha}"
        fi
    done

    CURRENT_SERVER_PID=""
    kill_server "$SERVER_PID"
    sleep 5
    echo ""
done

log "=== All alpha-sweep runs complete ==="
log "Results: $RESULTS"
echo ""
log "Analyze with: python3 $VLLM_SRC/scripts/analyze_alpha_sweep.py"
