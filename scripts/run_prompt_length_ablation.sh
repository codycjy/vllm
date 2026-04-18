#!/usr/bin/env bash
# Run the prompt-length ablation experiment (16 configurations).
#
# Must run on a GPU node with V100 and Singularity available.
#
# Usage:
#   ./scripts/run_prompt_length_ablation.sh
#
# Override to run a single config (debugging):
#   ONLY_BUCKET=medium ONLY_CONFIG=baseline ./scripts/run_prompt_length_ablation.sh
#
# Environment variables:
#   SERVER_WAIT_TIMEOUT  seconds to wait for server ready (default: 150)
#   SERVER_PORT          port for vLLM server (default: 8100)

set -euo pipefail

CURRENT_SERVER_PID=""
cleanup() {
    if [ -n "$CURRENT_SERVER_PID" ]; then
        echo ""
        log "Interrupted — killing server (PID $CURRENT_SERVER_PID)..."
        kill "$CURRENT_SERVER_PID" 2>/dev/null || true
    fi
    exit 1
}
trap cleanup INT TERM

# ── Paths ────────────────────────────────────────────────────────────────────
CONTAINER=/ocean/projects/cis250265p/xli45/opensource/containers/images/vllm.sif
VLLM_SRC=/ocean/projects/cis250265p/xli45/opensource/dev/vllm
MODELS=/ocean/projects/cis250265p/xli45/opensource/models
DATA=/ocean/projects/cis250265p/xli45/opensource/data/prompt_length
RESULTS=$VLLM_SRC/experiments/results/prompt_length_ablation

# ── Config ───────────────────────────────────────────────────────────────────
SERVER_PORT=${SERVER_PORT:-8100}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-150}
MODEL=$MODELS/Qwen3-8B       # main eval model per project plan
NUM_REQUESTS=200
MAX_TOKENS=32
NUM_GPU_BLOCKS=256   # 256 × 16 = 4096 token cache — forces eviction for long prompts
MAX_MODEL_LEN=4096   # Qwen3-8B supports long context; xlarge prompts ~1700 tok + 32 output

BUCKETS=(short medium long xlarge)
declare -A EVICTION=(
    [baseline]=lru
    [eviction]=adaptive
    [scheduling]=lru
    [joint]=adaptive
)
declare -A SCHEDULING=(
    [baseline]=fcfs
    [eviction]=fcfs
    [scheduling]=prefix_match
    [joint]=prefix_match
)
CONFIGS=(baseline eviction scheduling joint)

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local port=$1
    local timeout=$2
    local elapsed=0
    echo -n "[$(date '+%H:%M:%S')] Waiting for server (model loading, ~60-90s) "
    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo ""
            log "Server ready after ${elapsed}s"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    echo ""
    log "ERROR: server did not become ready in ${timeout}s — check $SERVER_LOG"
    return 1
}

kill_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        log "Stopping server (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

start_server() {
    local eviction=$1
    local scheduling=$2
    local port=$3

    SERVER_LOG="$RESULTS/server_${eviction}_${scheduling}.log"
    log "Starting server: eviction=$eviction scheduling=$scheduling port=$port"
    log "Server log: $SERVER_LOG"

    singularity exec --nv --writable-tmpfs "$CONTAINER" bash -c "
        cp -r $VLLM_SRC/vllm/* /usr/local/lib/python3.12/dist-packages/vllm/ 2>/dev/null
        cd /tmp
        CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL \
            --port $port \
            --enable-prefix-caching \
            --eviction-policy $eviction \
            --scheduling-policy $scheduling \
            --scheduling-max-wait 15 \
            --served-model-name default \
            --num-gpu-blocks-override $NUM_GPU_BLOCKS \
            --max-model-len $MAX_MODEL_LEN \
            --disable-log-requests \
            2>&1
    " > "$SERVER_LOG" 2>&1 &
    echo $!
}

run_collect() {
    local bucket=$1
    local config=$2
    local out_dir=$3
    local port=$4

    log "Running benchmark: bucket=$bucket config=$config → $out_dir"
    mkdir -p "$out_dir"

    # All configs use concurrency=8 for a fair comparison:
    # - 8 in-flight requests > cache capacity (for medium/long/xlarge) → server queues them
    # - prefix_match scheduler gets a non-trivial queue to reorder (real scheduling test)
    # - same load for all 4 configs → latency / hit-rate numbers are directly comparable
    # - vLLM handles over-capacity by queueing (not crashing), which is the realistic case
    local concurrency=8

    singularity exec "$CONTAINER" python3 "$VLLM_SRC/experiments/collect_metrics.py" \
        --server "http://localhost:${port}" \
        --dataset "$DATA/bucket_${bucket}.json" \
        --num-requests "$NUM_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$concurrency" \
        --output "$out_dir/metrics.jsonl"
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log "=== Prompt Length Ablation Experiment ==="
log "Results dir: $RESULTS"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER" >&2; exit 1
fi
for b in "${BUCKETS[@]}"; do
    if [ ! -f "$DATA/bucket_${b}.json" ]; then
        echo "ERROR: Data file missing: $DATA/bucket_${b}.json" >&2
        echo "Run: python3 $VLLM_SRC/scripts/prepare_length_buckets.py" >&2
        exit 1
    fi
done

nvidia-smi -L > /dev/null 2>&1 || { echo "ERROR: No GPU detected"; exit 1; }
mkdir -p "$RESULTS"

# ── Main loop ────────────────────────────────────────────────────────────────
TOTAL=0
DONE=0
for b in "${BUCKETS[@]}"; do
    for c in "${CONFIGS[@]}"; do
        TOTAL=$((TOTAL + 1))
    done
done

log "Total runs: $TOTAL (${#BUCKETS[@]} buckets × ${#CONFIGS[@]} configs)"
echo ""

for bucket in "${BUCKETS[@]}"; do
    # Skip if ONLY_BUCKET is set and doesn't match
    if [ -n "${ONLY_BUCKET:-}" ] && [ "$ONLY_BUCKET" != "$bucket" ]; then
        continue
    fi

    for config in "${CONFIGS[@]}"; do
        # Skip if ONLY_CONFIG is set and doesn't match
        if [ -n "${ONLY_CONFIG:-}" ] && [ "$ONLY_CONFIG" != "$config" ]; then
            continue
        fi

        DONE=$((DONE + 1))
        out_dir="$RESULTS/${bucket}_${config}"

        # Skip if already done
        if [ -f "$out_dir/metrics.summary.json" ]; then
            log "[$DONE/$TOTAL] SKIP (already exists): ${bucket}_${config}"
            continue
        fi

        log "[$DONE/$TOTAL] START: bucket=$bucket config=$config"

        eviction="${EVICTION[$config]}"
        scheduling="${SCHEDULING[$config]}"

        # Start server
        SERVER_PID=$(start_server "$eviction" "$scheduling" "$SERVER_PORT")
        CURRENT_SERVER_PID=$SERVER_PID
        log "Server PID: $SERVER_PID"

        # Wait for server to be ready; kill and skip on failure
        if ! wait_for_server "$SERVER_PORT" "$SERVER_WAIT_TIMEOUT"; then
            kill_server "$SERVER_PID"
            log "WARN: Skipping ${bucket}_${config} — server failed to start"
            continue
        fi

        # Run benchmark
        if run_collect "$bucket" "$config" "$out_dir" "$SERVER_PORT"; then
            log "[$DONE/$TOTAL] DONE: ${bucket}_${config}"
        else
            log "WARN: collect_metrics failed for ${bucket}_${config}"
        fi

        # Stop server
        CURRENT_SERVER_PID=""
        kill_server "$SERVER_PID"
        sleep 5   # brief pause before next server start

        echo ""
    done
done

log "=== All runs complete ==="
log "Results saved to: $RESULTS"
echo ""
log "Next step: python3 $VLLM_SRC/scripts/analyze_prompt_length.py \\"
log "    --results-dir $RESULTS \\"
log "    --output $RESULTS/plots"
