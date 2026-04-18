#!/usr/bin/env bash
# Run Scheme B (weighted mix) ablation: Scheme A vs Scheme B across 4 prompt-length buckets.
#
# Configs (5 × 4 buckets = 20 runs):
#   baseline   — LRU   + FCFS,         alpha n/a   (original vLLM)
#   scheme_a   — adaptive + FCFS,      alpha=1.0   (pure reuse count)
#   scheme_b   — adaptive + FCFS,      alpha=0.7   (70% reuse + 30% recency)
#   joint_a    — adaptive + prefix_match, alpha=1.0
#   joint_b    — adaptive + prefix_match, alpha=0.7
#
# Must run on a GPU node with V100 and Singularity available.
#
# Usage:
#   ./scripts/run_scheme_b_ablation.sh
#
# Single-run debug:
#   ONLY_BUCKET=medium ONLY_CONFIG=scheme_b ./scripts/run_scheme_b_ablation.sh
#
# Environment variables:
#   SERVER_WAIT_TIMEOUT  seconds to wait for server ready (default: 150)
#   SERVER_PORT          port for vLLM server (default: 8101)
#   ALPHA                override Scheme B alpha (default: 0.7)

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
RESULTS=$VLLM_SRC/experiments/results/scheme_b_ablation

# ── Config ───────────────────────────────────────────────────────────────────
SERVER_PORT=${SERVER_PORT:-8101}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-150}
ALPHA=${ALPHA:-0.7}              # Scheme B alpha (only used for scheme_b / joint_b)
MODEL=$MODELS/Qwen3-8B
NUM_REQUESTS=200
MAX_TOKENS=32
NUM_GPU_BLOCKS=256
MAX_MODEL_LEN=4096

BUCKETS=(short medium long xlarge)

# Per-config: eviction policy, alpha, scheduling policy
declare -A EVICTION=(
    [baseline]=lru
    [scheme_a]=adaptive
    [scheme_b]=adaptive
    [joint_a]=adaptive
    [joint_b]=adaptive
)
declare -A ALPHA_MAP=(
    [baseline]=1.0
    [scheme_a]=1.0
    [scheme_b]=$ALPHA
    [joint_a]=1.0
    [joint_b]=$ALPHA
)
declare -A SCHEDULING=(
    [baseline]=fcfs
    [scheme_a]=fcfs
    [scheme_b]=fcfs
    [joint_a]=prefix_match
    [joint_b]=prefix_match
)
CONFIGS=(baseline scheme_a scheme_b joint_a joint_b)

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
    local alpha=$2
    local scheduling=$3
    local port=$4

    SERVER_LOG="$RESULTS/server_${eviction}_a${alpha}_${scheduling}.log"
    log "Starting server: eviction=$eviction alpha=$alpha scheduling=$scheduling port=$port"
    log "Server log: $SERVER_LOG"

    singularity exec --nv --writable-tmpfs "$CONTAINER" bash -c "
        cp -r $VLLM_SRC/vllm/* /usr/local/lib/python3.12/dist-packages/vllm/ 2>/dev/null
        cd /tmp
        CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL \
            --port $port \
            --enable-prefix-caching \
            --eviction-policy $eviction \
            --eviction-alpha $alpha \
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

    singularity exec "$CONTAINER" python3 "$VLLM_SRC/experiments/collect_metrics.py" \
        --server "http://localhost:${port}" \
        --dataset "$DATA/bucket_${bucket}.json" \
        --num-requests "$NUM_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency 8 \
        --output "$out_dir/metrics.jsonl"
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log "=== Scheme B Ablation (alpha=${ALPHA}) ==="
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
TOTAL=$(( ${#BUCKETS[@]} * ${#CONFIGS[@]} ))
DONE=0

log "Total runs: $TOTAL (${#BUCKETS[@]} buckets × ${#CONFIGS[@]} configs)"
log "Configs: ${CONFIGS[*]}"
echo ""

for bucket in "${BUCKETS[@]}"; do
    if [ -n "${ONLY_BUCKET:-}" ] && [ "$ONLY_BUCKET" != "$bucket" ]; then
        continue
    fi

    for config in "${CONFIGS[@]}"; do
        if [ -n "${ONLY_CONFIG:-}" ] && [ "$ONLY_CONFIG" != "$config" ]; then
            continue
        fi

        DONE=$((DONE + 1))
        out_dir="$RESULTS/${bucket}_${config}"

        if [ -f "$out_dir/metrics.summary.json" ]; then
            log "[$DONE/$TOTAL] SKIP (already exists): ${bucket}_${config}"
            continue
        fi

        log "[$DONE/$TOTAL] START: bucket=$bucket config=$config"

        eviction="${EVICTION[$config]}"
        alpha="${ALPHA_MAP[$config]}"
        scheduling="${SCHEDULING[$config]}"

        SERVER_PID=$(start_server "$eviction" "$alpha" "$scheduling" "$SERVER_PORT")
        CURRENT_SERVER_PID=$SERVER_PID
        log "Server PID: $SERVER_PID"

        if ! wait_for_server "$SERVER_PORT" "$SERVER_WAIT_TIMEOUT"; then
            kill_server "$SERVER_PID"
            CURRENT_SERVER_PID=""
            log "WARN: Skipping ${bucket}_${config} — server failed to start"
            continue
        fi

        if run_collect "$bucket" "$config" "$out_dir" "$SERVER_PORT"; then
            log "[$DONE/$TOTAL] DONE: ${bucket}_${config}"
        else
            log "WARN: collect_metrics failed for ${bucket}_${config}"
        fi

        CURRENT_SERVER_PID=""
        kill_server "$SERVER_PID"
        sleep 5

        echo ""
    done
done

log "=== All runs complete ==="
log "Results saved to: $RESULTS"
echo ""
log "Next step: python3 $VLLM_SRC/scripts/analyze_scheme_b.py \\"
log "    --results-dir $RESULTS \\"
log "    --output $RESULTS/plots"
