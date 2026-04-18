#!/usr/bin/env bash
# Prompt-length ablation runner — INCREMENTAL by default.
#
# Layout: 6 logical configs, mapped to the existing directory names so old
# results in experiments/results/prompt_length_ablation/ are reused:
#
#   baseline   LRU   + FCFS              (dir: <bucket>_baseline)    [old]
#   scheduling LRU   + prefix_match      (dir: <bucket>_scheduling)  [old]
#   scheme_a   adaptive + FCFS, α=1.0    (dir: <bucket>_eviction)    [old, == Scheme A]
#   joint_a    adaptive + PFX,  α=1.0    (dir: <bucket>_joint)       [old, == Joint A]
#   scheme_b   adaptive + FCFS, α=ALPHA  (dir: <bucket>_scheme_b)    [NEW]
#   joint_b    adaptive + PFX,  α=ALPHA  (dir: <bucket>_joint_b)     [NEW]
#
# By default runs only scheme_b + joint_b (the two new ones). Existing runs
# are skipped because metrics.summary.json already exists.
#
# Usage:
#   ./scripts/run_prompt_length_ablation.sh                # only new configs (8 runs)
#   CONFIGS="baseline scheduling scheme_a scheme_b joint_a joint_b" \
#       ./scripts/run_prompt_length_ablation.sh            # force full re-run (24 runs)
#   ONLY_BUCKET=medium ./scripts/run_prompt_length_ablation.sh
#
# Environment variables:
#   ALPHA                Scheme B alpha (default: 0.7)
#   CONFIGS              space-separated list (default: "scheme_b joint_b")
#   SERVER_WAIT_TIMEOUT  seconds to wait for server ready (default: 150)
#   SERVER_PORT          port for vLLM server (default: 8100)

set -euo pipefail

CURRENT_SERVER_PID=""
SERVER_LOG=""
VLLM_PROC_PATTERN="vllm.entrypoints.openai.api_server"

# Always kill any lingering vllm server on script exit (normal, error, or Ctrl+C).
# Singularity wraps the real python process, so killing only $CURRENT_SERVER_PID
# leaves orphaned children holding the GPU. pkill by pattern catches them all.
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
ALPHA=${ALPHA:-0.7}
MODEL=$MODELS/Qwen3-8B
NUM_REQUESTS=200
MAX_TOKENS=32
NUM_GPU_BLOCKS=256
MAX_MODEL_LEN=4096

BUCKETS=(short medium long xlarge)

declare -A EVICTION=(
    [baseline]=lru
    [scheduling]=lru
    [scheme_a]=adaptive
    [scheme_b]=adaptive
    [joint_a]=adaptive
    [joint_b]=adaptive
)
declare -A ALPHA_CFG=(
    [baseline]=1.0
    [scheduling]=1.0
    [scheme_a]=1.0
    [scheme_b]=$ALPHA
    [joint_a]=1.0
    [joint_b]=$ALPHA
)
declare -A SCHED=(
    [baseline]=fcfs
    [scheduling]=prefix_match
    [scheme_a]=fcfs
    [scheme_b]=fcfs
    [joint_a]=prefix_match
    [joint_b]=prefix_match
)
# Map logical config name → on-disk dir suffix.
# Naming follows the convention: {policy}_a{alpha} where policy is one of
# baseline, sched_only, adaptive, joint. α=1.0 == pure reuse count; α<1 blends recency.
declare -A DIR_SUFFIX=(
    [baseline]=baseline
    [scheduling]=sched_only
    [scheme_a]=adaptive_a1.0
    [scheme_b]=adaptive_a0.7
    [joint_a]=joint_a1.0
    [joint_b]=joint_a0.7
)

# Default: full 6-config ablation
CONFIGS_STR=${CONFIGS:-"baseline scheduling scheme_a scheme_b joint_a joint_b"}
read -ra CONFIGS_ARR <<< "$CONFIGS_STR"

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
    # Kill Singularity wrapper first
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    # Kill orphaned python server process (Singularity wrapper doesn't cascade)
    pkill -TERM -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
    sleep 3
    if pgrep -f "$VLLM_PROC_PATTERN" > /dev/null 2>&1; then
        log "  still alive after TERM, sending KILL..."
        pkill -KILL -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
        sleep 1
    fi
    # Confirm port is free
    if curl -sf "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
        log "  WARN: port $SERVER_PORT still serving — another server alive?"
    fi
}

start_server() {
    local eviction=$1 alpha=$2 scheduling=$3 port=$4
    # SERVER_LOG is a global so wait_for_server can reference it
    SERVER_LOG="$RESULTS/server_${eviction}_a${alpha}_${scheduling}.log"
    # Log to stderr so $(start_server ...) captures only the PID
    log "Starting server: eviction=$eviction alpha=$alpha scheduling=$scheduling port=$port" >&2
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
    local bucket=$1 dir_suffix=$2 out_dir=$3 port=$4
    log "Running benchmark: bucket=$bucket → $out_dir"
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
log "=== Prompt Length Ablation (incremental, alpha=${ALPHA}) ==="
log "Results dir: $RESULTS"
log "Configs to run: ${CONFIGS_ARR[*]}"

[ -f "$CONTAINER" ] || { echo "ERROR: Container not found: $CONTAINER" >&2; exit 1; }
for b in "${BUCKETS[@]}"; do
    [ -f "$DATA/bucket_${b}.json" ] || {
        echo "ERROR: Data file missing: $DATA/bucket_${b}.json" >&2
        echo "Run: python3 $VLLM_SRC/scripts/prepare_length_buckets.py" >&2
        exit 1
    }
done
nvidia-smi -L > /dev/null 2>&1 || { echo "ERROR: No GPU detected"; exit 1; }
mkdir -p "$RESULTS"

# Kill any leftover vllm server from previous runs (prevents GPU OOM / port collision)
if pgrep -f "$VLLM_PROC_PATTERN" > /dev/null 2>&1; then
    log "WARN: leftover vllm server found, killing before starting..."
    pkill -TERM -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
    sleep 3
    pkill -KILL -f "$VLLM_PROC_PATTERN" 2>/dev/null || true
    sleep 2
fi

# ── Main loop ────────────────────────────────────────────────────────────────
TOTAL=$(( ${#BUCKETS[@]} * ${#CONFIGS_ARR[@]} ))
DONE=0
log "Total runs: $TOTAL (${#BUCKETS[@]} buckets × ${#CONFIGS_ARR[@]} configs)"
echo ""

for bucket in "${BUCKETS[@]}"; do
    [ -n "${ONLY_BUCKET:-}" ] && [ "$ONLY_BUCKET" != "$bucket" ] && continue

    for config in "${CONFIGS_ARR[@]}"; do
        [ -n "${ONLY_CONFIG:-}" ] && [ "$ONLY_CONFIG" != "$config" ] && continue

        DONE=$((DONE + 1))
        suffix="${DIR_SUFFIX[$config]:-$config}"
        out_dir="$RESULTS/${bucket}_${suffix}"

        if [ -f "$out_dir/metrics.summary.json" ]; then
            log "[$DONE/$TOTAL] SKIP (already exists): ${bucket}_${suffix} (config=$config)"
            continue
        fi

        log "[$DONE/$TOTAL] START: bucket=$bucket config=$config → dir=${bucket}_${suffix}"

        eviction="${EVICTION[$config]}"
        alpha="${ALPHA_CFG[$config]}"
        scheduling="${SCHED[$config]}"

        SERVER_PID=$(start_server "$eviction" "$alpha" "$scheduling" "$SERVER_PORT")
        CURRENT_SERVER_PID=$SERVER_PID
        log "Server PID: $SERVER_PID"

        if ! wait_for_server "$SERVER_PORT" "$SERVER_WAIT_TIMEOUT"; then
            kill_server "$SERVER_PID"; CURRENT_SERVER_PID=""
            log "WARN: Skipping ${bucket}_${config} — server failed to start"
            continue
        fi

        if run_collect "$bucket" "$suffix" "$out_dir" "$SERVER_PORT"; then
            log "[$DONE/$TOTAL] DONE: ${bucket}_${suffix}"
        else
            log "WARN: collect_metrics failed for ${bucket}_${suffix}"
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
log "Next step: python3 $VLLM_SRC/scripts/analyze_full_ablation.py \\"
log "    --results-dir $RESULTS"
