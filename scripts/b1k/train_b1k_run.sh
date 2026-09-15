#!/usr/bin/env bash
# Launch (or resume) a pi0.5 BEHAVIOR-1K training run described by a run definition file, detached-friendly.
#
#   scripts/b1k/train_b1k_run.sh scripts/b1k/runs/<exp>.env [auto|fresh|resume]
#
#   auto   (default) resume if the run's checkpoint directory already exists, otherwise start fresh
#   fresh  start from scratch (--overwrite: wipes an existing checkpoint directory of the same name!)
#   resume resume from the latest checkpoint (--resume)
#
# What it does:
#   1. sources $ENV_FILE if it exists (default /tmp/dev/env.sh: caches under /tmp, HF_TOKEN / WANDB_API_KEY,
#      WANDB_BASE_URL) and then the run definition file (KEY=VALUE, see scripts/b1k/runs/*.env),
#   2. waits until the GPUs in CUDA_VISIBLE_DEVICES are idle (no compute process, < 4 GiB used) so it never
#      fights another job for memory -- launch it early and it starts by itself,
#   3. runs scripts/b1k/train_b1k.py with the run's flags, output tee'd to $LOG_DIR/train-<EXP_NAME>.log
#      (default LOG_DIR=/tmp/dev/logs),
#   4. if the trainer dies, waits 60 s and relaunches with --resume; gives up after 3 consecutive failures that did
#      not produce a new checkpoint (so a persistent error does not loop forever).
# Meant to run inside tmux so that it survives the terminal / agent session that started it:
#   tmux new-session -d -s train-<exp> -c <OPENPI_DIR> 'scripts/b1k/train_b1k_run.sh scripts/b1k/runs/<exp>.env; exec bash'
# See docs/b1k_runs.md.
set -uo pipefail

RUN_ENV=${1:?usage: train_b1k_run.sh <run.env> [auto|fresh|resume]}
MODE=${2:-auto}
MAX_CONSECUTIVE_FAILURES=3
GPU_FREE_MIB=4096
GPU_IDLE_CHECKS=3        # consecutive idle checks (GPU_POLL_SECONDS apart) required before starting
GPU_POLL_SECONDS=20

ENV_FILE=${ENV_FILE:-/tmp/dev/env.sh}
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi
set -a
# shellcheck disable=SC1090
source "$RUN_ENV"
set +a
# OPENPI_DIR defaults to the checkout this script lives in.
OPENPI_DIR=${OPENPI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOG_DIR=${LOG_DIR:-/tmp/dev/logs}
LOG_FILE="$LOG_DIR/train-$EXP_NAME.log"
mkdir -p "$LOG_DIR"
CKPT_DIR="$OPENPI_DIR/outputs/checkpoints/$CONFIG_NAME/$EXP_NAME"

log() { echo "[$(date '+%F %T')] [launcher] $*" | tee -a "$LOG_FILE"; }

latest_step() {  # newest completed checkpoint step in CKPT_DIR, or -1
    local best=-1 d
    [ -d "$CKPT_DIR" ] || { echo -1; return; }
    for d in "$CKPT_DIR"/*/; do
        d=$(basename "$d")
        [[ "$d" =~ ^[0-9]+$ ]] && [ -f "$CKPT_DIR/$d/_CHECKPOINT_METADATA" ] && [ "$d" -gt "$best" ] && best=$d
    done
    echo "$best"
}

gpus_idle() {  # 0 if every GPU in CUDA_VISIBLE_DEVICES has < GPU_FREE_MIB used and no compute process anywhere
    local apps used idx
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "$apps" -eq 0 ] || return 1
    for idx in ${CUDA_VISIBLE_DEVICES//,/ }; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
        [ -n "$used" ] && [ "$used" -lt "$GPU_FREE_MIB" ] || return 1
    done
    return 0
}

wait_for_gpus() {
    local ok=0 waited=0
    while [ "$ok" -lt "$GPU_IDLE_CHECKS" ]; do
        if gpus_idle; then ok=$((ok + 1)); else
            if [ "$ok" -gt 0 ] || [ $((waited % 300)) -eq 0 ]; then
                log "waiting for GPUs $CUDA_VISIBLE_DEVICES to be idle (busy: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' | cut -c1-300))"
            fi
            ok=0
        fi
        sleep "$GPU_POLL_SECONDS"; waited=$((waited + GPU_POLL_SECONDS))
    done
    log "GPUs $CUDA_VISIBLE_DEVICES idle for $((GPU_IDLE_CHECKS * GPU_POLL_SECONDS)) s, starting"
}

run_once() {  # $1 = --overwrite | --resume | ""
    local mode_flag=$1
    cd "$OPENPI_DIR" || exit 1
    mkdir -p "$JAX_CACHE"
    export JAXTYPING_DISABLE=1 PYTHONUNBUFFERED=1
    export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES
    local cmd=("$VENV/bin/python" scripts/b1k/train_b1k.py "$CONFIG_NAME"
        --exp_name="$EXP_NAME" --project_name="$WANDB_PROJECT"
        --batch_size="$BATCH_SIZE" --grad_accum_steps="$GRAD_ACCUM_STEPS" --fsdp_devices="$FSDP_DEVICES"
        --model.remat-policy "$REMAT_POLICY" --model.max-token-len "$MAX_TOKEN_LEN"
        --num_workers="$NUM_WORKERS" --num_train_steps="$NUM_TRAIN_STEPS"
        --save_interval="$SAVE_INTERVAL" --max_to_keep="$MAX_TO_KEEP" --keep_period None
        --data.repo_id="$REPO_ID" --data.dataset-root="$DATASET_ROOT" --data.task-names ${TASK_NAMES//,/ })
    [ -n "$mode_flag" ] && cmd+=("$mode_flag")
    log "launching: ${cmd[*]}"
    log "env: WANDB_BASE_URL=${WANDB_BASE_URL:-<unset>} JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    return "${PIPESTATUS[0]}"
}

log "=== run $EXP_NAME (config $CONFIG_NAME), mode=$MODE, run.env=$RUN_ENV, pid $$ ==="
case "$MODE" in
    fresh) MODE_FLAG=--overwrite ;;
    resume) MODE_FLAG=--resume ;;
    auto) if [ -d "$CKPT_DIR" ]; then MODE_FLAG=--resume; else MODE_FLAG=""; fi ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
[ "$(latest_step)" -ge 0 ] && log "existing checkpoints in $CKPT_DIR, latest step $(latest_step)"

failures=0
step_at_last_failure=$(latest_step)
while true; do
    wait_for_gpus
    run_once "$MODE_FLAG"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        log "training finished successfully (latest checkpoint step $(latest_step))"
        exit 0
    fi
    now_step=$(latest_step)
    if [ "$now_step" -gt "$step_at_last_failure" ]; then failures=0; fi   # progress since last failure -> reset
    failures=$((failures + 1)); step_at_last_failure=$now_step
    log "trainer exited with code $rc (latest checkpoint step $now_step); consecutive failures without progress: $failures"
    if [ "$failures" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
        log "giving up after $failures consecutive failures; fix the problem and relaunch with mode 'resume'"
        exit "$rc"
    fi
    MODE_FLAG=--resume
    log "relaunching with --resume in 60 s"
    sleep 60
done
