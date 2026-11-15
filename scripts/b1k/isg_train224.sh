#!/usr/bin/env bash
# isg-train224 pi05 run on the batched replay's native 224 x 224 LeRobot v3 datasets (docs/b1k.md "isg-train224").
#
#   scripts/b1k/isg_train224.sh prepare   exact padding-free norm stats (compute_norm_stats.py) + the goal tables
#   scripts/b1k/isg_train224.sh train     the trainer (launch.sh; resumes from the newest save) + the checkpoint exporter
#                                         and Hugging Face upload monitor beside it
#   scripts/b1k/isg_train224.sh upload    exporter + upload monitor alone (resume their journals)
#   scripts/b1k/isg_train224.sh all       prepare, train
#
# PI_VARIANT selects the conditioning:
#   goal      goal image (head camera, the episode's own), the earlier pi runs' image regime with the role embedding
#   shufgoal  goal image drawn per sample among the training episodes of the episode's goal group (--shuffle-goal-image;
#             needs isg_meta/goal_groups.json)
#   text      no goal image; prompted with the task's text from meta/tasks.jsonl (--text-only)
#
# Recipe = the earlier pi runs' (pi05_b1k, FSDP 4, nothing_saveable remat, cuDNN attention, global batch 256, 16
# workers, 100k steps, save + validate every 5k with val batch 64 x 16, seed 0) with the isg-train224 changes:
# absolute grippers (no --data.gripper-action delta; state q1 + q2 in metres), exact statistics computed before the run
# (padded chunk steps excluded), --data.val-fraction 0.05 (the same episodes as ACT / DP), LR decay over the whole run
# (--lr_schedule.decay_steps = NUM_TRAIN_STEPS), 3 local saves (--max_to_keep 3 --keep_period None), online W&B.
#
# Environment (defaults in brackets):
#   DATASET     LeRobot v3 root (required)      REPO_ID [isg-train224] (names the norm-stats asset)
#   PI_VARIANT  goal | shufgoal | text [goal]   RUN [pi05-isg224-$PI_VARIANT-gbs256-s$SEED]
#   GOAL_SOURCE episode_last | goal_png [episode_last] (goal_png: the dataset's lossless isg_meta/goal_images PNGs)
#   OUT_ROOT    [/tmp/dev/runs/train224]        LOG_ROOT [$OUT_ROOT/logs]   B1K_GOAL_TABLE_DIR [$OUT_ROOT/goal_tables]
#   PYTHON      [.venv-jax/bin/python]          GPU [0,1,2,3]   CPUS [0-$(nproc - 1)]   FSDP [4]
#   NUM_TRAIN_STEPS [100000]   SAVE [5000]   SEED [0]   VAL_FRACTION [0.05]   SETTLE_STEPS [all]
#   TASK_NAMES  comma-separated tasks / task groups [the groups of isg_meta/task_groups.json: every task]
#   WANDB_ENTITY [kmy17518]   WANDB_PROJECT [b1k-isg-train224]   WANDB_MODE [online] (needs WANDB_API_KEY)
#   HF_REPO     [kmy17518/$RUN]   HF_PUBLIC [0]   HF_XET [1]   (needs HF_TOKEN with write access)
#   UPLOAD      [1]; 0 trains without the exporter / upload monitor (smoke tests)   LOG_INTERVAL [100]
set -euo pipefail
OPENPI_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${DATASET:?set DATASET to the LeRobot v3 root}"
SEED=${SEED:-0}
PI_VARIANT=${PI_VARIANT:-goal}
RUN=${RUN:-pi05-isg224-$PI_VARIANT-gbs256-s$SEED}
REPO_ID=${REPO_ID:-isg-train224}
OUT_ROOT=${OUT_ROOT:-/tmp/dev/runs/train224}
LOG_ROOT=${LOG_ROOT:-$OUT_ROOT/logs}
export B1K_GOAL_TABLE_DIR=${B1K_GOAL_TABLE_DIR:-$OUT_ROOT/goal_tables}
PYTHON=${PYTHON:-.venv-jax/bin/python}
# launch.sh preallocates 95% of each GPU for XLA: on GB300 (288 GB) that leaves NCCL ~14 GB, on an 80 GB H100 ~4 GB, and
# FSDP's first all-to-all fails with NCCL 'out of memory' (measured on 8 x H100). Leave 12% on GPUs under 100 GB.
if [[ -z ${XLA_PYTHON_CLIENT_MEM_FRACTION:-} ]]; then
    mib=$( (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true) | awk "NR == 1")
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$( ((${mib:-0} > 0 && ${mib:-0} < 100000)) && echo 0.88 || echo 0.95)
fi
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-100000}
SAVE=${SAVE:-5000}
HF_REPO=${HF_REPO:-kmy17518/$RUN}
CKPT=$OPENPI_DIR/outputs/checkpoints/pi05_b1k/$RUN
DIR=$OUT_ROOT/$RUN
TASK_NAMES=${TASK_NAMES:-$(python3 -c "import json, sys; print(','.join(json.load(open(sys.argv[1]))['groups']))" \
    "$DATASET/isg_meta/task_groups.json")}
# launch.sh passes the dataset (repo id, root, task names, prompt source) itself; `selection` is the rest of the data
# selection, `data` all of it (for the preparation scripts)
selection=(--data.settle-steps="${SETTLE_STEPS:-all}" --data.episode-split=auto --data.val-fraction="${VAL_FRACTION:-0.05}"
           --seed="$SEED")
data=(--data.repo_id="$REPO_ID" --data.dataset-root="$DATASET" --data.task-names ${TASK_NAMES//,/ } "${selection[@]}")
case $PI_VARIANT in
    goal|shufgoal)
        goal=(--data.conditioning-regime image --data.goal-views goal_image_0 --model.goal-image-keys goal_0_rgb
              --model.goal-role-embedding --data.goal-source "${GOAL_SOURCE:-episode_last}")
        [[ $PI_VARIANT == goal ]] || goal+=(--shuffle-goal-image) ;;
    text) goal=(--text-only --data.prompt-source task_description) ;;
    *) echo "PI_VARIANT must be goal, shufgoal or text" >&2; exit 2 ;;
esac
mkdir -p "$DIR" "$LOG_ROOT" "$B1K_GOAL_TABLE_DIR"

prepare() {
    cd "$OPENPI_DIR"
    # Statistics do not depend on the conditioning: every variant of the same data selection shares this asset.
    "$PYTHON" scripts/compute_norm_stats.py pi05_b1k --estimator exact "${data[@]}"
    "$PYTHON" scripts/b1k/prepare_goal_table.py pi05_b1k --exp_name="$RUN" "${data[@]}" "${goal[@]}"
}

upload() {
    cd "$OPENPI_DIR"
    local last=$((NUM_TRAIN_STEPS - 1))  # openpi's final save is step num_train_steps - 1
    local flags=(--run-dir "$DIR/upload" --staging-dir "$DIR/upload-staging" --repo-id "$HF_REPO" --run-id "$RUN"
                 --max-steps "$last" --eval-every "$SAVE" --poll-seconds 60)
    [[ ${HF_PUBLIC:-0} == 1 ]] && flags+=(--public)
    [[ ${HF_XET:-1} == 1 ]] && flags+=(--xet)
    mkdir -p "$DIR/upload/export_queue/eval" "$DIR/upload/export_queue/full" "$DIR/upload-staging"
    # The exporter tars each finalized orbax save (eval: params + assets; full: everything, for resuming) into the
    # queue; the uploader publishes them and deletes the queued tars. Only the newest full tar is queued / kept.
    "$PYTHON" -u scripts/b1k/export_checkpoints.py --checkpoint-dir "$CKPT" --run-dir "$DIR/upload" --poll-seconds 60 \
        --max-steps "$last" >> "$LOG_ROOT/export-$RUN.log" 2>&1 &
    local exporter=$!
    "$PYTHON" -u scripts/b1k/upload_checkpoints.py "${flags[@]}" >> "$LOG_ROOT/upload-$RUN.log" 2>&1 || true
    wait "$exporter" || true
}

train() {
    cd "$OPENPI_DIR"
    [[ -n $(find "outputs/assets/pi05_b1k/$REPO_ID" -name norm_stats.json 2>/dev/null) ]] \
        || { echo "no norm stats under outputs/assets/pi05_b1k/$REPO_ID: run '$0 prepare' first" >&2; exit 1; }
    local flags=()
    if compgen -G "$CKPT/[0-9]*" > /dev/null; then flags+=(--resume); elif [[ -d $CKPT ]]; then flags+=(--overwrite); fi
    local uploader='' rc=0
    if [[ ${UPLOAD:-1} == 1 ]]; then upload & uploader=$!; fi
    echo "[$(date -u +%FT%TZ)] $RUN: training (${flags[*]:-fresh start}); exporter + uploader pid $uploader -> $HF_REPO"
    env EXP_NAME="$RUN" BATCH_SIZE="${BATCH_SIZE:-256}" NUM_WORKERS="${NUM_WORKERS:-16}" NUM_TRAIN_STEPS="$NUM_TRAIN_STEPS" SAVE_INTERVAL="$SAVE" \
        MAX_TO_KEEP=3 GPU="${GPU:-0,1,2,3}" CPUS="${CPUS:-0-$(($(nproc) - 1))}" REPO_ID="$REPO_ID" DATASET_ROOT="$DATASET" \
        TASK_NAMES="$TASK_NAMES" PROMPT_SOURCE=$([[ $PI_VARIANT == text ]] && echo task_description || echo task_name) \
        OPENPI_REMAT_POLICY=nothing_saveable OPENPI_ATTENTION=cudnn WANDB_MODE="${WANDB_MODE:-online}" \
        WANDB_ENTITY="${WANDB_ENTITY:-kmy17518}" WANDB_PROJECT="${WANDB_PROJECT:-b1k-isg-train224}" WANDB_RUN_ID="$RUN" \
        WANDB_DIR="$DIR" WANDB_CONSOLE=off RUN_DIR="$DIR/run" LOG="$LOG_ROOT/$RUN.log" PYTHON="$PYTHON" \
        bash scripts/b1k/launch.sh "${flags[@]}" --fsdp-devices="${FSDP:-4}" "${selection[@]}" "${goal[@]}" \
        --val_log_interval="$SAVE" --val_batch_size=64 --val_num_batches=16 --log_interval="${LOG_INTERVAL:-100}" \
        --lr_schedule.decay_steps="$NUM_TRAIN_STEPS" || rc=$?
    if ((rc == 0)); then
        echo "[$(date -u +%FT%TZ)] $RUN: trainer finished; waiting for the last export / upload"
        [[ -z $uploader ]] || wait "$uploader" || true
    else
        echo "[$(date -u +%FT%TZ)] $RUN: trainer exited with $rc; exporter / uploader (pid $uploader) keep publishing" \
             "what is queued; '$0 train' resumes all three" >&2
    fi
    return "$rc"
}

case ${1:-} in
    prepare) prepare ;;
    train) train ;;
    upload) upload ;;
    all) prepare && train ;;
    *) sed -n '2,16p' "$0"; exit 2 ;;
esac
