#!/usr/bin/env bash
# Single-GPU pi0.5 BEHAVIOR-1K training launch for this checkout (branch my-clean), with the environment a
# Blackwell Ultra (B300) / aarch64 host needs. See docs/b1k.md, sections "Fine-tune" and "Blackwell Ultra (B300)
# and ARM (aarch64) hosts".
#
#   scripts/b1k/launch.sh [extra train_b1k.py flags, e.g. --resume]
#
# Everything is overridable through environment variables (defaults in brackets):
#   EXP_NAME       experiment / checkpoint name          [turning-on-radio-1gpu-bs${BATCH_SIZE}-clean-<today>]
#   BATCH_SIZE     [512]        NUM_WORKERS [16]          NUM_TRAIN_STEPS [300000]   MAX_TOKEN_LEN [112]
#   GPU            CUDA_VISIBLE_DEVICES [1]               CPUS  taskset core list     [30-59]
#   TASK_NAMES     [turning_on_radio]                     PROMPT_SOURCE               [task_name]
#   REPO_ID        [behavior-1k/2026-challenge-demos]     DATASET_ROOT [/tmp/dev/datasets/2026-challenge-demos]
#   WANDB_PROJECT  [b1k-challenge-2026-pi]                WANDB_RUN_ID (optional; W&B generates one if unset)
#   WANDB_ENTITY   [kmy17518]                             WANDB_MODE   [online]
#   WANDB_SERVER   [https://api.wandb.ai]  (exported as WANDB_BASE_URL; this host's /etc/environment points elsewhere)
#   RUN_DIR        scratch dir for this run               [/tmp/dev/runs/$EXP_NAME]
#   LOG            training log                          [/tmp/dev/logs/$EXP_NAME.train.log]
#   ENV_FILE       sourced first if it exists            [/tmp/dev/env.sh]
#
# The 2026-09-17 reference run (300k steps, GPU 1, CPUs 30-59) used BATCH_SIZE=576 NUM_WORKERS=8
# EXP_NAME=turning-on-radio-1gpu-bs576-300k-clean-20260917 WANDB_RUN_ID=piradio18clean, mirroring the `my`
# branch's turning-on-radio-1gpu-bs576-300k-20260916 run. Since the data-loader rewrite, batch 512 with 16 workers
# trains GPU-bound at 9.1 s/step on one B300 with trainer and workers confined to 30 cores.
#
# Run it detached:  tmux new-session -d -s pi-radio 'bash scripts/b1k/launch.sh; exec bash'
set -uo pipefail

ENV_FILE=${ENV_FILE:-/tmp/dev/env.sh}
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_DIR=${OPENPI_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
BATCH_SIZE=${BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-16}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-300000}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-112}
GPU=${GPU:-1}
CPUS=${CPUS:-30-59}
TASK_NAMES=${TASK_NAMES:-turning_on_radio}
PROMPT_SOURCE=${PROMPT_SOURCE:-task_name}
REPO_ID=${REPO_ID:-behavior-1k/2026-challenge-demos}
DATASET_ROOT=${DATASET_ROOT:-/tmp/dev/datasets/2026-challenge-demos}
EXP_NAME=${EXP_NAME:-turning-on-radio-1gpu-bs${BATCH_SIZE}-clean-$(date +%Y%m%d)}
RUN_DIR=${RUN_DIR:-/tmp/dev/runs/$EXP_NAME}
LOG=${LOG:-/tmp/dev/logs/$EXP_NAME.train.log}
mkdir -p "$RUN_DIR" "$(dirname "$LOG")"

# train_b1k.py at this commit writes the JAX compilation cache to ~/.cache/jax unconditionally
# (JAX_COMPILATION_CACHE_DIR is honoured only from `my` 5687c79 on). Point HOME at a scratch directory whose
# .cache/jax is the shared cache, so nothing is written under the real home directory.
JAX_CACHE=${JAX_COMPILATION_CACHE_DIR:-/tmp/.cache/jax}
mkdir -p "$JAX_CACHE" "$RUN_DIR/home/.cache"
ln -sfn "$JAX_CACHE" "$RUN_DIR/home/.cache/jax"
export HOME=$RUN_DIR/home

export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=${WANDB_MODE:-online} WANDB_ENTITY=${WANDB_ENTITY:-kmy17518} WANDB_BASE_URL=${WANDB_SERVER:-https://api.wandb.ai}
[ -n "${WANDB_RUN_ID:-}" ] && export WANDB_RUN_ID
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
# jax 0.5.3's XLA does not know the GB300 (compute capability 10.3): its Triton GEMM autotuner aborts the process at
# the first matmul, so route matmuls through cuBLAS. Harmless on GPUs XLA knows. (`my` sets this automatically.)
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

cd "$OPENPI_DIR" || exit 1
echo "[$(date '+%F %T')] launching $EXP_NAME (batch $BATCH_SIZE, $NUM_WORKERS workers, GPU $GPU, CPUs $CPUS, W&B run id ${WANDB_RUN_ID:-<auto>}) from $OPENPI_DIR @ $(git rev-parse --short HEAD)" | tee -a "$LOG"
# shellcheck disable=SC2086
taskset -c "$CPUS" .venv/bin/python -u scripts/b1k/train_b1k.py pi05_b1k \
    --exp_name="$EXP_NAME" \
    --project_name="${WANDB_PROJECT:-b1k-challenge-2026-pi}" \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --num_train_steps="$NUM_TRAIN_STEPS" \
    --save_interval=2500 \
    --keep_period None \
    --log_interval=10 \
    --model.max-token-len "$MAX_TOKEN_LEN" \
    --data.repo_id="$REPO_ID" \
    --data.dataset-root="$DATASET_ROOT" \
    --data.task-names ${TASK_NAMES//,/ } \
    --data.prompt-source "$PROMPT_SOURCE" \
    "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] trainer exited with code $rc" | tee -a "$LOG"
exit "$rc"
