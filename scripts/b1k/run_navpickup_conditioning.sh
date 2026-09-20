#!/usr/bin/env bash
# pi0.5 conditioning-regime runs on the two-task radio mixture (N / L / I-slot / LI-slot, plus the PI-ROLE switch),
# as thin presets over scripts/b1k/launch.sh (same environment variables: GPU, CPUS, BATCH_SIZE, NUM_TRAIN_STEPS, ...).
#
#   PI_CONDITION=none | language | image | image_language      bash scripts/b1k/run_navpickup_conditioning.sh [flags]
#   PI_ROLE=1   adds --model.goal-role-embedding (PI-ROLE) to the image conditions; default 0 = PI-SLOT only.
#   PI_GOAL_VIEWS  robot-config goal views, default goal_image_0 (head goal); the model's goal slots follow in order.
#
# Dataset: /tmp/dev/datasets/2026-challenge-demos-radio-navpickup-goal (merge of the nav-goal and pickup-goal
# skill-segment datasets; the goal image comes from their observation.goal_rgb.* streams = the episode's last frame).
# Language = the task name (--data.prompt-source task_name) after the fixed scaffold; the image regimes prompt with
# the fixed scaffold only. Norm stats must exist for the merged root first (same repo id / task names):
#   .venv-jax/bin/python scripts/compute_norm_stats.py pi05_b1k --data.repo_id=$REPO_ID \
#       --data.dataset-root=/tmp/dev/datasets/2026-challenge-demos-radio-navpickup-goal
# Unlike the ACT/DP smoke recipes, this preset has NOT been run on a GPU in this workspace (see docs/b1k.md).
set -uo pipefail
CONDITION=${PI_CONDITION:?set PI_CONDITION (none|language|image|image_language)}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export REPO_ID=${REPO_ID:-kmy17518/2026-challenge-demos-radio-navpickup-goal}
export DATASET_ROOT=${DATASET_ROOT:-/tmp/dev/datasets/2026-challenge-demos-radio-navpickup-goal}
export TASK_NAMES=${TASK_NAMES:-turning_on_radio-navigate_to_radio,turning_on_radio-pick_up_radio}
export PROMPT_SOURCE=${PROMPT_SOURCE:-task_name}
GOAL_VIEWS=${PI_GOAL_VIEWS:-goal_image_0}
ROLE=${PI_ROLE:-0}
slots=()
i=0
for _ in ${GOAL_VIEWS//,/ }; do slots+=("goal_${i}_rgb"); i=$((i + 1)); done
case "$CONDITION" in
    none|language) cond_flags=(--data.conditioning-regime "$CONDITION"); tag=$CONDITION ;;
    image|image_language)
        cond_flags=(--data.conditioning-regime "$CONDITION" --data.goal-views ${GOAL_VIEWS//,/ }
                    --model.goal-image-keys "${slots[@]}")
        tag="${CONDITION}-slot"
        if [[ "$ROLE" == 1 ]]; then cond_flags+=(--model.goal-role-embedding); tag="${tag}-role"; fi ;;
    *) printf 'Unknown PI_CONDITION %s\n' "$CONDITION" >&2; exit 2 ;;
esac
export EXP_NAME=${EXP_NAME:-navpickup-pi05-${tag}-bs${BATCH_SIZE:-512}-$(date +%Y%m%d)}
exec bash "$HERE/scripts/b1k/launch.sh" "${cond_flags[@]}" "$@"
