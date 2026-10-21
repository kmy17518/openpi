#!/usr/bin/env bash
# Launch a goal-conditioned pi0.5 variant (PI_ROLE=0 for PI-SLOT, PI_ROLE=1 for PI-ROLE).
#   PI_CONDITION=image|image_language [PI_ROLE=0|1] bash scripts/b1k/run_variant.sh [extra train_b1k.py flags]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PI_ROLE=${PI_ROLE:-0} OPENPI_DIR="$HERE"
exec bash "$HERE/scripts/b1k/run_navpickup_conditioning.sh" "$@"
