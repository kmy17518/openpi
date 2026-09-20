#!/usr/bin/env bash
# Variant branch goal-image-slot-role: pi0.5 image / image_language conditions with PI_ROLE=1, trained by this checkout's code.
#   PI_CONDITION=image|image_language bash scripts/b1k/run_variant.sh [extra train_b1k.py flags]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PI_ROLE=1 OPENPI_DIR="$HERE"
exec bash "$HERE/scripts/b1k/run_navpickup_conditioning.sh" "$@"
