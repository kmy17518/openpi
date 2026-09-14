#!/usr/bin/env bash
# Build the "venv-openpi-jax010" virtual environment: openpi on jax 0.10.2 + CUDA 13 (the fastest measured stack on
# GPUs newer than the pinned jax 0.5.3, whose XLA does not know their compute capability -- see docs/b1k.md).
#
#   scripts/b1k/setup_jax_cuda13_venv.sh [VENV_DIR] [--from-freeze | --from-recipe]
#
#   VENV_DIR        where to create the venv; default: ../venv-openpi-jax010 next to this checkout
#   --from-freeze   (default) install exactly the package set frozen in scripts/b1k/venv-openpi-jax010-freeze.txt
#                   (the environment the run single-task-turning-on-radio-bs512 trained in: Python 3.11.15,
#                   jax/jaxlib/jax-cuda13-plugin 0.10.2, flax 0.12.8, orbax-checkpoint 0.12.4, numpy 2.4.6,
#                   torch 2.7.1, transformers 5.5.4, lerobot wensi-ai/lerobot@c43f5811)
#   --from-recipe   re-resolve instead, following the recipe in docs/b1k.md ("A JAX that knows sm_103"): the pinned
#                   .venv must already exist (`GIT_LFS_SKIP_SMUDGE=1 uv sync`), its non-JAX packages are reused
#                   unpinned and the JAX / torch / vision stack is installed at the versions listed below
#
# The name says jax010 because 0.10.2 is the newest jax that runs on this repo's Python 3.11 (jax 0.11 requires
# Python 3.12); there is no jax 0.11 variant. Run anything in the venv with
#     JAXTYPING_DISABLE=1 JAX_COMPILATION_CACHE_DIR=/tmp/.cache/jax-010 $VENV_DIR/bin/python ...
# (flax 0.12 keeps nnx.Variable objects in the optimizer state, which TrainState's jaxtyping annotation rejects; the
# compile cache must not be shared with the pinned jax 0.5.3 venv). Requires `uv` (https://docs.astral.sh/uv/).
set -euo pipefail

OPENPI_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV_DIR=""
MODE=freeze
for arg in "$@"; do
    case "$arg" in
        --from-freeze) MODE=freeze ;;
        --from-recipe) MODE=recipe ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) VENV_DIR=$arg ;;
    esac
done
VENV_DIR=${VENV_DIR:-$(dirname "$OPENPI_DIR")/venv-openpi-jax010}
FREEZE="$OPENPI_DIR/scripts/b1k/venv-openpi-jax010-freeze.txt"

# Source the workspace environment if present: it keeps uv / pip / HF caches under /tmp on this machine.
if [ -f "${ENV_FILE:-/tmp/dev/env.sh}" ]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE:-/tmp/dev/env.sh}"
fi
command -v uv >/dev/null || { echo "uv not found; install it from https://docs.astral.sh/uv/" >&2; exit 1; }
if [ -e "$VENV_DIR" ]; then
    echo "$VENV_DIR already exists; delete it first (rm -rf \"$VENV_DIR\") to rebuild" >&2; exit 1
fi
export GIT_LFS_SKIP_SMUDGE=1   # lerobot is installed from git and carries LFS files that are not needed

echo "==> creating $VENV_DIR (Python 3.11, mode: $MODE) for $OPENPI_DIR"
uv venv --python 3.11 "$VENV_DIR"
PY="$VENV_DIR/bin/python"
# --no-config everywhere: otherwise uv applies this repo's [tool.uv] override-dependencies (ml-dtypes 0.4.1,
# tensorstore 0.1.74), which are incompatible with jax 0.10.
UV_PIP=(uv pip install --no-config --python "$PY")

if [ "$MODE" = freeze ]; then
    echo "==> installing the frozen package set from $FREEZE"
    # lerobot goes in with --no-deps (as in the recipe): its metadata pins setuptools<81 and numpy<2, both of which the
    # frozen environment deliberately violates.
    FROZEN=$(mktemp)
    grep -v '^lerobot' "$FREEZE" > "$FROZEN"
    "${UV_PIP[@]}" -r "$FROZEN"
    "${UV_PIP[@]}" --no-deps "$(grep '^lerobot' "$FREEZE")"
    rm -f "$FROZEN"
else
    [ -x "$OPENPI_DIR/.venv/bin/python" ] || { echo "--from-recipe needs the pinned .venv ($OPENPI_DIR/.venv): run GIT_LFS_SKIP_SMUDGE=1 uv sync first" >&2; exit 1; }
    BASE_NAMES=$(mktemp)
    # everything in the pinned venv except the JAX stack, torch and the nvidia-* wheels, unpinned
    uv pip freeze --python "$OPENPI_DIR/.venv/bin/python" \
        | grep -v -iE '^(jax|jaxlib|jax-cuda|flax|orbax|chex|optax|ml-dtypes|ml_dtypes|tensorstore|numpy|scipy|torch|torchvision|torchcodec|triton|nvidia-|openpi|-e |jaxtyping|equinox|augmax|treescope)' \
        | grep -v '^lerobot' | sed -E 's/==.*//' > "$BASE_NAMES"
    echo "==> installing jax[cuda13]==0.10.2 + flax/orbax and $(wc -l < "$BASE_NAMES") base packages (unpinned)"
    "${UV_PIP[@]}" "jax[cuda13]==0.10.2" "flax==0.12.8" "orbax-checkpoint==0.12.4" \
        "chex==0.1.92" optax "numpy>=2,<3" scipy "jaxtyping==0.2.36" equinox augmax treescope "torch==2.7.1" "torchvision==0.22.1" \
        "transformers==5.5.4" "av==15.1.0" "opencv-python==4.11.0.86" "opencv-python-headless==4.11.0.86" "draccus==0.10.0" \
        numpydantic -r "$BASE_NAMES"
    "${UV_PIP[@]}" --no-deps "lerobot @ git+https://github.com/wensi-ai/lerobot@c43f58116b975ae79af62714e1417b38facd4e37"
    rm -f "$BASE_NAMES"
fi

echo "==> installing openpi (editable) from $OPENPI_DIR"
(cd "$OPENPI_DIR" && "${UV_PIP[@]}" --no-deps -e . -e packages/openpi-client)

echo "==> smoke test (CPU only, does not touch the GPUs)"
JAX_PLATFORMS=cpu JAXTYPING_DISABLE=1 "$PY" - <<'EOF'
import importlib.metadata as md, sys
import jax, flax, orbax.checkpoint as ocp, torch
import openpi.training.config as _config
plugin = md.version("jax-cuda13-plugin")
print(f"python {sys.version.split()[0]}  jax {jax.__version__}  jax-cuda13-plugin {plugin}  flax {flax.__version__}  "
      f"orbax-checkpoint {ocp.__version__}  torch {torch.__version__}  openpi configs: {len(_config._CONFIGS)}")
assert jax.__version__ == plugin == "0.10.2", "unexpected jax version"
_config.get_config("pi05_b1k")
EOF

cat <<EOF

Done: $VENV_DIR
Use it with (see docs/b1k_runs.md):
    JAXTYPING_DISABLE=1 JAX_COMPILATION_CACHE_DIR=/tmp/.cache/jax-010 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \\
        $PY scripts/b1k/train_b1k.py ...
scripts/b1k/runs/<exp>.env should point VENV=$VENV_DIR
EOF
