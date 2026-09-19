#!/usr/bin/env bash
# Create the JAX 0.11 / CUDA 13 training environment (.venv-jax) that scripts/b1k/launch.sh uses by default on
# Blackwell Ultra (B300) hosts. See docs/b1k.md, "GPU step time: JAX 0.11, fused attention, gradient accumulation".
#
#   scripts/b1k/setup_venv_jax.sh [VENV_DIR]        default: <checkout>/.venv-jax
#
# Idempotent: re-running upgrades/fixes an existing venv. Needs `uv`; the interpreter (CPython 3.12) is downloaded
# by uv into $UV_PYTHON_INSTALL_DIR (set by /tmp/dev/env.sh to /tmp/uv/python) if it is not present yet.
#
# What it does, and why each step is shaped the way it is:
#   1. uv venv --python 3.12                        jax >= 0.11 needs Python >= 3.12
#   2. uv pip install of the requirements below (with the overrides below), run from a scratch directory: inside
#      the checkout, uv applies pyproject.toml's [tool.uv] override-dependencies, which pin ml-dtypes and
#      tensorstore to versions jax 0.11 rejects. The overrides also lift lerobot's numpy<2 pin and torch's exact
#      pins of older CUDA libraries. torch/torchvision come from PyTorch's cu130 index.
#   3. uv pip install --no-deps -e . -e packages/openpi-client
#   4. Install a site hook (openpi_cuda_preload.pth) that loads the CUDA 13 libraries JAX was built against before
#      `import torch` can load its own (older) copies: both carry the same sonames (libcudnn.so.9, libcublas.so.13),
#      the first loaded wins for the whole process, and XLA refuses cuDNN < 9.12.
#   5. Sanity check: versions, GPU visible, one matmul, cuDNN attention forward with head_dim 256.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
VENV=${1:-$REPO/.venv-jax}
ENV_FILE=${ENV_FILE:-/tmp/dev/env.sh}
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"  # cache/config redirects under /tmp, UV_PYTHON_INSTALL_DIR, UV_CACHE_DIR
fi
command -v uv > /dev/null || { echo "uv not found on PATH" >&2; exit 1; }

echo "== 1/5 virtualenv at $VENV"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
PY=$VENV/bin/python

echo "== 2/5 dependencies (resolved outside the checkout)"
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/setup-venv-jax.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT

# Direct dependencies of pyproject.toml with the JAX stack moved to versions that know the GPU.
# jax 0.11.2 breaks flax 0.12.9 (jax.experimental.hijax.HiPrimitive); keep the pair in sync when bumping.
cat > "$SCRATCH/requirements.txt" <<'EOF_REQ'
jax[cuda13]==0.11.1
flax==0.12.9
orbax-checkpoint==0.12.4
optax==0.2.8
chex==0.1.92
jaxtyping==0.2.36
ml-dtypes>=0.5
tensorstore>=0.1.84
numpy>=2.1,<2.4
scipy>=1.15

# PyTorch runs the data loader on the CPU only; the CUDA 13 build keeps one set of CUDA libraries in the venv.
torch==2.10.0+cu130
torchvision==0.25.0+cu130
triton==3.6.0

# unchanged direct dependencies of pyproject.toml
augmax>=0.3.4
dm-tree>=0.1.8
einops>=0.8.0
equinox>=0.11.8
flatbuffers>=24.3.25
fsspec[gcs]>=2024.6.0
imageio>=2.36.1
lerobot[dataset] @ git+https://github.com/wensi-ai/lerobot@c43f58116b975ae79af62714e1417b38facd4e37
ml_collections==1.0.0
numpydantic>=1.6.6
opencv-python>=4.10.0.84
pillow>=11.0.0
sentencepiece>=0.2.0
tqdm-loggable>=0.2
typing-extensions>=4.15
tyro>=0.9.5
wandb>=0.22.3
filelock>=3.16.1
beartype==0.19.0
treescope>=0.1.7
transformers==5.5.4
rich>=14.0.0
polars>=1.30.0

# openpi-client runtime and the test suite
websockets>=14
msgpack>=1.0
pytest>=8.3.4
pynvml>=12
EOF_REQ

# `--override`: replaces version constraints declared by other packages.
cat > "$SCRATCH/overrides.txt" <<'EOF_OVR'
# lerobot (release/b1k) pins numpy<2, jax 0.11 needs numpy>=2.1.
numpy>=2.1,<2.4
# pyproject.toml's [tool.uv] override-dependencies pin these for the jax 0.5.3 environment; jax 0.11 needs newer.
ml-dtypes>=0.5
tensorstore>=0.1.84
# torch 2.10.0+cu130 pins exact CUDA library versions (cuBLAS 13.1, cuDNN 9.15, runtime 13.0) that are older than
# what jaxlib 0.11.1 was built against; XLA refuses cuDNN < 9.12 and warns that cuBLAS < 13.2 can corrupt data
# (TMEM double free). torch never uses them here (CPU-only data loading), so install the versions JAX wants.
nvidia-cublas>=13.8
nvidia-cudnn-cu13>=9.26
nvidia-cuda-runtime>=13.4
nvidia-cuda-nvrtc>=13.4
nvidia-cuda-cupti>=13.4
nvidia-nvjitlink>=13.4
nvidia-cufft>=12.1
nvidia-cusolver>=12.1
nvidia-cusparse>=12.7
EOF_OVR

(
    cd "$SCRATCH"
    uv pip install --python "$PY" \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --override overrides.txt \
        -r requirements.txt
)

echo "== 3/5 openpi (editable, no dependency resolution)"
uv pip install --python "$PY" --no-deps -e "$REPO" -e "$REPO/packages/openpi-client"

echo "== 4/5 CUDA library preload hook"
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cat > "$SITE/_openpi_cuda_preload.py" <<'EOF'
"""Load the CUDA 13 libraries JAX/XLA was built against before `import torch` can load the older copies torch depends
on. Both carry the same sonames (libcudnn.so.9, libcublas.so.13, ...): whichever is loaded first serves the whole
process, and XLA refuses cuDNN < 9.12 and warns about cuBLAS < 13.2 (TMEM double-free bug). torch only runs the data
loader on the CPU here, so it never depends on those library versions itself.

Installed by scripts/b1k/setup_venv_jax.sh, activated by openpi_cuda_preload.pth; disable with
OPENPI_NO_CUDA_PRELOAD=1."""
import ctypes
import glob
import os

_ROOT = os.path.join(os.path.dirname(__file__), "nvidia")
_LIBS = [  # dependents after their dependencies; CUDA 13 wheels install under nvidia/cu13/lib, cuDNN under nvidia/cudnn
    "cu13/lib/libcudart.so.*",
    "cu13/lib/libnvrtc.so.*",
    "cu13/lib/libnvJitLink.so.*",
    "cu13/lib/libcublasLt.so.*",
    "cu13/lib/libcublas.so.*",
    "cudnn/lib/libcudnn.so.*",
    "cu13/lib/libcufft.so.*",
    "cu13/lib/libcusparse.so.*",
    "cu13/lib/libcusolver.so.*",
]
if not os.environ.get("OPENPI_NO_CUDA_PRELOAD"):
    for pattern in _LIBS:
        for path in sorted(glob.glob(os.path.join(_ROOT, pattern))):
            if path.count(".so.") == 1 and path.rsplit(".so.", 1)[1].isdigit():  # the soname link, e.g. libcublas.so.13
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
EOF
echo "import _openpi_cuda_preload" > "$SITE/openpi_cuda_preload.pth"
rm -f "$SITE/openpi_cudnn_preload.pth" "$SITE/_openpi_cudnn_preload.py"  # name used by the hand-made first version

echo "== 5/5 sanity check"
cd "$REPO"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
HOME=$SCRATCH "$PY" - <<'EOF'
import torch  # imported first on purpose: this is the order train_b1k.py ends up with (data loader before JAX init)
import jax, jax.numpy as jnp, flax, orbax.checkpoint, optax
from jax._src.lib import cuda_versions as cv

dev = jax.devices()[0]
print(f"jax {jax.__version__}  flax {flax.__version__}  torch {torch.__version__}  device {dev.device_kind}")
print(f"cuDNN {cv.cudnn_get_version()}  cuBLAS {cv.cublas_get_version()}  CUDA runtime {cv.cuda_runtime_get_version()}")
assert cv.cudnn_get_version() >= 91200, "cuDNN < 9.12 loaded: the preload hook did not take effect"
assert cv.cublas_get_version() >= 130200, "cuBLAS < 13.2 loaded (torch's copy): the preload hook did not take effect"
x = jnp.ones((2048, 2048), jnp.bfloat16)
assert float((x @ x)[0, 0]) == 2048.0
q = jax.random.normal(jax.random.key(0), (2, 912, 8, 256), jnp.bfloat16)
k = v = jax.random.normal(jax.random.key(1), (2, 912, 1, 256), jnp.bfloat16)
out = jax.nn.dot_product_attention(q, k, v, implementation="cudnn")
jax.block_until_ready(out)
import openpi.training.config  # noqa: F401  (openpi importable)
print("ok: GPU matmul, cuDNN fused attention (head_dim 256), openpi import")
EOF

echo "done. Use it with:  PYTHON=$(realpath --relative-to="$REPO" "$VENV")/bin/python bash scripts/b1k/launch.sh   (already the default)"
