"""Make the pinned JAX/XLA run on GPUs that are newer than it is.

jax 0.5.3 (``pyproject.toml``) predates NVIDIA's Blackwell Ultra GPUs (B300, CUDA compute capability 10.3).
Its XLA does not know that compute capability -- it warns ``Unknown compute capability 10.3. Defaulting to telling
LLVM that we're compiling for sm_101`` -- and while ordinary fusions still compile fine that way, XLA's Triton GEMM
emitter keeps generating Blackwell ``tcgen05`` instructions for that ``sm_101`` target, so ptxas rejects them and the
GEMM autotuner aborts the process (exit code 134) at the first jitted matrix multiply::

    F external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1071] Non-OK-status: executable.status()
    Status: INTERNAL: ptxas exited with non-zero error code 65280, output: ...
      error   : Instruction 'tcgen05.alloc' not supported on .target 'sm_101'

Everything else works (LLVM fusions go through ptxas for the real GPU, cuBLAS / cuDNN handle GEMMs and convolutions),
so disabling XLA's Triton GEMM fusions -- ``--xla_gpu_enable_triton_gemm=false``, matrix multiplies then run through
cuBLAS -- is all that is needed. :func:`configure_xla_flags` appends that flag to ``XLA_FLAGS`` when such a GPU is
visible. It must run before JAX creates its GPU backend, i.e. before the first ``jax.devices()`` /
``jax.device_count()`` / jitted call; merely importing jax is fine. An explicit ``--xla_gpu_enable_triton_gemm=...`` in
``XLA_FLAGS`` is left alone, so the behaviour can always be overridden from the shell.
"""

import ctypes
import logging
import os
import shlex

# Compute capabilities whose Triton GEMM fusions do not compile with the XLA bundled in the pinned jaxlib (see module
# docstring). 10.3 = Blackwell Ultra (B300). Extend when another unknown-to-XLA GPU shows the same abort.
TRITON_GEMM_BROKEN_COMPUTE_CAPABILITIES: frozenset[tuple[int, int]] = frozenset({(10, 3)})

TRITON_GEMM_FLAG = "--xla_gpu_enable_triton_gemm"

# CUDA driver API constants (cuda.h).
_CUDA_SUCCESS = 0
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


def visible_cuda_compute_capabilities() -> list[tuple[int, int]]:
    """``(major, minor)`` of every CUDA device visible to this process, in device order.

    Uses the CUDA driver API directly (honours ``CUDA_VISIBLE_DEVICES``) and creates no CUDA context, so it is safe to
    call before JAX or torch initialise and before data-loader workers are forked. Returns ``[]`` when there is no
    usable CUDA driver or no GPU.
    """
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return []
    if cuda.cuInit(0) != _CUDA_SUCCESS:
        return []
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != _CUDA_SUCCESS:
        return []
    capabilities = []
    for ordinal in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        if (
            cuda.cuDeviceGet(ctypes.byref(device), ordinal) == _CUDA_SUCCESS
            and cuda.cuDeviceGetAttribute(ctypes.byref(major), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
            == _CUDA_SUCCESS
            and cuda.cuDeviceGetAttribute(ctypes.byref(minor), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
            == _CUDA_SUCCESS
        ):
            capabilities.append((major.value, minor.value))
    return capabilities


def _mentions_triton_gemm_flag(xla_flags: str) -> bool:
    for flag in shlex.split(xla_flags):
        name = flag.split("=", 1)[0]
        if name in (TRITON_GEMM_FLAG, TRITON_GEMM_FLAG.replace("--", "--no", 1)):
            return True
    return False


def configure_xla_flags() -> str | None:
    """Append the XLA flags the visible GPUs need to ``XLA_FLAGS``; returns what was appended (``None`` if nothing).

    Call once, early, before JAX initialises its backend. No-op without a GPU that needs it, and when ``XLA_FLAGS``
    already sets ``--xla_gpu_enable_triton_gemm`` explicitly.
    """
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if _mentions_triton_gemm_flag(xla_flags):
        return None
    broken = sorted(set(visible_cuda_compute_capabilities()) & TRITON_GEMM_BROKEN_COMPUTE_CAPABILITIES)
    if not broken:
        return None
    flag = f"{TRITON_GEMM_FLAG}=false"
    os.environ["XLA_FLAGS"] = f"{xla_flags} {flag}".strip()
    logging.warning(
        "GPU compute capability %s is unknown to this XLA build (jax %s): disabling its Triton GEMM fusions (%s) so "
        "matrix multiplies run through cuBLAS instead of aborting in the GEMM autotuner. Set %s=true in XLA_FLAGS to "
        "override.",
        ", ".join(f"{major}.{minor}" for major, minor in broken),
        _jax_version(),
        flag,
        TRITON_GEMM_FLAG,
    )
    return flag


def _jax_version() -> str:
    try:
        from importlib.metadata import version

        return version("jax")
    except Exception:  # version is only used for the log line
        return "unknown"
