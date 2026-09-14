import os

import pytest

import openpi.shared.xla_gpu_compat as compat


@pytest.fixture
def capabilities(monkeypatch):
    """Pretend the given compute capabilities are visible instead of probing the CUDA driver."""

    def set_(ccs):
        monkeypatch.setattr(compat, "visible_cuda_compute_capabilities", lambda: list(ccs))

    return set_


def test_visible_cuda_compute_capabilities_probe_does_not_raise():
    # With or without a GPU / CUDA driver this must return a (possibly empty) list of (major, minor) pairs.
    for major, minor in compat.visible_cuda_compute_capabilities():
        assert isinstance(major, int)
        assert isinstance(minor, int)


def test_broken_gpu_appends_flag(monkeypatch, capabilities):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    capabilities([(10, 3)])
    assert compat.configure_xla_flags() == "--xla_gpu_enable_triton_gemm=false"
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=false"


def test_existing_flags_are_kept(monkeypatch, capabilities):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=2")
    capabilities([(10, 3), (10, 3)])
    assert compat.configure_xla_flags() == "--xla_gpu_enable_triton_gemm=false"
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_autotune_level=2 --xla_gpu_enable_triton_gemm=false"


@pytest.mark.parametrize(
    "preset",
    ["--xla_gpu_enable_triton_gemm=true", "--xla_gpu_enable_triton_gemm=false", "--noxla_gpu_enable_triton_gemm"],
)
def test_explicit_user_choice_wins(monkeypatch, capabilities, preset):
    monkeypatch.setenv("XLA_FLAGS", f"--xla_dump_to=/tmp/x {preset}")
    capabilities([(10, 3)])
    assert compat.configure_xla_flags() is None
    assert os.environ["XLA_FLAGS"] == f"--xla_dump_to=/tmp/x {preset}"


@pytest.mark.parametrize("ccs", [[], [(9, 0)], [(10, 0)], [(8, 0), (9, 0)]])
def test_supported_gpus_are_left_alone(monkeypatch, capabilities, ccs):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    capabilities(ccs)
    assert compat.configure_xla_flags() is None
    assert "XLA_FLAGS" not in os.environ
