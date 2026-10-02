"""The cuDNN attention back end of `gemma.attention_core` must match the reference XLA implementation.

Run on a GPU with cuDNN >= 9.11 (skipped otherwise):
    uv run pytest src/openpi/models/gemma_attention_test.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma


def _block_mask(batch, prefix, suffix, valid_prefix):
    """pi0-style mask: prefix attends to (unpadded) prefix, suffix attends to unpadded prefix + all of suffix."""
    total = prefix + suffix
    key_idx = jnp.arange(total)[None, :]
    key_valid = (key_idx < valid_prefix[:, None]) | (key_idx >= prefix)  # [B, S]
    query_is_prefix = jnp.arange(total) < prefix  # [T]
    key_is_suffix = jnp.arange(total) >= prefix  # [S]
    prefix_cannot_see_suffix = ~(query_is_prefix[:, None] & key_is_suffix[None, :])  # [T, S]
    return (key_valid[:, None, :] & prefix_cannot_see_suffix[None])[:, None]  # [B, 1, T, S]


def _inputs(batch=4, prefix=880, suffix=32, heads=8, kv_heads=1, head_dim=256):
    total = prefix + suffix
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(k1, (batch, total, heads, head_dim), jnp.bfloat16) * head_dim**-0.5
    k = jax.random.normal(k2, (batch, total, kv_heads, head_dim), jnp.bfloat16)
    v = jax.random.normal(k3, (batch, total, kv_heads, head_dim), jnp.bfloat16)
    valid_prefix = jnp.array([880, 840, 800, 700][:batch])
    return q, k, v, _block_mask(batch, prefix, suffix, valid_prefix)


def _run(impl, q, k, v, mask):
    def loss(q, k, v):
        out = gemma.attention_core(q, k, v, mask, num_kv_heads=k.shape[2])
        return jnp.sum(out.astype(jnp.float32) * jnp.arange(out.shape[-1]) / out.shape[-1]), out

    os.environ["OPENPI_ATTENTION"] = impl
    try:
        (_, out), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(q, k, v)
    finally:
        os.environ["OPENPI_ATTENTION"] = "xla"
    return np.asarray(out, np.float32), [np.asarray(g, np.float32) for g in grads]


def _cudnn_or_skip(q, k, v, mask):
    if jax.devices()[0].platform != "gpu":
        pytest.skip("needs a GPU")
    try:
        return _run("cudnn", q, k, v, mask)
    except NotImplementedError as e:
        pytest.skip(f"cudnn attention unavailable: {e}")


def test_cudnn_attention_matches_reference():
    q, k, v, mask = _inputs()
    ref_out, ref_grads = _run("xla", q, k, v, mask)
    out, grads = _cudnn_or_skip(q, k, v, mask)
    np.testing.assert_allclose(out, ref_out, atol=1e-2 * np.abs(ref_out).max(), rtol=0)
    for name, g, rg in zip(("dq", "dk", "dv"), grads, ref_grads, strict=True):
        np.testing.assert_allclose(g, rg, atol=2e-2 * np.abs(rg).max(), rtol=0, err_msg=name)


@pytest.mark.parametrize("scale", [16, 256, 4096])
def test_cudnn_attention_backward_with_saturated_rows(scale):
    """Trained pi05 action-expert rows saturate (one dominant key, logits in the 10^2..10^8 range); the backward must
    then agree with the reference instead of emitting the bf16 rounding residue of the output times |k| (the bug
    behind the turning-on-radio-4gpu-bs512-clean-20260919 divergence). Only the 32 suffix query rows are scaled, and
    k carries a large norm so that any spurious dS is visibly amplified."""
    q, k, v, mask = _inputs()
    q = q.at[:, 880:].multiply(scale)
    k = (k.astype(jnp.float32) * 64).astype(jnp.bfloat16)
    ref_out, ref_grads = _run("xla", q, k, v, mask)
    out, grads = _cudnn_or_skip(q, k, v, mask)
    np.testing.assert_allclose(out, ref_out, atol=1e-2 * np.abs(ref_out).max(), rtol=0)
    for name, g, rg in zip(("dq", "dk", "dv"), grads, ref_grads, strict=True):
        # Tolerance relative to the reference gradient's own scale: a wrong `delta` gives errors of 10^2-10^5 x.
        np.testing.assert_allclose(g, rg, atol=3e-2 * np.abs(rg).max(), rtol=0, err_msg=f"{name} at scale {scale}")


@pytest.mark.parametrize("gap", [6.0, 9.0, 13.0])
def test_cudnn_attention_backward_saturated_row_with_runner_up(gap):
    """The regime that broke training: a suffix row whose softmax is saturated on key 0 with a runner-up key 1 at
    probability e^-gap (2e-3 .. 2e-6) and large k / v. The true dq of such a row is O(e^-gap |dO| |v| |k|), i.e. tiny,
    while `delta = rowsum(dO * O)` with a bf16 O is off by ~2^-9 |dO| |v| independently of the gap -- multiplied by |k|
    that residue exceeded the true gradient by 10^2-10^5 x in trained pi05 checkpoints."""
    q, k, v, mask = _inputs()
    total, head_dim = q.shape[1], q.shape[-1]
    big, kappa = 400.0, 512.0
    unit = jnp.zeros((head_dim,), jnp.float32).at[0].set(1.0)
    # Suffix queries point along dim 0; the first two *suffix* keys (invisible to prefix rows, whose gradients thus
    # stay ordinary and set the scale below) are aligned with it at norms kappa and slightly less, giving logits `big`
    # and `big - gap`; every other key keeps its N(0, 1) entries (logits ~ N(0, 1)).
    q = q.at[:, 880:].set(jnp.broadcast_to(unit * (big / kappa), (q.shape[0], total - 880, q.shape[2], head_dim)).astype(jnp.bfloat16))
    k32 = k.astype(jnp.float32)
    k32 = k32.at[:, 880, 0].set(unit * kappa).at[:, 881, 0].set(unit * (kappa - gap * kappa / big))
    k = k32.astype(jnp.bfloat16)
    v = (v.astype(jnp.float32) * 256).astype(jnp.bfloat16)
    ref_out, ref_grads = _run("xla", q, k, v, mask)
    out, grads = _cudnn_or_skip(q, k, v, mask)
    np.testing.assert_allclose(out, ref_out, atol=1e-2 * np.abs(ref_out).max(), rtol=0)
    dq, dk, dv = grads
    ref_dq, ref_dk, ref_dv = ref_grads
    # Suffix-row dq must be as small as the reference says it is, measured against the ordinary prefix-row gradient.
    prefix_scale = np.abs(ref_dq[:, :880]).max()
    assert np.abs(ref_dq[:, 880:]).max() <= 0.1 * prefix_scale, "test construction: suffix rows are not saturated"
    assert np.abs(dq[:, 880:] - ref_dq[:, 880:]).max() <= 2e-2 * prefix_scale, f"spurious dq on saturated rows (gap {gap})"
    # The huge keys 0/1 also saturate some prefix rows, where both implementations show ordinary bf16 noise on a few
    # elements; compare those in norm.
    for name, g, rg in zip(("dq", "dk", "dv"), (dq, dk, dv), (ref_dq, ref_dk, ref_dv), strict=True):
        assert np.linalg.norm(g - rg) <= 5e-2 * np.linalg.norm(rg), f"{name} differs from the reference (gap {gap})"
