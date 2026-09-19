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


def test_cudnn_attention_matches_reference():
    if jax.devices()[0].platform != "gpu":
        pytest.skip("needs a GPU")
    q, k, v, mask = _inputs()
    ref_out, ref_grads = _run("xla", q, k, v, mask)
    try:
        out, grads = _run("cudnn", q, k, v, mask)
    except NotImplementedError as e:
        pytest.skip(f"cudnn attention unavailable: {e}")
    np.testing.assert_allclose(out, ref_out, atol=1e-2 * np.abs(ref_out).max(), rtol=0)
    for name, g, rg in zip(("dq", "dk", "dv"), grads, ref_grads, strict=True):
        np.testing.assert_allclose(g, rg, atol=2e-2 * np.abs(rg).max(), rtol=0, err_msg=name)
