"""Fused (flash) attention for the Gemma blocks on GPU.

The reference implementation in `gemma.attention_core` materialises the [B, heads, T, S] logits in f32 and pushes
them through four memory-bound XLA kernels per layer; at batch 512 x 912 tokens that is ~80 GB of HBM traffic per
layer and a third of the training step (see docs/b1k.md, "Profiling").

`cudnn_attention` replaces it with

* forward: cuDNN's fused flash-attention kernel through JAX's cuDNN custom call (head_dim 256 is supported for the
  forward pass on Hopper/Blackwell). The kernel also returns the per-row log-sum-exp, which we keep as the residual.
* backward: XLA's cuDNN backward is limited to head_dim <= 128 (`cuda_dnn.cc`, checked in JAX 0.11 / cuDNN 9.26), so
  the backward pass is written by hand from the standard flash-attention derivation, with the log-sum-exp from the
  forward pass (no softmax reductions) and with the query heads folded into the token dimension so that every
  operand of every matmul keeps the key axis minor: no layout transposes are generated.

The softmax correction term of that derivation, `delta_t = sum_s P_ts dP_ts`, is computed exactly from the (f32)
recomputed probabilities and `dP = dO V^T`, **not** as `rowsum(dO * O)` from the kernel's bf16 output as FlashAttention
does. The two are equal in exact arithmetic, but the bf16 rounding of O leaves a residue of ~2^-9 |dO||v| in `delta`.
For an attention row that has saturated (one dominant key -- the true softmax gradient is then ~0) that residue is the
whole of `dS = P (dP - delta)`, and it is multiplied by k (for dq) and q (for dk). Trained pi05 action-expert rows do
saturate: with the `rowsum(dO * O)` form, the gradient of a run with saturated rows was 10^2-10^7 times too large and
uncorrelated with the XLA reference, which drove the weights further into saturation -- a runaway visible as an
exponential grad-norm ramp (turning-on-radio-4gpu-bs512-clean-20260919, step ~17.8k). XLA's autodiff of the reference
implementation computes `delta` from the f32 probabilities and does not have this problem; neither does this form.

The mask is boolean `[B, 1, T, S]` (True = attend); `q` must already carry the 1/sqrt(head_dim) scale, as in the
Gemma code. Only grouped-query attention with `num_heads % num_kv_heads == 0` is supported, and inputs are bf16.
"""

import einops
import jax
import jax.numpy as jnp

try:  # private JAX API: returns the f32 log-sum-exp; the public wrapper casts it to the output dtype
    from jax._src.cudnn.fused_attention_stablehlo import MaskType as _MaskType
    from jax._src.cudnn.fused_attention_stablehlo import dot_product_attention as _cudnn_dpa
except ImportError:  # pragma: no cover
    _cudnn_dpa = None


def _cudnn_forward(q, k, v, mask):
    """Returns (out [B,T,N,H], lse [B,N,T] f32)."""
    if _cudnn_dpa is not None:
        out, lse = _cudnn_dpa(q, k, v, None, mask, None, None, scale=1.0, mask_type=_MaskType.NO_MASK, return_residual=True)
        return out, lse.astype(jnp.float32)
    out, lse = jax.nn.dot_product_attention(q, k, v, mask=mask, scale=1.0, implementation="cudnn", return_residual=True)
    return out, jnp.transpose(lse, (0, 2, 1)).astype(jnp.float32)


@jax.custom_vjp
def cudnn_attention(q, k, v, mask):
    out, _ = _cudnn_forward(q, k, v, mask)
    return out


def _fwd(q, k, v, mask):
    out, lse = _cudnn_forward(q, k, v, mask)
    return out, (q, k, v, mask, lse)


def _bwd(res, d_out):
    q, k, v, mask, lse = res
    B, T, N, H = q.shape
    K = k.shape[2]
    G = N // K
    dtype = q.dtype

    # Fold the G query heads of each KV head into the token axis: [B, K, T*G, H]. The mask and log-sum-exp are
    # broadcast/reshaped the same way, so all matmuls below are plain batched GEMMs with S as the minor dimension.
    qf = einops.rearrange(q, "B T (K G) H -> B K (T G) H", K=K)
    dof = einops.rearrange(d_out, "B T (K G) H -> B K (T G) H", K=K)
    kf = einops.rearrange(k, "B S K H -> B K S H")
    vf = einops.rearrange(v, "B S K H -> B K S H")
    lse_f = einops.rearrange(lse, "B (K G) T -> B K (T G)", K=K)  # [B, K, T*G]
    mask_f = einops.repeat(mask, "B 1 T S -> B 1 (T G) S", G=G)  # [B, 1, T*G, S]

    scores = jnp.einsum("bkth,bksh->bkts", qf, kf, preferred_element_type=jnp.float32)  # [B, K, T*G, S]
    probs = jnp.where(mask_f, jnp.exp(scores - lse_f[..., None]), 0.0)
    d_probs = jnp.einsum("bkth,bksh->bkts", dof, vf, preferred_element_type=jnp.float32)
    # Exact softmax correction sum_s P dP in f32 (see the module docstring: rowsum(dO * O) from the bf16 output is
    # off by the rounding of O, which for saturated rows *is* the gradient and blows up with |k|, |q|).
    delta = jnp.sum(probs * d_probs, axis=-1)  # [B, K, T*G]
    d_scores = (probs * (d_probs - delta[..., None])).astype(dtype)
    probs = probs.astype(dtype)

    dq = jnp.einsum("bkts,bksh->bkth", d_scores, kf, preferred_element_type=jnp.float32).astype(dtype)
    dk = jnp.einsum("bkts,bkth->bksh", d_scores, qf, preferred_element_type=jnp.float32).astype(dtype)
    dv = jnp.einsum("bkts,bkth->bksh", probs, dof, preferred_element_type=jnp.float32).astype(dtype)

    dq = einops.rearrange(dq, "B K (T G) H -> B T (K G) H", G=G)
    dk = einops.rearrange(dk, "B K S H -> B S K H")
    dv = einops.rearrange(dv, "B K S H -> B S K H")
    return dq, dk, dv, None


cudnn_attention.defvjp(_fwd, _bwd)
