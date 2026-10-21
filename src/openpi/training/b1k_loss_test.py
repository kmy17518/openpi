import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training import b1k_loss


def test_masked_mean_averages_the_steps_inside_the_episode():
    per_step_loss = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    is_pad = jnp.array([[False, False, True, True], [False, True, True, True]])
    assert float(b1k_loss.masked_mean(per_step_loss, is_pad)) == pytest.approx((1 + 2 + 5) / 3)
    # padded steps do not count, whatever their loss, and get no gradient
    assert float(b1k_loss.masked_mean(per_step_loss + 100.0 * is_pad, is_pad)) == pytest.approx(8 / 3)
    grads = jax.grad(lambda loss: b1k_loss.masked_mean(loss, is_pad))(per_step_loss)
    np.testing.assert_allclose(grads, np.where(is_pad, 0.0, 1 / 3), rtol=1e-6)
    # an all-False mask (dataset without one) is the plain mean; nothing to average when every step is padded
    assert float(b1k_loss.masked_mean(per_step_loss, jnp.zeros_like(is_pad))) == pytest.approx(4.5)
    assert float(b1k_loss.masked_mean(per_step_loss, jnp.ones_like(is_pad))) == 0.0
