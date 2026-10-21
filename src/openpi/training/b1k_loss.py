import jax.numpy as jnp


def masked_mean(per_step_loss, action_is_pad):
    valid = ~action_is_pad
    return jnp.sum(per_step_loss * valid) / jnp.maximum(jnp.sum(valid), 1)
