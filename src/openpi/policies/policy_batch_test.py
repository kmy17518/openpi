import flax.nnx as nnx
import jax
import numpy as np
import pytest

from openpi.policies import policy as _policy


class _NoisePolicyModel(nnx.Module):
    def __init__(self):
        self.action_horizon = 3
        self.action_dim = 2
        self.max_token_len = 1

    def sample_actions(self, rng, observation, *, noise=None):
        if noise is None:
            noise = jax.random.normal(rng, (observation.state.shape[0], self.action_horizon, self.action_dim))
        return noise + observation.state[:, None, : self.action_dim]


def _observation(offset: float) -> dict:
    return {
        "image": {"camera": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"camera": np.ones((), dtype=np.bool_)},
        "state": np.asarray([offset, offset + 1], dtype=np.float32),
    }


def test_infer_batch_preserves_sequential_rng_stream():
    sequential = _policy.Policy(_NoisePolicyModel(), rng=jax.random.key(7))
    batched = _policy.Policy(_NoisePolicyModel(), rng=jax.random.key(7))
    observations = [_observation(1.0), _observation(3.0), _observation(5.0)]

    expected = [sequential.infer(obs) for obs in observations]
    actual = batched.infer_batch(observations)

    for expected_item, actual_item in zip(expected, actual, strict=True):
        np.testing.assert_allclose(actual_item["actions"], expected_item["actions"], rtol=1e-6, atol=1e-6)

    np.testing.assert_array_equal(jax.random.key_data(batched._rng), jax.random.key_data(sequential._rng))  # noqa: SLF001
    # Subsequent identical-shaped calls use exactly the same key and computation.
    np.testing.assert_array_equal(
        batched.infer(_observation(9.0))["actions"],
        sequential.infer(_observation(9.0))["actions"],
    )


@pytest.mark.parametrize("shape", [(3, 2), (1, 3, 2)])
def test_configured_noise_is_preserved_and_broadcast(shape):
    noise = np.full(shape, 100, np.float32)
    batched = _policy.Policy(_NoisePolicyModel(), sample_kwargs={"noise": noise})
    result = batched.infer_batch([_observation(1), _observation(3)])
    for i, item in enumerate(result):
        np.testing.assert_array_equal(item["actions"], np.tile([101 + 2 * i, 102 + 2 * i], (3, 1)))
    single = _policy.Policy(_NoisePolicyModel(), sample_kwargs={"noise": noise})
    np.testing.assert_array_equal(single.infer(_observation(1))["actions"], result[0]["actions"])


def test_per_call_noise_overrides_configuration():
    policy = _policy.Policy(_NoisePolicyModel(), sample_kwargs={"noise": np.full((1, 3, 2), 100, np.float32)})
    result = policy.infer_batch([_observation(1)], noise=np.zeros((3, 2), np.float32))
    np.testing.assert_array_equal(result[0]["actions"], np.tile([1, 2], (3, 1)))


@pytest.mark.parametrize("shape", [(2,), (3, 3, 2), (1, 2, 3)])
def test_invalid_noise_shape_is_rejected(shape):
    policy = _policy.Policy(_NoisePolicyModel(), sample_kwargs={"noise": np.zeros(shape, np.float32)})
    with pytest.raises(ValueError, match="Batched noise has shape"):
        policy.infer_batch([_observation(1), _observation(3)])


def test_generated_noise_uses_exact_sequential_keys():
    policy = _policy.Policy(_NoisePolicyModel(), rng=jax.random.key(7))
    seen = []

    def capture(rng, observation, *, noise):
        seen.append(noise)
        return noise

    policy._sample_actions = capture  # noqa: SLF001
    policy.infer_batch([_observation(1), _observation(3)])
    rng = jax.random.key(7)
    expected = []
    for _ in range(2):
        rng, key = jax.random.split(rng)
        expected.append(jax.random.normal(key, (1, 3, 2)))
    np.testing.assert_array_equal(seen[0], np.concatenate(expected))
    np.testing.assert_array_equal(jax.random.key_data(policy._rng), jax.random.key_data(rng))  # noqa: SLF001


def test_policy_recorder_supports_batched_inference(tmp_path):
    class _BatchPolicy:
        def infer_batch(self, obs_list, *, noise=None):
            return [{"actions": np.asarray([i], dtype=np.float32)} for i, _ in enumerate(obs_list)]

    recorder = _policy.PolicyRecorder(_BatchPolicy(), str(tmp_path))
    results = recorder.infer_batch([{"state": np.asarray([1])}, {"state": np.asarray([2])}])

    assert len(results) == 2
    assert (tmp_path / "step_0.npy").is_file()
    assert (tmp_path / "step_1.npy").is_file()
