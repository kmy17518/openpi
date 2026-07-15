import flax.nnx as nnx
import jax
import numpy as np

from openpi.policies import policy as _policy


class _NoisePolicyModel(nnx.Module):
    def __init__(self):
        self.action_horizon = 3
        self.action_dim = 2
        self.max_token_len = 1

    def sample_actions(self, rng, observation, *, noise=None):
        if noise is None:
            noise = jax.random.normal(
                rng, (observation.state.shape[0], self.action_horizon, self.action_dim)
            )
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
        np.testing.assert_array_equal(actual_item["actions"], expected_item["actions"])

    # Consuming a batch of three must advance the key exactly as three sequential calls do.
    np.testing.assert_array_equal(
        batched.infer(_observation(9.0))["actions"],
        sequential.infer(_observation(9.0))["actions"],
    )


def test_policy_recorder_supports_batched_inference(tmp_path):
    class _BatchPolicy:
        def infer_batch(self, obs_list, *, noise=None):
            return [{"actions": np.asarray([i], dtype=np.float32)} for i, _ in enumerate(obs_list)]

    recorder = _policy.PolicyRecorder(_BatchPolicy(), str(tmp_path))
    results = recorder.infer_batch([{"state": np.asarray([1])}, {"state": np.asarray([2])}])

    assert len(results) == 2
    assert (tmp_path / "step_0.npy").is_file()
    assert (tmp_path / "step_1.npy").is_file()
