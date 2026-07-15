from collections.abc import Sequence
import inspect
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions_accepts_noise = "noise" in inspect.signature(model.sample_actions).parameters
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = jax.random.key(0) if rng is None else rng

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_batch(self, obs_list: list[dict], *, noise: np.ndarray | None = None) -> list[dict]:
        """Batched variant of `infer`.

        Runs a SINGLE model forward over `len(obs_list)` observations instead of one forward per
        observation. The input/output transforms assume a single, unbatched example, so they are
        applied per-example; only the (expensive) model call is batched. Returns a list of per-example
        output dicts aligned with `obs_list`. All observations must share the same structure/shapes.
        """
        if not obs_list:
            return []
        if (
            not self._is_pytorch_model
            and not self._sample_actions_accepts_noise
            and float(self._sample_kwargs.get("temperature", 0.0)) > 0.0
        ):
            # A stochastic autoregressive model accepts only one RNG key for the whole batch. Running it
            # batched would change the original one-key-per-example random stream, so retain the reference path.
            return [self.infer(obs) for obs in obs_list]

        # Apply the single-example input transforms to each observation, then stack into one batch.
        transformed = [self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in obs_list]
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0), *transformed)
            sample_rngs = []
            for _ in obs_list:
                self._rng, sample_rng = jax.random.split(self._rng)
                sample_rngs.append(sample_rng)
            sample_rng_or_pytorch_device = sample_rngs[-1]
        else:
            inputs = jax.tree.map(
                lambda *xs: torch.from_numpy(np.stack([np.asarray(x) for x in xs], axis=0)).to(self._pytorch_device),
                *transformed,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions. If provided, noise is expected to already be batched
        # as (batch, action_horizon, action_dim).
        sample_kwargs = dict(self._sample_kwargs)
        if noise is None and not self._is_pytorch_model and self._sample_actions_accepts_noise:
            # Match N sequential infer() calls exactly: each example gets the key it would have received
            # independently. A single batch RNG changes both this batch and the RNG stream of later calls.
            noise = jnp.concatenate(
                [
                    jax.random.normal(key, (1, self._model.action_horizon, self._model.action_dim))
                    for key in sample_rngs
                ],
                axis=0,
            )
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2 and len(obs_list) == 1:
                noise = noise[None, ...]
            if noise.shape[0] != len(obs_list):
                raise ValueError(f"Batched noise has batch size {noise.shape[0]}, expected {len(obs_list)}")
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        batched_outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            batched_outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), batched_outputs)
        else:
            batched_outputs = jax.tree.map(np.asarray, batched_outputs)

        # Split the batch back into per-example dicts and apply the single-example output transforms.
        results = []
        for i in range(len(obs_list)):
            example = jax.tree.map(lambda x, index=i: x[index], batched_outputs)
            example = self._output_transform(example)
            example["policy_timing"] = {"infer_ms": model_time * 1000}
            results.append(example)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        self._record(obs, results)
        return results

    def infer_batch(self, obs_list: list[dict], *, noise: np.ndarray | None = None) -> list[dict]:
        results = self._policy.infer_batch(obs_list, noise=noise)
        for obs, result in zip(obs_list, results, strict=True):
            self._record(obs, result)
        return results

    def _record(self, obs: dict, results: dict) -> None:
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
