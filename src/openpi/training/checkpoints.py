from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import List, Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.b1k_artifacts as _b1k_artifacts
import openpi.training.b1k_dataset as _b1k_dataset
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            asset_id = data_config.asset_id if isinstance(data_config.asset_id, str) else data_config.asset_id[0]
            _normalize.save(directory / asset_id, norm_stats)
            # Record which text of a LeRobot task the policy was prompted with (task name vs. description, see
            # openpi.training.b1k_dataset) so that serving prompts with the same kind of text.
            if data_config.prompt_from_task:
                _b1k_dataset.save_prompt_source(directory / asset_id, data_config.prompt_source)
            # Record the exact task prompts, inference-relevant model settings (e.g. max_token_len) and the
            # robot/action convention, so serving restores them and mismatched artifacts are rejected.
            if data_config.action_representation is not None:
                if data_config.inference_metadata is None:
                    raise ValueError("BEHAVIOR data loader is missing resolved inference metadata")
                _b1k_artifacts.save_metadata(directory / asset_id, data_config.inference_metadata)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    data_config = data_loader.data_config()
    if data_config.action_representation is not None:
        # Refuse to continue a run under a different data definition than the one that produced the checkpoint.
        restore_step = checkpoint_manager.latest_step() if step is None else step
        asset_id = data_config.asset_id if isinstance(data_config.asset_id, str) else data_config.asset_id[0]
        assets_dir = checkpoint_manager.directory / str(restore_step) / "assets" / asset_id
        metadata = _b1k_artifacts.load_metadata(assets_dir)
        _b1k_artifacts.validate_representation(
            metadata,
            data_config.action_representation,
            allow_legacy_assets=data_config.allow_legacy_assets,
            context="Resume checkpoint",
        )
        if metadata is not None and data_config.inference_metadata is not None:
            for key in ("task_prompts", "prompt_source", "model_type", "model"):
                if metadata.get(key) != data_config.inference_metadata.get(key):
                    raise ValueError(f"Resume checkpoint {key} differs from the current training configuration")
        _b1k_artifacts.validate_norm_stats(_normalize.load(assets_dir), data_config.norm_stats)

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str | List[str]) -> dict[str, _normalize.NormStats] | None:
    if isinstance(asset_id, list):
        # Only load the first asset_id assuming that the datasets are similar
        asset_id = asset_id[0]
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
