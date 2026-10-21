from collections.abc import Callable
import dataclasses
import functools
import importlib.util
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.policies.b1k_policy as _b1k_policy
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.b1k_figures as _b1k_figures
import openpi.training.b1k_loss as _b1k_loss
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

# Validation draws its flow-matching noise and times from this offset of --seed, apart from the training RNG.
VALIDATION_SEED_OFFSET = 1000
VALIDATION_FIGURE_SAMPLES = 4


def _validation_data(config: _config.TrainConfig, data_config: _config.DataConfig) -> tuple[_config.DataConfig, str]:
    """(data config, episode subset) of the validation frames, with the training norm stats: the training data's
    held-out episodes (--val-held-out), the episodes of --val-repo-id, or the --data.val-fraction episodes."""
    if config.val_held_out:
        return data_config, "held_out"
    if config.val_repo_id is None:
        if data_config.val_fraction:
            return data_config, "val"
        raise ValueError("--val-log-interval needs --data.val-fraction, --val-held-out or --val-repo-id")
    factory = dataclasses.replace(config.data, repo_id=config.val_repo_id, val_fraction=0.0)
    val_data_config = dataclasses.replace(
        factory.create(config.assets_dirs, config.model),
        norm_stats=data_config.norm_stats,
        norm_stats_metadata=data_config.norm_stats_metadata,
    )
    if config.val_episodes_index is not None:
        val_data_config = dataclasses.replace(
            val_data_config, dataset_kwargs={**val_data_config.dataset_kwargs, "episodes": config.val_episodes_index}
        )
    return val_data_config, "train"


def create_validation(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    mesh: jax.sharding.Mesh,
    train_state_sharding: Any,
    data_sharding: jax.sharding.NamedSharding,
    replicated_sharding: jax.sharding.NamedSharding,
) -> Callable[[training_utils.TrainState], dict[str, Any]]:
    """Validation loss and action figures, set up once: the same seeded frames spread over every validation episode
    and the same noise and flow-matching times at each evaluation, the masked loss of the params a checkpoint serves
    (the EMA params when training keeps them), batches sharded like the training data. The `val/actions/sample_<i>`
    figures show the predicted and ground-truth action chunks of fixed frames of different validation episodes in
    action units."""
    val_data_config, subset = _validation_data(config, data_config)
    batch_size = config.val_batch_size or config.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f"Validation batch size {batch_size} must be divisible by {jax.device_count()} devices")
    num_samples = VALIDATION_FIGURE_SAMPLES
    if importlib.util.find_spec("matplotlib") is None:
        logging.warning("matplotlib is not installed: no val/actions figures")
        num_samples = 0
    batches, samples, description = _data_loader.load_b1k_fixed_batches(
        val_data_config,
        config.model,
        subset=subset,
        batch_size=batch_size,
        num_batches=config.val_num_batches,
        seed=config.seed,
        num_workers=config.num_workers,
        num_samples=num_samples,
    )
    logging.info("Validation on %s", description)

    def validation_step(state: training_utils.TrainState, batch, rng: at.KeyArrayLike):
        params = state.params if state.ema_params is None else state.ema_params
        model = nnx.merge(state.model_def, params)
        model.eval()
        observation, actions, action_is_pad = batch
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
        return _b1k_loss.masked_mean(chunked_loss, action_is_pad)

    def sample_step(state: training_utils.TrainState, observation: _model.Observation, rng: at.KeyArrayLike):
        params = state.params if state.ema_params is None else state.ema_params
        model = nnx.merge(state.model_def, params)
        model.eval()
        return model.sample_actions(rng, observation)

    pvalidation_step = jax.jit(
        validation_step,
        in_shardings=(train_state_sharding, data_sharding, replicated_sharding),
        out_shardings=replicated_sharding,
    )
    # The few figure frames need not divide the device count, so they are replicated.
    psample_step = jax.jit(
        sample_step,
        in_shardings=(train_state_sharding, replicated_sharding, replicated_sharding),
        out_shardings=replicated_sharding,
    )
    val_rng = jax.random.key(config.seed + VALIDATION_SEED_OFFSET)
    rngs = [jax.random.fold_in(val_rng, i) for i in range(len(batches))]
    sample_rng = jax.random.fold_in(val_rng, len(batches))
    # Prediction and ground truth in action units, as served but without the gripper snap.
    to_action_units = _transforms.compose(
        [
            _transforms.Unnormalize(val_data_config.norm_stats, use_quantiles=val_data_config.use_quantile_norm),
            *(
                dataclasses.replace(transform, gripper_snap_threshold=None)
                if isinstance(transform, _b1k_policy.B1KOutputs)
                else transform
                for transform in val_data_config.data_transforms.outputs
            ),
        ]
    )

    def action_figures(state: training_utils.TrainState) -> dict[str, wandb.Image]:
        batch, titles = samples
        arrays = jax.tree.map(lambda x: jax.make_array_from_process_local_data(replicated_sharding, x), batch)
        with sharding.set_mesh(mesh):
            predicted = np.asarray(
                jax.device_get(psample_step(state, _model.Observation.from_dict(arrays), sample_rng)), np.float32
            )
        is_pad = batch.get("action_is_pad", np.zeros(batch["actions"].shape[:2], bool))
        figures = {}
        for i, title in enumerate(titles):
            state_i = np.asarray(batch["state"][i])
            prediction = to_action_units({"state": state_i, "actions": np.array(predicted[i])})["actions"]
            target = to_action_units({"state": state_i, "actions": np.array(batch["actions"][i])})["actions"]
            image = _b1k_figures.figure_image(
                _b1k_figures.action_chunk_figure(prediction, target, ~np.asarray(is_pad[i]), title)
            )
            figures[f"val/actions/sample_{i}"] = wandb.Image(image)
        return figures

    def validate(state: training_utils.TrainState) -> dict[str, Any]:
        losses = []
        for batch, rng in zip(batches, rngs, strict=True):
            arrays = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), batch)
            is_pad = arrays.get("action_is_pad")
            if is_pad is None:
                is_pad = jnp.zeros(arrays["actions"].shape[:2], bool, device=data_sharding)
            inputs = (_model.Observation.from_dict(arrays), arrays["actions"], is_pad)
            with sharding.set_mesh(mesh):
                losses.append(pvalidation_step(state, inputs, rng))
        info: dict[str, Any] = {"val/loss": float(np.mean(jax.device_get(losses)))}
        if samples is not None:
            info.update(action_figures(state))
        return info

    return validate


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            group="openpi"
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, at.Bool[at.Array, "*b ah"]],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        action_is_pad: at.Bool[at.Array, "*b ah"],
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return _b1k_loss.masked_mean(chunked_loss, action_is_pad)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, action_is_pad = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions, action_is_pad)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "valid_action_fraction": jnp.mean(jnp.logical_not(action_is_pad)),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_b1k_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        skip_norm_stats=False
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    validate = None
    if config.val_log_interval:
        validate = create_validation(
            config, data_loader.data_config(), mesh, train_state_sharding, data_sharding, replicated_sharding
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        if validate is not None and step % config.val_log_interval == 0:
            val_start = time.perf_counter()
            val_info = validate(train_state)
            val_info["val/seconds"] = time.perf_counter() - val_start
            pbar.write(
                f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in val_info.items() if isinstance(v, float))
            )
            wandb.log(val_info, step=step)
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
