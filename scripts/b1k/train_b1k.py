import dataclasses
import functools
import logging
import platform
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
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.xla_gpu_compat as _xla_gpu_compat
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def _extract_training_norm_stats(train_data_loader: _data_loader.DataLoader | None) -> dict | None:
    """Extract normalization stats from the training data loader."""
    if train_data_loader is not None and hasattr(train_data_loader, "data_config"):
        training_data_config = train_data_loader.data_config()
        if hasattr(training_data_config, "norm_stats") and training_data_config.norm_stats is not None:
            return training_data_config.norm_stats
    return None


def _prepare_validation_config(
    config: _config.TrainConfig, training_norm_stats: dict | None
) -> tuple[_config.TrainConfig, str, bool]:
    """
    Prepare validation configuration with training norm_stats.

    Returns:
        tuple: (validation_config, repo_id, use_norm_stats, actual_val_data_config)
    """
    val_config = dataclasses.replace(
        config,
        batch_size=config.val_batch_size or config.batch_size,
    )

    use_norm_stats = False
    if config.val_repo_id or hasattr(val_config.data, "repo_id"):
        repo_id = config.val_repo_id or (getattr(val_config.data, "repo_id", None) + "-val")
        episodes_index = config.val_episodes_index

        # Create validation data config by copying the training data config but changing repo_id
        val_base_config = dataclasses.replace(val_config.data.base_config, episodes_index=episodes_index)
        val_data_config = dataclasses.replace(val_config.data, repo_id=repo_id, base_config=val_base_config)
        val_config = dataclasses.replace(val_config, data=val_data_config)

        # Create the actual DataConfig using the factory and add norm_stats
        actual_val_data_config = val_config.data.create(val_config.assets_dirs, val_config.model)

        # Explicitly use norm_stats from training data loader for validation
        if training_norm_stats is not None:
            actual_val_data_config = dataclasses.replace(actual_val_data_config, norm_stats=training_norm_stats)
            logging.info("Copied norm_stats from training data loader to validation config")
            logging.info("Training norm_stats keys: %s", list(training_norm_stats.keys()))
            use_norm_stats = True
        else:
            logging.warning("No norm_stats found in training data loader - skipping normalization for validation")

        return val_config, repo_id, use_norm_stats, actual_val_data_config

    raise ValueError("No validation repository ID could be determined")



def _create_validation_data_loader(
    actual_val_data_config,
    val_config: _config.TrainConfig,
    *,
    use_norm_stats: bool,
    replicated_sharding: jax.sharding.NamedSharding,
) -> _data_loader.DataLoader:
    """Create validation data loader with custom wrapper for training norm_stats."""

    # Custom wrapper class to bridge our modified DataConfig with the expected DataLoader interface
    class ValidationDataLoader(_data_loader.DataLoader):
        """
        Custom data loader wrapper for validation that preserves training normalization stats.

        This class is necessary because:
        1. Validation datasets typically don't have their own norm_stats.json files
        2. We need to use the training dataset's normalization stats for consistent validation
        3. The standard create_data_loader() function expects a TrainConfig, but we have a custom DataConfig
        """

        def __init__(self, data_config, torch_data_loader):
            self._data_config = data_config
            self._torch_data_loader = torch_data_loader

        def data_config(self):
            return self._data_config

        def __iter__(self):
            for batch in self._torch_data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    val_dataset = _data_loader.create_b1k_dataset(actual_val_data_config, val_config.model.action_horizon)
    logging.info(f"Validation dataset created for {actual_val_data_config.repo_id}")
    val_dataset = _data_loader.transform_dataset(
        val_dataset, actual_val_data_config, skip_norm_stats=not use_norm_stats
    )

    val_torch_data_loader = _data_loader.TorchDataLoader(
        val_dataset,
        local_batch_size=val_config.batch_size // jax.process_count(),
        sharding=replicated_sharding,
        shuffle=False,
        num_batches=val_config.val_num_batches,
        num_workers=val_config.num_workers,
        seed=val_config.seed,
    )

    return ValidationDataLoader(actual_val_data_config, val_torch_data_loader)


def _compute_validation_losses(
    val_loader: _data_loader.DataLoader,
    train_state: training_utils.TrainState,
    mesh: jax.sharding.Mesh,
    train_state_sharding: jax.sharding.NamedSharding,
    replicated_sharding: jax.sharding.NamedSharding,
    config: _config.TrainConfig,
) -> float | None:
    """Compute validation losses over multiple batches."""

    # Define validation loss function (similar to train_step structure)
    @at.typecheck
    def val_loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
        return jnp.mean(chunked_loss)

    def validation_step(state, batch, rng):
        """Single validation step, aligned with train_step structure."""
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        observation, actions = batch
        val_rng = jax.random.fold_in(rng, state.step)

        return val_loss_fn(model, val_rng, observation, actions)

    # JIT compile the validation step
    pvalidation_step = jax.jit(
        validation_step,
        in_shardings=(train_state_sharding, replicated_sharding, replicated_sharding),
        out_shardings=replicated_sharding,
    )

    val_iter = iter(val_loader)
    losses = []
    # Use a separate RNG for validation to avoid interference with training RNG,
    # the specific seed offset is arbitrary.
    val_rng = jax.random.key(config.seed + 1000)

    for batch_idx in range(config.val_num_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        try:
            with sharding.set_mesh(mesh):
                loss = pvalidation_step(train_state, batch, val_rng)
            losses.append(jax.device_get(loss))
        except (RuntimeError, ValueError) as e:
            logging.warning("Error computing validation loss for batch %d: %s", batch_idx, e)
            continue

    if not losses:
        return None
    return float(jnp.mean(jnp.array(losses)))


def compute_validation_loss(
    config: _config.TrainConfig,
    train_state: training_utils.TrainState,
    mesh: jax.sharding.Mesh,
    train_state_sharding: jax.sharding.NamedSharding,
    replicated_sharding: jax.sharding.NamedSharding,
    train_data_loader: _data_loader.DataLoader | None = None,
) -> float | None:
    """Compute average validation loss over a few batches. Downloads validation dataset if missing locally."""
    # Extract training normalization stats
    training_norm_stats = _extract_training_norm_stats(train_data_loader)

    # Prepare validation configuration
    try:
        val_config, repo_id, use_norm_stats, actual_val_data_config = _prepare_validation_config(
            config, training_norm_stats
        )
    except ValueError:
        logging.warning("Could not determine validation repository ID, skipping validation loss.")
        return None

    # Check if validation dataset exists locally (same behavior as training dataset)
    is_local = True

    # Create validation data loader (this will use local data or trigger download if needed)
    try:
        val_loader = _create_validation_data_loader(
            actual_val_data_config, val_config, use_norm_stats=use_norm_stats, replicated_sharding=replicated_sharding
        )
        if not is_local:
            logging.info(f"Validation dataset downloaded successfully: {repo_id}")
    except Exception as e:
        logging.warning(f"Failed to create validation data loader for {repo_id}: {e}")
        logging.warning("Skipping validation loss computation.")
        return None

    # Compute and return validation losses
    return _compute_validation_losses(val_loader, train_state, mesh, train_state_sharding, replicated_sharding, config)


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
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)

    def loss_and_grads(micro_rng, observation, actions):
        # Build the module at the current trace level: nnx transforms only accept Variables created at the trace they
        # run in (and write the module's non-differentiated state, e.g. RNG counters, back into them afterwards), so
        # inside the `lax.scan` body below this has to happen per iteration, not once outside. The tree_map re-creates
        # the Variable objects (flax >= 0.11 keeps Variables inside `State`; on older flax it is a no-op copy).
        model = nnx.merge(state.model_def, jax.tree.map(lambda x: x, state.params))
        model.train()
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, micro_rng, observation, actions)
        return loss, grads, model

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    if config.grad_accum_steps == 1:
        loss, grads, model = loss_and_grads(train_rng, observation, actions)
    else:
        # Gradient accumulation. `batch` is laid out as [grad_accum_steps, batch_size / grad_accum_steps, ...] (see
        # `data_loader.TorchDataLoader`). With equal-sized micro-batches, the mean of their mean-loss gradients is the
        # gradient of the mean loss over the whole batch, so the update equals the full-batch one up to floating-point
        # summation order. Each micro-batch draws its own noise / timestep / augmentation randomness, exactly as the
        # samples of one full batch do. Only one micro-batch's activations are live at a time, which is what lets a
        # remat policy that saves more (`--model.remat-policy`) fit.
        trainable = state.params.filter(config.trainable_filter)

        def accumulate(carry, xs):
            loss_sum, grads_sum = carry
            micro_loss, micro_grads, _ = loss_and_grads(*xs)
            return (loss_sum + micro_loss, jax.tree.map(jnp.add, grads_sum, micro_grads)), None

        init = (jnp.zeros((), jnp.float32), jax.tree.map(jnp.zeros_like, trainable))
        micro_rngs = jax.random.split(train_rng, config.grad_accum_steps)
        (loss, grads), _ = jax.lax.scan(accumulate, init, (micro_rngs, observation, actions))
        loss = loss / config.grad_accum_steps
        grads = jax.tree.map(lambda g: g / config.grad_accum_steps, grads)
        model = nnx.merge(state.model_def, state.params)

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
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # Must precede the first JAX device query below: works around XLA aborting on GPUs newer than the pinned jax
    # (B300, compute capability 10.3) -- see openpi/shared/xla_gpu_compat.py.
    _xla_gpu_compat.configure_xla_flags()

    if config.batch_size % (jax.device_count() * config.grad_accum_steps) != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()} times "
            f"grad_accum_steps {config.grad_accum_steps}."
        )

    # Persistent compilation cache: JAX_COMPILATION_CACHE_DIR if set, else jax's default location under ~/.cache.
    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path(jax.config.jax_compilation_cache_dir or "~/.cache/jax").expanduser()),
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # Sharding of the batches the data loader produces: [B, ...] or, with gradient accumulation,
    # [grad_accum_steps, B / grad_accum_steps, ...] with the sample axis data-parallel.
    batch_sharding = _data_loader.micro_batch_sharding(data_sharding, config.grad_accum_steps)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        max_to_keep=config.max_to_keep,
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

    # Log images from first batch to sanity check. Images are [B, h, w, c] or, with gradient accumulation,
    # [grad_accum_steps, B / grad_accum_steps, h, w, c]; index the i-th sample of either layout.
    def sample_image(img, i):
        return img[i] if img.ndim == 4 else img[i // img.shape[1], i % img.shape[1]]

    num_samples = int(np.prod(next(iter(batch[0].images.values())).shape[:-3]))
    images_to_log = [
        wandb.Image(np.concatenate([np.array(sample_image(img, i)) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, num_samples))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    if config.prefetch_batches > 0:
        # From here on the batches are pulled in a background thread (the torch loader's worker processes were started
        # above, in the main thread, by the first `next`), so the hand-off overlaps with the running train step.
        data_iter = _data_loader.PrefetchIterator(data_iter, depth=config.prefetch_batches)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, batch_sharding),
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
        if config.val_log_interval and step % config.val_log_interval == 0:
            val_loss = compute_validation_loss(
                config, train_state, mesh, train_state_sharding, replicated_sharding, data_loader
            )
            if val_loss is not None:
                wandb.log({"val_loss": val_loss}, step=step)
                logging.info("Validation loss at step %d: %.4f", step, val_loss)
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())