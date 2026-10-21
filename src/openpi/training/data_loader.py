from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.b1k_artifacts as _b1k_artifacts
import openpi.training.b1k_dataset as _b1k_dataset
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.lerobot_compat as _lerobot_compat
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_b1k_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig | None = None,
    *,
    subset: str = "train",
) -> Dataset:
    """Create a behavior dataset for training.

    The dataset is read from the local root `data_config.dataset_root` (the full download or a per-task partial
    download of the challenge demos). If `data_config.task_names` is set, only the episodes of those tasks are
    loaded (resolved against `meta/tasks.parquet` / `meta/episodes/`, so a misspelled name or a partial download
    of other tasks fails here rather than mid-training). Prompts follow `data_config.prompt_source` (task name or
    natural-language description); when `model_config` is given, prompts that would not fit its `max_token_len`
    are rejected up front. With an episode split, `subset` selects its `train` or `held_out` episodes; with
    `data_config.val_fraction`, `train` excludes the validation episodes and `val` selects them.
    """
    if data_config.dataset_root is None:
        raise ValueError("dataset_root is not set. Pass --data.dataset-root=<local LeRobot root> for B1K datasets.")
    if isinstance(data_config.repo_id, list):
        if data_config.task_names:
            raise NotImplementedError("task_names is not supported with a list of repo_ids (MultiLeRobotDataset).")
        dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(
            repo_id=data_config.repo_id[0], root=os.path.join(data_config.dataset_root, data_config.repo_id[0])
        )
        dataset_kwargs = {"repo_ids": data_config.repo_id, **data_config.dataset_kwargs}
    else:
        dataset_meta = _b1k_dataset.B1KDatasetMetadata(data_config.repo_id, data_config.dataset_root)
        dataset_kwargs = {"repo_id": data_config.repo_id, **data_config.dataset_kwargs}
        goal_camera_keys = dataset_kwargs.pop("goal_camera_keys", None)
        expanded_tasks = _b1k_dataset.expand_task_groups(data_config.dataset_root, data_config.task_names)
        if data_config.task_names and expanded_tasks != list(data_config.task_names):
            logging.info("--data.task-names %s select the tasks %s", list(data_config.task_names), expanded_tasks)
        episode_split = _b1k_dataset.resolve_episode_split(
            data_config.dataset_root,
            getattr(data_config, "episode_split", "auto"),
            expanded_tasks,
            subset="train" if subset == "val" else subset,
        )
        if episode_split is not None:
            logging.info("Episode split %s", _b1k_dataset.describe_episode_split(episode_split))
            # every task of the file without task names; held_out drops the tasks that have no held-out episode
            expanded_tasks = list(episode_split["tasks"])
        elif subset == "held_out":
            raise ValueError(
                f"The {subset} episodes need an episode split (--data.episode-split); {data_config.dataset_root} has none"
            )
        task_names = _b1k_dataset.normalize_task_names(expanded_tasks)
        episodes = dataset_kwargs.get("episodes")
        if episode_split is not None:
            split_eps = list(episode_split["episodes"])
            episodes = split_eps if episodes is None else sorted(set(split_eps) & {int(ep) for ep in episodes})
        val_split = None
        if subset != "held_out":
            val_split = _b1k_dataset.resolve_val_split(
                dataset_meta,
                task_names,
                getattr(data_config, "val_fraction", 0.0),
                getattr(data_config, "val_seed", None),
                episodes=episodes,
            )
            if val_split is not None:
                logging.info("Validation split %s", _b1k_dataset.describe_val_split(val_split))
                episodes = val_split[f"{subset}_episodes"]
            elif subset == "val":
                raise ValueError("The val episodes need --data.val-fraction > 0")
        if task_names:
            # Every requested task must have episodes, also within an explicit `episodes` selection.
            subset = _b1k_dataset.select_task_subset(dataset_meta, task_names, episodes=episodes)
            dataset_kwargs["episodes"] = list(subset.episode_indices)
            logging.info(
                "Task subset %s (task_index %s): %d of %d episodes under %s",
                list(subset.task_names),
                sorted(subset.task_indices),
                len(dataset_kwargs["episodes"]),
                len(dataset_meta.episodes),
                data_config.dataset_root,
            )
        elif episodes is not None:
            dataset_kwargs["episodes"] = sorted({int(ep) for ep in episodes})

        settle_spec = _b1k_dataset.parse_settle_steps(getattr(data_config, "settle_steps", "all"))
        if settle_spec != "all":
            if data_config.data_cls is _b1k_dataset.B1KLeRobotDataset:
                dataset_kwargs["settle_steps"] = settle_spec
            else:
                raise NotImplementedError(f"settle_steps={settle_spec!r} requires B1KLeRobotDataset")

        goal_views = tuple(getattr(data_config, "goal_views", ()) or ())
        resolved_goal_source: str | None = None
        if goal_views:
            goal_source = getattr(data_config, "goal_source", None)
            if goal_source is None:
                # Auto-detect: use dedicated `observation.goal_rgb.*` streams if the dataset ships them, else `episode_last`.
                wanted_goal_streams = set(goal_camera_keys or ())
                available_streams = set(dataset_meta.video_keys) | set(dataset_meta.camera_keys)
                resolved_goal_source = (
                    "dataset_goal_stream"
                    if wanted_goal_streams and wanted_goal_streams.issubset(available_streams)
                    else "episode_last"
                )
            elif goal_source == "goal_key":
                wanted_goal_streams = set(goal_camera_keys or ())
                available_streams = set(dataset_meta.video_keys) | set(dataset_meta.camera_keys)
                if missing_streams := sorted(wanted_goal_streams - available_streams):
                    raise ValueError(
                        f"{data_config.dataset_root}: no metadata for video streams {missing_streams}; "
                        "goal_source=goal_key needs the dataset's observation.goal_rgb.* streams "
                        "(use --data.goal-source episode_last otherwise)"
                    )
                resolved_goal_source = "goal_key"
            elif goal_source == "episode_last":
                resolved_goal_source = "episode_last"
            else:
                raise ValueError(f"Unknown goal_source {goal_source!r}; expected one of {_b1k_dataset.GOAL_SOURCES}")
            if resolved_goal_source == "episode_last":
                if data_config.data_cls is not _b1k_dataset.B1KLeRobotDataset or not goal_camera_keys:
                    raise NotImplementedError("goal_source=episode_last requires B1KLeRobotDataset")
                dataset_kwargs["goal_camera_keys"] = goal_camera_keys
                if "video_keys" in dataset_kwargs and dataset_kwargs["video_keys"] is not None:
                    dataset_kwargs["video_keys"] = [
                        k for k in dataset_kwargs["video_keys"] if k not in set(goal_camera_keys)
                    ]

    dataset = data_config.data_cls(
        root=data_config.dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        **dataset_kwargs,
    )

    recorded_prompts: dict[str, str] = {}
    if data_config.prompt_from_task:
        if isinstance(data_config.repo_id, list):
            if data_config.action_representation is not None:
                raise NotImplementedError("Checkpoint prompt provenance requires a single BEHAVIOR dataset root")
            prompts = _lerobot_compat.tasks_from_metadata(dataset_meta)
        else:
            # Only the tasks actually trained on need a prompt: the selected subset, else every task on disk.
            required = (
                set(subset.task_indices)
                if task_names
                else set(_b1k_dataset.episode_task_indices(dataset_meta))
            )
            prompts = _b1k_dataset.task_prompts(dataset_meta, data_config.prompt_source, required)
            logging.info(
                "Prompting with %s, e.g. %r", data_config.prompt_source, prompts[min(required)] if required else None
            )
            names = _b1k_dataset.task_names_by_index(dataset_meta.tasks)
            recorded_prompts = {names[i]: prompts[i] for i in required}
            if model_config is not None:
                # The prompt is tokenized with the extracted (unpadded) state; count that, not the model's action_dim.
                representation = data_config.action_representation
                state_dim = (
                    None
                    if representation is None
                    else sum(1 if group["is_eef"] else len(group["indices"]) for group in representation["proprio"])
                )
                # Under a conditioning regime, ComposePrompt formats the final text fed to the tokenizer.
                regime = getattr(data_config, "conditioning_regime", None)
                if regime is not None:
                    scaffold = _config.GOAL_PROMPT_SCAFFOLD if goal_views else _config.PLAIN_PROMPT_SCAFFOLD
                    include_task = regime in ("language", "image_language")
                    checked_prompts = {
                        i: (f"{scaffold} {prompts[i]}".strip() if include_task else scaffold) for i in required
                    }
                else:
                    checked_prompts = {i: prompts[i] for i in required}
                _b1k_dataset.check_prompt_token_lengths(checked_prompts, model_config, state_dim=state_dim)
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(prompts)])
    if data_config.action_representation is not None and model_config is not None:
        # Recorded into new checkpoints (`b1k_metadata.json`) so serving restores the exact prompts and model settings.
        settle_record = getattr(dataset, "settle", None) or getattr(
            getattr(dataset, "_dataset", None), "settle", None
        )
        dataset.inference_metadata = _b1k_artifacts.inference_metadata(
            data_config,
            model_config,
            recorded_prompts,
            goal_source=resolved_goal_source if not isinstance(data_config.repo_id, list) else None,
            episode_split=episode_split if not isinstance(data_config.repo_id, list) else None,
            settle_steps=settle_record,
            val_split=val_split if not isinstance(data_config.repo_id, list) else None,
        )

    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(repo_id)
    dataset = _lerobot_compat.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(_lerobot_compat.tasks_from_metadata(dataset_meta))]
        )

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def _missing_norm_stats_message(data_config: _config.DataConfig) -> str:
    subset = f" (task subset {list(data_config.task_names)})" if data_config.task_names else ""
    return (
        f"Normalization stats not found for asset id {data_config.asset_id!r}{subset}. Make sure to run "
        "`scripts/compute_norm_stats.py <config_name>` first, with the same --data.* flags as training."
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(_missing_norm_stats_message(data_config))
        _b1k_artifacts.validate_representation(
            data_config.norm_stats_metadata,
            data_config.action_representation,
            allow_legacy_assets=data_config.allow_legacy_assets,
            context=f"Normalization statistics for {data_config.asset_id}",
        )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(_missing_norm_stats_message(data_config))
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_b1k_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions, jax.Array]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_b1k_dataset(
        data_config=data_config, action_horizon=config.model.action_horizon, model_config=config.model
    )
    data_config = dataclasses.replace(data_config, inference_metadata=getattr(dataset, "inference_metadata", None))
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    return B1KDataLoaderImpl(data_config, data_loader)


def load_b1k_fixed_batches(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    *,
    subset: str,
    batch_size: int,
    num_batches: int,
    seed: int,
    num_workers: int = 0,
    num_samples: int = 0,
) -> tuple[list[dict], tuple[dict, list[str]] | None, str]:
    """Transformed batches of a fixed selection of frames of the `subset` episodes, loaded once.

    `num_batches` batches of `batch_size` frames (fewer when the episodes hold fewer frames) spread over every
    episode by `b1k_dataset.spread_frame_indices` with `seed`, so each call returns the same frames. Transforms and
    norm stats are those of `data_config`. Returns the collated numpy batches; one collated batch of `num_samples`
    frames of different episodes (`b1k_dataset.figure_frame_indices`) with a title per frame, or None without
    samples; and a description for logs.
    """
    dataset = create_b1k_dataset(data_config, model_config.action_horizon, model_config, subset=subset)
    frames = getattr(dataset, "_dataset", dataset)
    episode_index = frames.hf_dataset.data.column("episode_index").to_numpy()
    rows = _b1k_dataset.spread_frame_indices(episode_index, batch_size * num_batches, seed)
    rows = rows[: len(rows) // batch_size * batch_size]
    if not rows:
        raise ValueError(f"The {subset} episodes hold {len(episode_index)} frames, fewer than a batch of {batch_size}")
    transformed = transform_dataset(dataset, data_config)
    samples = None
    if num_samples:
        sample_rows = _b1k_dataset.figure_frame_indices(episode_index, num_samples, seed)
        task_names = _b1k_dataset.task_names_by_index(frames.meta.tasks)
        task_index = frames.hf_dataset.data.column("task_index").to_numpy()
        frame_index = frames.hf_dataset.data.column("frame_index").to_numpy()
        titles = [
            f"{task_names[int(task_index[row])]}  episode {episode_index[row]}  frame {frame_index[row]}"
            for row in sample_rows
        ]
        samples = (_collate_fn([transformed[row] for row in sample_rows]), titles)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(transformed, rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, len(rows) // batch_size),
        multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
        collate_fn=_collate_fn,
        worker_init_fn=_worker_init_fn,
    )
    batches = list(loader)
    description = (
        f"{len(rows)} frames ({len(batches)} batches of {batch_size}, seed {seed}) of the {subset} episodes: "
        f"{len(np.unique(episode_index))} episodes, {len(episode_index)} frames"
    )
    return batches, samples, description


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Keep JAX inside a data-loader worker on the CPU, single-threaded, and off the training GPU.

    The only JAX work in a worker is `image_tools.resize_with_pad` (`ResizeImages`). Workers are spawned with the
    trainer's environment (`CUDA_VISIBLE_DEVICES`, `JAX_PLATFORMS=cuda`), so by default each one would create a CUDA
    context on the training GPU for that resize and XLA:CPU would use every core for its thread pool -- with 8-32
    workers on a CPU quota that oversubscribes the cores. jax is already imported here (the dataset was unpickled),
    but its backend is not initialised until the first operation, so the platform can still be chosen.
    """
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    flags = os.environ.get("XLA_FLAGS", "")
    if "intra_op_parallelism_threads" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1".strip()
    try:
        jax.config.update("jax_platforms", "cpu")
    except RuntimeError:  # backend already initialised in this worker; keep whatever it is
        logging.warning("Data-loader worker %d: JAX backend already initialised, cannot pin it to the CPU", worker_id)


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]


class B1KDataLoaderImpl(DataLoaderImpl):
    """Yields `(observation, actions, action_is_pad)`; `action_is_pad` (bool `[B, H]`) marks the action-chunk steps
    past the episode end, all False when the dataset has no mask."""

    def __iter__(self):
        for batch in self._data_loader:
            actions = batch["actions"]
            is_pad = batch.get("action_is_pad")
            if is_pad is None:
                is_pad = jnp.zeros(actions.shape[:2], bool, device=getattr(actions, "sharding", None))
            yield _model.Observation.from_dict(batch), actions, is_pad
