"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

    uv run scripts/compute_norm_stats.py <config_name> [--max-frames N] [--data.* overrides]

The config's data settings can be overridden like for `scripts/b1k/train_b1k.py`, e.g. for the
BEHAVIOR-1K challenge demos:

    uv run scripts/compute_norm_stats.py pi05_b1k \\
        --data.repo_id=behavior-1k/2026-challenge-demos --data.dataset-root=$DATA_ROOT \\
        --data.task-names turning_on_radio

Stats of a task subset are written under `outputs/assets/<config>/<repo_id>/task_subsets/<key>/`
(computed over the selected tasks' episodes only), stats of the whole dataset under
`outputs/assets/<config>/<repo_id>/`; training and serving resolve the same directory from the
same `--data.*` flags.

For LeRobot datasets the `state` / `actions` batches are read straight from the parquet files
(see `openpi.training.lerobot_lowdim`) instead of going through the data loader, which decodes
every camera stream of every frame. This is orders of magnitude faster and yields bit-identical
statistics; `--no-fast` uses the regular data loader instead (e.g. to cross-check).
"""

from collections.abc import Iterable, Sequence
import dataclasses
import logging
import sys
import time

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.b1k_dataset as _b1k_dataset
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.lerobot_compat as _lerobot_compat
import openpi.training.lerobot_lowdim as _lerobot_lowdim
import openpi.transforms as transforms

# `LeRobotDataset` constructor arguments that do not affect the non-visual items and are therefore irrelevant to the
# low-dim reader.
_LOWDIM_IRRELEVANT_DATASET_KWARGS = frozenset(
    {
        "tolerance_s",
        "video_backend",
        "download_videos",
        "return_uint8",
        "video_keys",
        "image_transforms",
        "revision",
        "force_cache_sync",
    }
)


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def resolve_asset_id(data_config: _config.DataConfig) -> str:
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Data config must have an asset_id or repo_id")
    if isinstance(asset_id, list):
        if not asset_id:
            raise ValueError("Data config asset_id/repo_id list cannot be empty")
        return asset_id[0]
    return asset_id


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_b1k_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_b1k_dataset(data_config=data_config, action_horizon=action_horizon)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _create_lowdim_dataset(
    data_config: _config.DataConfig, action_horizon: int
) -> _lerobot_lowdim.LowDimLeRobotDataset:
    """The low-dim counterpart of the dataset that `create_b1k_dataloader` / `create_torch_dataloader` iterate over.

    Raises `NotImplementedError` for data configs the low-dim reader cannot mirror.
    """
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if isinstance(data_config.repo_id, list):
        raise NotImplementedError("multiple repo_ids (MultiLeRobotDataset)")
    if data_config.repo_id == "fake":
        raise NotImplementedError("fake dataset")

    def make(meta, **kwargs) -> _lerobot_lowdim.LowDimLeRobotDataset:
        return _lerobot_lowdim.LowDimLeRobotDataset(
            meta,
            delta_timestamps={
                key: [t / meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
            prompt_from_task=data_config.prompt_from_task,
            **kwargs,
        )

    if data_config.dataset_root is None:
        # Mirrors `_data_loader.create_torch_dataset`: `LeRobotDataset(repo_id, episodes=episodes_index)`.
        meta = _lerobot_compat.LeRobotDatasetMetadata(data_config.repo_id)
        if not _lerobot_lowdim.episode_metadata_is_positional(meta):
            raise NotImplementedError("episode metadata not indexed by position")
        episodes = data_config.episodes_index
        return make(
            meta,
            episodes=episodes,
            num_frames=meta.total_frames if episodes is None else None,
            frame_index_is_position=episodes is None,
        )

    # Mirrors `_data_loader.create_b1k_dataset`.
    dataset_kwargs = dict(data_config.dataset_kwargs)
    episodes = dataset_kwargs.pop("episodes", None)
    if unsupported := set(dataset_kwargs) - _LOWDIM_IRRELEVANT_DATASET_KWARGS:
        raise NotImplementedError(f"dataset_kwargs {sorted(unsupported)}")
    tolerance_s = dataset_kwargs.get("tolerance_s", 1e-4)
    if data_config.task_names:
        subset = _b1k_dataset.select_task_subset(
            _b1k_dataset.B1KDatasetMetadata(data_config.repo_id, data_config.dataset_root),
            data_config.task_names,
            episodes=episodes,
        )
        episodes = list(subset.episode_indices)
    elif episodes is not None:
        episodes = sorted({int(episode) for episode in episodes})

    if data_config.data_cls is _b1k_dataset.B1KLeRobotDataset:
        meta = _b1k_dataset.B1KDatasetMetadata(data_config.repo_id, data_config.dataset_root)
        if episodes is not None and not episodes:
            raise ValueError("No episode selected.")
        return make(
            meta,
            tolerance_s=tolerance_s,
            episodes=episodes,
            data_files=_lerobot_lowdim.referenced_data_files(meta, episodes),
        )
    if data_config.data_cls is _lerobot_compat.LeRobotDataset:
        meta = _lerobot_compat.LeRobotDatasetMetadata(data_config.repo_id, root=data_config.dataset_root)
        if not _lerobot_lowdim.episode_metadata_is_positional(meta):
            raise NotImplementedError("episode metadata not indexed by position")
        return make(
            meta,
            tolerance_s=tolerance_s,
            episodes=episodes,
            num_frames=meta.total_frames if episodes is None else None,
            frame_index_is_position=episodes is None,
        )
    raise NotImplementedError(f"data_cls {data_config.data_cls.__name__}")


def create_lowdim_batches(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None,
    keys: Sequence[str],
) -> tuple[Iterable[dict[str, np.ndarray]], int] | None:
    """Batches of `keys` of a LeRobot dataset, read from its parquet files without decoding any video.

    Produces the same batches, in the same order, as the `TorchDataLoader` built by `create_b1k_dataloader` /
    `create_torch_dataloader`, so the resulting statistics are bit-identical (see `openpi.training.lerobot_lowdim`).
    Returns None, after logging the reason, if the data config is not a LeRobot dataset the low-dim reader can mirror.
    """
    try:
        dataset = _create_lowdim_dataset(data_config, action_horizon)
    except (NotImplementedError, ValueError, FileNotFoundError) as e:
        logging.warning("Cannot read the dataset without decoding videos (%s); using the regular data loader.", e)
        return None
    if len(dataset) < batch_size:
        raise ValueError(f"Local batch size ({batch_size}) is larger than the dataset size ({len(dataset)}).")
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ]
    )
    per_frame_mode = not _lerobot_lowdim.batched_transform_matches_per_frame(dataset, transform, keys)
    logging.info(
        "Reading %s of %d frames (%d episodes) from the parquet files of %s, transforms applied %s%s",
        f"{num_batches * batch_size} shuffled" if shuffle else "all",
        len(dataset),
        len(dataset.spans),
        dataset.root,
        "per frame" if per_frame_mode else "per episode",
        f" (task subset {list(data_config.task_names)})" if data_config.task_names else "",
    )
    if shuffle:
        batch_indices = _lerobot_lowdim.shuffled_batch_indices(len(dataset), batch_size, num_batches)
        batches = _lerobot_lowdim.iter_gathered_batches(
            dataset, transform, keys, batch_indices, per_frame_mode=per_frame_mode
        )
    else:
        batches = _lerobot_lowdim.iter_sequential_batches(
            dataset, transform, keys, batch_size, num_batches, per_frame_mode=per_frame_mode
        )
    return batches, num_batches


@dataclasses.dataclass(frozen=True)
class Args:
    """Arguments of compute_norm_stats.py; the positional config name selects the defaults."""

    # Name of the training config the stats are for (selected as the CLI subcommand).
    config_name: tyro.conf.Suppress[str]
    # The config's data settings. Override them like for train_b1k.py, e.g. --data.repo_id, --data.dataset-root,
    # --data.task-names (B1K).
    data: _config.DataConfigFactory
    # If set, compute the stats over at most this many (randomly sampled) frames instead of the whole dataset.
    max_frames: int | None = None
    # Read the state / action columns straight from the LeRobot parquet files instead of going through the data loader
    # (which decodes every camera stream). Same batches, same statistics, orders of magnitude faster; --no-fast uses
    # the regular data loader (also used automatically for datasets the fast path does not support).
    fast: bool = True


def cli(argv: Sequence[str] | None = None) -> Args:
    """Parse `<config_name> [--max-frames N] [--data.* overrides]` (same shape as scripts/train.py).

    The former flag form `--config-name <name>` (upstream README) is still accepted and rewritten to the
    positional form.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, arg in enumerate(argv):
        if arg == "--config-name" and i + 1 < len(argv):
            argv = [argv[i + 1], *argv[:i], *argv[i + 2 :]]
            break
        if arg.startswith("--config-name="):
            argv = [arg.split("=", 1)[1], *argv[:i], *argv[i + 1 :]]
            break
    return tyro.extras.overridable_config_cli(
        {
            name: (name, Args(config_name=name, data=config.data))
            for name, config in _config._CONFIGS_DICT.items()  # noqa: SLF001
        },
        args=argv,
    )


def main(args: Args):
    config = dataclasses.replace(_config.get_config(args.config_name), data=args.data)
    max_frames = args.max_frames
    data_config = config.data.create(config.assets_dirs, config.model)
    keys = ["state", "actions"]

    start_time = time.monotonic()
    lowdim_batches = None
    if args.fast and data_config.rlds_data_dir is None:
        lowdim_batches = create_lowdim_batches(
            data_config, config.model.action_horizon, config.batch_size, max_frames, keys
        )
    if lowdim_batches is not None:
        data_loader, num_batches = lowdim_batches
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    elif data_config.dataset_root is not None:
        data_loader, num_batches = create_b1k_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.num_workers, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    logging.info("Computed stats over %d batches in %.1fs", num_batches, time.monotonic() - start_time)

    asset_id = resolve_asset_id(data_config)
    output_path = config.assets_dirs / asset_id
    if data_config.task_names:
        print(f"Stats computed over task subset {list(data_config.task_names)} only")
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    if data_config.action_representation is not None:
        from openpi.training import b1k_artifacts

        b1k_artifacts.save_metadata(output_path, b1k_artifacts.norm_metadata(data_config.action_representation))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)  # basicConfig is a no-op when an import already installed a handler.
    main(cli())
