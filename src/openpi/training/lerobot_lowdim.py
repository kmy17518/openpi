"""Video-free iteration over the low-dimensional features of a local LeRobot dataset.

`LeRobotDataset.__getitem__` decodes every camera stream of a frame, which takes hundreds of milliseconds per frame
and is by far the dominant cost when only low-dimensional features are needed (e.g. to compute the normalization
statistics of `state` and `actions`). This module instead reads the non-visual columns straight from the dataset's
parquet files and rebuilds, one episode at a time, exactly the items that `LeRobotDataset` returns for them:

* `delta_timestamps` windows (e.g. the action chunk) are gathered with the same clamping at episode boundaries and
  come with the same `<key>_is_pad` masks,
* numeric columns become torch tensors with the dtypes `LeRobotDataset` produces,
* the `task` string and, if requested, the `prompt` (see `PromptFromLeRobotTask`) are added,
* visual features are replaced by small placeholder tensors since they are not needed for the requested keys.

The data transforms are applied to a whole episode at once (openpi transforms index the last axes and act
element-wise along the leading frame axis; this is verified against per-frame application before relying on it) and
the result is re-batched exactly like `openpi.training.data_loader.TorchDataLoader` would, sequentially or shuffled,
so that order-dependent consumers such as `RunningStats` see the same batches and produce bit-identical results.
"""

from collections.abc import Iterator, Sequence
import dataclasses
import logging
import pathlib
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import openpi.training.lerobot_compat as _lerobot_compat
import openpi.transforms as _transforms

try:
    from lerobot.datasets.feature_utils import check_delta_timestamps
    from lerobot.datasets.feature_utils import get_delta_indices
except ImportError:  # Older LeRobot layout.
    from lerobot.datasets.utils import check_delta_timestamps
    from lerobot.datasets.utils import get_delta_indices

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class EpisodeSpan:
    """Location of one episode's frames."""

    episode_index: int
    first_index: int  # Absolute frame index (`index` column) of the episode's first frame.
    end_index: int  # One past the absolute frame index of the episode's last frame.
    data_file: pathlib.Path  # Parquet file that stores the episode's rows.
    offset: int  # Position of the episode's first frame in this dataset.

    def __len__(self) -> int:
        return self.end_index - self.first_index


class LowDimLeRobotDataset:
    """Episode-level access to the non-visual features of a local LeRobot dataset.

    Frame `i` of this dataset corresponds to item `i` of `LeRobotDataset(root=meta.root, episodes=episodes,
    delta_timestamps=..., tolerance_s=...)` (or of `openpi.training.b1k_dataset.B1KLeRobotDataset`, which reads the
    given `data_files` only): the selected episodes' rows, in the order of the data files and of the rows within them.
    Raises `ValueError` if the files do not have the layout those datasets rely on (each episode stored as a
    contiguous run of rows at the position its metadata announces, `index` column contiguous within a file), so that
    callers can fall back to the regular data loader.
    """

    def __init__(
        self,
        meta: Any,
        *,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        prompt_from_task: bool = False,
        episodes: Sequence[int] | None = None,
        data_files: Sequence[pathlib.Path | str] | None = None,
        num_frames: int | None = None,
        frame_index_is_position: bool = False,
    ):
        """
        Args:
            meta: A `LeRobotDatasetMetadata` of a dataset whose files are available under `meta.root`.
            delta_timestamps: Same as for `LeRobotDataset`; windows are gathered for these keys.
            tolerance_s: Same as for `LeRobotDataset` (only used to validate `delta_timestamps`).
            prompt_from_task: Add a `prompt` entry from the task index, like `PromptFromLeRobotTask` does.
            episodes: `episode_index` values to include; None selects every episode in the metadata.
            data_files: The parquet files to read (relative to the root, or absolute), in order; None reads every
                `data/*/*.parquet` file in sorted order like `LeRobotDataset` does.
            num_frames: Value reported by `len()`; defaults to the number of selected frames. `LeRobotDataset`
                without an episode filter reports `meta.total_frames` from `info.json` instead.
            frame_index_is_position: Require every frame's `index` to equal its position in the concatenated data
                files. `LeRobotDataset` without an episode filter assumes this when it gathers `delta_timestamps`
                windows (`B1KLeRobotDataset` and episode-filtered datasets map indices explicitly instead).
        """
        self.meta = meta
        self.root = pathlib.Path(meta.root)
        self.delta_indices: dict[str, list[int]] = {}
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)
        self.visual_keys: list[str] = [*meta.video_keys, *meta.image_keys]
        # Position -> task name, like `LeRobotDataset` (`meta.tasks.iloc[task_index].name`).
        self._task_names = np.asarray([str(name) for name in meta.tasks.index], dtype=str)
        self._prompts = _lerobot_compat.tasks_from_metadata(meta) if prompt_from_task else None
        self.spans = self._build_spans(episodes, data_files)
        if frame_index_is_position and any(span.offset != span.first_index for span in self.spans):
            raise ValueError(f"Frame indices under {self.root} do not match the row positions of the data files.")
        self._spans_by_file: dict[pathlib.Path, list[EpisodeSpan]] = {}
        for span in self.spans:
            self._spans_by_file.setdefault(span.data_file, []).append(span)
        num_selected = sum(len(span) for span in self.spans)
        self._num_frames = num_selected if num_frames is None else int(num_frames)
        if self._num_frames > num_selected:
            raise ValueError(f"{self._num_frames} frames requested but the selected episodes only hold {num_selected}.")
        self._cached_file: tuple[pathlib.Path, dict[str, np.ndarray], int] | None = None

    def __len__(self) -> int:
        return self._num_frames

    def _build_spans(
        self, episodes: Sequence[int] | None, data_files: Sequence[pathlib.Path | str] | None
    ) -> list[EpisodeSpan]:
        if data_files is None:
            # Same file enumeration as LeRobot's `load_nested_dataset`.
            files = sorted((self.root / "data").glob("*/*.parquet"))
        else:
            files = [self.root / path for path in data_files]
        if not files:
            raise FileNotFoundError(f"No parquet files found under {self.root / 'data'}")
        if missing_files := [str(path) for path in files if not path.is_file()]:
            raise FileNotFoundError(f"Missing data files: {missing_files[:5]}")
        file_positions = {path.resolve(): position for position, path in enumerate(files)}

        episode_table = self.meta.episodes
        episode_indices = _metadata_column(episode_table, "episode_index")
        first_indices = _metadata_column(episode_table, "dataset_from_index")
        end_indices = _metadata_column(episode_table, "dataset_to_index")
        chunk_indices = _metadata_column(episode_table, "data/chunk_index")
        file_indices = _metadata_column(episode_table, "data/file_index")
        selected = None if episodes is None else {int(episode) for episode in episodes}
        if selected is not None and (missing := selected - {int(episode) for episode in episode_indices}):
            raise ValueError(f"Episodes {sorted(missing)[:10]} are not in the episode metadata of {self.root}.")

        located = []
        for episode_index, first_index, end_index, chunk_index, file_index in zip(
            episode_indices, first_indices, end_indices, chunk_indices, file_indices, strict=True
        ):
            if selected is not None and int(episode_index) not in selected:
                continue
            if end_index < first_index:
                raise ValueError(f"Episode {episode_index} has a negative length.")
            path = (self.root / self.meta.data_path.format(chunk_index=chunk_index, file_index=file_index)).resolve()
            if (position := file_positions.get(path)) is None:
                raise ValueError(f"The data file of episode {episode_index} ({path}) is not among the loaded files.")
            located.append((position, int(first_index), int(end_index), int(episode_index)))

        # Rows are concatenated file by file, and within a file stored in frame order (verified when loading).
        located.sort()
        spans, offset = [], 0
        for position, first_index, end_index, episode_index in located:
            spans.append(EpisodeSpan(episode_index, first_index, end_index, files[position], offset))
            offset += end_index - first_index
        return spans

    def _file_columns(self, span: EpisodeSpan) -> tuple[dict[str, np.ndarray], int]:
        """Non-visual columns of the parquet file holding `span` (cached for the last file) and its first frame index."""
        if self._cached_file is None or self._cached_file[0] != span.data_file:
            schema = pq.read_schema(span.data_file)
            names = [name for name in schema.names if name not in self.visual_keys]
            table = pq.read_table(span.data_file, columns=names)
            columns = {name: _column_to_numpy(table.column(name)) for name in names}
            if missing := {"index", "episode_index", "task_index"} - set(columns):
                raise ValueError(f"{span.data_file} lacks the LeRobot columns {sorted(missing)}.")
            if table.num_rows == 0:
                raise ValueError(f"{span.data_file} is empty.")
            first_index = int(columns["index"][0])
            if not np.array_equal(columns["index"], np.arange(first_index, first_index + table.num_rows)):
                raise ValueError(f"Frame indices in {span.data_file} are not contiguous.")
            # Every selected episode of this file must be exactly the run of rows its metadata announces.
            episode_column = columns["episode_index"]
            for file_span in self._spans_by_file[span.data_file]:
                rows = slice(file_span.first_index - first_index, file_span.end_index - first_index)
                if (
                    rows.start < 0
                    or rows.stop > table.num_rows
                    or np.any(episode_column[rows] != file_span.episode_index)
                    or np.count_nonzero(episode_column == file_span.episode_index) != len(file_span)
                ):
                    raise ValueError(
                        f"Rows of episode {file_span.episode_index} in {span.data_file} do not match its metadata."
                    )
            self._cached_file = (span.data_file, columns, first_index)
        return self._cached_file[1], self._cached_file[2]

    def episode_items(
        self, span: EpisodeSpan, positions: np.ndarray | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Items of the frames of one episode, before any data transform.

        Args:
            span: The episode.
            positions: Frame positions relative to the episode start (all frames if None).

        Returns:
            `(per_frame, shared)`: `per_frame` maps each item key to an array/tensor with a leading frame axis (frame
            `t` of the episode item is `{**{k: v[t] for k, v in per_frame.items()}, **shared}`); `shared` holds the
            placeholders for the visual features.
        """
        columns, file_first_index = self._file_columns(span)
        local = slice(span.first_index - file_first_index, span.end_index - file_first_index)
        length = len(span)
        positions = np.arange(length) if positions is None else np.asarray(positions, dtype=np.int64)

        per_frame: dict[str, Any] = {}
        for name, values in columns.items():
            episode_values = values[local]
            if name in self.delta_indices:
                # Same as LeRobotDataset._get_query_indices: clamp the window to the episode and flag padding.
                query = positions[:, None] + np.asarray(self.delta_indices[name], dtype=np.int64)[None, :]
                per_frame[name] = _to_torch(episode_values[np.clip(query, 0, length - 1)])
                per_frame[f"{name}_is_pad"] = torch.from_numpy((query < 0) | (query >= length))
            else:
                per_frame[name] = _to_torch(episode_values[positions])

        task_indices = columns["task_index"][local][positions]
        per_frame["task"] = self._task_names[task_indices]
        if self._prompts is not None:
            per_frame["prompt"] = self._lookup_prompts(task_indices)

        shared: dict[str, Any] = {}
        for key in self.visual_keys:
            # LeRobotDataset returns float32 (C, H, W) frames; use 1x1 pixel stand-ins.
            if key in self.delta_indices:
                deltas = np.asarray(self.delta_indices[key], dtype=np.int64)
                query = positions[:, None] + deltas[None, :]
                per_frame[f"{key}_is_pad"] = torch.from_numpy((query < 0) | (query >= length))
                shared[key] = torch.zeros((len(deltas), 3, 1, 1), dtype=torch.float32)
            else:
                shared[key] = torch.zeros((3, 1, 1), dtype=torch.float32)
        return per_frame, shared

    def _lookup_prompts(self, task_indices: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(task_indices, return_inverse=True)
        prompts = []
        for task_index in unique:
            if (prompt := self._prompts.get(int(task_index))) is None:
                raise ValueError(f"task_index={int(task_index)} not found in task mapping: {self._prompts}")
            prompts.append(prompt)
        return np.asarray(prompts, dtype=str)[inverse]


def transform_episode(
    per_frame: dict[str, Any],
    shared: dict[str, Any],
    transform: _transforms.DataTransformFn,
    keys: Sequence[str],
    *,
    per_frame_mode: bool = False,
) -> dict[str, np.ndarray]:
    """Apply `transform` to the items of `episode_items` and return the requested output keys as (num_frames, ...) arrays.

    In `per_frame_mode` the transform is applied to every frame separately and the outputs are stacked like
    `TorchDataLoader`'s collate function does; otherwise it is applied once to the whole episode.
    """
    if per_frame_mode:
        num_frames = len(next(iter(per_frame.values())))
        outputs = [transform({**{k: v[t] for k, v in per_frame.items()}, **shared}) for t in range(num_frames)]
        return {key: np.stack([np.asarray(output[key]) for output in outputs], axis=0) for key in keys}
    output = transform({**per_frame, **shared})
    return {key: np.asarray(output[key]) for key in keys}


def batched_transform_matches_per_frame(
    dataset: LowDimLeRobotDataset,
    transform: _transforms.DataTransformFn,
    keys: Sequence[str],
    *,
    max_frames_per_side: int = 128,
) -> bool:
    """Check on the first episode that applying `transform` to a whole episode equals applying it frame by frame.

    Frames at both ends of the episode are used so that padded `delta_timestamps` windows are covered.
    """
    if (span := next((span for span in dataset.spans if len(span) > 0), None)) is None:
        return True
    positions = np.unique(
        np.concatenate(
            [
                np.arange(min(len(span), max_frames_per_side)),
                np.arange(max(0, len(span) - max_frames_per_side), len(span)),
            ]
        )
    )
    # Transforms may modify their inputs in place, so build the items twice.
    try:
        batched = transform_episode(*dataset.episode_items(span, positions), transform, keys)
    except Exception as e:  # Any failure means the transforms cannot be applied to a whole episode.
        logger.warning("Transforms cannot be applied to a whole episode at once (%r); using per-frame mode.", e)
        return False
    per_frame = transform_episode(*dataset.episode_items(span, positions), transform, keys, per_frame_mode=True)
    for key in keys:
        a, b = batched[key], per_frame[key]
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            logger.warning("Batched transform differs from per-frame transform for key %r; using per-frame mode.", key)
            return False
    return True


def iter_sequential_batches(
    dataset: LowDimLeRobotDataset,
    transform: _transforms.DataTransformFn,
    keys: Sequence[str],
    batch_size: int,
    num_batches: int,
    *,
    per_frame_mode: bool = False,
) -> Iterator[dict[str, np.ndarray]]:
    """The first `num_batches` batches of `TorchDataLoader(transformed_dataset, batch_size, shuffle=False)`.

    I.e. frames `0 .. num_batches * batch_size` in order, `batch_size` consecutive frames per batch.
    """
    num_frames = num_batches * batch_size
    if num_frames > len(dataset):
        raise ValueError(f"Requested {num_frames} frames but the dataset only has {len(dataset)}.")
    buffers: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    buffered = 0
    emitted = 0
    for span in dataset.spans:
        if span.offset >= num_frames:
            break
        if len(span) == 0:
            continue
        outputs = transform_episode(*dataset.episode_items(span), transform, keys, per_frame_mode=per_frame_mode)
        if span.offset + len(span) > num_frames:
            outputs = {key: value[: num_frames - span.offset] for key, value in outputs.items()}
        for key in keys:
            buffers[key].append(outputs[key])
        buffered += len(outputs[keys[0]])
        if buffered < batch_size:
            continue
        num_full = buffered // batch_size
        merged = {
            key: np.concatenate(buffers[key], axis=0) if len(buffers[key]) > 1 else buffers[key][0] for key in keys
        }
        for i in range(num_full):
            yield {key: np.ascontiguousarray(merged[key][i * batch_size : (i + 1) * batch_size]) for key in keys}
        emitted += num_full
        buffers = {key: [merged[key][num_full * batch_size :]] for key in keys}
        buffered -= num_full * batch_size
    if emitted != num_batches:
        raise RuntimeError(f"Produced {emitted} batches instead of {num_batches}.")


def iter_gathered_batches(
    dataset: LowDimLeRobotDataset,
    transform: _transforms.DataTransformFn,
    keys: Sequence[str],
    batch_indices: Sequence[np.ndarray],
    *,
    per_frame_mode: bool = False,
) -> Iterator[dict[str, np.ndarray]]:
    """Batches made of the frames listed in `batch_indices` (e.g. from `shuffled_batch_indices`), in that order.

    The needed frames are computed episode by episode and kept in memory, so this suits subsampled runs.
    """
    if len(batch_indices) == 0:
        return
    all_indices = np.unique(np.concatenate([np.asarray(indices, dtype=np.int64) for indices in batch_indices]))
    if all_indices[0] < 0 or all_indices[-1] >= len(dataset):
        raise IndexError(f"Frame indices must be in [0, {len(dataset)}).")
    offsets = np.asarray([span.offset for span in dataset.spans])
    span_of_frame = np.searchsorted(offsets, all_indices, side="right") - 1

    gathered: dict[str, np.ndarray] = {}
    for span_position in np.unique(span_of_frame):
        rows = np.nonzero(span_of_frame == span_position)[0]
        span = dataset.spans[span_position]
        outputs = transform_episode(
            *dataset.episode_items(span, all_indices[rows] - span.offset),
            transform,
            keys,
            per_frame_mode=per_frame_mode,
        )
        for key in keys:
            if key not in gathered:
                gathered[key] = np.empty((len(all_indices), *outputs[key].shape[1:]), dtype=outputs[key].dtype)
            gathered[key][rows] = outputs[key]

    for indices in batch_indices:
        rows = np.searchsorted(all_indices, np.asarray(indices, dtype=np.int64))
        yield {key: gathered[key][rows] for key in keys}


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, num_items: int):
        self._num_items = num_items

    def __getitem__(self, index: int) -> int:
        return index

    def __len__(self) -> int:
        return self._num_items


def shuffled_batch_indices(num_frames: int, batch_size: int, num_batches: int, *, seed: int = 0) -> list[np.ndarray]:
    """Frame indices of the first `num_batches` batches of a shuffled `TorchDataLoader`.

    Builds the torch `DataLoader` exactly like `openpi.training.data_loader.TorchDataLoader(dataset,
    local_batch_size=batch_size, shuffle=True, seed=seed)` does (seeded generator, `RandomSampler`, `drop_last`), so
    the order is the same for any number of loader workers. Limited to a single pass over the data: the order of
    subsequent passes depends on whether the loader uses persistent workers. The equivalence is pinned down by a test
    in `lerobot_lowdim_test.py`.
    """
    if num_frames < batch_size:
        raise ValueError(f"Local batch size ({batch_size}) is larger than the dataset size ({num_frames}).")
    if num_batches > num_frames // batch_size:
        raise ValueError(f"{num_batches} batches of {batch_size} exceed one pass over {num_frames} frames.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        _IndexDataset(num_frames),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        collate_fn=lambda items: np.asarray(items, dtype=np.int64),
    )
    batches: list[np.ndarray] = []
    for indices in loader:
        if len(batches) == num_batches:
            break
        batches.append(indices)
    return batches


def referenced_data_files(meta: Any, episodes: Sequence[int] | None = None) -> list[pathlib.Path]:
    """Data files (relative to the root, sorted, de-duplicated) holding the given episodes (all if None).

    This is the file selection of `openpi.training.b1k_dataset.B1KLeRobotDataset`.
    """
    episode_table = meta.episodes
    episode_indices = _metadata_column(episode_table, "episode_index")
    chunk_indices = _metadata_column(episode_table, "data/chunk_index")
    file_indices = _metadata_column(episode_table, "data/file_index")
    selected = None if episodes is None else {int(episode) for episode in episodes}
    return sorted(
        {
            pathlib.Path(meta.data_path.format(chunk_index=chunk_index, file_index=file_index))
            for episode_index, chunk_index, file_index in zip(episode_indices, chunk_indices, file_indices, strict=True)
            if selected is None or int(episode_index) in selected
        }
    )


def episode_metadata_is_positional(meta: Any) -> bool:
    """Whether row `i` of the episode metadata is episode `i`, which `LeRobotDataset` relies on (`meta.episodes[i]`)."""
    episode_indices = _metadata_column(meta.episodes, "episode_index")
    return bool(np.array_equal(episode_indices, np.arange(len(episode_indices))))


def _metadata_column(episodes: Any, name: str) -> np.ndarray:
    """A column of the episode metadata (a HF `Dataset` or a pandas `DataFrame`, depending on the LeRobot version)."""
    if hasattr(episodes, "data") and hasattr(episodes.data, "column"):
        return np.asarray(episodes.data.column(name).to_numpy())
    return np.asarray(episodes[name])


def _column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    """Convert a parquet column to a (num_rows, *feature_shape) array; (nested) fixed-length lists are flattened."""
    array = column.combine_chunks()
    shape = []
    while pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        if pa.types.is_fixed_size_list(array.type):
            length = array.type.list_size
        else:
            lengths = np.diff(array.offsets.to_numpy())
            if len(lengths) > 0 and np.any(lengths != lengths[0]):
                raise ValueError("Ragged list columns are not supported.")
            length = int(lengths[0]) if len(lengths) > 0 else 0
        shape.append(length)
        array = array.flatten()
    return array.to_numpy(zero_copy_only=False).reshape(len(column), *shape)


def _to_torch(values: np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert like LeRobot's `hf_transform_to_torch` (`torch.tensor` of Python scalars, strings kept) would."""
    if values.dtype == np.bool_:
        return torch.from_numpy(np.ascontiguousarray(values))
    if np.issubdtype(values.dtype, np.integer):
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64))
    if np.issubdtype(values.dtype, np.floating):
        return torch.from_numpy(np.ascontiguousarray(values, dtype=_numpy_dtype(torch.get_default_dtype())))
    return values  # Strings and other objects are passed through.


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    return torch.empty((), dtype=dtype).numpy().dtype
