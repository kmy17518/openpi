"""BEHAVIOR-1K challenge demos (LeRobot v3.0) as a local openpi training dataset.

The demos ship as one 100-task dataset on the Hub (``behavior-1k/2026-challenge-demos``), one chunk per
task (``chunk-000`` = task 0 = ``turning_on_radio``, ...). A training root is either the complete download or
the per-task partial download from the challenge docs (``data/<chunk>/**``, ``meta/episodes/<chunk>/**``,
``videos/*/<chunk>/**`` plus the dataset-wide ``meta/info.json`` / ``stats.json`` / ``tasks.parquet``), and
training may be restricted to some tasks with ``task_names`` -- the same way on both layouts.

Two things in ``lerobot.datasets.LeRobotDataset`` break on a partial download of a middle chunk, and this
module works around both without touching the lerobot fork:

* it addresses episode metadata by *row position* (``meta.episodes[episode_index]``), which only equals
  ``episode_index`` when every chunk is present -- chunk-042 alone has 200 rows carrying indices 8400..8599;
* it treats the local root as an incomplete Hub cache when it holds fewer episodes than ``meta/info.json``
  announces and falls back to ``snapshot_download`` (3.3 TB for this dataset), and reports lengths from the
  dataset-wide totals in ``info.json`` instead of the rows on disk.

:class:`B1KLeRobotDataset` is a drop-in replacement for ``LeRobotDataset`` as ``DataConfig.data_cls`` that
reads local roots only, looks episodes up by ``episode_index``, and loads only the data files holding the
selected episodes (a one-task subset of the full root opens 6 parquet files instead of 955).
"""

from collections.abc import Iterable, Sequence
import dataclasses
import hashlib
import json
import logging
import pathlib
import re
from typing import Any, Literal, get_args

import datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import hf_transform_to_torch
from lerobot.utils.import_utils import get_safe_default_video_backend
from lerobot.utils.utils import SuppressProgressBars
import numpy as np
import pyarrow.dataset as pa_ds
import torch

from openpi.configs.tasks import TASK_REGISTRY

# Which text of a BEHAVIOR task the policy is conditioned on (the two kinds shipped per task in the demos'
# ``meta/tasks.jsonl``; see :func:`task_prompts`):
#   task_name        -- the LeRobot task string of ``meta/tasks.parquet``, for the challenge demos the snake_case
#                       id, e.g. ``turning_on_radio`` (stock openpi ``prompt_from_task`` behavior);
#   task_description -- the natural-language instruction, e.g. "Turn on the radio receiver that's on the table in
#                       the living room.", from ``meta/tasks.jsonl`` (fallback: openpi's ``TASK_REGISTRY["b1k"]``).
PromptSource = Literal["task_name", "task_description"]
PROMPT_SOURCES: tuple[str, ...] = get_args(PromptSource)
DEFAULT_PROMPT_SOURCE: PromptSource = "task_name"
TASKS_JSONL_FILENAME = "tasks.jsonl"
# Written next to ``norm_stats.json`` in a checkpoint's assets so serving can pick the same kind of text.
PROMPT_SOURCE_FILENAME = "prompt_source.json"
TASK_REGISTRY_BUCKET = "b1k"

# Norm stats of a task subset live under ``<repo_id>/task_subsets/<key>/`` (mirrors the GR00T baseline's
# ``meta/task_subsets/<key>/``), so per-task statistics never shadow the dataset-wide ones and different
# subsets of the same dataset do not clobber each other.
TASK_SUBSETS_DIR_NAME = "task_subsets"
_TASK_SUBSET_KEY_SEPARATOR = "+"
_TASK_SUBSET_KEY_SAFE_NAME = re.compile(r"[A-Za-z0-9_.\-]+")
_TASK_SUBSET_KEY_MAX_LEN = 96


def normalize_task_names(task_names: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Canonical ``task_names`` value: sorted unique tuple, or ``None`` for "all tasks"."""
    if task_names is None:
        return None
    if isinstance(task_names, str):
        task_names = [task_names]
    names = tuple(sorted({str(name) for name in task_names}))
    return names or None


def task_subset_key(task_names: str | Iterable[str]) -> str:
    """Directory name for a task subset: ``picking_up_trash+turning_on_radio`` when the names are short
    filesystem-safe identifiers, a content hash otherwise."""
    names = normalize_task_names(task_names)
    if names is None:
        raise ValueError("task_subset_key needs at least one task name")
    joined = _TASK_SUBSET_KEY_SEPARATOR.join(names)
    if len(joined) <= _TASK_SUBSET_KEY_MAX_LEN and all(_TASK_SUBSET_KEY_SAFE_NAME.fullmatch(name) for name in names):
        return joined
    digest = hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()[:16]
    return f"sha256-{digest}"


def task_subset_asset_id(repo_id: str, task_names: str | Iterable[str] | None) -> str:
    """Asset id (norm stats directory) for ``repo_id`` restricted to ``task_names``; ``repo_id`` itself when
    ``task_names`` is empty."""
    names = normalize_task_names(task_names)
    if names is None:
        return repo_id
    return f"{repo_id}/{TASK_SUBSETS_DIR_NAME}/{task_subset_key(names)}"


@dataclasses.dataclass(frozen=True)
class TaskSubset:
    """Episodes of a dataset restricted to ``task_names`` (see :func:`select_task_subset`)."""

    task_names: tuple[str, ...]
    task_indices: frozenset[int]
    episode_indices: tuple[int, ...]


def select_task_indices(tasks: Any, task_names: Iterable[str]) -> frozenset[int]:
    """Resolve task names to ``task_index`` values through the dataset's ``meta/tasks.parquet``.

    ``tasks`` is ``LeRobotDatasetMetadata.tasks``: a DataFrame indexed by the task string (for the challenge
    demos the snake_case task name, e.g. ``turning_on_radio``) with a ``task_index`` column. Unknown names
    raise ``ValueError`` listing the available ones.
    """
    by_name: dict[str, set[int]] = {}
    for task_str, task_index in zip(tasks.index, tasks["task_index"], strict=True):
        by_name.setdefault(str(task_str), set()).add(int(task_index))
    selected: set[int] = set()
    unknown: list[str] = []
    for name in normalize_task_names(task_names) or ():
        if name in by_name:
            selected |= by_name[name]
        else:
            unknown.append(name)
    if unknown:
        available = sorted(by_name)
        shown = ", ".join(available[:20]) + (f", ... ({len(available)} tasks total)" if len(available) > 20 else "")
        raise ValueError(f"Unknown task(s) {unknown}; meta/tasks.parquet has: {shown}")
    return frozenset(selected)


def select_task_subset(meta: LeRobotDatasetMetadata, task_names: str | Iterable[str]) -> TaskSubset:
    """Resolve ``task_names`` against a dataset root's ``meta/`` (works on a partial download).

    Episodes are matched on their ``task_index`` when the episode metadata carries it (the challenge demos
    do), else on LeRobot's per-episode ``tasks`` strings. Raises ``ValueError`` for an unknown name or when
    no episode on disk belongs to the selected tasks (e.g. a partial download of *other* tasks).
    """
    names = normalize_task_names(task_names)
    if names is None:
        raise ValueError("select_task_subset needs at least one task name")
    task_indices = select_task_indices(meta.tasks, names)
    episodes = meta.episodes
    episode_indices = _column(episodes, "episode_index")
    if "task_index" in episodes.column_names:
        episode_tasks = [int(task) for task in _column(episodes, "task_index")]
        selected = [int(ep) for ep, task in zip(episode_indices, episode_tasks, strict=True) if task in task_indices]
        on_disk = f"task_index {sorted(set(episode_tasks))}"
    else:
        task_strings = {
            str(task_str)
            for task_str, idx in zip(meta.tasks.index, meta.tasks["task_index"], strict=True)
            if int(idx) in task_indices
        }
        selected = [
            int(ep)
            for ep, tasks in zip(episode_indices, _column(episodes, "tasks"), strict=True)
            if task_strings & {str(t) for t in (tasks or ())}
        ]
        on_disk = "other tasks"
    if not selected:
        raise ValueError(
            f"No episodes of task(s) {list(names)} (task_index {sorted(task_indices)}) in {meta.root}: the "
            f"{len(episode_indices)} episodes on disk belong to {on_disk} -- is this a partial download of other tasks?"
        )
    return TaskSubset(task_names=names, task_indices=task_indices, episode_indices=tuple(sorted(selected)))


def episode_task_indices(meta: LeRobotDatasetMetadata) -> set[int]:
    """``task_index`` values of the episodes on disk (falls back to every task of ``meta/tasks.parquet`` when the
    episode metadata does not carry ``task_index``)."""
    if "task_index" in meta.episodes.column_names:
        return {int(task) for task in _column(meta.episodes, "task_index")}
    return set(task_names_by_index(meta.tasks))


def task_names_by_index(tasks: Any) -> dict[int, str]:
    """``task_index -> task string`` of ``LeRobotDatasetMetadata.tasks`` (``meta/tasks.parquet``)."""
    return {
        int(task_index): str(task_str) for task_str, task_index in zip(tasks.index, tasks["task_index"], strict=True)
    }


def load_task_descriptions(root: str | pathlib.Path, task_names: dict[int, str]) -> dict[int, str] | None:
    """``task_index -> natural-language description`` from ``meta/tasks.jsonl``, or None if the file is absent.

    Each line is ``{"task_index", "task_name", "task"}`` (``task`` being the description). Rows are checked
    against ``task_names`` (``meta/tasks.parquet``) so a stale or foreign ``tasks.jsonl`` cannot silently attach
    the wrong instruction to a task.
    """
    path = pathlib.Path(root) / "meta" / TASKS_JSONL_FILENAME
    if not path.is_file():
        return None
    descriptions: dict[int, str] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_index = int(row["task_index"])
            expected = task_names.get(task_index)
            if expected is not None and "task_name" in row and str(row["task_name"]) != expected:
                raise ValueError(
                    f"{path} disagrees with meta/tasks.parquet on task_index {task_index}: "
                    f"{row['task_name']!r} vs {expected!r}"
                )
            descriptions[task_index] = str(row["task"])
    return descriptions


def task_prompts(
    meta: LeRobotDatasetMetadata,
    prompt_source: str = DEFAULT_PROMPT_SOURCE,
    required_task_indices: Iterable[int] | None = None,
) -> dict[int, str]:
    """``task_index -> prompt text`` for ``PromptFromLeRobotTask``.

    ``task_name``: the task string of ``meta/tasks.parquet``. ``task_description``: the instruction of the
    dataset's ``meta/tasks.jsonl``; for tasks it lacks (or if the file is absent, e.g. an older download),
    openpi's task registry (``configs/tasks/b1k.py``) supplies the description by task name. Raises if a
    description is missing for any of ``required_task_indices`` (default: every task of the dataset).
    """
    if prompt_source not in PROMPT_SOURCES:
        raise ValueError(f"Unknown prompt_source {prompt_source!r}; choose from {list(PROMPT_SOURCES)}")
    names = task_names_by_index(meta.tasks)
    if prompt_source == "task_name":
        return names
    descriptions = load_task_descriptions(meta.root, names) or {}
    registry = TASK_REGISTRY.get(TASK_REGISTRY_BUCKET, {})
    for task_index, name in names.items():
        if task_index not in descriptions and name in registry:
            descriptions[task_index] = registry[name]
    required = set(names) if required_task_indices is None else {int(i) for i in required_task_indices}
    if missing := sorted(required - set(descriptions)):
        raise ValueError(
            f"No task description for task_index {missing[:10]} ({[names.get(i, '?') for i in missing[:10]]}): "
            f"neither {pathlib.Path(meta.root) / 'meta' / TASKS_JSONL_FILENAME} nor openpi's task registry "
            f"(src/openpi/configs/tasks/{TASK_REGISTRY_BUCKET}.py) has one. Download meta/tasks.jsonl or use "
            "--data.prompt-source task_name."
        )
    return {task_index: descriptions[task_index] for task_index in names if task_index in descriptions}


def check_prompt_token_lengths(prompts: dict[int, str], model_config: Any) -> None:
    """Fail fast if a prompt cannot fit ``model_config.max_token_len`` (PaliGemma-tokenized pi0 / pi05 models).

    ``PaligemmaTokenizer`` truncates over-long prompts from the end -- for pi05, whose prompt is
    ``Task: <text>, State: <discretized state>;\\nAction:``, that drops state digits and the ``Action:`` marker, so
    the policy would silently lose its proprioception on those tasks. The challenge demos' task descriptions run
    up to ~105 tokens, which next to a 32-dim state does not always fit the default ``max_token_len=200``. The
    state is assumed worst case (every dimension a 3-digit bin).
    """
    model_type = getattr(model_config, "model_type", None)
    if model_type is None or model_type.value not in ("pi0", "pi05"):
        return
    from openpi.models import tokenizer as _tokenizer  # local import: fetches the tokenizer model on first use

    max_len = int(model_config.max_token_len)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=max(8 * max_len, 4096))  # long enough to never truncate
    state = np.full(int(model_config.action_dim), 1.0) if getattr(model_config, "discrete_state_input", False) else None
    too_long: dict[int, tuple[str, int]] = {}
    for task_index, prompt in prompts.items():
        length = int(tokenizer.tokenize(prompt, state=state)[1].sum())
        if length > max_len:
            too_long[task_index] = (prompt, length)
    if too_long:
        needed = max(length for _, length in too_long.values())
        shown = "; ".join(
            f"task_index {i}: {length} tokens ({p[:60]!r}...)" for i, (p, length) in list(too_long.items())[:5]
        )
        raise ValueError(
            f"{len(too_long)} task prompt(s) exceed max_token_len={max_len}"
            f"{' together with the discretized state' if state is not None else ''} and would be truncated: {shown}. "
            f"Pass --model.max-token-len {needed} (or more), or prompt with --data.prompt-source task_name."
        )


def save_prompt_source(assets_dir: Any, prompt_source: str) -> None:
    """Record ``prompt_source`` in a checkpoint's assets directory (next to ``norm_stats.json``)."""
    if prompt_source not in PROMPT_SOURCES:
        raise ValueError(f"Unknown prompt_source {prompt_source!r}; choose from {list(PROMPT_SOURCES)}")
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / PROMPT_SOURCE_FILENAME).write_text(json.dumps({"prompt_source": prompt_source}))


def load_prompt_source(assets_dir: str | pathlib.Path) -> str | None:
    """The ``prompt_source`` a checkpoint was trained with, or None for checkpoints that predate the record."""
    path = pathlib.Path(assets_dir) / PROMPT_SOURCE_FILENAME
    if not path.is_file():
        return None
    prompt_source = json.loads(path.read_text())["prompt_source"]
    if prompt_source not in PROMPT_SOURCES:
        raise ValueError(f"{path} holds unknown prompt_source {prompt_source!r}; choose from {list(PROMPT_SOURCES)}")
    return prompt_source


def _column(table: Any, name: str) -> list:
    """A column of an HF dataset (or the proxy below) as a Python list."""
    return table.data.column(name).to_pylist()


class _EpisodesByIndex:
    """Episode-metadata table addressed by ``episode_index`` rather than row position.

    ``LeRobotDatasetMetadata.episodes`` is the HF dataset of the ``meta/episodes/chunk-*/file-*.parquet``
    rows, and lerobot's reader looks records up as ``meta.episodes[episode_index]`` -- correct only while row
    position == ``episode_index``, i.e. on a complete download. This wrapper resolves non-negative integer
    keys through an ``episode_index -> row`` map and otherwise behaves like the wrapped dataset (``len``,
    iteration, column access, ``.filter`` and every other attribute).
    """

    def __init__(self, table: datasets.Dataset):
        self._table = table
        episode_indices = table.data.column("episode_index").to_pylist()
        self._row_by_episode = {int(ep): row for row, ep in enumerate(episode_indices)}
        if len(self._row_by_episode) != len(episode_indices):
            raise ValueError("Duplicate episode_index values in the episode metadata")

    @property
    def episode_indices(self) -> list[int]:
        """Sorted ``episode_index`` values present in the metadata."""
        return sorted(self._row_by_episode)

    def __getitem__(self, key):
        if isinstance(key, int | np.integer) and not isinstance(key, bool) and key >= 0:
            row = self._row_by_episode.get(int(key))
            if row is None:
                raise IndexError(
                    f"episode_index {int(key)} is not in the episode metadata on disk "
                    f"({len(self._row_by_episode)} episodes present)"
                )
            return self._table[row]
        return self._table[key]

    def __len__(self) -> int:
        return len(self._table)

    def __iter__(self):
        return iter(self._table)

    def __getattr__(self, name: str):
        # Only reached for attributes not found on the proxy. Never resolve dunders / private names through
        # the wrapped table: pickle probes e.g. ``__setstate__`` on a half-built instance (dataloader workers).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._table, name)


class B1KDatasetMetadata(LeRobotDatasetMetadata):
    """``LeRobotDatasetMetadata`` for a *local* root that may be a per-task partial download.

    Never downloads from the Hub (an incomplete root is an error, not a cache miss), and exposes
    ``.episodes`` addressed by ``episode_index`` (see :class:`_EpisodesByIndex`).
    """

    def __init__(self, repo_id: str, root: str | pathlib.Path):
        root = pathlib.Path(root)
        # Validate up front: the parent treats a missing meta file as a Hub cache miss and starts with a Hub
        # API call, which would surface as a confusing network/auth error for a plain bad path.
        required = [root / "meta" / "info.json", root / "meta" / "tasks.parquet"]
        missing = [str(path) for path in required if not path.is_file()]
        if not any((root / "meta" / "episodes").glob("*/*.parquet")):
            missing.append(str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet"))
        if missing:
            raise FileNotFoundError(
                f"{root} is not a (complete) LeRobot v3.0 dataset root, missing: {', '.join(missing)}. openpi reads "
                "B1K datasets from local disk only; download the dataset (or one task's chunk plus meta/) first."
            )
        super().__init__(repo_id, root=root)

    def _pull_from_repo(self, allow_patterns=None, ignore_patterns=None) -> None:
        raise FileNotFoundError(
            f"Refusing to download {self.repo_id} from the Hub into {self.root}: openpi reads B1K datasets from "
            "local disk only."
        )

    def _load_metadata(self) -> None:
        super()._load_metadata()
        self.episodes = _EpisodesByIndex(self.episodes)

    def _episode(self, ep_index: int) -> dict:
        try:
            return self.episodes[ep_index]
        except IndexError as e:
            raise IndexError(f"{e} (dataset root {self.root})") from e

    def get_data_file_path(self, ep_index: int) -> pathlib.Path:
        ep = self._episode(ep_index)
        return pathlib.Path(self.data_path.format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]))

    def get_video_file_path(self, ep_index: int, vid_key: str) -> pathlib.Path:
        ep = self._episode(ep_index)
        return pathlib.Path(
            self.video_path.format(
                video_key=vid_key,
                chunk_index=ep[f"videos/{vid_key}/chunk_index"],
                file_index=ep[f"videos/{vid_key}/file_index"],
            )
        )


class _B1KDatasetReader(DatasetReader):
    """``DatasetReader`` over an explicit list of data files, with lengths and frame-index mapping derived
    from the rows actually loaded instead of the dataset-wide totals in ``meta/info.json``."""

    def __init__(self, *args, data_files: Sequence[pathlib.Path], video_keys: Iterable[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_files = [str(path) for path in data_files]
        # Video streams to decode; None = every stream in the dataset (lerobot's behavior). Requested keys that are
        # not video streams (e.g. image columns stored in the parquet files) need no decoding and are ignored here.
        self._video_keys: frozenset[str] | None = None
        if video_keys is not None:
            self._video_keys = frozenset(video_keys) & set(self._meta.video_keys)

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        if self._video_backend == "pyav" and quiet_pyav_logging():
            # Once per process (data-loader workers included): see quiet_pyav_logging.
            logging.info("Disabled PyAV's Python log forwarding for the pyav video backend")
        if self._video_keys is not None:
            query_timestamps = {key: ts for key, ts in query_timestamps.items() if key in self._video_keys}
        return super()._query_videos(query_timestamps, ep_idx)

    def _load_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self._meta.features)
        filters = pa_ds.field("episode_index").isin(list(self.episodes)) if self.episodes is not None else None
        with SuppressProgressBars():
            hf_dataset = datasets.Dataset.from_parquet(self._data_files, filters=filters, features=features)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _build_index_mapping(self) -> None:
        # Frame lookups by absolute ``index`` (delta timestamps) only work positionally when the loaded rows
        # are exactly 0..N-1. Otherwise (task subset, or a partial download whose first frame is
        # 98,415,414) map absolute index -> row. Rows are unique and ascending, so first/last suffice.
        self._absolute_to_relative_idx = None
        index_column = self.hf_dataset.data.column("index")
        num_rows = len(index_column)
        if num_rows == 0:
            raise ValueError("No frames loaded from the selected data files")
        if index_column[0].as_py() == 0 and index_column[num_rows - 1].as_py() == num_rows - 1:
            return
        indices = index_column.to_numpy()
        self._absolute_to_relative_idx = dict(zip(indices.tolist(), range(num_rows), strict=True))

    @property
    def num_frames(self) -> int:
        if self.hf_dataset is None:
            raise RuntimeError("Dataset not loaded")
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else len(self._meta.episodes)


def _referenced_files(
    meta: LeRobotDatasetMetadata, episode_indices: Sequence[int] | None
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """(data files, video files) referenced by the given episodes (all episodes on disk if ``None``),
    relative to the dataset root, sorted and de-duplicated."""
    episodes = meta.episodes
    all_episodes = _column(episodes, "episode_index")
    if episode_indices is None:
        rows = range(len(all_episodes))
    else:
        keep = {int(ep) for ep in episode_indices}
        rows = [row for row, ep in enumerate(all_episodes) if int(ep) in keep]

    def files(pattern: str, chunk_column: str, file_column: str, **extra) -> set[pathlib.Path]:
        chunks, files_ = _column(episodes, chunk_column), _column(episodes, file_column)
        return {pathlib.Path(pattern.format(chunk_index=chunks[r], file_index=files_[r], **extra)) for r in rows}

    data_files = files(meta.data_path, "data/chunk_index", "data/file_index")
    video_files: set[pathlib.Path] = set()
    for key in meta.video_keys:
        video_files |= files(meta.video_path, f"videos/{key}/chunk_index", f"videos/{key}/file_index", video_key=key)
    return sorted(data_files), sorted(video_files)


def default_video_backend() -> str:
    """lerobot's default decoder, but only ``torchcodec`` if it actually loads.

    lerobot picks ``torchcodec`` whenever the package is *installed*; when its FFmpeg / torch ABI does not
    match (e.g. an aarch64 venv without matching FFmpeg shared libraries) every ``__getitem__`` would fail,
    so fall back to ``pyav`` in that case.
    """
    backend = get_safe_default_video_backend()
    if backend == "torchcodec":
        try:
            import torchcodec  # noqa: F401
        except Exception as e:  # torchcodec raises RuntimeError/OSError when its shared libraries do not load
            logging.warning(
                "torchcodec is installed but cannot be loaded (%s); decoding videos with pyav", str(e).splitlines()[0]
            )
            return "pyav"
    return backend


def quiet_pyav_logging() -> bool:
    """Drop FFmpeg's log messages inside PyAV instead of forwarding them to Python. Returns True if it changed.

    ``torchvision.io`` (imported by lerobot) calls ``av.logging.set_level(av.logging.ERROR)`` at import time. That
    installs PyAV's Python log callback, which acquires the GIL for *every* FFmpeg log call -- including the tens of
    thousands of TRACE-level atom messages libavformat emits while parsing an mp4 header -- so with the ``pyav``
    backend every ``av.open`` in the reader's per-camera decode threads takes seconds instead of milliseconds
    (measured here: 6 concurrent opens 6 ms -> 2.3 s; a sample 0.1 s -> 3.8 s). PyAV's default is to ignore FFmpeg
    logs entirely (``set_level(None)`` installs a no-op C callback); restore that. Only PyAV's error-message
    bookkeeping is lost. Must run in every process that decodes (data-loader workers re-import torchvision).
    """
    import av.logging

    if av.logging.get_level() is None:
        return False
    av.logging.set_level(None)
    return True


def _check_files_exist(root: pathlib.Path, files: Sequence[pathlib.Path], what: str) -> None:
    missing = [str(path) for path in files if not (root / path).is_file()]
    if missing:
        shown = ", ".join(missing[:5]) + (f", ... ({len(missing)} total)" if len(missing) > 5 else "")
        raise FileNotFoundError(
            f"{len(missing)} {what} file(s) referenced by the selected episodes are missing under {root}: {shown}"
        )


class B1KLeRobotDataset(torch.utils.data.Dataset):
    """Local LeRobot v3.0 dataset: the complete root or a per-task partial download, optionally restricted
    to ``task_names`` and/or explicit ``episodes`` (``episode_index`` values; intersected when both are given).

    Accepts the ``LeRobotDataset`` read-mode constructor arguments used by openpi (``repo_id``, ``root``,
    ``episodes``, ``delta_timestamps``, ``tolerance_s``, ``video_backend``, ``image_transforms``,
    ``return_uint8``); items are produced by lerobot's ``DatasetReader`` exactly as ``LeRobotDataset`` does, except
    that ``video_keys`` restricts which video streams are decoded (default: all of them). The challenge demos carry
    six streams per frame (RGB and depth for three cameras) and the R1Pro policy consumes the three RGB ones, so
    decoding only those halves the per-sample cost of the data loader.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | pathlib.Path,
        *,
        task_names: str | Iterable[str] | None = None,
        episodes: Sequence[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        image_transforms: Any = None,
        return_uint8: bool = False,
        video_keys: Iterable[str] | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = pathlib.Path(root)
        self.meta = B1KDatasetMetadata(repo_id, self.root)
        self.task_names = normalize_task_names(task_names)
        self.task_subset = select_task_subset(self.meta, self.task_names) if self.task_names else None

        selected: list[int] | None = None
        if self.task_subset is not None:
            selected = list(self.task_subset.episode_indices)
        if episodes is not None:
            requested = sorted({int(ep) for ep in episodes})
            on_disk = set(self.meta.episodes.episode_indices)
            if missing := [ep for ep in requested if ep not in on_disk]:
                raise ValueError(f"Episodes {missing[:10]}{'...' if len(missing) > 10 else ''} are not in {self.root}")
            selected = requested if selected is None else sorted(set(selected) & set(requested))
            if not selected:
                raise ValueError(
                    f"No episode of task(s) {list(self.task_names or ())} among episodes {requested[:10]}..."
                )
        self.episodes = selected

        data_files, video_files = _referenced_files(self.meta, self.episodes)
        _check_files_exist(self.root, data_files, "data")
        _check_files_exist(self.root, video_files, "video")

        self.reader = _B1KDatasetReader(
            meta=self.meta,
            root=self.root,
            episodes=self.episodes,
            tolerance_s=tolerance_s,
            video_backend=video_backend or default_video_backend(),
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            return_uint8=return_uint8,
            data_files=[self.root / path for path in data_files],
            video_keys=video_keys,
        )
        self.reader.load_and_activate()
        logging.info(
            "B1KLeRobotDataset(%s): %d episodes, %d frames, %d data files under %s%s; decoding %s",
            repo_id,
            self.num_episodes,
            self.num_frames,
            len(data_files),
            self.root,
            f" (task subset {list(self.task_names)})" if self.task_names else "",
            f"{len(self.video_keys)} of {len(self.meta.video_keys)} video streams: {sorted(self.video_keys)}"
            if video_keys is not None
            else f"all {len(self.meta.video_keys)} video streams",
        )

    @property
    def video_keys(self) -> list[str]:
        """Video streams decoded per item (all of the dataset's unless ``video_keys`` restricted them)."""
        keys = self.reader._video_keys  # noqa: SLF001
        return list(self.meta.video_keys) if keys is None else [k for k in self.meta.video_keys if k in keys]

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def num_frames(self) -> int:
        return self.reader.num_frames

    @property
    def num_episodes(self) -> int:
        return self.reader.num_episodes

    @property
    def hf_dataset(self) -> datasets.Dataset:
        return self.reader.hf_dataset

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        return self.reader.get_item(idx)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(repo_id={self.repo_id!r}, root={str(self.root)!r}, "
            f"task_names={self.task_names}, num_episodes={self.num_episodes}, num_frames={self.num_frames})"
        )
