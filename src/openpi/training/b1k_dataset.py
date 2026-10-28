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

import collections
from collections.abc import Iterable, Sequence
import dataclasses
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import logging
import math
import pathlib
import re
from typing import Any, Literal, get_args

import av
import datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import hf_transform_to_torch
from lerobot.utils.import_utils import get_safe_default_video_backend
from lerobot.utils.utils import SuppressProgressBars
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
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

# Goal images (goal-image conditioning): one image per selected camera view, no observation-history axis.
#   episode_last -- the last recorded frame of the episode's own camera stream (the settled terminal state, also
#                   when --data.settle-steps cuts the episode earlier; works on any LeRobot v3 root, including the
#                   challenge demos and ISG);
#   goal_key     -- a dedicated goal video stream shipped by the dataset (`observation.goal_rgb.<camera>_camera_0`,
#                   static per episode, as in the radio skill-segment datasets);
#   norobot_last -- the episode's last recorded state re-rendered with the robot hidden (the dataset's sidecar
#                   `isg_meta/goal_norobot/<camera>/<episode:06d>.png` + manifest.json, format isg-goal-norobot/v1,
#                   from train85/work/goal_norobot/render_goals.py): the same camera and scene state as episode_last
#                   without the robot's arms and torso in the view.
GoalSource = Literal["episode_last", "goal_key", "norobot_last"]
GOAL_SOURCES: tuple[str, ...] = get_args(GoalSource)
DEFAULT_GOAL_SOURCE: GoalSource = "episode_last"
GOAL_NOROBOT_DIR = "isg_meta/goal_norobot"
GOAL_NOROBOT_FORMAT = "isg-goal-norobot/v1"

EPISODE_SPLIT_FILE = "isg_meta/train_split.json"
EPISODE_SPLIT_FORMAT = "isg-episode-split/v1"
EPISODE_SPLIT_SUBSETS = ("train", "held_out")
TASK_GROUPS_FILE = "isg_meta/task_groups.json"
TASK_GROUPS_FORMAT = "isg-task-groups/v1"
SETTLE_WINDOWS_FILE = "isg_meta/settle_windows.json"
SETTLE_WINDOWS_FORMAT = "isg-settle-windows/v1"
# Same record and episodes as ACT's `resolve_val_split` (b1k_training.py) for the same seed, tasks and fraction.
VAL_SPLIT_FORMAT = "isg-val-split/v1"

# Norm stats of a task subset live under ``<repo_id>/task_subsets/<key>/`` (mirrors the GR00T baseline's
# ``meta/task_subsets/<key>/``), so per-task statistics never shadow the dataset-wide ones and different
# subsets of the same dataset do not clobber each other.
TASK_SUBSETS_DIR_NAME = "task_subsets"
SETTLE_STEPS_DIR_NAME = "settle_steps"
GRIPPER_ACTION_DIR_NAME = "gripper_action"
VAL_FRACTION_DIR_NAME = "val_fraction"
SEED_DIR_NAME = "seed"
_TASK_SUBSET_KEY_SEPARATOR = "+"
_TASK_SUBSET_KEY_SAFE_NAME = re.compile(r"[A-Za-z0-9_.\-]+")
_TASK_SUBSET_KEY_MAX_LEN = 96


def expand_task_groups(
    dataset_path: str | pathlib.Path | None, task_names: str | Iterable[str] | None
) -> list[str] | None:
    """``task_names`` with every task group of ``<root>/isg_meta/task_groups.json`` replaced by the tasks it reaches.

    A group lists task names and/or other groups. ``None`` (every task) stays ``None``, and without a group file
    the names are returned unchanged; otherwise the tasks come back without duplicates in order of first appearance.
    """
    if task_names is None:
        return None
    names = [str(task_names)] if isinstance(task_names, str) else [str(name) for name in task_names]
    if dataset_path is None:
        return names
    path = pathlib.Path(dataset_path) / TASK_GROUPS_FILE
    if not path.is_file():
        return names
    content = json.loads(path.read_text())
    groups = content.get("groups")
    if content.get("format") != TASK_GROUPS_FORMAT or not isinstance(groups, dict):
        raise ValueError(f"{path} is not an {TASK_GROUPS_FORMAT} task group file (format {content.get('format')!r})")
    tasks = {
        row.get("task", row.get("__index_level_0__"))
        for row in pq.read_table(pathlib.Path(dataset_path) / "meta/tasks.parquet").to_pylist()
    }

    def expand(name: str, parents: tuple[str, ...]) -> list[str]:
        if name in tasks:
            if name in groups:
                raise ValueError(f"{name!r} is both a task and a task group in {path}")
            return [name]
        if name not in groups:
            if parents:
                raise ValueError(f"Task group {parents[-1]!r} in {path} lists unknown task or group {name!r}")
            raise ValueError(
                f"Unknown task or task group {name!r}; groups in {path}: {sorted(groups)}; tasks: {sorted(tasks)}"
            )
        if name in parents:
            raise ValueError(f"Task group cycle {' -> '.join([*parents, name])} in {path}")
        members = groups[name]
        if not isinstance(members, list) or not members or not all(isinstance(member, str) for member in members):
            raise ValueError(f"Task group {name!r} in {path} must be a nonempty list of task or group names")
        return [task for member in members for task in expand(member, (*parents, name))]

    return list(dict.fromkeys(task for name in names for task in expand(name, ())))


def resolve_episode_split(
    dataset_path: str | pathlib.Path | None,
    spec: str = "auto",
    task_names: str | Iterable[str] | None = None,
    *,
    subset: str = "train",
) -> dict[str, Any] | None:
    """Training episodes of ``--data.episode-split SPEC`` as a JSON-serializable record, or None without a split.

    ``auto`` reads ``<root>/isg_meta/train_split.json`` when that file exists (no split otherwise), ``none``
    disables the split, any other value is the path of a split file. The selected tasks are ``task_names``
    (expanded through ``isg_meta/task_groups.json`` when present) or, when omitted, every task of the file;
    the episodes are the sorted union of their ``subset`` lists (``train``, or ``held_out`` for validation, which
    skips the selected tasks that have no held-out episode with a warning).
    """
    if subset not in EPISODE_SPLIT_SUBSETS:
        raise ValueError(f"Episode split subset must be one of {EPISODE_SPLIT_SUBSETS}, got {subset!r}")
    if spec == "none" or dataset_path is None:
        if spec not in ("auto", "none") and dataset_path is None:
            path = pathlib.Path(spec)
        else:
            return None
    else:
        path = pathlib.Path(dataset_path) / EPISODE_SPLIT_FILE if spec == "auto" else pathlib.Path(spec)
    if not path.is_file():
        if spec == "auto":
            return None
        raise FileNotFoundError(f"Episode split file {path} not found")
    path = path.resolve()
    content = path.read_bytes()
    split = json.loads(content)
    if split.get("format") != EPISODE_SPLIT_FORMAT or not isinstance(split.get("tasks"), dict):
        raise ValueError(f"{path} is not an {EPISODE_SPLIT_FORMAT} episode split (format {split.get('format')!r})")
    expanded = expand_task_groups(dataset_path, task_names) if task_names is not None else None
    tasks = list(dict.fromkeys(expanded or split["tasks"]))
    missing = [name for name in tasks if name not in split["tasks"]]
    if missing:
        raise ValueError(f"Selected tasks {missing} have no entry in episode split {path}")
    if subset == "held_out":
        if empty := [name for name in tasks if not split["tasks"][name].get(subset)]:
            logging.warning("Tasks %s have no held-out episodes in %s; validating without them", empty, path)
            tasks = [name for name in tasks if name not in empty]
        if not tasks:
            raise ValueError(f"None of the selected tasks has held-out episodes in {path}")
    episodes = sorted({int(episode) for name in tasks for episode in split["tasks"][name][subset]})
    return {
        "file": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "format": split["format"],
        "name": split.get("name"),
        "subset": subset,
        "tasks": tasks,
        "episodes": episodes,
    }


def spread_frame_indices(episode_index: Sequence[int] | np.ndarray, count: int, seed: int) -> list[int]:
    """``count`` distinct row positions spread over every episode, the same for the same rows and ``seed``.

    ``episode_index`` is the episode of each row (rows of an episode contiguous). Each episode gets a share of
    ``count`` proportional to its length (at least one row while ``count`` covers every episode); its rows are
    drawn one per equal-length stretch of the episode at a seeded offset. Every row when ``count`` covers them all.
    """
    episode_index = np.asarray(episode_index)
    if count >= len(episode_index):
        return list(range(len(episode_index)))
    _, starts, lengths = np.unique(episode_index, return_index=True, return_counts=True)
    order = np.argsort(starts)
    starts, lengths = starts[order], lengths[order]
    if np.any(np.diff(starts) != lengths[:-1]):
        raise ValueError("Rows of an episode must be contiguous")
    share = count * lengths / lengths.sum()
    quota = np.floor(share).astype(int)
    if count >= len(lengths):
        quota = np.maximum(quota, 1)
    # Hand out what is left to the largest remainders (taking back from the largest shares when the minimum of one
    # row per episode overshoots), never beyond an episode's length.
    while (left := count - quota.sum()) != 0:
        if left > 0:
            room = np.where(quota < lengths, share - quota, -np.inf)
            quota[int(np.argmax(room))] += 1
        else:
            spare = np.where(quota > 1, quota - share, -np.inf)
            quota[int(np.argmax(spare))] -= 1
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    for start, length, n in zip(starts, lengths, quota, strict=True):
        if n == 0:
            continue
        edges = np.linspace(0, length, n + 1)
        low, high = np.ceil(edges[:-1]).astype(int), np.ceil(edges[1:]).astype(int)
        rows.extend(int(start + lo + rng.integers(hi - lo)) for lo, hi in zip(low, high, strict=True))
    return rows


def figure_frame_indices(episode_index: Sequence[int] | np.ndarray, count: int, seed: int) -> list[int]:
    """Row positions of one seeded frame in each of ``count`` seeded episodes (every episode when there are fewer),
    the same for the same rows and ``seed``."""
    episode_index = np.asarray(episode_index)
    episodes = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    picks = np.sort(rng.choice(len(episodes), min(count, len(episodes)), replace=False))
    return [int(rng.choice(np.flatnonzero(episode_index == episodes[i]))) for i in picks]


def describe_episode_split(split: dict[str, Any] | None) -> str:
    """One line for logs and errors: name, file, number of episodes and tasks of a split record."""
    if split is None:
        return "no episode split"
    return (
        f"{split['name']} ({split['file']}, sha256 {split['sha256'][:12]}): {len(split['episodes'])} "
        f"{split['subset']} episodes of tasks {split['tasks']}"
    )


def split_val_episodes(
    episodes_by_task: dict[str, Iterable[int]], val_fraction: float, seed: int
) -> dict[str, dict[str, list[int]]]:
    """``{task: {"train": [...], "val": [...]}}``: each task's episodes permuted with
    ``SeedSequence([seed, sha256(task)])``, the first ``round(N * val_fraction)`` (at least 1, at most N - 1) of
    them validation episodes."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"--data.val-fraction must be in [0, 1); got {val_fraction}")
    by_task = {}
    for task_name, task_episodes in episodes_by_task.items():
        episodes = sorted(int(ep) for ep in task_episodes)
        if len(episodes) < 2:
            raise ValueError(
                f"Task {task_name!r} has {len(episodes)} episode(s); at least 2 are required for "
                f"--data.val-fraction {val_fraction}"
            )
        n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_fraction)))
        task_seed = int.from_bytes(hashlib.sha256(task_name.encode()).digest()[:8], "little")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), task_seed]))
        perm = [episodes[i] for i in rng.permutation(len(episodes))]
        by_task[task_name] = {"train": sorted(perm[n_val:]), "val": sorted(perm[:n_val])}
    return by_task


def resolve_val_split(
    meta: LeRobotDatasetMetadata,
    task_names: Iterable[str] | None,
    val_fraction: float,
    seed: int | None,
    *,
    episodes: Iterable[int] | None = None,
) -> dict[str, Any] | None:
    """Per-task train/val episode split of ``--data.val-fraction`` as a JSON-serializable record, or None when the
    fraction is 0.

    The candidates of a task are its episodes on disk, restricted to ``task_names`` (every task when None) and to
    ``episodes`` (e.g. an episode split's ``train`` list) when given; see :func:`split_val_episodes`.
    """
    if not val_fraction:
        return None
    if seed is None:
        raise ValueError("--data.val-fraction needs the training seed (TrainConfig.seed)")
    names = task_names_by_index(meta.tasks)
    if "task_index" in meta.episodes.column_names:
        episode_tasks = [int(task) for task in _column(meta.episodes, "task_index")]
    else:
        by_name = {name: index for index, name in names.items()}
        episode_tasks = []
        for tasks in _column(meta.episodes, "tasks"):
            if not tasks or len(tasks) != 1 or str(tasks[0]) not in by_name:
                raise ValueError(f"{meta.root}: each episode needs one known task or an explicit task_index")
            episode_tasks.append(by_name[str(tasks[0])])
    selected = None if task_names is None else set(task_names)
    allowed = None if episodes is None else {int(ep) for ep in episodes}
    by_task: dict[int, list[int]] = {}
    for ep, task in zip(_column(meta.episodes, "episode_index"), episode_tasks, strict=True):
        if (selected is None or names[task] in selected) and (allowed is None or int(ep) in allowed):
            by_task.setdefault(task, []).append(int(ep))
    tasks = split_val_episodes({names[task]: by_task[task] for task in sorted(by_task)}, val_fraction, seed)
    return {
        "format": VAL_SPLIT_FORMAT,
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "tasks": tasks,
        "train_episodes": sorted(ep for entry in tasks.values() for ep in entry["train"]),
        "val_episodes": sorted(ep for entry in tasks.values() for ep in entry["val"]),
    }


def describe_val_split(val_split: dict[str, Any] | None) -> str:
    """One line for logs: fraction, seed, train/val episode counts and task count."""
    if val_split is None:
        return "no validation split"
    return (
        f"val_fraction={val_split['val_fraction']} (seed {val_split['seed']}): "
        f"{len(val_split['train_episodes'])} train / {len(val_split['val_episodes'])} val episodes "
        f"across {len(val_split['tasks'])} tasks"
    )


def parse_settle_steps(spec: str | int | float) -> str:
    """Canonical ``--data.settle-steps`` SPEC: ``all``, a number of settle frames (``'0'``, ``'10'``) or a decimal
    fraction of each episode's program length (``'0.2'``; ``'1.0'`` is the whole program length, ``'1'`` one frame).
    Raises ``ValueError`` otherwise.
    """
    text = str(spec).strip()
    if text == "all":
        return text
    if re.fullmatch(r"[0-9]+", text):
        return str(int(text))
    if re.fullmatch(r"[0-9]*\.[0-9]+|[0-9]+\.", text):
        value = format(Decimal(text).normalize(), "f")
        return value if "." in value else f"{value}.0"
    raise ValueError(
        "--settle-steps must be all, a number of frames (0, 10) or a decimal fraction of the program "
        f"length (0.1, 0.2); got {spec!r}"
    )


def settle_frames(spec: str, program_length: int) -> int:
    """Settle frames that a canonical SPEC other than ``all`` requests after ``program_length`` program frames."""
    return math.ceil(Fraction(spec) * program_length) if "." in spec else int(spec)


def apply_settle_steps(
    dataset_path: str | pathlib.Path,
    spec: str | int | float,
    episodes: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Cut episodes (``meta/episodes`` rows; ``length`` and ``dataset_to_index`` change in place) to ``--settle-steps SPEC``.

    ``all`` keeps every recorded frame and returns ``None``. Otherwise each episode keeps its program -- the first
    ``program_length`` frames listed by ``<root>/isg_meta/settle_windows.json`` -- followed by SPEC settle frames,
    capped at the recorded window; returns the record saved with runs and checkpoints.
    """
    spec = parse_settle_steps(spec)
    if spec == "all":
        return None
    path = pathlib.Path(dataset_path).resolve() / SETTLE_WINDOWS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"--settle-steps {spec} needs {path}, the program length of every episode; only datasets "
            "whose episodes end with a recorded settle window ship it"
        )
    content = path.read_bytes()
    windows = json.loads(content)
    if windows.get("format") != SETTLE_WINDOWS_FORMAT or not isinstance(windows.get("episodes"), dict):
        raise ValueError(f"{path} is not an {SETTLE_WINDOWS_FORMAT} file (format {windows.get('format')!r})")
    recorded = settle = capped = 0
    for ep in episodes:
        entry = windows["episodes"].get(str(ep["episode_index"]))
        if not entry or entry.get("length") != ep["length"] or not 0 < entry.get("program_length", 0) <= ep["length"]:
            raise ValueError(
                f"{path} has no settle window for episode {ep['episode_index']} of length {ep['length']} "
                f"(entry {entry}); regenerate it for this dataset"
            )
        program = int(entry["program_length"])
        wanted = settle_frames(spec, program)
        kept = min(wanted, int(ep["length"]) - program)
        capped += wanted > kept
        recorded += int(ep["length"])
        settle += kept
        ep["length"] = program + kept
        ep["dataset_to_index"] = int(ep["dataset_from_index"]) + ep["length"]
    if capped:
        logging.warning(
            "--settle-steps %s exceeds the recorded settle window of %d of %d episodes; they keep their whole window",
            spec,
            capped,
            len(episodes),
        )
    return {
        "spec": spec,
        "file": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "format": SETTLE_WINDOWS_FORMAT,
        "frames": sum(int(ep["length"]) for ep in episodes),
        "settle_frames": settle,
        "recorded_frames": recorded,
        "capped_episodes": capped,
    }


def describe_settle_steps(settle: dict[str, Any] | None) -> str:
    """One line for logs and errors: kept, settle and recorded frames of a settle record."""
    if settle is None:
        return "all (every recorded frame)"
    return (
        f"{settle['spec']} ({settle['file']}, sha256 {settle['sha256'][:12]}): {settle['frames']} of "
        f"{settle['recorded_frames']} recorded frames, {settle['settle_frames']} of them settle frames; "
        f"{settle['capped_episodes']} episodes keep their whole recorded window"
    )


def normalize_task_names(
    task_names: str | Iterable[str] | None,
    *,
    dataset_root: str | pathlib.Path | None = None,
) -> tuple[str, ...] | None:
    """Canonical ``task_names`` value: sorted unique tuple (with task groups expanded when ``dataset_root`` has
    ``isg_meta/task_groups.json``), or ``None`` for "all tasks"."""
    if task_names is None:
        return None
    if isinstance(task_names, str):
        # Accept comma-separated task / group names (e.g. from shell scripts) as well as individual strings.
        raw = [part for part in task_names.split(",") if part] if "," in task_names else [task_names]
    else:
        raw = [part for item in task_names for part in (str(item).split(",") if "," in str(item) else [str(item)]) if part]
    if not raw:
        return None
    if dataset_root is not None:
        raw = expand_task_groups(dataset_root, raw) or []
    names = tuple(sorted({str(name) for name in raw}))
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


def task_subset_asset_id(
    repo_id: str,
    task_names: str | Iterable[str] | None,
    *,
    settle_steps: str = "all",
    gripper_action: str = "absolute",
    val_fraction: float = 0.0,
    seed: int | None = None,
) -> str:
    """Asset id (norm stats directory) for ``repo_id`` restricted to ``task_names`` and ``settle_steps``, under
    ``gripper_action``, without the validation episodes of ``val_fraction`` and ``seed``; ``repo_id`` itself when
    ``task_names`` is empty, ``settle_steps == 'all'``, the gripper actions are absolute and ``val_fraction`` is 0."""
    names = normalize_task_names(task_names)
    spec = parse_settle_steps(settle_steps)
    asset_id = repo_id if names is None else f"{repo_id}/{TASK_SUBSETS_DIR_NAME}/{task_subset_key(names)}"
    if spec != "all":
        asset_id = f"{asset_id}/{SETTLE_STEPS_DIR_NAME}/{spec}"
    if gripper_action != "absolute":
        asset_id = f"{asset_id}/{GRIPPER_ACTION_DIR_NAME}/{gripper_action}"
    if val_fraction:
        asset_id = f"{asset_id}/{VAL_FRACTION_DIR_NAME}/{float(val_fraction)}/{SEED_DIR_NAME}/{seed}"
    return asset_id


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


def select_task_subset(
    meta: LeRobotDatasetMetadata,
    task_names: str | Iterable[str],
    *,
    episodes: Iterable[int] | None = None,
) -> TaskSubset:
    """Select episodes, requiring coverage of every requested task on this local root.

    Match ``task_index`` when present, otherwise resolve episode task strings through ``meta/tasks.parquet``.
    ``episodes`` optionally restricts the available episodes before checking task coverage. Raises ``ValueError``
    for an unknown name or when any selected task has no episode on disk (e.g. a partial download of *other*
    tasks) -- a requested task is never silently omitted.
    """
    names = normalize_task_names(task_names)
    if names is None:
        raise ValueError("select_task_subset needs at least one task name")
    task_indices = select_task_indices(meta.tasks, names)
    by_name: dict[str, set[int]] = {}
    for name, index in zip(meta.tasks.index, meta.tasks["task_index"], strict=True):
        by_name.setdefault(str(name), set()).add(int(index))
    episode_indices = _column(meta.episodes, "episode_index")
    if "task_index" in meta.episodes.column_names:
        episode_tasks = [{int(task)} for task in _column(meta.episodes, "task_index")]
    else:
        episode_tasks = [
            {index for task in (tasks or ()) for index in by_name.get(str(task), ())}
            for tasks in _column(meta.episodes, "tasks")
        ]
    allowed = None if episodes is None else {int(ep) for ep in episodes}
    selected = []
    available: set[int] = set()
    for ep, tasks in zip(episode_indices, episode_tasks, strict=True):
        if allowed is not None and int(ep) not in allowed:
            continue
        available.update(tasks)
        if tasks & task_indices:
            selected.append(int(ep))
    if missing := task_indices - available:
        missing_names = [name for name in names if by_name[name] & missing]
        selection = "" if allowed is None else " within the requested episode selection"
        raise ValueError(
            f"No episodes of task(s) {missing_names} (task_index {sorted(missing)}) in {meta.root}{selection}: the "
            f"available episodes belong to task_index {sorted(available)} -- is this a partial download of other tasks?"
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


def check_prompt_token_lengths(prompts: dict[int, str], model_config: Any, state_dim: int | None = None) -> None:
    """Fail fast if a prompt cannot fit ``model_config.max_token_len`` (PaliGemma-tokenized pi0 / pi05 models).

    ``PaligemmaTokenizer`` truncates over-long prompts from the end -- for pi05, whose prompt is
    ``Task: <text>, State: <discretized state>;\\nAction:``, that drops state digits and the ``Action:`` marker, so
    the policy would silently lose its proprioception on those tasks. The challenge demos' task descriptions run
    up to ~105 tokens, which next to the state does not always fit the default ``max_token_len=200``. The
    state is assumed worst case (every dimension a 3-digit bin). ``state_dim`` is the extracted state length
    before model padding (23 for R1Pro): pi05 tokenizes the prompt *before* ``PadStatesAndActions``, so counting
    the model's padded ``action_dim`` (32) would over-estimate the prompt by ~36 tokens. If omitted, use the model's
    action dimension.
    """
    model_type = getattr(model_config, "model_type", None)
    if model_type is None or model_type.value not in ("pi0", "pi05"):
        return
    from openpi.models import tokenizer as _tokenizer  # local import: fetches the tokenizer model on first use

    max_len = int(model_config.max_token_len)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=max(8 * max_len, 4096))  # long enough to never truncate
    state_dim = int(model_config.action_dim) if state_dim is None else int(state_dim)
    state = np.full(state_dim, 1.0) if getattr(model_config, "discrete_state_input", False) else None
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
    if hasattr(table, "column") and callable(table.column):
        return table.column(name)
    return table.data.column(name).to_pylist()


class _EpisodesByIndex:
    """Episode-metadata table addressed by ``episode_index`` rather than row position.

    ``LeRobotDatasetMetadata.episodes`` is the HF dataset of the ``meta/episodes/chunk-*/file-*.parquet``
    rows, and lerobot's reader looks records up as ``meta.episodes[episode_index]`` -- correct only while row
    position == ``episode_index``, i.e. on a complete download. This wrapper resolves non-negative integer
    keys through an ``episode_index -> row`` map and otherwise behaves like the wrapped dataset (``len``,
    iteration, column access, ``.filter`` and every other attribute).
    """

    def __init__(self, table: datasets.Dataset, overrides: dict[int, dict[str, Any]] | None = None):
        self._table = table
        episode_indices = table.data.column("episode_index").to_pylist()
        self._row_by_episode = {int(ep): row for row, ep in enumerate(episode_indices)}
        if len(self._row_by_episode) != len(episode_indices):
            raise ValueError("Duplicate episode_index values in the episode metadata")
        self._overrides: dict[int, dict[str, Any]] = {
            int(ep): dict(values) for ep, values in (overrides or {}).items()
        }

    def set_overrides(self, overrides: dict[int, dict[str, Any]]) -> None:
        self._overrides = {int(ep): dict(values) for ep, values in overrides.items()}

    def column(self, name: str) -> list:
        values = self._table.data.column(name).to_pylist()
        if not self._overrides:
            return values
        eps = self._table.data.column("episode_index").to_pylist()
        return [
            self._overrides[int(ep)][name] if int(ep) in self._overrides and name in self._overrides[int(ep)] else val
            for ep, val in zip(eps, values, strict=True)
        ]

    @property
    def episode_indices(self) -> list[int]:
        """Sorted ``episode_index`` values present in the metadata."""
        return sorted(self._row_by_episode)

    def __getitem__(self, key):
        if isinstance(key, int | np.integer) and not isinstance(key, bool) and key >= 0:
            ep = int(key)
            row = self._row_by_episode.get(ep)
            if row is None:
                raise IndexError(
                    f"episode_index {ep} is not in the episode metadata on disk "
                    f"({len(self._row_by_episode)} episodes present)"
                )
            record = dict(self._table[row])
            if ep in self._overrides:
                record.update(self._overrides[ep])
            return record
        if isinstance(key, int | np.integer) and not isinstance(key, bool) and key < 0:
            record = dict(self._table[int(key)])
            ep = int(record["episode_index"])
            if ep in self._overrides:
                record.update(self._overrides[ep])
            return record
        if isinstance(key, str) and self._overrides:
            return self.column(key)
        return self._table[key]

    def __len__(self) -> int:
        return len(self._table)

    def __iter__(self):
        for record in self._table:
            ep = int(record["episode_index"])
            if ep in self._overrides:
                record = {**record, **self._overrides[ep]}
            yield record

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


def quiet_pyav_logging() -> None:
    """Restore PyAV's default (no-op) FFmpeg log callback, once per process.

    ``torchvision.io`` -- imported by lerobot -- calls ``av.logging.set_level(av.logging.ERROR)`` at import time,
    which installs PyAV's Python-side log callback. That callback takes the GIL for every FFmpeg log line, and
    libavformat emits tens of thousands of trace lines while parsing the index of a packed 200 MB mp4. With six
    camera streams opened concurrently in threads (lerobot's ``_query_videos``) the GIL round-trips make each
    ``av.open`` take ~2 s instead of ~2 ms (measured on the challenge demos), i.e. essentially all of a sample's
    loading time. ``set_level(None)`` removes the callback; only PyAV's error-message bookkeeping is lost.
    """
    if not getattr(quiet_pyav_logging, "done", False):
        av.logging.set_level(None)
        quiet_pyav_logging.done = True  # type: ignore[attr-defined]


class VideoFrameReader:
    """Random access into packed LeRobot v3 videos with one long-lived decoder per file.

    A LeRobot v3.0 root packs many episodes into one mp4 per camera; a training sample touches one frame of each
    camera at a random position. lerobot's PyAV path re-opens the file for every access (parsing the index of a
    200 MB file costs ~10 ms), converts *every* frame it decodes on the way from the keyframe to RGB, and returns
    float32. This reader keeps an LRU of open containers per process (like the video readers of the diffusion
    policy, ACT and GR00T baselines), decodes with one FFmpeg thread so many data-loader workers do not
    oversubscribe the CPUs, converts only the frames that were asked for, and returns them as uint8 HWC -- which is
    what the B1K image transforms turn lerobot's float32 CHW frames back into anyway, bit for bit.

    Frame selection replicates ``lerobot.datasets.video_utils.decode_video_frames_pyav`` exactly: seek to the
    keyframe at or before the first requested timestamp, decode forward until a frame at or past the last one, and
    take, for each requested timestamp, the closest decoded frame (which must lie strictly within ``tolerance_s``).
    Decoding from a keyframe yields the same pictures whether or not the container was just opened, so the frames
    are identical to lerobot's.

    Instances are created lazily per process (``__getstate__`` drops the open containers), so a dataset holding one
    can be pickled into data-loader workers.
    """

    def __init__(self, *, max_open_files: int = 32, decoder_threads: int = 1):
        if max_open_files < 1 or decoder_threads < 0:
            raise ValueError("max_open_files must be >= 1 and decoder_threads >= 0")
        self.max_open_files = max_open_files
        self.decoder_threads = decoder_threads
        self._containers: collections.OrderedDict[str, Any] = collections.OrderedDict()

    def __getstate__(self):
        return {"max_open_files": self.max_open_files, "decoder_threads": self.decoder_threads}

    def __setstate__(self, state):
        self.__init__(**state)

    def close(self) -> None:
        for container in self._containers.values():
            container.close()
        self._containers.clear()

    def _container(self, path: str):
        container = self._containers.get(path)
        if container is None:
            quiet_pyav_logging()
            container = av.open(path)
            stream = container.streams.video[0]
            if self.decoder_threads:
                # Frame/slice threading never changes decoded pixels; one thread per worker process scales best.
                stream.thread_type = "SLICE" if self.decoder_threads == 1 else "AUTO"
                stream.codec_context.thread_count = self.decoder_threads
            self._containers[path] = container
            while len(self._containers) > self.max_open_files:
                self._containers.popitem(last=False)[1].close()
        self._containers.move_to_end(path)
        return container

    def read(self, path: str | pathlib.Path, timestamps: Sequence[float], tolerance_s: float) -> np.ndarray:
        """Frames at ``timestamps`` (seconds in the video's own time base) as uint8 ``[T, H, W, 3]`` RGB."""
        path = str(path)
        first_ts, last_ts = min(timestamps), max(timestamps)
        container = self._container(path)
        stream = container.streams.video[0]
        # Same call as lerobot: offset in av.time_base units, land on the keyframe at or before first_ts.
        container.seek(int(first_ts * av.time_base), backward=True)
        decoded: list[tuple[float, Any]] = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            decoded.append((current_ts, frame))
            if current_ts >= last_ts:
                break
        if not decoded:
            raise ValueError(f"No frames could be decoded from {path} in the timestamp range [{first_ts}, {last_ts}].")
        loaded_ts = np.asarray([ts for ts, _ in decoded], dtype=np.float64)
        query_ts = np.asarray(timestamps, dtype=np.float64)
        distances = np.abs(query_ts[:, None] - loaded_ts[None, :])
        closest = distances.argmin(axis=1)
        if not (distances[np.arange(len(query_ts)), closest] < tolerance_s).all():
            raise ValueError(
                f"Query timestamps {query_ts.tolist()} are farther than tolerance_s={tolerance_s} from the decoded "
                f"frames {loaded_ts.tolist()} of {path}"
            )
        converted: dict[int, np.ndarray] = {}
        for index in closest.tolist():
            if index not in converted:
                converted[index] = decoded[index][1].to_ndarray(format="rgb24")  # (H, W, 3) uint8, same as lerobot
        return np.stack([converted[index] for index in closest.tolist()])


class _B1KDatasetReader(DatasetReader):
    """``DatasetReader`` over an explicit list of data files, with lengths and frame-index mapping derived
    from the rows actually loaded instead of the dataset-wide totals in ``meta/info.json``.

    ``video_keys`` restricts decoding to those camera streams (the B1K robot configs use the three RGB cameras;
    the depth streams of the challenge demos are not decoded at all). Selected non-depth streams are read with
    :class:`VideoFrameReader`; depth streams, and everything when ``fast_video_reader`` is off, go through
    lerobot's decoder (``video_backend``).
    """

    def __init__(
        self,
        *args,
        data_files: Sequence[pathlib.Path],
        video_keys: Iterable[str] | None = None,
        fast_video_reader: bool = True,
        decoder_threads: int = 1,
        max_open_videos: int = 32,
        kept_indices: set[int] | None = None,
        goal_table: dict[int, dict[str, np.ndarray]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._data_files = [str(path) for path in data_files]
        self._kept_indices = kept_indices
        self._goal_table = goal_table
        available = list(self._meta.video_keys)
        if video_keys is None:
            self._video_keys = available
        else:
            requested = {str(key) for key in video_keys}
            if unknown := sorted(requested - set(available) - set(self._meta.camera_keys)):
                # Training configs request the robot's cameras without knowing the dataset; a camera the dataset lacks
                # surfaces as a missing key in the repack transform, with the dataset's keys listed here.
                logging.warning("Video keys %s are not streams of this dataset (which has %s); ignoring them", unknown, available)
            self._video_keys = [key for key in available if key in requested]
        depth = set(self._meta.depth_keys)
        self._fast_video_keys = frozenset(k for k in self._video_keys if k not in depth) if fast_video_reader else frozenset()
        self._frame_reader = VideoFrameReader(max_open_files=max_open_videos, decoder_threads=decoder_threads)

    @property
    def video_keys(self) -> list[str]:
        return list(self._video_keys)

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        # Only the requested streams; everything else lerobot would decode (e.g. depth) is skipped entirely.
        wanted = {key: ts for key, ts in query_timestamps.items() if key in self._video_keys}
        slow = {key: ts for key, ts in wanted.items() if key not in self._fast_video_keys}
        result = super()._query_videos(slow, ep_idx) if slow else {}
        if len(wanted) > len(slow):
            ep = self._meta.episodes[ep_idx]
            for key, query_ts in wanted.items():
                if key in slow:
                    continue
                from_timestamp = ep[f"videos/{key}/from_timestamp"]
                path = self.root / self._meta.get_video_file_path(ep_idx, key)
                frames = self._frame_reader.read(path, [from_timestamp + ts for ts in query_ts], self._tolerance_s)
                result[key] = torch.from_numpy(frames).squeeze(0)  # (H, W, 3) uint8, or (T, H, W, 3) for several
        return result

    def get_item(self, idx) -> dict:
        item = super().get_item(idx)
        if self._goal_table is not None:
            ep_idx = int(item["episode_index"])
            for key, frame in self._goal_table[ep_idx].items():
                item[key] = torch.from_numpy(frame)
        return item

    def _load_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self._meta.features)
        filters = pa_ds.field("episode_index").isin(list(self.episodes)) if self.episodes is not None else None
        if self._kept_indices is not None:
            idx_filter = pa_ds.field("index").isin(sorted(self._kept_indices))
            filters = idx_filter if filters is None else (filters & idx_filter)
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
    ``return_uint8``); items are produced by lerobot's ``DatasetReader`` exactly as ``LeRobotDataset`` does,
    except for the video frames:

    - ``video_keys`` selects the camera streams to decode (default: every video stream of the dataset). Training
      configs pass the robot config's cameras, so e.g. the challenge demos' depth streams are never decoded.
    - Selected RGB streams are read by :class:`VideoFrameReader` (``fast_video_reader``, default on): one
      long-lived single-threaded decoder per file and worker, only the requested frames converted, uint8 HWC
      output -- the same pixels as lerobot's float32 CHW frames, ~100x cheaper per sample on the challenge demos.
      ``decoder_threads`` / ``max_open_videos`` tune it; depth streams always use lerobot's decoder.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | pathlib.Path,
        *,
        task_names: str | Iterable[str] | None = None,
        episodes: Sequence[int] | None = None,
        episode_split: str = "none",
        settle_steps: str | int | float = "all",
        goal_camera_keys: dict[str, str] | None = None,
        goal_source: GoalSource = DEFAULT_GOAL_SOURCE,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        image_transforms: Any = None,
        return_uint8: bool = False,
        video_keys: Iterable[str] | None = None,
        fast_video_reader: bool = True,
        decoder_threads: int = 1,
        max_open_videos: int = 32,
    ):
        super().__init__()
        if image_transforms is not None and (video_keys is not None or fast_video_reader):
            raise ValueError(
                "image_transforms are applied by lerobot to every camera stream as float CHW frames; use them only "
                "with video_keys=None and fast_video_reader=False, or apply them in the data transforms instead."
            )
        self.repo_id = repo_id
        self.root = pathlib.Path(root)
        self.meta = B1KDatasetMetadata(repo_id, self.root)
        expanded_tasks = expand_task_groups(self.root, task_names)
        if task_names is not None and expanded_tasks != (
            [task_names] if isinstance(task_names, str) else list(task_names)
        ):
            logging.info("--data.task-names %s select the tasks %s", list(task_names), expanded_tasks)
        self.episode_split = resolve_episode_split(self.root, episode_split, expanded_tasks)
        if self.episode_split is not None:
            logging.info("Episode split %s", describe_episode_split(self.episode_split))
            if expanded_tasks is None:
                expanded_tasks = list(self.episode_split["tasks"])
        self.task_names = normalize_task_names(expanded_tasks)
        self.task_subset = select_task_subset(self.meta, self.task_names) if self.task_names else None

        selected: list[int] | None = None
        if self.task_subset is not None:
            selected = list(self.task_subset.episode_indices)
        if self.episode_split is not None:
            split_eps = list(self.episode_split["episodes"])
            on_disk = set(self.meta.episodes.episode_indices)
            if missing := [ep for ep in split_eps if ep not in on_disk]:
                raise ValueError(f"Episodes {missing[:10]}{'...' if len(missing) > 10 else ''} are not in {self.root}")
            selected = split_eps if selected is None else sorted(set(selected) & set(split_eps))
            if self.task_subset is not None:
                self.task_subset = select_task_subset(self.meta, self.task_names, episodes=selected)
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
            if self.task_subset is not None:
                # Every requested task must still be covered after the explicit episode filter.
                self.task_subset = select_task_subset(self.meta, self.task_names, episodes=selected)
        self.episodes = selected

        episode_ids = self.episodes if self.episodes is not None else self.meta.episodes.episode_indices
        episode_dicts = [dict(self.meta.episodes[ep]) for ep in episode_ids]
        self.recorded_ranges: dict[int, tuple[int, int]] = {
            int(ep["episode_index"]): (int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
            for ep in episode_dicts
        }
        self.settle = apply_settle_steps(self.root, settle_steps, episode_dicts)
        kept_indices: set[int] | None = None
        if self.settle is not None:
            logging.info("Settle steps %s", describe_settle_steps(self.settle))
            self.meta.episodes.set_overrides(
                {
                    int(ep["episode_index"]): {
                        "length": int(ep["length"]),
                        "dataset_to_index": int(ep["dataset_to_index"]),
                    }
                    for ep in episode_dicts
                }
            )
            kept_indices = {
                idx
                for ep in episode_dicts
                for idx in range(int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
            }

        data_files, video_files = _referenced_files(self.meta, self.episodes)
        _check_files_exist(self.root, data_files, "data")
        _check_files_exist(self.root, video_files, "video")

        goal_table: dict[int, dict[str, np.ndarray]] | None = None
        self.goal_source = goal_source
        if goal_camera_keys:
            if goal_source == "norobot_last":
                goal_table = self._load_norobot_goal_table(episode_ids, goal_camera_keys)
            else:
                goal_table = self._build_goal_table(episode_ids, goal_camera_keys, tolerance_s)

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
            fast_video_reader=fast_video_reader,
            decoder_threads=decoder_threads,
            max_open_videos=max_open_videos,
            kept_indices=kept_indices,
            goal_table=goal_table,
        )
        self.reader.load_and_activate()
        logging.info(
            "B1KLeRobotDataset(%s): %d episodes, %d frames, %d data files under %s%s; decoding %s%s",
            repo_id,
            self.num_episodes,
            self.num_frames,
            len(data_files),
            self.root,
            f" (task subset {list(self.task_names)})" if self.task_names else "",
            self.reader.video_keys,
            " with VideoFrameReader" if self.reader._fast_video_keys else " with lerobot's decoder",  # noqa: SLF001
        )

    def _build_goal_table(
        self,
        episode_ids: Sequence[int],
        goal_camera_keys: dict[str, str],
        tolerance_s: float,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Decode the last recorded frame of each requested camera stream per episode once at construction.

        ``goal_camera_keys`` maps ``output_key`` (e.g. ``observation.goal_rgb.zed_link_camera_0``) to the camera's
        video stream key (e.g. ``observation.rgb.zed_link_camera_0``). Uses ``self.recorded_ranges`` so that even
        when ``--data.settle-steps`` cuts the episode earlier, the goal image stays the last *recorded* (settled)
        frame of the episode, matching ACT's ``episode_last`` semantics.
        """
        by_file: dict[pathlib.Path, list[int]] = {}
        for ep_id in episode_ids:
            by_file.setdefault(self.root / self.meta.get_data_file_path(ep_id), []).append(int(ep_id))

        last_timestamps: dict[int, float] = {}
        for path, eps in sorted(by_file.items()):
            target_indices = {self.recorded_ranges[ep_id][1] - 1: ep_id for ep_id in eps}
            table = pq.read_table(
                path,
                columns=["index", "timestamp"],
                filters=[("index", "in", sorted(target_indices))],
            )
            for idx_val, ts_val in zip(table["index"].to_pylist(), table["timestamp"].to_pylist(), strict=True):
                last_timestamps[target_indices[int(idx_val)]] = float(ts_val)
            if missing := [ep_id for ep_id in eps if ep_id not in last_timestamps]:
                raise ValueError(f"Missing last-frame timestamp for episode(s) {missing} in {path}")

        reader = VideoFrameReader(max_open_files=16, decoder_threads=1)
        table_out: dict[int, dict[str, np.ndarray]] = {}
        try:
            for ep_id in episode_ids:
                ep = self.meta.episodes[ep_id]
                ts = last_timestamps[int(ep_id)]
                ep_goals: dict[str, np.ndarray] = {}
                for out_key, cam_key in goal_camera_keys.items():
                    from_ts = float(ep[f"videos/{cam_key}/from_timestamp"])
                    video_path = self.root / self.meta.get_video_file_path(ep_id, cam_key)
                    frame = reader.read(video_path, [from_ts + ts], tolerance_s)[0]
                    ep_goals[out_key] = frame
                table_out[int(ep_id)] = ep_goals
        finally:
            reader.close()
        logging.info(
            "Goal table (episode_last): %d episodes x %d views decoded",
            len(table_out),
            len(goal_camera_keys),
        )
        return table_out

    def _load_norobot_goal_table(
        self, episode_ids: Sequence[int], goal_camera_keys: dict[str, str]
    ) -> dict[int, dict[str, np.ndarray]]:
        """Read each requested camera's robot-free goal image per episode from the dataset's ``isg_meta/goal_norobot``
        sidecar (``manifest.json`` + ``<camera>/<episode:06d>.png``), the same shape as the camera stream's frames.

        ``goal_camera_keys`` maps ``output_key`` to the camera's video stream key as for ``episode_last``; the camera
        name is the stream key's middle part (``observation.rgb.zed_link_camera_0`` -> ``zed_link``).
        """
        from PIL import Image

        sidecar = self.root / GOAL_NOROBOT_DIR
        manifest_path = sidecar / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"goal_source=norobot_last needs {manifest_path} (the robot-free goal images rendered from the demos' "
                "last states: train85/work/goal_norobot); this dataset copy has none"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != GOAL_NOROBOT_FORMAT:
            raise ValueError(f"{manifest_path}: format {manifest.get('format')!r}, expected {GOAL_NOROBOT_FORMAT}")
        entries = manifest["episodes"]
        table_out: dict[int, dict[str, np.ndarray]] = {}
        for ep_id in episode_ids:
            entry = entries.get(str(int(ep_id)))
            if entry is None:
                raise ValueError(f"{manifest_path} has no robot-free goal image for episode {ep_id}")
            ep_goals: dict[str, np.ndarray] = {}
            for out_key, cam_key in goal_camera_keys.items():
                camera = cam_key.removeprefix("observation.rgb.").removesuffix("_camera_0")
                if camera not in entry:
                    raise ValueError(f"{manifest_path}: episode {ep_id} has no {camera} goal image (cameras {sorted(entry)})")
                path = sidecar / entry[camera]["path"]
                frame = np.asarray(Image.open(path).convert("RGB"))
                expected = tuple(self.meta.features[cam_key]["shape"])
                if frame.shape != expected:
                    raise ValueError(f"{path}: shape {frame.shape}, the {cam_key} stream has {expected}")
                ep_goals[out_key] = frame
            table_out[int(ep_id)] = ep_goals
        logging.info(
            "Goal table (norobot_last, %s): %d episodes x %d views read from %s",
            manifest.get("render", {}).get("renderer", "renders of the demos' last states"),
            len(table_out),
            len(goal_camera_keys),
            sidecar,
        )
        return table_out

    @property
    def video_keys(self) -> list[str]:
        """Camera streams that are decoded for each item."""
        return self.reader.video_keys

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
