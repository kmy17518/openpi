"""Tests for openpi.training.b1k_dataset on a small synthetic LeRobot v3.0 dataset.

The synthetic dataset mimics the challenge demos' layout -- one chunk per task, global ``episode_index`` and frame
``index`` -- so that a "partial download" (only the chunk of a middle task on disk) can be exercised without the
real 3 TB dataset.
"""

import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from openpi.training import b1k_dataset

FPS = 10
STATE_DIM, ACTION_DIM = 4, 3
TASKS = ["turning_on_radio", "picking_up_trash", "chop_an_onion"]  # task_index 0, 1, 2 == chunk index
EPISODES_PER_TASK = 2
FRAMES_PER_EPISODE = 12
HORIZON = 5


def _write_synthetic_root(root: pathlib.Path, chunks: list[int]) -> None:
    """Write ``meta/`` (dataset-wide) plus the data / episode-metadata files of ``chunks`` only."""
    meta = root / "meta"
    (meta / "episodes").mkdir(parents=True)
    (root / "data").mkdir()
    features = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": None},
        "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "features": features,
        "total_episodes": len(TASKS) * EPISODES_PER_TASK,
        "total_frames": len(TASKS) * EPISODES_PER_TASK * FRAMES_PER_EPISODE,
        "total_tasks": len(TASKS),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "robot_type": "R1Pro",
        "splits": {"train": f"0:{len(TASKS) * EPISODES_PER_TASK}"},
    }
    (meta / "info.json").write_text(json.dumps(info))
    tasks = pd.DataFrame({"task_index": range(len(TASKS))}, index=pd.Index(TASKS, name="task"))
    tasks.to_parquet(meta / "tasks.parquet")

    for chunk in chunks:
        episode_records, frames = [], []
        for demo in range(EPISODES_PER_TASK):
            episode_index = chunk * EPISODES_PER_TASK + demo
            start = episode_index * FRAMES_PER_EPISODE
            episode_records.append(
                {
                    "episode_index": episode_index,
                    "tasks": [TASKS[chunk]],
                    "task_index": chunk,
                    "length": FRAMES_PER_EPISODE,
                    "data/chunk_index": chunk,
                    "data/file_index": 0,
                    "dataset_from_index": start,
                    "dataset_to_index": start + FRAMES_PER_EPISODE,
                    "meta/episodes/chunk_index": chunk,
                    "meta/episodes/file_index": 0,
                }
            )
            for frame in range(FRAMES_PER_EPISODE):
                index = start + frame
                frames.append(
                    {
                        "observation.state": np.full(STATE_DIM, index, dtype=np.float32),
                        "action": np.full(ACTION_DIM, index, dtype=np.float32),
                        "timestamp": np.float32(frame / FPS),
                        "frame_index": frame,
                        "episode_index": episode_index,
                        "index": index,
                        "task_index": chunk,
                    }
                )
        (meta / "episodes" / f"chunk-{chunk:03d}").mkdir()
        pd.DataFrame(episode_records).to_parquet(meta / "episodes" / f"chunk-{chunk:03d}" / "file-000.parquet")
        (root / "data" / f"chunk-{chunk:03d}").mkdir()
        pd.DataFrame(frames).to_parquet(root / "data" / f"chunk-{chunk:03d}" / "file-000.parquet")


@pytest.fixture
def full_root(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "full"
    _write_synthetic_root(root, chunks=list(range(len(TASKS))))
    return root


@pytest.fixture
def partial_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Per-task partial download of the middle task (chunk-001 = picking_up_trash, episodes 2 and 3)."""
    root = tmp_path / "partial"
    _write_synthetic_root(root, chunks=[1])
    return root


def test_task_subset_key_and_asset_id():
    assert b1k_dataset.normalize_task_names(None) is None
    assert b1k_dataset.normalize_task_names([]) is None
    assert b1k_dataset.normalize_task_names("b") == ("b",)
    assert b1k_dataset.normalize_task_names(["b", "a", "b"]) == ("a", "b")
    assert b1k_dataset.task_subset_key(["turning_on_radio", "picking_up_trash"]) == "picking_up_trash+turning_on_radio"
    assert b1k_dataset.task_subset_key(["has space"]).startswith("sha256-")
    assert b1k_dataset.task_subset_key(["x" * 100]).startswith("sha256-")
    assert b1k_dataset.task_subset_asset_id("org/demos", None) == "org/demos"
    assert b1k_dataset.task_subset_asset_id("org/demos", ["chop_an_onion"]) == "org/demos/task_subsets/chop_an_onion"
    with pytest.raises(ValueError, match="at least one task name"):
        b1k_dataset.task_subset_key([])


def test_metadata_addresses_episodes_by_index_on_partial_root(partial_root: pathlib.Path):
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", partial_root)
    assert len(meta.episodes) == EPISODES_PER_TASK
    assert meta.episodes.episode_indices == [2, 3]
    assert meta.episodes[3]["episode_index"] == 3  # by episode_index, not row position
    assert meta.episodes[-1]["episode_index"] == 3  # negative keys stay positional
    assert list(meta.episodes["episode_index"]) == [2, 3]  # column access delegates to the table
    assert meta.get_data_file_path(2) == pathlib.Path("data/chunk-001/file-000.parquet")
    assert meta.filter_episodes(lambda ep: ep["task_index"] == 1) == [2, 3]
    with pytest.raises(IndexError, match="episode_index 0 is not in the episode metadata"):
        meta.get_data_file_path(0)
    # pickle probes dunders on a half-built instance; the proxy must not recurse into the wrapped table.
    assert getattr(meta.episodes, "__setstate__", None) is None
    assert pickle.loads(pickle.dumps(meta.episodes))[2]["episode_index"] == 2


def test_metadata_never_downloads(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="not a \\(complete\\) LeRobot v3.0 dataset root"):
        b1k_dataset.B1KDatasetMetadata("org/demos", tmp_path / "missing")
    root = tmp_path / "no_episodes"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    (root / "meta" / "tasks.parquet").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="meta/episodes"):
        b1k_dataset.B1KDatasetMetadata("org/demos", root)


def test_select_task_subset(full_root: pathlib.Path, partial_root: pathlib.Path):
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", full_root)
    subset = b1k_dataset.select_task_subset(meta, ["chop_an_onion", "turning_on_radio"])
    assert subset.task_names == ("chop_an_onion", "turning_on_radio")
    assert subset.task_indices == frozenset({0, 2})
    assert subset.episode_indices == (0, 1, 4, 5)

    with pytest.raises(ValueError, match="Unknown task\\(s\\) \\['chop_onion'\\]"):
        b1k_dataset.select_task_subset(meta, "chop_onion")

    partial = b1k_dataset.B1KDatasetMetadata("org/demos", partial_root)
    assert b1k_dataset.select_task_subset(partial, "picking_up_trash").episode_indices == (2, 3)
    with pytest.raises(
        ValueError, match="No episodes of task\\(s\\) \\['turning_on_radio'\\].*belong to task_index \\[1\\]"
    ):
        b1k_dataset.select_task_subset(partial, "turning_on_radio")


def _episode_indices(ds: b1k_dataset.B1KLeRobotDataset) -> set[int]:
    return {int(ds[i]["episode_index"]) for i in range(len(ds))}


def test_dataset_task_subset_on_full_root(full_root: pathlib.Path):
    delta_timestamps = {"action": [t / FPS for t in range(HORIZON)]}
    ds = b1k_dataset.B1KLeRobotDataset(
        "org/demos", full_root, task_names="picking_up_trash", delta_timestamps=delta_timestamps
    )
    assert ds.num_episodes == EPISODES_PER_TASK
    assert len(ds) == EPISODES_PER_TASK * FRAMES_PER_EPISODE
    assert _episode_indices(ds) == {2, 3}
    first = ds[0]
    assert int(first["index"]) == 2 * FRAMES_PER_EPISODE
    assert first["task"] == "picking_up_trash"
    assert first["action"].shape == (HORIZON, ACTION_DIM)
    # The action window is read by absolute frame index (values equal the frame index by construction).
    assert torch.equal(first["action"][:, 0], torch.arange(HORIZON, dtype=torch.float32) + 2 * FRAMES_PER_EPISODE)
    last = ds[len(ds) - 1]
    assert bool(last["action_is_pad"][-1])  # padded past the episode end

    ds_all = b1k_dataset.B1KLeRobotDataset("org/demos", full_root)
    assert len(ds_all) == len(TASKS) * EPISODES_PER_TASK * FRAMES_PER_EPISODE
    assert ds_all.reader._absolute_to_relative_idx is None  # noqa: SLF001 -- rows are 0..N-1, no mapping needed


def test_dataset_partial_root_with_and_without_task_names(partial_root: pathlib.Path):
    delta_timestamps = {"action": [t / FPS for t in range(HORIZON)]}
    ds = b1k_dataset.B1KLeRobotDataset("org/demos", partial_root, delta_timestamps=delta_timestamps)
    ds_named = b1k_dataset.B1KLeRobotDataset(
        "org/demos", partial_root, task_names=["picking_up_trash"], delta_timestamps=delta_timestamps
    )
    for dataset in (ds, ds_named):
        assert len(dataset) == EPISODES_PER_TASK * FRAMES_PER_EPISODE  # rows on disk, not info.json's total_frames
        assert dataset.num_episodes == EPISODES_PER_TASK
        assert _episode_indices(dataset) == {2, 3}
        item = dataset[FRAMES_PER_EPISODE + 1]  # second frame of episode 3
        assert int(item["episode_index"]) == 3
        assert int(item["frame_index"]) == 1
        assert torch.equal(
            item["action"][:, 0], torch.arange(HORIZON, dtype=torch.float32) + 3 * FRAMES_PER_EPISODE + 1
        )
    assert torch.equal(ds[5]["action"], ds_named[5]["action"])

    with pytest.raises(ValueError, match="No episodes of task"):
        b1k_dataset.B1KLeRobotDataset("org/demos", partial_root, task_names=["turning_on_radio"])
    with pytest.raises(ValueError, match="not in"):
        b1k_dataset.B1KLeRobotDataset("org/demos", partial_root, episodes=[0])


def test_dataset_explicit_episodes_intersect_task_subset(full_root: pathlib.Path):
    ds = b1k_dataset.B1KLeRobotDataset("org/demos", full_root, task_names=["chop_an_onion"], episodes=[5, 0])
    assert ds.episodes == [5]
    assert _episode_indices(ds) == {5}
    with pytest.raises(ValueError, match="No episode of task"):
        b1k_dataset.B1KLeRobotDataset("org/demos", full_root, task_names=["chop_an_onion"], episodes=[0])


def test_dataset_pickles_for_dataloader_workers(partial_root: pathlib.Path):
    ds = b1k_dataset.B1KLeRobotDataset("org/demos", partial_root, task_names="picking_up_trash")
    clone = pickle.loads(pickle.dumps(ds))
    assert len(clone) == len(ds)
    assert torch.equal(clone[7]["observation.state"], ds[7]["observation.state"])


DESCRIPTIONS = {
    "turning_on_radio": "Turn on the radio receiver that's on the table in the living room.",
    "picking_up_trash": "Put the three cans of soda from the living room inside the trash can in the kitchen.",
    "chop_an_onion": "Dice the onion.",
}


def _write_tasks_jsonl(root: pathlib.Path, names: list[str] = TASKS) -> None:
    lines = [
        json.dumps({"task_index": TASKS.index(name), "task_name": name, "task": DESCRIPTIONS[name]}) for name in names
    ]
    (root / "meta" / "tasks.jsonl").write_text("\n".join(lines) + "\n")


def test_task_prompts_task_name(full_root: pathlib.Path):
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", full_root)
    assert b1k_dataset.task_prompts(meta) == dict(enumerate(TASKS))
    assert b1k_dataset.task_prompts(meta, "task_name") == dict(enumerate(TASKS))
    assert b1k_dataset.episode_task_indices(meta) == {0, 1, 2}
    with pytest.raises(ValueError, match="Unknown prompt_source"):
        b1k_dataset.task_prompts(meta, "description")


def test_task_prompts_task_description_from_tasks_jsonl(full_root: pathlib.Path):
    _write_tasks_jsonl(full_root)
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", full_root)
    prompts = b1k_dataset.task_prompts(meta, "task_description")
    assert prompts == {i: DESCRIPTIONS[name] for i, name in enumerate(TASKS)}
    assert prompts[2] == "Dice the onion."  # the dataset's file wins over the registry copy

    # A tasks.jsonl that disagrees with tasks.parquet on a task's name is rejected.
    (full_root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task_name": "picking_up_trash", "task": "wrong"}) + "\n"
    )
    with pytest.raises(ValueError, match="disagrees with meta/tasks.parquet on task_index 0"):
        b1k_dataset.task_prompts(meta, "task_description")


def test_task_prompts_task_description_registry_fallback(full_root: pathlib.Path, monkeypatch):
    registry = {"turning_on_radio": "Radio.", "picking_up_trash": "Trash."}
    monkeypatch.setitem(b1k_dataset.TASK_REGISTRY, b1k_dataset.TASK_REGISTRY_BUCKET, registry)
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", full_root)  # no meta/tasks.jsonl
    # Only the tasks being trained on need a description.
    assert b1k_dataset.task_prompts(meta, "task_description", required_task_indices=[0, 1]) == {
        0: "Radio.",
        1: "Trash.",
    }
    with pytest.raises(ValueError, match="No task description for task_index \\[2\\] \\(\\['chop_an_onion'\\]\\)"):
        b1k_dataset.task_prompts(meta, "task_description")
    # The dataset's file fills the gap and takes precedence over the registry.
    _write_tasks_jsonl(full_root, ["chop_an_onion", "turning_on_radio"])
    assert b1k_dataset.task_prompts(meta, "task_description") == {
        0: DESCRIPTIONS["turning_on_radio"],
        1: "Trash.",
        2: "Dice the onion.",
    }


def test_check_prompt_token_lengths():
    from openpi.models import pi0_config

    pi05 = pi0_config.Pi0Config(pi05=True, action_dim=32, max_token_len=200)
    assert pi05.discrete_state_input
    short = {0: "turning_on_radio", 1: "Turn on the radio receiver that's on the table in the living room."}
    b1k_dataset.check_prompt_token_lengths(short, pi05)  # fits next to a worst-case 32-dim state
    long_prompt = " ".join(["walk to the kitchen and open the fridge"] * 12)  # ~110 tokens
    with pytest.raises(
        ValueError, match="exceed max_token_len=200 together with the discretized state.*--model.max-token-len"
    ):
        b1k_dataset.check_prompt_token_lengths({**short, 2: long_prompt}, pi05)
    b1k_dataset.check_prompt_token_lengths(
        {2: long_prompt}, pi0_config.Pi0Config(pi05=True, action_dim=32, max_token_len=300)
    )

    pi0 = pi0_config.Pi0Config(action_dim=32, max_token_len=48)  # no state in the prompt, but a 48-token budget
    b1k_dataset.check_prompt_token_lengths(short, pi0)
    with pytest.raises(ValueError, match="exceed max_token_len=48 and would be truncated"):
        b1k_dataset.check_prompt_token_lengths({2: long_prompt}, pi0)


def test_prompt_source_record_round_trip(tmp_path: pathlib.Path):
    assets_dir = tmp_path / "assets" / "org" / "demos"
    assert b1k_dataset.load_prompt_source(assets_dir) is None  # checkpoints from before the record
    b1k_dataset.save_prompt_source(assets_dir, "task_description")
    assert b1k_dataset.load_prompt_source(assets_dir) == "task_description"
    with pytest.raises(ValueError, match="Unknown prompt_source"):
        b1k_dataset.save_prompt_source(assets_dir, "instruction")


def test_dataset_missing_data_file_fails_fast(full_root: pathlib.Path):
    (full_root / "data" / "chunk-002" / "file-000.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="1 data file\\(s\\).*chunk-002"):
        b1k_dataset.B1KLeRobotDataset("org/demos", full_root, task_names="chop_an_onion")
    # Other tasks are unaffected: only the selected episodes' files are needed.
    assert len(b1k_dataset.B1KLeRobotDataset("org/demos", full_root, task_names="turning_on_radio")) > 0
