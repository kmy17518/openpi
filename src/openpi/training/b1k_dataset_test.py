"""Tests for openpi.training.b1k_dataset on a small synthetic LeRobot v3.0 dataset.

The synthetic dataset mimics the challenge demos' layout -- one chunk per task, global ``episode_index`` and frame
``index`` -- so that a "partial download" (only the chunk of a middle task on disk) can be exercised without the
real 3 TB dataset.
"""

import collections
import dataclasses
import json
import pathlib
import pickle
from unittest import mock

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


def _write_synthetic_root(
    root: pathlib.Path, chunks: list[int], state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM
) -> None:
    """Write ``meta/`` (dataset-wide) plus the data / episode-metadata files of ``chunks`` only."""
    meta = root / "meta"
    (meta / "episodes").mkdir(parents=True)
    (root / "data").mkdir()
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
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
                        "observation.state": np.full(state_dim, index, dtype=np.float32),
                        "action": np.full(action_dim, index, dtype=np.float32),
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


@pytest.mark.parametrize("with_task_index", [True, False])
def test_subset_requires_every_requested_task(partial_root: pathlib.Path, *, with_task_index: bool):
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", partial_root)
    if not with_task_index:
        meta.episodes = meta.episodes.remove_columns("task_index")
    with pytest.raises(ValueError, match="No episodes of task") as error:
        b1k_dataset.select_task_subset(meta, TASKS)
    message = str(error.value)
    assert "No episodes of task(s) ['chop_an_onion', 'turning_on_radio']" in message
    assert "task_index [0, 2]" in message
    assert str(partial_root) in message


@pytest.mark.parametrize("with_task_index", [True, False])
def test_subset_accepts_task_aliases(full_root: pathlib.Path, *, with_task_index: bool):
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", full_root)
    meta.tasks.loc["radio_alias", "task_index"] = 0
    if not with_task_index:
        meta.episodes = meta.episodes.remove_columns("task_index")
    subset = b1k_dataset.select_task_subset(meta, ["radio_alias", "turning_on_radio", "picking_up_trash"])
    assert subset.task_indices == frozenset({0, 1})
    assert subset.episode_indices == (0, 1, 2, 3)


def test_partial_task_subset_fails_before_reader(partial_root: pathlib.Path, monkeypatch):
    reader = mock.Mock(side_effect=AssertionError("reader must not be constructed"))
    monkeypatch.setattr(b1k_dataset, "_B1KDatasetReader", reader)
    with pytest.raises(ValueError, match="No episodes of task"):
        b1k_dataset.B1KLeRobotDataset("org/demos", partial_root, task_names=TASKS)
    reader.assert_not_called()


def test_partial_task_subset_fails_before_stats_dataset(partial_root: pathlib.Path, monkeypatch):
    from openpi.training import config
    from scripts import compute_norm_stats

    reader = mock.Mock(side_effect=AssertionError("stats dataset must not be constructed"))
    monkeypatch.setattr(compute_norm_stats._lerobot_lowdim, "LowDimLeRobotDataset", reader)  # noqa: SLF001
    data_config = config.DataConfig(repo_id="org/demos", dataset_root=str(partial_root), task_names=TASKS)
    with pytest.raises(ValueError, match="No episodes of task"):
        compute_norm_stats._create_lowdim_dataset(data_config, HORIZON)  # noqa: SLF001
    reader.assert_not_called()


@pytest.mark.parametrize("fast", [True, False])
def test_partial_task_subset_never_writes_stats(partial_root: pathlib.Path, monkeypatch, *, fast: bool):
    from openpi.training import config
    from scripts import compute_norm_stats

    data_config = config.DataConfig(repo_id="org/demos", dataset_root=str(partial_root), task_names=TASKS)
    factory = config.LeRobotB1KDataConfig()
    monkeypatch.setattr(config.LeRobotB1KDataConfig, "create", lambda *_args: data_config)
    save = mock.Mock(side_effect=AssertionError("stats must not be written"))
    monkeypatch.setattr(compute_norm_stats.normalize, "save", save)
    args = compute_norm_stats.Args(config_name="pi05_b1k", data=factory, fast=fast)
    with pytest.raises(ValueError, match="No episodes of task"):
        compute_norm_stats.main(args)
    save.assert_not_called()


@pytest.mark.parametrize("with_task_index", [True, False])
def test_explicit_episode_subset_requires_every_task(full_root: pathlib.Path, *, with_task_index: bool):
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", full_root)
    if not with_task_index:
        meta.episodes = meta.episodes.remove_columns("task_index")
    with pytest.raises(ValueError, match="picking_up_trash.*task_index \\[1\\].*requested episode selection"):
        b1k_dataset.select_task_subset(meta, TASKS[:2], episodes=[0, 1])
    assert b1k_dataset.select_task_subset(meta, TASKS[:2], episodes=[0, 2]).episode_indices == (0, 2)


def test_dataset_explicit_episodes_cannot_drop_requested_tasks(full_root: pathlib.Path):
    with pytest.raises(ValueError, match="picking_up_trash.*requested episode selection"):
        b1k_dataset.B1KLeRobotDataset("org/demos", full_root, task_names=TASKS[:2], episodes=[0, 1])


@pytest.mark.parametrize("entrypoint", ["training", "stats"])
def test_explicit_episode_selection_checked_before_dataset(full_root: pathlib.Path, monkeypatch, entrypoint: str):
    from openpi.training import config
    from openpi.training import data_loader
    from scripts import compute_norm_stats

    reader = mock.Mock(side_effect=AssertionError("dataset must not be constructed"))
    data_config = config.DataConfig(
        repo_id="org/demos", dataset_root=str(full_root), task_names=TASKS[:2], dataset_kwargs={"episodes": [0, 1]}
    )
    if entrypoint == "training":
        create = data_loader.create_b1k_dataset
        data_config = dataclasses.replace(data_config, data_cls=reader)
    else:
        monkeypatch.setattr(compute_norm_stats._lerobot_lowdim, "LowDimLeRobotDataset", reader)  # noqa: SLF001
        create = compute_norm_stats._create_lowdim_dataset  # noqa: SLF001
    with pytest.raises(ValueError, match="picking_up_trash.*requested episode selection"):
        create(data_config, HORIZON)
    reader.assert_not_called()


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


def test_prompt_length_check_uses_extracted_state_dimension(monkeypatch):
    from openpi.models import pi0_config
    from openpi.models import tokenizer

    tokenize = mock.Mock(side_effect=lambda _prompt, state: (None, np.ones(len(state), dtype=bool)))
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", lambda **_kwargs: mock.Mock(tokenize=tokenize))
    model = pi0_config.Pi0Config(pi05=True, action_dim=32, max_token_len=23)
    b1k_dataset.check_prompt_token_lengths({0: "instruction"}, model, state_dim=23)
    assert tokenize.call_args.kwargs["state"].shape == (23,)
    with pytest.raises(ValueError, match="32 tokens"):
        b1k_dataset.check_prompt_token_lengths({0: "instruction"}, model)


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


# ---- video decoding -----------------------------------------------------------------------------------------------

VIDEO_FRAMES = 40


def _write_test_video(path: pathlib.Path, *, seed: int, fps: int = FPS, gop: int = 4) -> None:
    """A short inter-coded video (H.264, GOP ``gop``, no B-frames -- the challenge demos' structure) with a distinct
    picture per frame, so that picking a wrong frame is detectable."""
    import av

    rng = np.random.default_rng(seed)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": str(gop), "bf": "0", "crf": "18"}
        for i in range(VIDEO_FRAMES):
            picture = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            picture[:16, :, :] = i * 6  # frame-number bar
            frame = av.VideoFrame.from_ndarray(picture, format="rgb24")  # pts assigned by PyAV: frame i at i / fps
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_video_frame_reader_matches_lerobot_decoder(tmp_path: pathlib.Path):
    from lerobot.datasets import video_utils

    path = tmp_path / "cam.mp4"
    _write_test_video(path, seed=0)
    reader = b1k_dataset.VideoFrameReader(max_open_files=4)
    rng = np.random.default_rng(1)
    for _ in range(12):
        frames = sorted(rng.choice(VIDEO_FRAMES, size=rng.integers(1, 4), replace=False).tolist())
        timestamps = [f / FPS for f in frames]
        ours = reader.read(path, timestamps, tolerance_s=1e-4)
        theirs = video_utils.decode_video_frames_pyav(path, timestamps, tolerance_s=1e-4, return_uint8=True)
        assert ours.dtype == np.uint8
        assert ours.shape == (len(frames), 64, 64, 3)
        np.testing.assert_array_equal(ours, theirs.permute(0, 2, 3, 1).numpy())
        # The frame-number bar identifies the picture: no off-by-one frame from the seek.
        assert [round(float(pic[0, 0, 0]) / 6) for pic in ours] == frames
    assert list(reader._containers) == [str(path)]  # noqa: SLF001  -- one long-lived container per file
    # Out-of-tolerance requests are rejected like lerobot rejects them.
    with pytest.raises(ValueError, match="tolerance"):
        reader.read(path, [0.5 / FPS], tolerance_s=1e-4)
    reader.close()


def test_video_frame_reader_lru_and_pickling(tmp_path: pathlib.Path):
    paths = [tmp_path / f"cam{i}.mp4" for i in range(3)]
    for i, path in enumerate(paths):
        _write_test_video(path, seed=i)
    reader = b1k_dataset.VideoFrameReader(max_open_files=2)
    for path in paths:
        reader.read(path, [0.0], tolerance_s=1e-4)
    assert list(reader._containers) == [str(paths[1]), str(paths[2])]  # noqa: SLF001  -- oldest evicted
    reader.read(paths[1], [0.0], tolerance_s=1e-4)
    assert list(reader._containers) == [str(paths[2]), str(paths[1])]  # noqa: SLF001  -- most recent last
    clone = pickle.loads(pickle.dumps(reader))  # what a spawned data-loader worker receives
    assert clone._containers == {}  # noqa: SLF001
    assert (clone.max_open_files, clone.decoder_threads) == (2, 1)
    np.testing.assert_array_equal(clone.read(paths[0], [3 / FPS], tolerance_s=1e-4), reader.read(paths[0], [3 / FPS], 1e-4))
    reader.close()
    clone.close()


def _add_video_streams(root: pathlib.Path, keys: dict[str, bool]) -> None:
    """Declare video features (``key -> is_depth``) on a synthetic root and point every episode at file-000."""
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    for key, is_depth in keys.items():
        info["features"][key] = {
            "dtype": "video",
            "shape": [64, 64, 1 if is_depth else 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": FPS, "video.is_depth_map": is_depth},
        }
    info_path.write_text(json.dumps(info))
    for episode_path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        episodes = pd.read_parquet(episode_path)
        for key in keys:
            episodes[f"videos/{key}/chunk_index"] = episodes["data/chunk_index"]
            episodes[f"videos/{key}/file_index"] = 0
            episodes[f"videos/{key}/from_timestamp"] = 2.5  # camera time = row timestamp + this offset
        episodes.to_parquet(episode_path)


def test_dataset_decodes_only_selected_rgb_streams(full_root: pathlib.Path, monkeypatch):
    from lerobot.datasets import dataset_reader

    rgb, depth = "observation.rgb.head", "observation.depth.head"
    _add_video_streams(full_root, {rgb: False, depth: True})
    monkeypatch.setattr(dataset_reader.DepthEncoderConfig, "from_video_info", mock.Mock())
    reads: list[tuple[str, list[float]]] = []

    def fake_read(self, path, timestamps, tolerance_s):
        reads.append((str(path), list(timestamps)))
        return np.full((len(timestamps), 64, 64, 3), 7, dtype=np.uint8)

    monkeypatch.setattr(b1k_dataset.VideoFrameReader, "read", fake_read)
    lerobot_query = mock.Mock(side_effect=AssertionError("lerobot's decoder must not be used for RGB streams"))
    monkeypatch.setattr(dataset_reader.DatasetReader, "_query_videos", lerobot_query)
    for path in [full_root / "videos" / key / f"chunk-{chunk:03d}" / "file-000.mp4" for key in (rgb, depth) for chunk in range(3)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    ds = b1k_dataset.B1KLeRobotDataset("org/demos", full_root, video_keys=[rgb], task_names="picking_up_trash")
    assert ds.video_keys == [rgb]
    item = ds[3]  # frame 3 of episode 2
    assert item[rgb].dtype == torch.uint8
    assert tuple(item[rgb].shape) == (64, 64, 3)
    assert depth not in item  # never decoded, never requested
    assert item["task"] == "picking_up_trash"
    assert reads == [(str(full_root / "videos" / rgb / "chunk-001" / "file-000.mp4"), [pytest.approx(2.5 + 3 / FPS)])]
    lerobot_query.assert_not_called()
    # Depth streams (or the fast reader switched off) go through lerobot's decoder.
    lerobot_query.side_effect = None
    lerobot_query.return_value = {depth: torch.zeros(1, 64, 64)}
    ds_depth = b1k_dataset.B1KLeRobotDataset("org/demos", full_root, video_keys=[rgb, depth], task_names="picking_up_trash")
    item = ds_depth[0]
    # Only the depth stream is handed to lerobot, with the row timestamp (it applies from_timestamp itself).
    assert lerobot_query.call_args.args[0] == {depth: [pytest.approx(0.0)]}
    assert item[depth].shape == (1, 64, 64)
    assert item[rgb].shape == (64, 64, 3)
    # Keys the dataset does not have are ignored (training configs request the robot's cameras blindly).
    assert b1k_dataset.B1KLeRobotDataset("org/demos", full_root, video_keys=["observation.rgb.nope", rgb]).video_keys == [rgb]


def _write_task_groups(root: pathlib.Path, groups: dict, fmt: str = "isg-task-groups/v1") -> pathlib.Path:
    path = root / "isg_meta/task_groups.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": fmt, "groups": groups}))
    return path


def _write_episode_split(
    root: pathlib.Path,
    train_episodes: dict[str, list[int]],
    fmt: str = "isg-episode-split/v1",
    held_out: dict[str, list[int]] | None = None,
) -> pathlib.Path:
    path = root / "isg_meta/train_split.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = {task: {"train": train, "held_out": (held_out or {}).get(task, [])} for task, train in train_episodes.items()}
    path.write_text(json.dumps({"format": fmt, "name": "tiny", "tasks": tasks}))
    return path


def _write_settle_windows(root: pathlib.Path, programs: dict[int, int], fmt: str = "isg-settle-windows/v1") -> pathlib.Path:
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", root)
    lengths = {int(ep["episode_index"]): int(ep["length"]) for ep in meta.episodes}
    path = root / "isg_meta/settle_windows.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format": fmt,
                "episodes": {
                    str(ep): {"length": lengths[ep], "program_length": prog} for ep, prog in programs.items()
                },
            }
        )
    )
    return path


def test_task_groups_expand_and_validate(full_root: pathlib.Path):
    assert b1k_dataset.expand_task_groups(full_root, None) is None
    assert b1k_dataset.expand_task_groups(full_root, ["picking_up_trash", "typo"]) == ["picking_up_trash", "typo"]
    _write_task_groups(
        full_root,
        {
            "radios": ["turning_on_radio"],
            "pair": ["radios", "picking_up_trash"],
            "all": ["pair", "chop_an_onion"],
        },
    )
    assert b1k_dataset.expand_task_groups(full_root, "radios") == ["turning_on_radio"]
    assert b1k_dataset.expand_task_groups(full_root, ["all"]) == [
        "turning_on_radio",
        "picking_up_trash",
        "chop_an_onion",
    ]
    assert b1k_dataset.expand_task_groups(full_root, ["picking_up_trash", "pair", "turning_on_radio"]) == [
        "picking_up_trash",
        "turning_on_radio",
    ]
    assert b1k_dataset.normalize_task_names(["pair"], dataset_root=full_root) == (
        "picking_up_trash",
        "turning_on_radio",
    )
    with pytest.raises(ValueError, match="Unknown task or task group 'typo'"):
        b1k_dataset.expand_task_groups(full_root, ["turning_on_radio", "typo"])
    for groups, error in [
        ({"outer": ["inner"], "inner": ["typo"]}, "Task group 'inner' in .* lists unknown task or group 'typo'"),
        ({"outer": ["inner"], "inner": ["outer"]}, "Task group cycle outer -> inner -> outer"),
        (
            {"outer": ["turning_on_radio"], "turning_on_radio": ["picking_up_trash"]},
            "'turning_on_radio' is both a task and a task group",
        ),
        ({"outer": []}, "Task group 'outer' .* must be a nonempty list"),
        ({"outer": "turning_on_radio"}, "Task group 'outer' .* must be a nonempty list"),
    ]:
        _write_task_groups(full_root, groups)
        with pytest.raises(ValueError, match=error):
            b1k_dataset.expand_task_groups(full_root, ["outer"])
    _write_task_groups(full_root, {}, fmt="isg-task-groups/v0")
    with pytest.raises(ValueError, match="not an isg-task-groups/v1 task group file"):
        b1k_dataset.expand_task_groups(full_root, ["turning_on_radio"])


def test_episode_split_resolution_and_dataset_filtering(full_root: pathlib.Path, tmp_path: pathlib.Path):
    assert b1k_dataset.resolve_episode_split(full_root, "auto") is None
    _write_episode_split(full_root, {"turning_on_radio": [1], "picking_up_trash": [3], "chop_an_onion": [4]})
    split = b1k_dataset.resolve_episode_split(full_root, "auto")
    assert split is not None and split["episodes"] == [1, 3, 4]
    assert b1k_dataset.resolve_episode_split(full_root, "none") is None
    assert b1k_dataset.resolve_episode_split(full_root, "auto", ["picking_up_trash"])["episodes"] == [3]

    ds = b1k_dataset.B1KLeRobotDataset(
        "org/demos", full_root, task_names=["turning_on_radio", "picking_up_trash"], episode_split="auto"
    )
    assert ds.episodes == [1, 3]
    assert _episode_indices(ds) == {1, 3}
    assert len(ds) == 2 * FRAMES_PER_EPISODE

    with pytest.raises(FileNotFoundError, match="missing.json"):
        b1k_dataset.resolve_episode_split(full_root, str(tmp_path / "missing.json"))


def test_held_out_episodes_skip_tasks_without_any(full_root: pathlib.Path, caplog):
    from openpi.training import config
    from openpi.training import data_loader

    _write_episode_split(
        full_root,
        {"turning_on_radio": [0], "picking_up_trash": [2, 3], "chop_an_onion": [4]},
        held_out={"turning_on_radio": [1], "chop_an_onion": [5]},
    )
    split = b1k_dataset.resolve_episode_split(full_root, "auto", subset="held_out")
    assert split["subset"] == "held_out" and split["episodes"] == [1, 5]
    assert split["tasks"] == ["turning_on_radio", "chop_an_onion"]
    assert "['picking_up_trash'] have no held-out episodes" in caplog.text
    assert b1k_dataset.resolve_episode_split(full_root, "auto")["episodes"] == [0, 2, 3, 4]
    with pytest.raises(ValueError, match="None of the selected tasks has held-out episodes"):
        b1k_dataset.resolve_episode_split(full_root, "auto", ["picking_up_trash"], subset="held_out")
    with pytest.raises(ValueError, match="subset must be one of"):
        b1k_dataset.resolve_episode_split(full_root, "auto", subset="test")

    dc = config.DataConfig(
        repo_id="org/demos",
        dataset_root=str(full_root),
        task_names=["turning_on_radio", "picking_up_trash", "chop_an_onion"],
        episode_split="auto",
        action_sequence_keys=("action",),
    )
    held_out = data_loader.create_b1k_dataset(dc, HORIZON, subset="held_out")
    assert held_out.episodes == [1, 5] and len(held_out) == 2 * FRAMES_PER_EPISODE
    assert data_loader.create_b1k_dataset(dc, HORIZON).episodes == [0, 2, 3, 4]
    with pytest.raises(ValueError, match="need an episode split"):
        data_loader.create_b1k_dataset(dataclasses.replace(dc, episode_split="none"), HORIZON, subset="held_out")


def test_spread_frame_indices_cover_every_episode_and_repeat_with_the_seed():
    episode_index = np.repeat([7, 3, 9, 4], [12, 12, 3, 30])
    rows = b1k_dataset.spread_frame_indices(episode_index, 10, seed=0)
    assert len(set(rows)) == 10 and rows == sorted(rows)
    # shares of 10 proportional to 12:12:3:30 frames, at least one per episode
    assert collections.Counter(episode_index[rows].tolist()) == {7: 2, 3: 2, 9: 1, 4: 5}
    assert rows == b1k_dataset.spread_frame_indices(episode_index, 10, seed=0)
    assert rows != b1k_dataset.spread_frame_indices(episode_index, 10, seed=1)
    long_episode = [row - 27 for row in rows if episode_index[row] == 4]
    assert [position // 6 for position in long_episode] == [0, 1, 2, 3, 4]  # one per stretch of 6 frames
    assert set(episode_index[b1k_dataset.spread_frame_indices(episode_index, 2, seed=0)].tolist()) == {7, 4}
    assert b1k_dataset.spread_frame_indices(episode_index, 57, seed=0) == list(range(57))
    with pytest.raises(ValueError, match="contiguous"):
        b1k_dataset.spread_frame_indices([0, 1, 0], 2, seed=0)


def test_figure_frame_indices_pick_different_episodes_and_repeat_with_the_seed():
    episode_index = np.repeat([7, 3, 9, 4, 5, 8], [12, 12, 3, 30, 5, 9])
    rows = b1k_dataset.figure_frame_indices(episode_index, 4, seed=0)
    assert len(set(episode_index[rows].tolist())) == 4
    assert rows == b1k_dataset.figure_frame_indices(episode_index, 4, seed=0)
    assert rows != b1k_dataset.figure_frame_indices(episode_index, 4, seed=1)
    every_episode = b1k_dataset.figure_frame_indices(episode_index, 9, seed=0)
    assert set(episode_index[every_episode].tolist()) == {3, 4, 5, 7, 8, 9}


# Validation episodes of ACT's resolve_val_split (act b1k_training.py) for these tasks, sizes and seeds, computed with
# ACT's own function on a catalog of these episodes.
_ACT_TASKS = {0: "picking_up_trash", 1: "iors_small-turning_on_radio", 2: "chop_an_onion", 3: "x"}
_ACT_SIZES = {0: 2, 1: 7, 2: 25, 3: 131}
_ACT_BASES = {0: 40, 1: 0, 2: 500, 3: 2000}
# fmt: off
_ACT_VAL_EPISODES = {
    (0, 0.1): {
        "picking_up_trash": [40],
        "iors_small-turning_on_radio": [12],
        "chop_an_onion": [522, 542],
        "x": [2000, 2010, 2082, 2102, 2136, 2138, 2150, 2178, 2234, 2240, 2286, 2306, 2340],
    },
    (0, 0.3): {
        "picking_up_trash": [40],
        "iors_small-turning_on_radio": [12, 16],
        "chop_an_onion": [510, 522, 534, 542, 548, 558, 566, 570],
        "x": [2000, 2004, 2010, 2016, 2036, 2048, 2066, 2070, 2076, 2082, 2102, 2106, 2124, 2136, 2138, 2150, 2166,
              2172, 2178, 2190, 2192, 2198, 2204, 2216, 2228, 2234, 2240, 2286, 2294, 2306, 2310, 2318, 2322, 2324,
              2328, 2330, 2340, 2346, 2364],
    },
    (42, 0.1): {
        "picking_up_trash": [40],
        "iors_small-turning_on_radio": [4],
        "chop_an_onion": [510, 516],
        "x": [2042, 2046, 2130, 2144, 2178, 2214, 2216, 2222, 2246, 2300, 2352, 2378, 2388],
    },
    (42, 0.3): {
        "picking_up_trash": [40],
        "iors_small-turning_on_radio": [4, 16],
        "chop_an_onion": [510, 516, 518, 528, 536, 564, 566, 572],
        "x": [2040, 2042, 2046, 2058, 2070, 2078, 2096, 2124, 2126, 2130, 2136, 2138, 2144, 2160, 2178, 2186, 2192,
              2210, 2214, 2216, 2222, 2226, 2228, 2246, 2270, 2276, 2280, 2300, 2318, 2322, 2334, 2352, 2354, 2358,
              2364, 2370, 2378, 2388, 2390],
    },
}
# fmt: on


@pytest.mark.parametrize(("seed", "fraction"), list(_ACT_VAL_EPISODES))
def test_val_split_matches_act(seed: int, fraction: float):
    episodes = {
        _ACT_TASKS[task]: [_ACT_BASES[task] + 3 * k + k % 2 for k in range(size)] for task, size in _ACT_SIZES.items()
    }
    split = b1k_dataset.split_val_episodes(episodes, fraction, seed)
    assert {name: entry["val"] for name, entry in split.items()} == _ACT_VAL_EPISODES[(seed, fraction)]
    for name, entry in split.items():
        assert sorted(entry["train"] + entry["val"]) == episodes[name]
    with pytest.raises(ValueError, match="at least 2"):
        b1k_dataset.split_val_episodes({"one": [3]}, fraction, seed)
    with pytest.raises(ValueError, match="val-fraction"):
        b1k_dataset.split_val_episodes(episodes, 1.0, seed)


def test_val_fraction_splits_training_and_validation_episodes(full_root: pathlib.Path, monkeypatch):
    from openpi.training import config
    from openpi.training import data_loader
    from scripts import compute_norm_stats

    tasks = ["turning_on_radio", "picking_up_trash", "chop_an_onion"]
    dc = config.DataConfig(
        repo_id="org/demos",
        dataset_root=str(full_root),
        task_names=tasks,
        action_sequence_keys=("action",),
        val_fraction=0.5,
        val_seed=3,
    )
    meta = b1k_dataset.B1KDatasetMetadata("org/demos", full_root)
    split = b1k_dataset.resolve_val_split(meta, tasks, 0.5, 3)
    assert split["format"] == "isg-val-split/v1"
    assert len(split["val_episodes"]) == 3
    assert sorted(split["train_episodes"] + split["val_episodes"]) == [0, 1, 2, 3, 4, 5]
    expected = b1k_dataset.split_val_episodes({"picking_up_trash": [2, 3]}, 0.5, 3)
    assert split["tasks"]["picking_up_trash"] == expected["picking_up_trash"]
    assert data_loader.create_b1k_dataset(dc, HORIZON).episodes == split["train_episodes"]
    assert data_loader.create_b1k_dataset(dc, HORIZON, subset="val").episodes == split["val_episodes"]
    reader = mock.Mock()
    monkeypatch.setattr(compute_norm_stats._lerobot_lowdim, "LowDimLeRobotDataset", reader)  # noqa: SLF001
    compute_norm_stats._create_lowdim_dataset(dc, HORIZON)  # noqa: SLF001
    assert reader.call_args.kwargs["episodes"] == split["train_episodes"]
    assert b1k_dataset.resolve_val_split(meta, tasks, 0.0, 3) is None
    with pytest.raises(ValueError, match="need --data.val-fraction"):
        data_loader.create_b1k_dataset(dataclasses.replace(dc, val_fraction=0.0), HORIZON, subset="val")
    assert b1k_dataset.task_subset_asset_id("org/demos", ["b"], gripper_action="delta", val_fraction=0.1, seed=7) == (
        "org/demos/task_subsets/b/gripper_action/delta/val_fraction/0.1/seed/7"
    )


def test_train_config_seed_selects_the_validation_split():
    from openpi.training import config

    train_config = config.get_config("pi05_b1k")
    assert dataclasses.replace(train_config, seed=5).data.val_seed == 5
    seeded = dataclasses.replace(train_config, seed=5, data=dataclasses.replace(train_config.data, val_fraction=0.1))
    assert seeded.data.val_seed == 5
    assert seeded.data.val_fraction == 0.1


def test_action_is_pad_reaches_the_training_batch_and_flags_the_program_end(tmp_path: pathlib.Path, monkeypatch):
    from lerobot.datasets import dataset_reader
    from openpi.training import config
    from openpi.training import data_loader

    root = tmp_path / "r1pro"
    _write_synthetic_root(root, chunks=[0], state_dim=61, action_dim=23)
    cameras = [
        "observation.rgb.zed_link_camera_0",
        "observation.rgb.left_realsense_link_camera_0",
        "observation.rgb.right_realsense_link_camera_0",
    ]
    _add_video_streams(root, dict.fromkeys(cameras, False))
    monkeypatch.setattr(dataset_reader.DepthEncoderConfig, "from_video_info", mock.Mock())
    for camera in cameras:
        path = root / "videos" / camera / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        b1k_dataset.VideoFrameReader, "read", lambda self, path, ts, tol: np.zeros((len(ts), 64, 64, 3), np.uint8)
    )
    programs = {0: 5, 1: 4}  # of 12 recorded frames each
    _write_settle_windows(root, programs)
    model = config.get_config("pi05_b1k").model
    dc = config.LeRobotB1KDataConfig(
        repo_id="org/demos",
        dataset_root=str(root),
        robot_config_name="b1k/R1Pro",
        settle_steps="0",
        base_config=config.DataConfig(prompt_from_task=True),
    ).create(root / "assets", model)
    dataset = data_loader.create_b1k_dataset(dc, HORIZON, model)
    assert len(dataset) == sum(programs.values())
    loader = data_loader.B1KDataLoaderImpl(
        dc,
        data_loader.TorchDataLoader(
            data_loader.transform_dataset(dataset, dc, skip_norm_stats=True), local_batch_size=len(dataset), num_batches=1
        ),
    )
    observation, actions, action_is_pad = next(iter(loader))
    # padded from the program end (not the recorded end): frame f of a program of n frames pads steps t >= n - f
    expected = [[frame + t >= n for t in range(HORIZON)] for n in programs.values() for frame in range(n)]
    assert np.asarray(action_is_pad).tolist() == expected
    assert actions.shape == (len(dataset), HORIZON, model.action_dim)
    assert observation.state.shape[0] == len(dataset)


def test_settle_steps_parse_and_cut_episodes(full_root: pathlib.Path):
    for spec, canonical in [
        ("all", "all"),
        ("0", "0"),
        ("010", "10"),
        (10, "10"),
        ("0.20", "0.2"),
        (".5", "0.5"),
        ("1.", "1.0"),
        ("1.0", "1.0"),
        ("0.0", "0.0"),
    ]:
        assert b1k_dataset.parse_settle_steps(spec) == canonical
    for bad in ("-1", "1e-1", "0.2%", "half", "", ".", "none"):
        with pytest.raises(ValueError, match="--settle-steps must be all, a number of frames"):
            b1k_dataset.parse_settle_steps(bad)
    assert b1k_dataset.settle_frames("0.55", 100) == 55
    assert b1k_dataset.settle_frames("0.2", 7) == 2 and b1k_dataset.settle_frames("1", 7) == 1

    with pytest.raises(FileNotFoundError, match="settle_windows.json"):
        b1k_dataset.B1KLeRobotDataset("org/demos", full_root, settle_steps="0")

    programs = {0: 5, 1: 4, 2: 6, 3: 3, 4: 8, 5: 7}
    _write_settle_windows(full_root, programs)
    delta_timestamps = {"action": [t / FPS for t in range(HORIZON)]}
    ds0 = b1k_dataset.B1KLeRobotDataset(
        "org/demos", full_root, task_names="turning_on_radio", settle_steps="0", delta_timestamps=delta_timestamps
    )
    assert ds0.settle is not None and ds0.settle["spec"] == "0"
    assert ds0.settle["frames"] == 9 and ds0.settle["recorded_frames"] == 24
    assert len(ds0) == 9
    # Last kept frame of episode 0 is frame_index 4 (the 5th frame); its action chunk is clamped at frame 4 and padded.
    item4 = ds0[4]
    assert int(item4["episode_index"]) == 0 and int(item4["frame_index"]) == 4
    assert item4["action_is_pad"].tolist() == [False, True, True, True, True]
    assert torch.all(item4["action"][:, 0] == 4.0)
    # Next item in ds0 is frame 0 of episode 1 (index 12).
    item5 = ds0[5]
    assert int(item5["episode_index"]) == 1 and int(item5["frame_index"]) == 0

    # Low-dim norm-stats reader produces identical items and length under settle_steps + task_groups + episode_split.
    from openpi.training import config
    from openpi.training import data_loader
    from scripts import compute_norm_stats

    _write_task_groups(full_root, {"radio_group": ["turning_on_radio"]})
    _write_episode_split(full_root, {"turning_on_radio": [0, 1]})
    dc = config.DataConfig(
        repo_id="org/demos",
        dataset_root=str(full_root),
        task_names=["radio_group"],
        episode_split="auto",
        settle_steps="0",
        action_sequence_keys=("action",),
    )
    lowdim = compute_norm_stats._create_lowdim_dataset(dc, HORIZON)  # noqa: SLF001
    train_ds = data_loader.create_b1k_dataset(dc, HORIZON)
    assert len(lowdim) == len(train_ds) == 9
    for idx in range(len(lowdim)):
        span = lowdim.spans[0 if idx < 5 else 1]
        pos = idx if idx < 5 else idx - 5
        per_frame, _ = lowdim.episode_items(span, np.asarray([pos]))
        expected = train_ds[idx]
        assert torch.equal(per_frame["action"][0], expected["action"])
        assert torch.equal(per_frame["action_is_pad"][0], expected["action_is_pad"])


def test_episode_last_goal_source_keeps_last_recorded_frame_when_settle_steps_cuts_episode(
    full_root: pathlib.Path, monkeypatch
):
    from lerobot.datasets import dataset_reader
    from openpi.training import config
    from openpi.training import data_loader

    rgb = "observation.rgb.zed_link_camera_0"
    _add_video_streams(full_root, {rgb: False})
    monkeypatch.setattr(dataset_reader.DepthEncoderConfig, "from_video_info", mock.Mock())
    for chunk in range(3):
        path = full_root / "videos" / rgb / f"chunk-{chunk:03d}" / "file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fake_read(self, path, timestamps, tolerance_s):
        # Encode the requested timestamp (in 0.1s ticks) into pixel [0, 0, 0] so we can check which frame was decoded.
        return np.stack([np.full((64, 64, 3), round(ts * 10), dtype=np.uint8) for ts in timestamps])

    monkeypatch.setattr(b1k_dataset.VideoFrameReader, "read", fake_read)
    _write_settle_windows(full_root, {0: 5, 1: 4, 2: 6, 3: 3, 4: 8, 5: 7})

    model = config.get_config("pi05_b1k").model
    model = dataclasses.replace(model, goal_image_keys=("goal_0_rgb",))
    dc = config.LeRobotB1KDataConfig(
        repo_id="org/demos",
        dataset_root=str(full_root),
        robot_config_name="b1k/R1Pro",
        task_names=["turning_on_radio"],
        settle_steps="0",
        goal_views=("goal_image_0",),
        goal_source="episode_last",
        conditioning_regime="image",
        base_config=config.DataConfig(prompt_from_task=True),
    ).create(full_root / "assets", model)
    ds = data_loader.create_b1k_dataset(dc, HORIZON, model)
    assert len(ds) == 9
    assert ds.inference_metadata["conditioning"]["goal_source"] == "episode_last"
    assert ds.inference_metadata["conditioning"]["settle_steps"] == "0"
    # Episode 0 has 12 recorded frames (last recorded frame_index = 11, timestamp = 1.1s, plus from_timestamp = 2.5s -> 3.6s -> 36).
    sample = ds[2]
    assert int(sample["observation.goal_rgb.zed_link_camera_0"][0, 0, 0]) == 36

