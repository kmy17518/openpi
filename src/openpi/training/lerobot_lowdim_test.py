import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import openpi.training.b1k_dataset as _b1k_dataset
import openpi.training.data_loader as _data_loader
import openpi.training.lerobot_compat as _lerobot_compat
import openpi.training.lerobot_lowdim as _lowdim
import openpi.transforms as _transforms

FPS = 10
HORIZON = 5
STATE_DIM, ACTION_DIM = 6, 4
# Episode lengths (some shorter than the action horizon, so windows get clamped) and their tasks.
EPISODES = [(12, "task_a"), (3, "task_b"), (17, "task_a"), (1, "task_b"), (9, "task_b"), (14, "task_a")]
KEYS = ["state", "actions"]


class _RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _data_transforms() -> list[_transforms.DataTransformFn]:
    # Representative of the norm-stats transform chain: repack, delta actions (in-place on torch tensors), drop strings.
    return [
        _transforms.RepackTransform({"state": "observation.state", "actions": "action", "prompt": "prompt"}),
        _transforms.DeltaActions(mask=[True, True, False, False]),
        _RemoveStrings(),
    ]


def _delta_timestamps() -> dict[str, list[float]]:
    return {"action": [t / FPS for t in range(HORIZON)]}


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory) -> pathlib.Path:
    """A small video-free LeRobot v3.0 dataset written with LeRobot itself, spread over two data files."""
    root = tmp_path_factory.mktemp("lerobot") / "lowdim"
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    dataset = _lerobot_compat.LeRobotDataset.create(
        "test/lowdim", fps=FPS, features=features, root=root, use_videos=False
    )
    rng = np.random.default_rng(0)
    for length, task in EPISODES:
        for _ in range(length):
            dataset.add_frame(
                {
                    "observation.state": rng.standard_normal(STATE_DIM).astype(np.float32),
                    "action": rng.standard_normal(ACTION_DIM).astype(np.float32),
                    "task": task,
                }
            )
        dataset.save_episode()
    dataset.finalize()

    # Move the last two episodes into a second data file (the writer only rolls files over by size).
    data_file = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_file)
    episode_index = table.column("episode_index").to_numpy()
    second = episode_index >= len(EPISODES) - 2
    pq.write_table(table.filter(pa.array(~second)), data_file)
    pq.write_table(table.filter(pa.array(second)), root / "data/chunk-000/file-001.parquet")
    episodes_file = next((root / "meta/episodes").glob("*/*.parquet"))
    episodes = pq.read_table(episodes_file)
    file_index = np.where(episodes.column("episode_index").to_numpy() >= len(EPISODES) - 2, 1, 0)
    episodes = episodes.set_column(
        episodes.schema.get_field_index("data/file_index"), "data/file_index", pa.array(file_index, pa.int64())
    )
    pq.write_table(episodes, episodes_file)
    return root


def _reference_dataset(root: pathlib.Path, episodes: list[int] | None = None) -> _data_loader.Dataset:
    # Same wrapping as `create_b1k_dataset` + the norm-stats transforms.
    dataset = _lerobot_compat.LeRobotDataset(
        "test/lowdim", root=root, delta_timestamps=_delta_timestamps(), episodes=episodes
    )
    tasks = _lerobot_compat.tasks_from_metadata(dataset.meta)
    dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks)])
    return _data_loader.TransformedDataset(dataset, _data_transforms())


def _reference_batches(dataset, batch_size: int, num_batches: int, *, shuffle: bool) -> list[dict[str, np.ndarray]]:
    loader = _data_loader.TorchDataLoader(
        dataset, local_batch_size=batch_size, shuffle=shuffle, num_batches=num_batches, framework="pytorch"
    )
    return [{key: np.asarray(batch[key]) for key in KEYS} for batch in loader]


def _assert_same_batches(actual, expected) -> None:
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        for key in KEYS:
            assert a[key].shape == b[key].shape, key
            assert a[key].dtype == b[key].dtype, key
            assert a[key].tobytes() == b[key].tobytes(), key


def _lowdim_dataset(root: pathlib.Path, **kwargs) -> _lowdim.LowDimLeRobotDataset:
    meta = _lerobot_compat.LeRobotDatasetMetadata("test/lowdim", root=root)
    return _lowdim.LowDimLeRobotDataset(meta, delta_timestamps=_delta_timestamps(), prompt_from_task=True, **kwargs)


def test_batched_transform_matches_per_frame(dataset_root):
    dataset = _lowdim_dataset(dataset_root)
    assert _lowdim.batched_transform_matches_per_frame(dataset, _transforms.compose(_data_transforms()), KEYS)


class _FrameOnlyTransform(_transforms.DataTransformFn):
    """A transform that only works on single frames (like `PromptFromLeRobotTask`'s `int(...)`)."""

    def __call__(self, x: dict) -> dict:
        x["state"] = x["state"] * float(x["state"][0])  # Fails on a batch: `float()` needs a single element.
        return x


def test_transforms_that_only_work_per_frame_fall_back(dataset_root):
    reference = _data_loader.TransformedDataset(_reference_dataset(dataset_root), [_FrameOnlyTransform()])
    batch_size = 7
    num_batches = len(reference) // batch_size
    expected = _reference_batches(reference, batch_size, num_batches, shuffle=False)

    dataset = _lowdim_dataset(dataset_root)
    transform = _transforms.compose([*_data_transforms(), _FrameOnlyTransform()])
    assert not _lowdim.batched_transform_matches_per_frame(dataset, transform, KEYS)
    actual = list(
        _lowdim.iter_sequential_batches(dataset, transform, KEYS, batch_size, num_batches, per_frame_mode=True)
    )
    _assert_same_batches(actual, expected)


@pytest.mark.parametrize("per_frame_mode", [False, True])
def test_sequential_batches_match_lerobot_dataset(dataset_root, per_frame_mode):
    reference = _reference_dataset(dataset_root)
    batch_size = 7
    num_batches = len(reference) // batch_size
    expected = _reference_batches(reference, batch_size, num_batches, shuffle=False)

    dataset = _lowdim_dataset(
        dataset_root, num_frames=sum(length for length, _ in EPISODES), frame_index_is_position=True
    )
    assert len(dataset) == len(reference)
    actual = list(
        _lowdim.iter_sequential_batches(
            dataset,
            _transforms.compose(_data_transforms()),
            KEYS,
            batch_size,
            num_batches,
            per_frame_mode=per_frame_mode,
        )
    )
    _assert_same_batches(actual, expected)


def test_shuffled_batches_match_lerobot_dataset(dataset_root):
    reference = _reference_dataset(dataset_root)
    batch_size, num_batches = 4, 5
    expected = _reference_batches(reference, batch_size, num_batches, shuffle=True)

    dataset = _lowdim_dataset(dataset_root)
    batch_indices = _lowdim.shuffled_batch_indices(len(dataset), batch_size, num_batches)
    actual = list(_lowdim.iter_gathered_batches(dataset, _transforms.compose(_data_transforms()), KEYS, batch_indices))
    _assert_same_batches(actual, expected)


def test_episode_subset_matches_lerobot_dataset(dataset_root):
    episodes = [1, 2, 5]
    reference = _reference_dataset(dataset_root, episodes=episodes)
    batch_size = 5
    num_batches = len(reference) // batch_size
    expected = _reference_batches(reference, batch_size, num_batches, shuffle=False)

    dataset = _lowdim_dataset(dataset_root, episodes=episodes)
    assert len(dataset) == len(reference) == sum(EPISODES[i][0] for i in episodes)
    actual = list(
        _lowdim.iter_sequential_batches(dataset, _transforms.compose(_data_transforms()), KEYS, batch_size, num_batches)
    )
    _assert_same_batches(actual, expected)


def test_task_subset_matches_b1k_dataset(dataset_root):
    # `B1KLeRobotDataset` reads only the data files referenced by the selected episodes.
    reference = _b1k_dataset.B1KLeRobotDataset(
        "test/lowdim", dataset_root, task_names=["task_b"], delta_timestamps=_delta_timestamps()
    )
    tasks = _lerobot_compat.tasks_from_metadata(reference.meta)
    reference = _data_loader.TransformedDataset(reference, [_transforms.PromptFromLeRobotTask(tasks)])
    reference = _data_loader.TransformedDataset(reference, _data_transforms())
    batch_size = 3
    num_batches = len(reference) // batch_size
    expected = _reference_batches(reference, batch_size, num_batches, shuffle=False)

    meta = _b1k_dataset.B1KDatasetMetadata("test/lowdim", dataset_root)
    episodes = list(_b1k_dataset.select_task_subset(meta, ["task_b"]).episode_indices)
    assert episodes == [1, 3, 4]
    dataset = _lowdim.LowDimLeRobotDataset(
        meta,
        delta_timestamps=_delta_timestamps(),
        prompt_from_task=True,
        episodes=episodes,
        data_files=_lowdim.referenced_data_files(meta, episodes),
    )
    assert len(dataset) == len(reference)
    actual = list(
        _lowdim.iter_sequential_batches(dataset, _transforms.compose(_data_transforms()), KEYS, batch_size, num_batches)
    )
    _assert_same_batches(actual, expected)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_shuffled_batch_indices_match_torch_data_loader(num_workers):
    num_frames, batch_size, num_batches, seed = 50, 8, 6, 3  # All batches of one pass (drop_last).
    loader = _data_loader.TorchDataLoader(
        _lowdim._IndexDataset(num_frames),  # noqa: SLF001
        local_batch_size=batch_size,
        shuffle=True,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="pytorch",
    )
    expected = [np.asarray(batch) for batch in loader]
    actual = _lowdim.shuffled_batch_indices(num_frames, batch_size, num_batches, seed=seed)
    assert len(actual) == len(expected) == num_batches
    for a, b in zip(actual, expected, strict=True):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="exceed one pass"):
        _lowdim.shuffled_batch_indices(num_frames, batch_size, num_batches + 1, seed=seed)


def test_items_match_lerobot_items(dataset_root):
    dataset = _lerobot_compat.LeRobotDataset("test/lowdim", root=dataset_root, delta_timestamps=_delta_timestamps())
    lowdim = _lowdim_dataset(dataset_root)
    span = lowdim.spans[1]  # 3 frames, shorter than the horizon.
    per_frame, shared = lowdim.episode_items(span)
    assert shared == {}
    for t in range(len(span)):
        expected = dataset[span.offset + t]
        for key in [
            "observation.state",
            "action",
            "action_is_pad",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
        ]:
            assert torch.equal(per_frame[key][t], expected[key]), key
            assert per_frame[key][t].dtype == expected[key].dtype, key
        assert per_frame["task"][t] == expected["task"]
        assert per_frame["prompt"][t] == expected["task"]


def test_rejects_inconsistent_layout(dataset_root):
    with pytest.raises(ValueError, match="frames requested"):
        _lowdim_dataset(dataset_root, num_frames=10_000)
    with pytest.raises(ValueError, match="not in the episode metadata"):
        _lowdim_dataset(dataset_root, episodes=[42])
    # Reading the files in the wrong order breaks the "frame index == row position" assumption of LeRobotDataset.
    files = sorted(path.relative_to(dataset_root) for path in (dataset_root / "data").glob("*/*.parquet"))
    _lowdim_dataset(dataset_root, data_files=files, frame_index_is_position=True)
    with pytest.raises(ValueError, match="row positions"):
        _lowdim_dataset(dataset_root, data_files=files[::-1], frame_index_is_position=True)
