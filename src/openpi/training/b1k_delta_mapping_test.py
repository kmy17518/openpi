import copy

import numpy as np
import pytest
import tyro

from openpi import transforms
from openpi.configs.robots.b1k import R1Pro
from openpi.configs.robots.base_config import RobotConfig
from openpi.configs.robots.base_config import StateActionConfig
from openpi.training import config


def _robot(*actions: StateActionConfig, state_sizes: tuple[int, ...] = (2, 2)) -> RobotConfig:
    return RobotConfig(
        name="test",
        robot_type="test",
        observations={},
        action_key="action",
        action_dim=sum(len(action.indices) for action in actions),
        action=list(actions),
        proprio=[StateActionConfig(name=f"state_{i}", indices=list(range(size))) for i, size in enumerate(state_sizes)],
    )


def _mappings(robot):
    return config.LeRobotB1KDataConfig()._build_delta_mappings(robot)  # noqa: SLF001


@pytest.mark.parametrize("explicit_first", [True, False])
def test_explicit_mapping_reserves_state_dimensions(*, explicit_first: bool):
    left = StateActionConfig("left", [0, 1], needs_delta_comp=True, delta_state_indices=[0, 1])
    right = StateActionConfig("right", [2, 3], needs_delta_comp=True)
    robot = _robot(*(left, right) if explicit_first else (right, left))
    mappings = _mappings(robot)
    assert {tuple(a): s for a, s in mappings} == {(0, 1): [0, 1], (2, 3): [2, 3]}
    state = np.array([1, 2, 10, 20], dtype=np.float32)
    actions = np.array([[11, 22, 110, 220], [12, 24, 130, 240]], dtype=np.float32)
    result = transforms.MappedDeltaActions(mappings)({"state": state, "actions": actions.copy()})
    np.testing.assert_array_equal(result["actions"], [[10, 20, 100, 200], [11, 22, 120, 220]])
    np.testing.assert_array_equal(transforms.MappedAbsoluteActions(mappings)(result)["actions"], actions)


def test_partial_explicit_mapping_blocks_whole_group_from_inference():
    robot = _robot(
        StateActionConfig("partial", [0], needs_delta_comp=True, delta_state_indices=[1]),
        StateActionConfig("inferred", [1, 2], needs_delta_comp=True),
    )
    mappings = _mappings(robot)
    assert mappings == [([0], [1]), ([1, 2], [2, 3])]
    result = transforms.MappedDeltaActions(mappings)(
        {"state": np.array([1, 2, 10, 20]), "actions": np.array([[5, 110, 220]])}
    )
    np.testing.assert_array_equal(result["actions"], [[3, 100, 200]])


def test_inference_rejects_only_partially_occupied_candidate():
    robot = _robot(
        StateActionConfig("partial", [0], needs_delta_comp=True, delta_state_indices=[1]),
        StateActionConfig("inferred", [1, 2], needs_delta_comp=True),
        state_sizes=(2,),
    )
    with pytest.raises(ValueError, match="Could not find an unoccupied state slice"):
        _mappings(robot)


def test_explicit_maps_can_share_state_for_distinct_actions():
    robot = _robot(
        StateActionConfig("left", [0, 1], needs_delta_comp=True, delta_state_indices=[0, 1]),
        StateActionConfig("mimic", [2, 3], needs_delta_comp=True, delta_state_indices=[0, 1]),
        StateActionConfig("inferred", [4, 5], needs_delta_comp=True),
    )
    mappings = _mappings(robot)
    assert mappings == [([0, 1], [0, 1]), ([2, 3], [0, 1]), ([4, 5], [2, 3])]
    result = transforms.MappedDeltaActions(mappings)(
        {"state": np.array([1, 2, 10, 20]), "actions": np.full((1, 6), 100)}
    )
    np.testing.assert_array_equal(result["actions"], [[99, 98, 99, 98, 90, 80]])


@pytest.mark.parametrize("indices", [[-1, 0], [0, 4], [0, 0], [0, 1.5], [0, True]])
def test_reject_invalid_explicit_state_indices(indices):
    robot = _robot(StateActionConfig("arm", [0, 1], needs_delta_comp=True, delta_state_indices=indices))
    with pytest.raises(ValueError, match="delta_state_indices"):
        _mappings(robot)


@pytest.mark.parametrize("indices", [[-1, 0], [0, 2], [0, 0], [0, 1.5], [0, True]])
def test_reject_invalid_action_indices(indices):
    robot = _robot(StateActionConfig("arm", indices, needs_delta_comp=True))
    with pytest.raises(ValueError, match="action indices"):
        _mappings(robot)


def test_reject_duplicate_action_dimensions_across_delta_groups():
    robot = _robot(
        StateActionConfig("left", [0, 1], needs_delta_comp=True),
        StateActionConfig("right", [1, 2], needs_delta_comp=True),
    )
    with pytest.raises(ValueError, match="already mapped action indices"):
        _mappings(robot)


def test_reject_mismatched_delta_mapping_lengths():
    robot = _robot(StateActionConfig("arm", [0, 1], needs_delta_comp=True))
    robot.action[0].delta_state_indices = [0]
    with pytest.raises(ValueError, match="2 action indices but 1 delta_state_indices"):
        _mappings(robot)


def test_r1pro_partial_torso_mapping_subtracts_only_commanded_joints():
    robot = copy.deepcopy(R1Pro)
    mappings = _mappings(robot)
    assert mappings == [
        ([3, 4, 5], [3, 4, 5]),
        (list(range(7, 14)), list(range(7, 14))),
        (list(range(15, 22)), list(range(15, 22))),
    ]
    state = np.arange(1, 24, dtype=np.float32)
    actions = np.full((2, 23), 100, dtype=np.float32)
    expected = actions.copy()
    expected[:, 3:6] -= state[3:6]
    expected[:, 7:14] -= state[7:14]
    expected[:, 15:22] -= state[15:22]
    result = transforms.MappedDeltaActions(mappings)({"state": state, "actions": actions})
    np.testing.assert_array_equal(result["actions"], expected)
    assert np.all(result["actions"][:, 6] == 100)


@pytest.mark.parametrize("extra_delta", [True, False])
def test_data_config_records_resolved_representation(tmp_path, monkeypatch, *, extra_delta: bool):
    from openpi.models import pi0_config

    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *_args: transforms.Group())
    factory = config.LeRobotB1KDataConfig(
        repo_id="org/demos", robot_config_name="b1k/R1Pro", extra_delta_transform=extra_delta, allow_legacy_assets=True
    )
    data = factory.create(tmp_path, pi0_config.Pi0Config(pi05=True))
    assert data.action_representation["delta_mappings"] == (
        [[a, s] for a, s in _mappings(R1Pro)] if extra_delta else []
    )
    assert data.allow_legacy_assets
    assert data.inference_metadata is None
    assert data.norm_stats_metadata is None


@pytest.mark.parametrize("repo_id", ["org/demos", ["org/demos", "org/other"]])
def test_data_config_loads_asset_metadata(tmp_path, repo_id):
    from openpi.models import pi0_config
    from openpi.training import b1k_artifacts

    metadata = b1k_artifacts.norm_metadata({"robot_type": "test"})
    b1k_artifacts.save_metadata(tmp_path / "assets" / "org/demos", metadata)
    factory = config.LeRobotB1KDataConfig(
        repo_id=repo_id, assets=config.AssetsConfig(assets_dir=str(tmp_path / "assets"))
    )
    data = factory.create_base_config(tmp_path / "unused", pi0_config.Pi0Config(pi05=True))
    assert data.norm_stats_metadata == metadata
    assert data.norm_stats is None


def test_legacy_assets_cli_flag():
    factory = tyro.cli(
        config.LeRobotB1KDataConfig,
        args=["--repo-id", "org/demos", "--robot-config-name", "b1k/R1Pro", "--allow-legacy-assets"],
    )
    assert factory.allow_legacy_assets
