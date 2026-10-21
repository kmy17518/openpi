"""BEHAVIOR input/output adapters: gripper actions (absolute / delta), served gripper clamp and snap, padding mask."""

import numpy as np
import pytest

from openpi import transforms
from openpi.configs.robots.base_config import ROBOT_REGISTRY
import openpi.configs.robots.b1k  # noqa: F401  (registers b1k/R1Pro)
from openpi.models import model as _model
import openpi.models.pi0_config as _pi0_config
from openpi.policies import b1k_policy
from openpi.training import b1k_artifacts
from openpi.training import b1k_dataset
from openpi.training import config as _config

HORIZON = 6


def _sample(rng: np.random.Generator) -> dict:
    proprio = rng.uniform(-1, 1, 61).astype(np.float32)
    proprio[[24, 25]] = [0.012, 0.015]  # left fingers holding an object
    proprio[[49, 50]] = [0.0, 0.0]  # right gripper closed
    actions = rng.uniform(-1, 1, (HORIZON, 23)).astype(np.float32)
    actions[:, 14] = [-0.5, -0.47, -0.46, -1.0, 0.2, 1.0]
    actions[:, 22] = -1.0
    return {
        **{f"observation/image_{i}": np.zeros((224, 224, 3), np.uint8) for i in range(3)},
        "observation/state": proprio,
        "actions": actions,
        "action_is_pad": np.array([False] * 4 + [True] * 2),
        "prompt": "turning_on_radio",
    }


def test_delta_gripper_state_is_in_command_units():
    robot = ROBOT_REGISTRY["b1k/R1Pro"]
    sample = _sample(np.random.default_rng(0))
    absolute = b1k_policy.B1KInputs(robot_config=robot, model_type=_model.ModelType.PI05)(dict(sample))
    delta = b1k_policy.B1KInputs(robot_config=robot, model_type=_model.ModelType.PI05, gripper_action="delta")(
        dict(sample)
    )
    np.testing.assert_allclose(absolute["state"][[14, 22]], [0.027, 0.0], atol=1e-6)
    # u = 20 * (q1 + q2) - 1: 0.027 -> -0.46 (holding), 0 -> -1 (closed), 0.1 -> +1 (open)
    np.testing.assert_allclose(delta["state"][[14, 22]], [-0.46, -1.0], atol=1e-5)
    np.testing.assert_allclose(b1k_policy.gripper_state_in_command_units(np.float32(0.1)), 1.0, atol=1e-6)
    other = [i for i in range(23) if i not in (14, 22)]
    np.testing.assert_array_equal(delta["state"][other], absolute["state"][other])
    np.testing.assert_array_equal(delta["action_is_pad"], sample["action_is_pad"])
    assert delta["action_is_pad"].dtype == bool


def test_delta_then_absolute_reproduces_the_dataset_command():
    robot = ROBOT_REGISTRY["b1k/R1Pro"]
    absolute_mappings = _config.LeRobotB1KDataConfig(robot_config_name="b1k/R1Pro")._build_delta_mappings(robot)  # noqa: SLF001
    mappings = _config.LeRobotB1KDataConfig(robot_config_name="b1k/R1Pro", gripper_action="delta")._build_delta_mappings(  # noqa: SLF001
        robot
    )
    assert mappings == [*absolute_mappings, ([14], [14]), ([22], [22])]
    inputs = b1k_policy.B1KInputs(robot_config=robot, model_type=_model.ModelType.PI05, gripper_action="delta")
    outputs = b1k_policy.B1KOutputs(action_dim=23)
    for seed in range(3):
        sample = _sample(np.random.default_rng(seed))
        command = sample["actions"].copy()
        encoded = transforms.MappedDeltaActions(mappings)(inputs({**sample, "actions": command.copy()}))
        # gripper targets become deltas against the gripper state in command units
        np.testing.assert_allclose(encoded["actions"][:, 14], command[:, 14] + 0.46, atol=1e-5)
        np.testing.assert_allclose(encoded["actions"][:, 22], 0.0, atol=1e-6)  # closed hand pushing: exactly 0
        decoded = outputs(transforms.MappedAbsoluteActions(mappings)({"state": encoded["state"], "actions": encoded["actions"]}))
        np.testing.assert_allclose(decoded["actions"], command, atol=1e-5)


def test_b1k_outputs_clamp_grippers_and_snap_them_open_above_the_threshold():
    actions = np.zeros((4, 32), np.float32)
    actions[:, 14] = [1.3, 0.95, 0.85, -1.4]
    actions[:, 22] = [0.9, 0.91, -0.2, -1.0]
    actions[:, 3] = 5.0  # non-gripper dims are never clamped
    snapped = b1k_policy.B1KOutputs(action_dim=23, gripper_snap_threshold=0.9, gripper_dims=(14, 22))({"actions": actions})
    assert snapped["actions"].shape == (4, 23)
    np.testing.assert_array_equal(snapped["actions"][:, 14], np.float32([1.0, 1.0, 0.85, -1.0]))
    np.testing.assert_array_equal(snapped["actions"][:, 22], np.float32([0.9, 1.0, -0.2, -1.0]))
    np.testing.assert_array_equal(snapped["actions"][:, 3], 5.0)
    assert actions[0, 14] == np.float32(1.3)  # the model output is not modified in place
    raw = b1k_policy.B1KOutputs(action_dim=23, gripper_dims=(14, 22))({"actions": actions})
    np.testing.assert_array_equal(raw["actions"], actions[:, :23])


def test_gripper_action_is_recorded_in_the_representation_and_asset_id(tmp_path):
    robot = ROBOT_REGISTRY["b1k/R1Pro"]
    absolute = b1k_artifacts.action_representation(robot, extra_delta_transform=True)
    assert "gripper_action" not in absolute  # unchanged for absolute artifacts written before the option existed
    delta = b1k_artifacts.action_representation(robot, extra_delta_transform=True, gripper_action="delta")
    assert delta == {**absolute, "gripper_action": "delta"}
    assert b1k_dataset.task_subset_asset_id("org/demos", ["b", "a"], settle_steps="0") == (
        "org/demos/task_subsets/a+b/settle_steps/0"
    )
    assert b1k_dataset.task_subset_asset_id("org/demos", ["b", "a"], settle_steps="0", gripper_action="delta") == (
        "org/demos/task_subsets/a+b/settle_steps/0/gripper_action/delta"
    )
    assert b1k_dataset.task_subset_asset_id("org/demos", None, gripper_action="delta") == "org/demos/gripper_action/delta"

    model = _pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")

    def data_config(**kwargs):
        return _config.LeRobotB1KDataConfig(
            repo_id="org/demos",
            base_config=_config.DataConfig(prompt_from_task=True),
            robot_config_name="b1k/R1Pro",
            task_names=["a"],
            **kwargs,
        ).create(tmp_path, model)

    config = data_config(gripper_action="delta", settle_steps="0")
    assert config.asset_id == "org/demos/task_subsets/a/settle_steps/0/gripper_action/delta"
    assert config.action_representation == delta
    assert config.repack_transforms.inputs[0].structure["action_is_pad"] == "action_is_pad"
    (inputs,) = [t for t in config.data_transforms.inputs if isinstance(t, b1k_policy.B1KInputs)]
    (to_delta,) = [t for t in config.data_transforms.inputs if isinstance(t, transforms.MappedDeltaActions)]
    (to_absolute, outputs) = config.data_transforms.outputs
    assert inputs.gripper_action == "delta" and ([14], [14]) in to_delta.mappings and ([22], [22]) in to_delta.mappings
    assert isinstance(to_absolute, transforms.MappedAbsoluteActions) and to_absolute.mappings == to_delta.mappings
    assert outputs.gripper_dims == (14, 22) and outputs.gripper_snap_threshold == 0.9
    default = data_config()
    assert default.asset_id == "org/demos/task_subsets/a" and default.action_representation == absolute
    assert data_config(gripper_snap_threshold=None).data_transforms.outputs[-1].gripper_snap_threshold is None
    with pytest.raises(ValueError, match="gripper_action must be one of"):
        data_config(gripper_action="binary")
    with pytest.raises(ValueError, match="needs extra_delta_transform"):
        data_config(gripper_action="delta", extra_delta_transform=False)
