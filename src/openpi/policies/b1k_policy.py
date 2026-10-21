from collections.abc import Sequence
import dataclasses
import typing
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.configs.robots.base_config import RobotConfig

# Gripper action dims (the robot config's end-effector action groups): `absolute` commands, or `delta` commands
# relative to the gripper state in command units (`--data.gripper-action`).
GripperAction = Literal["absolute", "delta"]
GRIPPER_ACTIONS: tuple[str, ...] = typing.get_args(GripperAction)
# Served gripper commands above this value are snapped to +1 (fully open), after clamping to [-1, 1].
DEFAULT_GRIPPER_SNAP_THRESHOLD = 0.9


def gripper_state_in_command_units(width):
    """Summed finger positions ``q1 + q2`` as a gripper command: 0 (closed) -> -1, 0.1 (both fingers open) -> +1."""
    return 20.0 * width - 1.0


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/image_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(
    proprio_data, robot_config: RobotConfig, gripper_action: GripperAction = "absolute"
) -> np.ndarray:
    """Extract state from proprioception data based on robot configuration.

    We assume perfect correlation for the two gripper fingers.

    Args:
        proprio_data: Raw proprioception data
        robot_config: RobotConfig instance containing robot configuration
        gripper_action: ``delta`` expresses each gripper state in command units, the reference of delta gripper actions

    Returns:
        Extracted state array
    """
    state = []
    for proprio in robot_config.proprio:
        if proprio.is_eef:
            # Sum the gripper finger positions to get a single width value
            width = proprio_data[..., proprio.indices].sum(axis=-1, keepdims=True)
            state.append(gripper_state_in_command_units(width) if gripper_action == "delta" else width)
        else:
            state.append(proprio_data[..., proprio.indices])
    return np.concatenate(state, axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1KInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Robot configuration object
    robot_config: RobotConfig = dataclasses.field(default=None)

    # Goal-image conditioning: robot-config goal views ("goal_image_0", ...) read from `observation/<key>` and
    # emitted as the model's goal image slots (_model.GOAL_IMAGE_KEYS, same order), masks True. Empty = no goals.
    goal_views: tuple[str, ...] = ()

    gripper_action: GripperAction = "absolute"

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data, self.robot_config, self.gripper_action)
        if "actions" in data:
            action = data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        images, image_masks = [], []
        for camera_id in self.robot_config.observations:
            images.append(_parse_image(data[f"observation/{camera_id}"]))
            image_masks.append(np.True_)
        while len(images) < 3:
            images.append(np.zeros_like(images[0]))
            image_masks.append(np.False_)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                for view, slot in zip(self.goal_views, _model.GOAL_IMAGE_KEYS[: len(self.goal_views)], strict=True):
                    # goal slots after the cameras, in the fixed slot order; one image per sample, no history axis
                    images.append(_parse_image(data[f"observation/{view}"]))
                    image_masks.append(np.True_)
                    names += (slot,)
                images = tuple(images)
                image_masks = tuple(image_masks)
            case _model.ModelType.PI0_FAST:
                if self.goal_views:
                    raise ValueError("Goal-image slots are implemented for the pi0 / pi05 models, not pi0-FAST")
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = tuple(images)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action
        if "action_is_pad" in data:
            # action-chunk steps past the episode end, excluded from the training loss
            inputs["action_is_pad"] = np.asarray(data["action_is_pad"], dtype=bool)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


def snap_grippers(actions: np.ndarray, gripper_dims: Sequence[int], threshold: float | None) -> np.ndarray:
    """Clamp the gripper commands to [-1, 1] and set those above ``threshold`` to +1; unchanged without a threshold."""
    if threshold is None or not gripper_dims:
        return actions
    actions = np.array(actions, copy=True)
    dims = list(gripper_dims)
    grippers = np.clip(actions[..., dims], -1.0, 1.0)
    grippers[grippers > threshold] = 1.0
    actions[..., dims] = grippers
    return actions


@dataclasses.dataclass(frozen=True)
class B1KOutputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    # Absolute gripper commands (after MappedAbsoluteActions) are clamped and snapped open above this threshold
    # (`snap_grippers`); None serves them unchanged.
    gripper_snap_threshold: float | None = None
    gripper_dims: tuple[int, ...] = ()

    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        actions = np.asarray(data["actions"][:, : self.action_dim])
        return {"actions": snap_grippers(actions, self.gripper_dims, self.gripper_snap_threshold)}
