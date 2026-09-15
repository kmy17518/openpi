from .base_config import ObservationConfig, StateActionConfig, RobotConfig, register_robot


# R1Pro Robot Configuration
# Dual-arm mobile manipulator with base, torso, and multiple camera views
R1Pro = RobotConfig(
    name="robot_r1",
    robot_type="R1Pro",
    observations={
        "image_0": ObservationConfig(
            name="head",
            obs_key="robot_r1::robot_r1:zed_link:Camera:0::rgb",
            dataset_key="observation.rgb.zed_link_camera_0",
            resolution=[240, 240]
        ),
        "image_1": ObservationConfig(
            name="left_wrist",
            obs_key="robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.left_realsense_link_camera_0",
            resolution=[240, 240]
        ),
        "image_2": ObservationConfig(
            name="right_wrist",
            obs_key="robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.right_realsense_link_camera_0",
            resolution=[240, 240]
        ),
    },
    action_key="action",
    action_dim=23,
    action=[
        StateActionConfig(name="base", indices=list(range(3))),
        # Trunk joints 1-3 are predicted as deltas from the measured trunk position, which sits at dims 3..5 of the
        # extracted state (`trunk_qpos` below, after the 3 `base_qvel` dims). The explicit mapping is required:
        # size-based matching would pair this 3-dim group with the 3-dim `base_qvel` slice.
        StateActionConfig(
            name="torso", indices=list(range(3, 6)), needs_delta_comp=True, delta_state_indices=[3, 4, 5]
        ),
        # Trunk joint 4 is never commanded in the challenge demos: action[6] is identically 0 in all 100 tasks, while
        # its measured position (observation.state[56]) occasionally yields under load by up to ~0.15 rad. As a delta
        # the training target would be -state, i.e. ~0 in 99.99% of frames with rare outliers, so its q01/q99
        # normalization range collapses to ~+-0.001 and the outliers normalize to |z| > 100. Kept absolute, the
        # target is a constant that normalizes to a constant and un-normalizes back to 0 at serving time.
        StateActionConfig(name="torso_joint4", indices=[6]),
        StateActionConfig(name="left_arm", indices=list(range(7, 14)), needs_delta_comp=True),
        StateActionConfig(name="left_gripper", indices=[14], is_eef=True),
        StateActionConfig(name="right_arm", indices=list(range(15, 22)), needs_delta_comp=True),
        StateActionConfig(name="right_gripper", indices=[22], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="base_qvel", indices=list(range(0, 3))),
        StateActionConfig(name="trunk_qpos", indices=list(range(53, 57))),
        StateActionConfig(name="left_arm_qpos", indices=list(range(3, 10))),
        StateActionConfig(name="left_gripper_qpos", indices=list(range(24, 26)), is_eef=True),
        StateActionConfig(name="right_arm_qpos", indices=list(range(28, 35))),
        StateActionConfig(name="right_gripper_qpos", indices=list(range(49, 51)), is_eef=True),
    ],
)

# Register robots in the global registry
register_robot("b1k/R1Pro", R1Pro)
