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
    # Goal-image views (goal-image conditioning): the goal frame of a camera. The radio skill-segment datasets ship
    # them as static per-episode clips under observation.goal_rgb.*; requests carry them under goal::<camera key>.
    goals={
        "goal_image_0": ObservationConfig(
            name="goal_head",
            obs_key="goal::robot_r1::robot_r1:zed_link:Camera:0::rgb",
            dataset_key="observation.goal_rgb.zed_link_camera_0",
            resolution=[240, 240]
        ),
        "goal_image_1": ObservationConfig(
            name="goal_left_wrist",
            obs_key="goal::robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
            dataset_key="observation.goal_rgb.left_realsense_link_camera_0",
            resolution=[240, 240]
        ),
        "goal_image_2": ObservationConfig(
            name="goal_right_wrist",
            obs_key="goal::robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
            dataset_key="observation.goal_rgb.right_realsense_link_camera_0",
            resolution=[240, 240]
        ),
    },
    action_key="action",
    action_dim=23,
    action=[
        StateActionConfig(name="base", indices=list(range(3))),
        StateActionConfig(name="torso", indices=list(range(3, 7)), needs_delta_comp=True),
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
