#!/usr/bin/env python3
"""
Expert Baseline Launch File

Starts the same MuJoCo simulation as RL.launch.py but runs the
ExpertStateMachine controller instead of the RL learner.

This gives you a fair comparison: same sim, same physics, same reward
structure, same episode timing — only the controller is different.

Usage:
    # Run on its own domain so it doesn't interfere with RL sims
    ROS_DOMAIN_ID=20 ros2 launch senior_project expert_baseline.launch.py

    # With SAM3 goal generator (for goal-seeking comparison)
    ROS_DOMAIN_ID=20 ros2 launch senior_project expert_baseline.launch.py \
        use_sam3:=true target:=person

    # Headless, no RViz (for parallel benchmarking)
    ROS_DOMAIN_ID=20 ros2 launch senior_project expert_baseline.launch.py \
        use_rviz:=false use_mujoco_viewer:=false

    # Then monitor everything:
    python reward_monitor.py --scan-start 0 --scan-end 25 \
        --experiment ~/experiments/expert_vs_rl
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # =========================================================================
    # Arguments
    # =========================================================================

    ld.add_action(DeclareLaunchArgument("ns", default_value="stretch"))
    ld.add_action(DeclareLaunchArgument(
        "use_rviz", default_value="false", choices=["true", "false"],
        description="Launch RViz (default off for headless baseline runs)",
    ))
    ld.add_action(DeclareLaunchArgument(
        "use_mujoco_viewer", default_value="false", choices=["true", "false"],
    ))
    ld.add_action(DeclareLaunchArgument(
        "use_cameras", default_value="true", choices=["true", "false"],
    ))
    ld.add_action(DeclareLaunchArgument(
        "mujoco_xml",
        default_value=os.path.expanduser(
            "~/ament_ws/src/stretch_ros2/senior_project/config/scene.xml"
        ),
        description="Path to MuJoCo scene XML",
    ))

    # Sim args
    ld.add_action(DeclareLaunchArgument(
        "broadcast_odom_tf", default_value="True", choices=["True", "False"],
    ))
    ld.add_action(DeclareLaunchArgument(
        "fail_out_of_range_goal", default_value="False", choices=["False", "True"],
    ))
    ld.add_action(DeclareLaunchArgument(
        "mode", default_value="navigation",
        choices=["position", "navigation", "trajectory", "gamepad"],
    ))

    # Topic configuration (same defaults as RL.launch.py)
    ld.add_action(DeclareLaunchArgument("odom_topic", default_value="/stretch/odom"))
    ld.add_action(DeclareLaunchArgument("cmd_topic", default_value="/stretch/cmd_vel"))
    ld.add_action(DeclareLaunchArgument("lidar_topic", default_value="/stretch/scan"))
    ld.add_action(DeclareLaunchArgument("imu_topic", default_value="imu"))
    ld.add_action(DeclareLaunchArgument("goal_topic", default_value="goal"))

    # Expert-specific args
    ld.add_action(DeclareLaunchArgument(
        "total_episodes", default_value="0",
        description="Episodes to run (0 = run forever)",
    ))
    ld.add_action(DeclareLaunchArgument(
        "total_steps", default_value="0",
        description="Max steps (0 = no limit)",
    ))

    # SAM3 goal generator (optional — for goal-seeking comparison)
    ld.add_action(DeclareLaunchArgument(
        "use_sam3", default_value="false", choices=["true", "false"],
        description="Start SAM3 goal generator for goal-seeking tasks",
    ))
    ld.add_action(DeclareLaunchArgument("target", default_value="person"))
    ld.add_action(DeclareLaunchArgument("sam3_server_port", default_value="8100"))
    ld.add_action(DeclareLaunchArgument("confidence_threshold", default_value="0.2"))

    # Robocasa stubs (for MuJoCo driver compatibility)
    ld.add_action(DeclareLaunchArgument("robocasa_task", default_value="PnPCounterToCab"))
    ld.add_action(DeclareLaunchArgument("robocasa_layout", default_value="Random"))
    ld.add_action(DeclareLaunchArgument("robocasa_style", default_value="Random"))

    # =========================================================================
    # URDF + Robot State Publisher (identical to RL.launch.py)
    # =========================================================================

    robot_description_file = Path(
        get_package_share_directory("stretch_description")
    ) / "urdf" / "stretch.urdf"

    mesh_root = get_package_share_directory("stretch_description")

    with open(robot_description_file, "r") as f:
        robot_description_content = f.read()

    robot_description_content = robot_description_content.replace(
        'filename="./meshes/', f'filename="file://{mesh_root}/meshes/'
    )

    ld.add_action(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            namespace=LaunchConfiguration("ns"),
            output="both",
            parameters=[
                {"robot_description": robot_description_content},
                {"publish_frequency": 30.0},
            ],
            arguments=["--ros-args", "--log-level", "error"],
        )
    )

    ld.add_action(
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            namespace=LaunchConfiguration("ns"),
            output="log",
            parameters=[
                {"source_list": ["/stretch/joint_states"]},
                {"rate": 30.0},
            ],
            arguments=["--ros-args", "--log-level", "error"],
        )
    )

    # =========================================================================
    # RViz (optional)
    # =========================================================================

    ld.add_action(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=[],
            remappings=[
                ("/move_base_simple/goal", "/goal"),
                ("/goal_pose", "/goal"),
            ],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    # =========================================================================
    # MuJoCo Simulation Driver (identical to RL.launch.py)
    # =========================================================================

    driver_params = [
        {
            "rate": 100.0,
            "timeout": 0.5,
            "broadcast_odom_tf": LaunchConfiguration("broadcast_odom_tf"),
            "fail_out_of_range_goal": LaunchConfiguration("fail_out_of_range_goal"),
            "mode": LaunchConfiguration("mode"),
            "use_mujoco_viewer": LaunchConfiguration("use_mujoco_viewer"),
            "use_cameras": LaunchConfiguration("use_cameras"),
            "mujoco_xml": LaunchConfiguration("mujoco_xml"),
            "use_robocasa": False,
            "robocasa_task": LaunchConfiguration("robocasa_task"),
            "robocasa_layout": LaunchConfiguration("robocasa_layout"),
            "robocasa_style": LaunchConfiguration("robocasa_style"),
        }
    ]
    ld.add_action(
        Node(
            package="stretch_simulation",
            executable="stretch_mujoco_driver",
            emulate_tty=True,
            output="screen",
            remappings=[
                ("cmd_vel", "/stretch/cmd_vel"),
                ("joint_states", "/stretch/joint_states"),
                ("/scan_filtered", "/stretch/scan"),
                ("odom", "/stretch/odom"),
            ],
            parameters=driver_params,
        )
    )

    # =========================================================================
    # Expert Baseline Controller (replaces the RL learner)
    # =========================================================================

    expert_proc = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "senior_project.expert_baseline_node",
            "--ns",
            LaunchConfiguration("ns"),
            "--odom-topic",
            LaunchConfiguration("odom_topic"),
            "--cmd-topic",
            LaunchConfiguration("cmd_topic"),
            "--lidar-topic",
            LaunchConfiguration("lidar_topic"),
            "--imu-topic",
            LaunchConfiguration("imu_topic"),
            "--goal-topic",
            LaunchConfiguration("goal_topic"),
            "--total-episodes",
            LaunchConfiguration("total_episodes"),
            "--total-steps",
            LaunchConfiguration("total_steps"),
        ],
        output="screen",
    )

    # Delay so sim is alive first (same as RL.launch.py)
    ld.add_action(TimerAction(period=2.0, actions=[expert_proc]))

    # =========================================================================
    # SAM3 Goal Generator (optional — for goal-seeking comparison)
    # =========================================================================

    # If use_sam3 is true, start the goal generator so the expert has goals
    # to navigate toward, making the comparison with RL goal-seeking fair.
    home = os.path.expanduser("~")
    sam3_goal_cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"source {home}/ament_ws/install/setup.bash && "
        f"ros2 run senior_project sam3_goal_generator "
        f"--ros-args "
        f"-p target:=$(ros2 launch --print-arguments | echo person) "  # fallback
        f"-p auto_publish_goal:=true "
        f"-p depth_mode:=auto "
        f"-p server_url:=http://localhost:8100"
    )

    # Simpler: use ExecuteProcess with the module directly
    sam3_goal_proc = ExecuteProcess(
        cmd=[
            "python3", "-m", "senior_project.sam3_goal_generator",
            "--ros-args",
            "-p", ["target:=", LaunchConfiguration("target")],
            "-p", "auto_publish_goal:=true",
            "-p", "depth_mode:=auto",
            "-p", ["server_url:=http://localhost:", LaunchConfiguration("sam3_server_port")],
            "-p", ["confidence_threshold:=", LaunchConfiguration("confidence_threshold")],
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_sam3")),
    )
    ld.add_action(TimerAction(period=5.0, actions=[sam3_goal_proc]))

    # =========================================================================
    # Info
    # =========================================================================

    ld.add_action(
        LogInfo(msg=""),
    )
    ld.add_action(
        LogInfo(msg="============================================================"),
    )
    ld.add_action(
        LogInfo(msg="  EXPERT BASELINE — State Machine Controller"),
    )
    ld.add_action(
        LogInfo(msg="  Same sim, same rewards, same episodes as RL.launch.py"),
    )
    ld.add_action(
        LogInfo(msg="  Phase tag: EXPERT_BASELINE (visible in reward_monitor)"),
    )
    ld.add_action(
        LogInfo(msg="============================================================"),
    )
    ld.add_action(
        LogInfo(msg=""),
    )

    return ld
