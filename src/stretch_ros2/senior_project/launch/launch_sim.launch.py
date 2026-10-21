#!/usr/bin/env python3
"""
Plain simulation launch: MuJoCo sim + robot model + RViz + MuJoCo viewer.
No learner, no expert, no reward monitor. Drive it yourself (teleop / cmd_vel).

Reuses RL.launch.py with run_rl:=false so the sim setup exists in one place.

Usage:
    ros2 launch senior_project launch_sim.launch.py
    ros2 launch senior_project launch_sim.launch.py mode:=navigation use_rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument(
        "mode", default_value="position", choices=["position", "navigation", "trajectory", "gamepad"]))
    ld.add_action(DeclareLaunchArgument("use_rviz", default_value="true", choices=["true", "false"]))
    ld.add_action(DeclareLaunchArgument("use_mujoco_viewer", default_value="true", choices=["true", "false"]))
    ld.add_action(DeclareLaunchArgument("use_cameras", default_value="false", choices=["true", "false"]))
    ld.add_action(DeclareLaunchArgument(
        "mujoco_xml", default_value="",
        description="MuJoCo world XML (empty = driver default world, same as RL.launch.py)"))

    rl_launch = os.path.join(get_package_share_directory("senior_project"), "launch", "RL.launch.py")
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rl_launch),
        launch_arguments={
            "run_rl": "false",
            "random_spawn": "false",
            "use_reward_monitor": "false",
            "mode": LaunchConfiguration("mode"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "use_mujoco_viewer": LaunchConfiguration("use_mujoco_viewer"),
            "use_cameras": LaunchConfiguration("use_cameras"),
            "mujoco_xml": LaunchConfiguration("mujoco_xml"),
        }.items(),
    ))
    return ld
