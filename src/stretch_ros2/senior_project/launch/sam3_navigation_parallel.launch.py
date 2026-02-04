#!/usr/bin/env python3
"""
SAM3 Navigation - PARALLEL ONLY (ament_python friendly)

This launch file ONLY starts multi-sim training by calling the installed
console script:

  ros2 run senior_project parallel_runner ...

Usage:
  ros2 launch senior_project sam3_navigation_parallel.launch.py
  ros2 launch senior_project sam3_navigation_parallel.launch.py num_sims:=4 target:=cup headless:=true
  ros2 launch senior_project sam3_navigation_parallel.launch.py extra:="use_mono_depth_server=true mono_depth_model=depth_anything"
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument(
        "num_sims", default_value="4",
        description="Number of parallel simulations"
    ))
    ld.add_action(DeclareLaunchArgument(
        "target", default_value="person",
        description="Target object to navigate to"
    ))
    ld.add_action(DeclareLaunchArgument(
        "total_steps", default_value="200000",
        description="Total training steps per simulation"
    ))
    ld.add_action(DeclareLaunchArgument(
        "rollout_steps", default_value="2048",
        description="Rollout steps per update"
    ))
    ld.add_action(DeclareLaunchArgument(
        "ckpt_dir",
        default_value="~/ament_ws/src/stretch_ros2/senior_project/parallel_training",
        description="Base checkpoint directory"
    ))
    ld.add_action(DeclareLaunchArgument(
        "stagger_delay", default_value="10.0",
        description="Delay between launching sims (seconds)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "headless", default_value="true",
        description="Headless mode propagated into each sim launch (true/false)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "merge_on_exit", default_value="false",
        description="If true, runner merges checkpoints on exit"
    ))
    ld.add_action(DeclareLaunchArgument(
        "extra", default_value="",
        description="Extra launch args forwarded to each sim as key=value pairs (space-separated)"
    ))

    ld.add_action(LogInfo(msg=""))
    ld.add_action(LogInfo(msg="============================================================"))
    ld.add_action(LogInfo(msg="  SAM3 Navigation - PARALLEL ONLY"))
    ld.add_action(LogInfo(msg="============================================================"))
    ld.add_action(LogInfo(msg=""))

    ld.add_action(OpaqueFunction(function=_start_parallel_runner))
    return ld


def _start_parallel_runner(context):
    num_sims = LaunchConfiguration("num_sims").perform(context)
    target = LaunchConfiguration("target").perform(context)
    total_steps = LaunchConfiguration("total_steps").perform(context)
    rollout_steps = LaunchConfiguration("rollout_steps").perform(context)
    ckpt_dir = os.path.expanduser(LaunchConfiguration("ckpt_dir").perform(context))
    stagger_delay = LaunchConfiguration("stagger_delay").perform(context)
    headless = LaunchConfiguration("headless").perform(context)
    merge_on_exit = LaunchConfiguration("merge_on_exit").perform(context).strip().lower()
    extra = LaunchConfiguration("extra").perform(context).strip()

    # IMPORTANT: use --launch_file (runner arg) and point to the SINGLE-sim launch file
    cmd = [
        "ros2", "run", "senior_project", "parallel_runner",
        "--num_sims", str(num_sims),
        "--target", str(target),
        "--total_steps", str(total_steps),
        "--rollout_steps", str(rollout_steps),
        "--ckpt_dir", str(ckpt_dir),
        "--stagger_delay", str(stagger_delay),
        "--headless", str(headless),
        "--package", "senior_project",
        "--launch_file", "sam3_navigation.launch.py",
    ]

    if merge_on_exit in ("true", "1", "yes", "on"):
        cmd.append("--merge_on_exit")

    if extra:
        cmd.append("--extra")
        cmd.extend(extra.split())

    return [
        LogInfo(msg=f"[parallel] cmd: {' '.join(cmd)}"),
        ExecuteProcess(
            cmd=cmd,
            output="screen",
            name="parallel_runner",
        ),
    ]
