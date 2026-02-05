#!/usr/bin/env python3
"""
SAM3 Navigation - PARALLEL ONLY (ament_python friendly)

This launch file starts:
  1. A SINGLE shared SAM3 server in a dedicated domain (sam3_domain)
  2. Multiple parallel simulations, each in their own domain (sim_base_domain + sim_index)

Usage:
  ros2 launch senior_project sam3_navigation_parallel.launch.py
  ros2 launch senior_project sam3_navigation_parallel.launch.py num_sims:=4 target:=cup headless:=true
  ros2 launch senior_project sam3_navigation_parallel.launch.py sam3_domain:=0 sim_base_domain:=10
  ros2 launch senior_project sam3_navigation_parallel.launch.py extra:="use_mono_depth_server=true mono_depth_model=depth_anything"
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = LaunchDescription()

    # ─────────────────────────────────────────────────────────────────────────
    # Domain configuration
    # ─────────────────────────────────────────────────────────────────────────
    ld.add_action(DeclareLaunchArgument(
        "sam3_domain", default_value="0",
        description="ROS_DOMAIN_ID for the shared SAM3 server"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sim_base_domain", default_value="10",
        description="Base ROS_DOMAIN_ID for simulations (sim N uses base + N)"
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # SAM3 Server configuration
    # ─────────────────────────────────────────────────────────────────────────
    ld.add_action(DeclareLaunchArgument(
        "sam3_server_package", default_value="senior_project",
        description="Package containing the SAM3 server node"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sam3_server_executable", default_value="sam3_server",
        description="SAM3 server executable name"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sam3_startup_delay", default_value="5.0",
        description="Seconds to wait for SAM3 server before starting sims"
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation / training configuration
    # ─────────────────────────────────────────────────────────────────────────
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
    ld.add_action(LogInfo(msg="  SAM3 Navigation - PARALLEL ONLY (Shared SAM3 Server)"))
    ld.add_action(LogInfo(msg="============================================================"))
    ld.add_action(LogInfo(msg=""))

    # Start SAM3 server first, then sims after a delay
    ld.add_action(OpaqueFunction(function=_start_sam3_server))
    ld.add_action(OpaqueFunction(function=_start_parallel_runner_delayed))

    return ld


def _start_sam3_server(context):
    """Start the shared SAM3 server in its dedicated domain."""
    import os
    home = os.path.expanduser("~")

    sam3_domain = LaunchConfiguration("sam3_domain").perform(context)

    return [
        LogInfo(msg=f"[SAM3] Starting shared SAM3 server in domain {sam3_domain}"),
        ExecuteProcess(
            cmd=[
                "bash", "-c",
                f"python3 {home}/ament_ws/src/stretch_ros2/senior_project/senior_project/sam3_server.py"
            ],
            output="screen",
            name="sam3_server_shared",
            additional_env={"ROS_DOMAIN_ID": str(sam3_domain)},
        ),
    ]


def _start_parallel_runner_delayed(context):
    """Start the parallel runner after SAM3 server has initialized."""
    sam3_startup_delay = float(LaunchConfiguration("sam3_startup_delay").perform(context))
    sam3_domain = LaunchConfiguration("sam3_domain").perform(context)
    sim_base_domain = LaunchConfiguration("sim_base_domain").perform(context)

    num_sims = LaunchConfiguration("num_sims").perform(context)
    target = LaunchConfiguration("target").perform(context)
    total_steps = LaunchConfiguration("total_steps").perform(context)
    rollout_steps = LaunchConfiguration("rollout_steps").perform(context)
    ckpt_dir = os.path.expanduser(LaunchConfiguration("ckpt_dir").perform(context))
    stagger_delay = LaunchConfiguration("stagger_delay").perform(context)
    headless = LaunchConfiguration("headless").perform(context)
    merge_on_exit = LaunchConfiguration("merge_on_exit").perform(context).strip().lower()
    extra = LaunchConfiguration("extra").perform(context).strip()

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
        # Domain configuration for the runner
        "--sam3_domain", str(sam3_domain),
        "--sim_base_domain", str(sim_base_domain),
    ]

    if merge_on_exit in ("true", "1", "yes", "on"):
        cmd.append("--merge_on_exit")

    if extra:
        cmd.append("--extra")
        cmd.extend(extra.split())

    return [
        TimerAction(
            period=sam3_startup_delay,
            actions=[
                LogInfo(msg=f"[parallel] SAM3 domain: {sam3_domain}, Sim base domain: {sim_base_domain}"),
                LogInfo(msg=f"[parallel] cmd: {' '.join(cmd)}"),
                ExecuteProcess(
                    cmd=cmd,
                    output="screen",
                    name="parallel_runner",
                ),
            ],
        ),
    ]