#!/usr/bin/env python3
"""
SAM3 Navigation - PARALLEL with DEDICATED SAM3 Servers

This launch file starts:
  1. Multiple SAM3 servers (one per sim) on ports 8100, 8101, 8102, etc.
  2. Multiple parallel simulations, each in their own domain with dedicated SAM3 server

Usage:
  ros2 launch senior_project sam3_navigation_parallel.launch.py
  ros2 launch senior_project sam3_navigation_parallel.launch.py num_sims:=4 target:=cup headless:=true
  ros2 launch senior_project sam3_navigation_parallel.launch.py sam3_mode:=shared  # Fall back to single shared server
"""

import os

from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration

from launch import LaunchDescription


def generate_launch_description():
    ld = LaunchDescription()

    # ─────────────────────────────────────────────────────────────────────────
    # Domain configuration
    # ─────────────────────────────────────────────────────────────────────────
    ld.add_action(DeclareLaunchArgument(
        "sam3_domain", default_value="0",
        description="ROS_DOMAIN_ID for SAM3 servers (only used if sam3_mode=shared)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sim_base_domain", default_value="10",
        description="Base ROS_DOMAIN_ID for simulations (sim N uses base + N)"
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # SAM3 Server configuration
    # ─────────────────────────────────────────────────────────────────────────
    ld.add_action(DeclareLaunchArgument(
        "sam3_mode", default_value="dedicated",
        description="SAM3 server mode: 'dedicated' (one per sim) or 'shared' (single server)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sam3_base_port", default_value="8100",
        description="Base port for SAM3 servers (sim N uses base_port + N)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sam3_server_package", default_value="senior_project",
        description="Package containing the SAM3 server node"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sam3_server_executable", default_value="sam3_server",
        description="SAM3 server executable name"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sam3_startup_delay", default_value="15.0",
        description="Seconds to wait for SAM3 servers before starting sims"
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation / training configuration
    # ─────────────────────────────────────────────────────────────────────────
    ld.add_action(DeclareLaunchArgument(
        "num_sims", default_value="1",
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
        default_value=os.path.expanduser("~/ament_ws/src/stretch_ros2/senior_project/parallel_training"),
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
    ld.add_action(LogInfo(msg="  SAM3 Navigation - PARALLEL with Dedicated SAM3 Servers"))
    ld.add_action(LogInfo(msg="============================================================"))
    ld.add_action(LogInfo(msg=""))

    # Start SAM3 server(s) first, then sims after a delay
    ld.add_action(OpaqueFunction(function=_start_sam3_servers))
    ld.add_action(OpaqueFunction(function=_start_parallel_runner_delayed))

    return ld


def _start_sam3_servers(context):
    """Start SAM3 server(s) - either dedicated (one per sim) or shared (single)."""
    sam3_mode = LaunchConfiguration("sam3_mode").perform(context)
    sam3_base_port = int(LaunchConfiguration("sam3_base_port").perform(context))
    num_sims = int(LaunchConfiguration("num_sims").perform(context))
    sam3_domain = LaunchConfiguration("sam3_domain").perform(context)
    home = os.path.expanduser("~")

    actions = []

    if sam3_mode == "dedicated":
        # Start one SAM3 server per sim
        actions.append(LogInfo(msg=f"[SAM3] Starting {num_sims} DEDICATED SAM3 servers (ports {sam3_base_port}-{sam3_base_port + num_sims - 1})"))
        
        for sim_id in range(num_sims):
            port = sam3_base_port + sim_id
            cmd_string = (
                f"source /opt/ros/humble/setup.bash && "
                f"source {home}/ament_ws/install/setup.bash && "
                f"python3 -m senior_project.sam3_server --port {port}"
            )
            
            actions.append(LogInfo(msg=f"[SAM3] Server {sim_id}: port {port}"))
            actions.append(ExecuteProcess(
                cmd=["bash", "-c", cmd_string],
                output="screen",
                name=f"sam3_server_{sim_id}",
                additional_env={"ROS_DOMAIN_ID": str(sam3_domain)},
            ))
    else:
        # Shared mode - single server
        actions.append(LogInfo(msg=f"[SAM3] Starting SHARED SAM3 server on port {sam3_base_port}"))
        cmd_string = (
            f"source /opt/ros/humble/setup.bash && "
            f"source {home}/ament_ws/install/setup.bash && "
            f"python3 -m senior_project.sam3_server --port {sam3_base_port}"
        )
        actions.append(ExecuteProcess(
            cmd=["bash", "-c", cmd_string],
            output="screen",
            name="sam3_server_shared",
            additional_env={"ROS_DOMAIN_ID": str(sam3_domain)},
        ))

    return actions


def _start_parallel_runner_delayed(context):
    """Start the parallel runner after SAM3 servers have initialized."""
    sam3_startup_delay = float(LaunchConfiguration("sam3_startup_delay").perform(context))
    sam3_domain = LaunchConfiguration("sam3_domain").perform(context)
    sim_base_domain = LaunchConfiguration("sim_base_domain").perform(context)
    sam3_mode = LaunchConfiguration("sam3_mode").perform(context)
    sam3_base_port = LaunchConfiguration("sam3_base_port").perform(context)

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
        # SAM3 server configuration
        "--sam3_mode", str(sam3_mode),
        "--sam3_base_port", str(sam3_base_port),
    ]

    if merge_on_exit in ("true", "1", "yes", "on"):
        cmd.append("--merge_on_exit")

    if extra:
        cmd.extend(["--extra", extra])

    return [
        LogInfo(msg=f"[parallel] SAM3 mode: {sam3_mode}, base port: {sam3_base_port}"),
        LogInfo(msg=f"[parallel] Sim base domain: {sim_base_domain}"),
        LogInfo(msg=f"[parallel] cmd: {' '.join(cmd)}"),
        TimerAction(
            period=sam3_startup_delay,
            actions=[
                ExecuteProcess(
                    cmd=cmd,
                    output="screen",
                    name="parallel_runner",
                ),
            ],
        ),
    ]