#!/usr/bin/env python3
"""
Grid arena: N Stretch robots in N walled rooms inside ONE MuJoCo simulator,
trained by ONE learner with one replay buffer.

Asks before starting (skipped when given on the command line):
  What should the robot go to?   -> target:=chair
  How many robots? [8]           -> num_robots:=8

Starts: SAM3 server, arena_sim (stretch_simulation), one SAM3 goal generator
per robot (namespace r{i}), the learner in --num-robots mode, the reward
dashboard, and the terminal status panel.

  ros2 launch senior_project grid_navigation.launch.py
  ros2 launch senior_project grid_navigation.launch.py target:=chair num_robots:=4
  ros2 launch senior_project grid_navigation.launch.py target:=chair eval_episodes:=50

The single-robot launch files (sam3_navigation, RL, launch_sim) are unchanged.
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from senior_project.prompt import ask_count, ask_target


def generate_launch_description():
    # Ask before anything starts (target:= / num_robots:= skip the questions)
    target = ask_target(sys.argv, sys.stdin.isatty())
    num_robots = ask_count(sys.argv, sys.stdin.isatty())

    args = [
        ("target", target, "Object every robot should go to"),
        ("num_robots", str(num_robots), "Robots (rooms) in the arena"),
        ("total_steps", "1000000", "Training transitions, summed over all robots"),
        ("ckpt_dir", os.path.expanduser("~/rl_checkpoints/grid"), "Kept apart from single-robot runs"),
        ("load_ckpt", "", "Checkpoint to start from instead of <ckpt_dir>"),
        ("eval_episodes", "0", "K > 0: evaluate the saved policy for K episodes in total, then stop"),
        ("seed", "42", ""),
        ("random_spawn", "true", "Random free start pose for every robot, every episode"),
        ("room_xml", "", "Room layout (empty = senior_project/config/scene.xml)"),
        ("status_panel", "true", "Live status panel in this terminal; node output goes to log files"),
        ("use_reward_monitor", "true", "Web dashboard on 127.0.0.1:5555"),
        ("start_sam3_server", "true", "false = use a SAM3 server that is already running"),
        ("sam3_server_port", "8100", ""),
        ("confidence_threshold", "0.2", ""),
        ("goal_offset", "0.5", ""),
        ("depth_mode", "auto", ""),
    ]
    ld = LaunchDescription([DeclareLaunchArgument(k, default_value=v, description=d) for k, v, d in args])
    ld.add_action(OpaqueFunction(function=_setup))
    return ld


def _setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    n = int(arg("num_robots"))
    panel = arg("status_panel") == "true"
    # dict form: plain "log" still sends stderr (where ROS logs go) to the screen
    out = {"both": "log"} if panel else "screen"
    # One shared SAM3 server: scale each robot's detection rate so the total
    # stays near 8 requests/s (2 Hz each up to 4 robots, 1 Hz at 8)
    rate = 2.0 * min(1.0, 4.0 / n)
    sam3_url = f"http://localhost:{arg('sam3_server_port')}"
    ckpt_dir = os.path.expanduser(arg("ckpt_dir"))
    actions = []

    if arg("start_sam3_server") == "true":
        actions.append(ExecuteProcess(
            cmd=["python3", "-m", "senior_project.sam3_server", "--port", arg("sam3_server_port")],
            output=out, name="sam3_server"))

    actions.append(Node(
        package="stretch_simulation", executable="arena_sim", name="arena_sim", output=out,
        parameters=[{
            "num_robots": n,
            "camera_hz": rate,
            "use_cameras": True,
            "random_spawn": arg("random_spawn") == "true",
            "seed": int(arg("seed")),
            "room_xml": arg("room_xml"),
        }]))

    generators = []
    for i in range(n):
        generators.append(Node(
            package="senior_project", executable="sam3_goal_generator", name="sam3_goal_generator",
            namespace=f"r{i}", output=out,
            parameters=[{
                "server_url": sam3_url,
                "target": arg("target"),
                "ns": f"r{i}",
                "rate": rate,
                "confidence_threshold": float(arg("confidence_threshold")),
                "goal_offset": float(arg("goal_offset")),
                "auto_publish_goal": True,
                "depth_mode": arg("depth_mode"),
                "min_distance": 0.5,
                "max_distance": 5.0,
                "rgb_topic": f"/r{i}/camera/color/image_raw",
                "depth_topic": f"/r{i}/camera/depth/image_rect_raw",
                "camera_info_topic": f"/r{i}/camera/color/camera_info",
                # arena_sim publishes r{i}/odom -> r{i}/camera_color_optical_frame
                "use_tf": True,
                "odom_frame": f"r{i}/odom",
                "camera_frame": f"r{i}/camera_color_optical_frame",
                "show_gui": False,
            }]))
    actions.append(TimerAction(period=8.0, actions=generators))

    learner = ExecuteProcess(
        cmd=["python3", "-m", "senior_project.learner_node",
             "--num-robots", str(n),
             "--total-steps", arg("total_steps"),
             "--ckpt-dir", ckpt_dir,
             "--load-ckpt", arg("load_ckpt"),
             "--seed", arg("seed"),
             "--eval-episodes", arg("eval_episodes")],
        output=out, name="learner")
    actions.append(TimerAction(period=5.0, actions=[learner]))

    if arg("use_reward_monitor") == "true":
        monitor = os.path.join(get_package_share_directory("senior_project"), "scripts", "reward_monitor.py")
        actions.append(TimerAction(period=8.0, actions=[ExecuteProcess(
            cmd=["python3", monitor, "--experiment", os.path.join(ckpt_dir, "monitor"),
                 "--namespaces", *[f"r{i}" for i in range(n)]],
            output=out, name="reward_monitor")]))

    if panel:
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=learner,
            on_exit=[Shutdown(reason="learner exited -- full node logs are in $ROS_LOG_DIR")],
        )))
        # Redirected output (docker logs, nohup): summary lines, no redraws, and a
        # pipe instead of a pty (a pty turns every "\n" into "\r\n" -> ^M in the log)
        tty = sys.stdout.isatty()
        actions.append(ExecuteProcess(
            cmd=["python3", "-m", "senior_project.status_panel",
                 "--num-robots", str(n), "--total-steps", arg("total_steps"), "--target", arg("target"),
                 *([] if tty else ["--plain"])],
            output="screen", emulate_tty=tty, output_format="{line}", name="status_panel"))
    return actions
