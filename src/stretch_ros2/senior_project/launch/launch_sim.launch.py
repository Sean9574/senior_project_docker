#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from platform import system

import launch_ros
import launch_ros.parameter_descriptions
from ament_index_python import get_package_share_path  # still used for stretch_simulation RViz cfg
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription

# Optional: fix RViz/QT issues on Linux
if system() == "Linux":
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so"
    os.environ["GTK_PATH"] = ""

# Robocasa helpers (as in your code)
from stretch_mujoco.robocasa_gen import choose_layout, choose_style, get_styles, layouts


def generate_launch_description():
    ld = LaunchDescription()

    # --------- Arguments (kept) ---------
    ld.add_action(DeclareLaunchArgument("broadcast_odom_tf", default_value="True", choices=["True", "False"]))
    ld.add_action(DeclareLaunchArgument("fail_out_of_range_goal", default_value="False", choices=["True", "False"]))
    ld.add_action(DeclareLaunchArgument(
        "mode", default_value="position", choices=["position", "navigation", "trajectory", "gamepad"]
    ))
    ld.add_action(DeclareLaunchArgument("use_rviz", default_value="true", choices=["true", "false"]))
    ld.add_action(DeclareLaunchArgument("use_mujoco_viewer", default_value="true", choices=["true", "false"]))
    ld.add_action(DeclareLaunchArgument("use_cameras", default_value="false", choices=["true", "false"]))
    ld.add_action(DeclareLaunchArgument("use_robocasa", default_value="true", choices=["true", "false"]))
    ld.add_action(DeclareLaunchArgument("robocasa_task", default_value="PnPCounterToCab"))
    ld.add_action(DeclareLaunchArgument(
        "robocasa_layout", default_value="Random", choices=["Random"] + list(layouts.values()))
    )
    ld.add_action(DeclareLaunchArgument(
        "robocasa_style", default_value="Random", choices=["Random"] + list(get_styles().values()))
    )

    # --------- Optional interactive Robocasa selection (kept) ---------
    use_robocasa = "use_robocasa:=false" not in sys.argv
    robocasa_layout_val = None
    robocasa_style_val = None
    if use_robocasa and "--show-args" not in sys.argv:
        args_string = " ".join(sys.argv)
        if "robocasa_layout" not in args_string:
            print("\n\nYou have not specified a `robocasa_layout` argument, choose a layout:\n")
            robocasa_layout_val = layouts[choose_layout()]
            print(f"{robocasa_layout_val=}")
        if "robocasa_style" not in args_string:
            print("\n\nYou have not specified a `robocasa_style` argument, choose a style:\n")
            robocasa_style_val = get_styles()[choose_style()]
            print(f"{robocasa_style_val=}")

    # --------- HARD-POINT robot description to ament_ws source tree ---------
    # If your file name is different, update robot_description_file accordingly.
    robot_description_file = Path("/home/sean/ament_ws/src/stretch_ros2/stretch_description/urdf/stretch.urdf")

    if not robot_description_file.is_file():
        ld.add_action(LogInfo(msg=(
            f"\nERROR: Expected calibrated URDF not found at:\n  {robot_description_file}\n"
            "Please verify the filename under ament_ws/src/stretch_ros2/stretch_description/urdf\n"
            "and update 'robot_description_file' in this launch file."
        )))
        return ld

    with open(robot_description_file, "r") as f:
        robot_description_content = f.read()

    # --------- Joint State Publisher ---------
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        output="log",
        parameters=[
            {"source_list": ["/stretch/joint_states"]},
            {"rate": 30.0},
            {"robot_description": robot_description_content},
        ],
        arguments=["--ros-args", "--log-level", "error"],
    )
    ld.add_action(joint_state_publisher)

    # --------- Robot State Publisher ---------
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            {"robot_description": robot_description_content},
            {"publish_frequency": 30.0},
        ],
        arguments=["--ros-args", "--log-level", "error"],
    )
    ld.add_action(robot_state_publisher)

    # --------- MuJoCo simulator driver ---------
    stretch_driver_params = [{
        "rate": 30.0,
        "timeout": 0.5,
        "broadcast_odom_tf": LaunchConfiguration("broadcast_odom_tf"),
        "fail_out_of_range_goal": LaunchConfiguration("fail_out_of_range_goal"),
        "mode": LaunchConfiguration("mode"),
        "use_mujoco_viewer": LaunchConfiguration("use_mujoco_viewer"),
        "use_cameras": LaunchConfiguration("use_cameras"),
        "use_robocasa": LaunchConfiguration("use_robocasa"),
        "robocasa_task": LaunchConfiguration("robocasa_task"),
        "robocasa_layout": robocasa_layout_val if robocasa_layout_val is not None else LaunchConfiguration("robocasa_layout"),
        "robocasa_style":  robocasa_style_val if robocasa_style_val is not None else LaunchConfiguration("robocasa_style"),
    }]

    ld.add_action(Node(
        package="stretch_simulation",
        executable="stretch_mujoco_driver",
        emulate_tty=True,
        output="screen",
        remappings=[
            ("cmd_vel", "/stretch/cmd_vel"),
            ("joint_states", "/stretch/joint_states"),
        ],
        parameters=stretch_driver_params,
    ))

    return ld
