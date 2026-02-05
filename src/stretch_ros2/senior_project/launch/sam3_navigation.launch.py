#!/usr/bin/env python3
"""
SAM3 Navigation Launch File

Starts everything needed for SAM3-based object navigation with multiple
depth estimation methods:
  1. SAM3 server (object detection/segmentation) - OPTIONAL, can use shared server
  2. Mono depth server (optional, for RGB-only depth estimation)
  3. SAM3 goal generator  (supports depth camera, mono depth, or bbox heuristic)
  4. RL simulation + learner
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    ld = LaunchDescription()
    pkg_dir = get_package_share_directory("senior_project")

    # ===================== ARGUMENTS =====================

    ld.add_action(DeclareLaunchArgument(
        "sim_id", default_value="0",
        description="Simulation ID (used for port offsets in parallel mode)"
    ))

    # SAM3 server configuration
    ld.add_action(DeclareLaunchArgument(
        "start_sam3_server", default_value="true",
        description="Whether to start SAM3 server (false when using shared server)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sam3_domain", default_value="0",
        description="(Legacy) ROS_DOMAIN_ID where SAM3 server is running (HTTP server is domain-agnostic)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sam3_server_host", default_value="localhost",
        description="SAM3 server hostname"
    ))
    ld.add_action(DeclareLaunchArgument(
        "sam3_server_port", default_value="8100",
        description="SAM3 server port"
    ))

    # SAM3 args
    ld.add_action(DeclareLaunchArgument(
        "target", default_value="person",
        description="Object to find and navigate to"
    ))
    ld.add_action(DeclareLaunchArgument(
        "confidence_threshold", default_value="0.2",
        description="SAM3 detection confidence threshold"
    ))
    ld.add_action(DeclareLaunchArgument(
        "goal_offset", default_value="0.5",
        description="Distance to stop from object (meters)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "auto_publish_goal", default_value="true",
        description="Automatically publish goal when object found"
    ))

    # Depth estimation args
    ld.add_action(DeclareLaunchArgument(
        "depth_mode", default_value="auto",
        description="Depth estimation mode: auto, depth, mono, bbox"
    ))
    ld.add_action(DeclareLaunchArgument(
        "use_mono_depth_server", default_value="false",
        description="Start monocular depth estimation server"
    ))
    ld.add_action(DeclareLaunchArgument(
        "mono_depth_model", default_value="auto",
        description="Mono depth model: auto, depth_anything, midas, simple"
    ))
    ld.add_action(DeclareLaunchArgument(
        "start_mono_depth_server", default_value="true",
        description="Whether to start mono depth server (false when using shared server)"
    ))
    ld.add_action(DeclareLaunchArgument(
        "mono_depth_port", default_value="8101",
        description="Mono depth server port"
    ))

    # RL args
    ld.add_action(DeclareLaunchArgument("ns", default_value="stretch"))
    ld.add_action(DeclareLaunchArgument("run_rl", default_value="true"))
    ld.add_action(DeclareLaunchArgument("use_rviz", default_value="true"))
    ld.add_action(DeclareLaunchArgument("use_mujoco_viewer", default_value="false"))
    ld.add_action(DeclareLaunchArgument("use_cameras", default_value="true"))
    ld.add_action(DeclareLaunchArgument("use_reward_monitor", default_value="false"))
    ld.add_action(DeclareLaunchArgument("total_steps", default_value="200000"))
    ld.add_action(DeclareLaunchArgument("rollout_steps", default_value="2048"))
    ld.add_action(DeclareLaunchArgument(
        "ckpt_dir", default_value=os.path.expanduser("~/rl_checkpoints")
    ))
    ld.add_action(DeclareLaunchArgument("load_ckpt", default_value=""))
    ld.add_action(DeclareLaunchArgument("headless", default_value="false"))

    # Camera topics
    ld.add_action(DeclareLaunchArgument(
        "rgb_topic", default_value="/camera/color/image_raw"
    ))
    ld.add_action(DeclareLaunchArgument(
        "depth_topic", default_value="/camera/depth/image_rect_raw"
    ))
    ld.add_action(DeclareLaunchArgument(
        "camera_info_topic", default_value="/camera/color/camera_info"
    ))

    # Configure at runtime
    ld.add_action(OpaqueFunction(function=_configure_launch))

    return ld


def _configure_launch(context):
    pkg_dir = get_package_share_directory("senior_project")

    # Runtime values
    sim_id = LaunchConfiguration("sim_id").perform(context)
    start_sam3 = LaunchConfiguration("start_sam3_server").perform(context).lower() in ("true", "1", "yes")
    sam3_host = LaunchConfiguration("sam3_server_host").perform(context)
    sam3_port = LaunchConfiguration("sam3_server_port").perform(context)

    start_mono = LaunchConfiguration("start_mono_depth_server").perform(context).lower() in ("true", "1", "yes")
    use_mono = LaunchConfiguration("use_mono_depth_server").perform(context).lower() in ("true", "1", "yes")
    mono_port = LaunchConfiguration("mono_depth_port").perform(context)

    actions = []

    # URLs
    sam3_url = f"http://{sam3_host}:{sam3_port}"
    mono_url = f"http://{sam3_host}:{mono_port}"

    # ===================== INFO =====================

    actions.append(LogInfo(msg=""))
    actions.append(LogInfo(msg="============================================================"))
    actions.append(LogInfo(msg=f"  SAM3 Navigation - Sim {sim_id}"))
    actions.append(LogInfo(msg="============================================================"))
    actions.append(LogInfo(msg=""))
    actions.append(LogInfo(msg="  Depth Modes:"))
    actions.append(LogInfo(msg="    auto  - depth camera -> mono depth -> bbox size"))
    actions.append(LogInfo(msg="    depth - depth camera only"))
    actions.append(LogInfo(msg="    mono  - monocular depth estimation"))
    actions.append(LogInfo(msg="    bbox  - bounding box size heuristic"))
    actions.append(LogInfo(msg=""))

    # ===================== SAM3 SERVER (NO CONDA) =====================

    if start_sam3:
        actions.append(LogInfo(msg=f"  [0s]  SAM3 server starting on port {sam3_port}"))

        # IMPORTANT:
        # - No conda
        # - No exporting tokens here (we want to inherit env from docker --env-file)
        # - Pass the port explicitly
        
        sam3_server = ExecuteProcess(
        cmd=['bash', '-lc',
            # make sure token env vars exist (your --env-file already does this)
            'python3 -m senior_project.sam3_server --port ' + sam3_port
        ],
        output='screen',
        name='sam3_server',
    )

        actions.append(sam3_server)

        # Set SAM3 prompt after server starts
        set_prompt = ExecuteProcess(
            cmd=[
                "bash", "-lc",
                f"sleep 15 && curl -sS -X POST \"{sam3_url}/prompt/$(echo \\\"$TARGET\\\" | sed 's/ /%20/g')\""
            ],
            output="screen",
            additional_env={"TARGET": LaunchConfiguration("target")},
        )
        actions.append(LogInfo(msg="  [15s] SAM3 prompt being set"))
        actions.append(set_prompt)

    else:
        actions.append(LogInfo(msg=f"  [---] Using shared SAM3 server at {sam3_url}"))

        # Still set prompt on shared server
        set_prompt = ExecuteProcess(
            cmd=[
                "bash", "-lc",
                f"sleep 5 && curl -sS -X POST \"{sam3_url}/prompt/$(echo \\\"$TARGET\\\" | sed 's/ /%20/g')\""
            ],
            output="screen",
            additional_env={"TARGET": LaunchConfiguration("target")},
        )
        actions.append(set_prompt)

    # ===================== MONO DEPTH SERVER (NO CONDA) =====================

    if use_mono and start_mono:
        actions.append(LogInfo(msg=f"  [0s]  Mono depth server starting on port {mono_port}"))

        mono_depth_server = ExecuteProcess(
            cmd=[
                "bash", "-lc",
                f"python3 /home/stretch/ament_ws/src/stretch_ros2/senior_project/senior_project/mono_depth_server.py "
                f"--model \"$MONO_MODEL\" --port {mono_port}"
            ],
            output="screen",
            name="mono_depth_server",
            additional_env={"MONO_MODEL": LaunchConfiguration("mono_depth_model")},
        )
        actions.append(mono_depth_server)

    elif use_mono:
        actions.append(LogInfo(msg=f"  [---] Using shared mono depth server at {mono_url}"))

    # ===================== RL SIMULATION =====================

    actions.append(LogInfo(msg="  [5s]  RL simulation starting"))

    rl_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, "launch", "RL.launch.py")
        ),
        launch_arguments={
            "ns": LaunchConfiguration("ns"),
            "run_rl": LaunchConfiguration("run_rl"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "use_mujoco_viewer": LaunchConfiguration("use_mujoco_viewer"),
            "use_cameras": LaunchConfiguration("use_cameras"),
            "use_reward_monitor": LaunchConfiguration("use_reward_monitor"),
            "total_steps": LaunchConfiguration("total_steps"),
            "rollout_steps": LaunchConfiguration("rollout_steps"),
            "ckpt_dir": LaunchConfiguration("ckpt_dir"),
            "load_ckpt": LaunchConfiguration("load_ckpt"),
            "headless": LaunchConfiguration("headless"),
        }.items(),
    )
    actions.append(TimerAction(period=5.0, actions=[rl_launch]))

    # ===================== SAM3 GOAL GENERATOR =====================

    actions.append(LogInfo(msg="  [25s] Goal generator starting"))

    sam3_goal_node = Node(
        package="senior_project",
        executable="sam3_goal_generator",
        name="sam3_goal_generator",
        output="screen",
        parameters=[{
            "server_url": sam3_url,
            "mono_depth_url": mono_url,
            "target": LaunchConfiguration("target"),
            "confidence_threshold": LaunchConfiguration("confidence_threshold"),
            "goal_offset": LaunchConfiguration("goal_offset"),
            "auto_publish_goal": LaunchConfiguration("auto_publish_goal"),
            "depth_mode": LaunchConfiguration("depth_mode"),
            "ns": LaunchConfiguration("ns"),
            "rate": 2.0,
            "min_distance": 0.5,
            "max_distance": 5.0,
            "rgb_topic": LaunchConfiguration("rgb_topic"),
            "depth_topic": LaunchConfiguration("depth_topic"),
            "camera_info_topic": LaunchConfiguration("camera_info_topic"),
        }],
    )
    actions.append(TimerAction(period=25.0, actions=[sam3_goal_node]))

    # ===================== HELP =====================

    actions.append(LogInfo(msg="  Change target: ros2 topic pub --once \\"))
    actions.append(LogInfo(msg="    /sam3_goal_generator/set_target std_msgs/String \"data: 'chair'\""))
    actions.append(LogInfo(msg=""))

    return actions
