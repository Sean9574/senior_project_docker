#!/usr/bin/env python3
"""
SAM3 Navigation Launch File

Starts everything needed for SAM3-based object navigation with multiple
depth estimation methods:
  1. SAM3 server (object detection/segmentation)
  2. Mono depth server (optional, for RGB-only depth estimation)
  3. SAM3 goal generator  (supports depth camera, mono depth, or bbox heuristic)
  4. RL simulation + learner

Usage:
    # With depth camera (default)
    ros2 launch senior_project sam3_navigation.launch.py target:="cup"
    
    # RGB-only with monocular depth estimation
    ros2 launch senior_project sam3_navigation.launch.py \
        target:="cup" depth_mode:="mono" use_mono_depth_server:="true"
    
    # RGB-only with bounding box size heuristic (no ML)
    ros2 launch senior_project sam3_navigation.launch.py \
        target:="cup" depth_mode:="bbox"

Depth Modes:
    auto  - Try depth camera first, fall back to mono depth, then bbox size
    depth - Depth camera only (fails if no depth camera)
    mono  - Monocular depth estimation only
    bbox  - Bounding box size heuristic only (no depth camera needed, no ML)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    ld = LaunchDescription()
    home = os.path.expanduser("~")
    pkg_dir = get_package_share_directory('senior_project')
    
    # ===================== ARGUMENTS =====================
    
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
    
    # RL args
    ld.add_action(DeclareLaunchArgument("ns", default_value="stretch"))
    ld.add_action(DeclareLaunchArgument("run_rl", default_value="true"))
    ld.add_action(DeclareLaunchArgument("use_rviz", default_value="true"))
    ld.add_action(DeclareLaunchArgument("use_mujoco_viewer", default_value="false"))
    ld.add_action(DeclareLaunchArgument("use_cameras", default_value="true"))
    ld.add_action(DeclareLaunchArgument("use_reward_monitor", default_value="true"))
    ld.add_action(DeclareLaunchArgument("total_steps", default_value="200000"))
    ld.add_action(DeclareLaunchArgument("rollout_steps", default_value="2048"))
    ld.add_action(DeclareLaunchArgument(
        "ckpt_dir", default_value=os.path.expanduser("~/rl_checkpoints")
    ))
    ld.add_action(DeclareLaunchArgument("load_ckpt", default_value=""))
    
    # Camera topics (for non-standard setups)
    ld.add_action(DeclareLaunchArgument(
        "rgb_topic", default_value="/camera/color/image_raw"
    ))
    ld.add_action(DeclareLaunchArgument(
        "depth_topic", default_value="/camera/depth/image_rect_raw"
    ))
    ld.add_action(DeclareLaunchArgument(
        "camera_info_topic", default_value="/camera/color/camera_info"
    ))
    
    # ===================== SAM3 SERVER =====================
    
    sam3_server = ExecuteProcess(
        cmd=['bash', '-c',
            f'source {home}/miniconda3/etc/profile.d/conda.sh && '
            f'conda activate sam3 && '
            f'python {home}/ament_ws/src/stretch_ros2/senior_project/senior_project/sam3_server.py'
        ],
        output='screen',
        name='sam3_server',
    )
    ld.add_action(sam3_server)
    
    # Set SAM3 prompt after server starts
    set_prompt = ExecuteProcess(
        cmd=['bash', '-c',
            'sleep 15 && curl -X POST "http://localhost:8100/prompt/$(echo $TARGET | sed \'s/ /%20/g\')"'
        ],
        output='screen',
        additional_env={'TARGET': LaunchConfiguration("target")},
    )
    ld.add_action(set_prompt)
    
    # ===================== MONO DEPTH SERVER (OPTIONAL) =====================
    
    mono_depth_server = ExecuteProcess(
        cmd=['bash', '-c',
            f'source {home}/miniconda3/etc/profile.d/conda.sh && '
            f'conda activate depth && '  # Assumes a 'depth' conda env, adjust as needed
            f'python {home}/ament_ws/src/stretch_ros2/senior_project/senior_project/mono_depth_server.py '
            f'--model $MONO_MODEL'
        ],
        output='screen',
        name='mono_depth_server',
        additional_env={'MONO_MODEL': LaunchConfiguration("mono_depth_model")},
        condition=IfCondition(LaunchConfiguration("use_mono_depth_server")),
    )
    ld.add_action(mono_depth_server)
    
    # ===================== RL SIMULATION =====================
    
    rl_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'RL.launch.py')
        ),
        launch_arguments={
            'ns': LaunchConfiguration('ns'),
            'run_rl': LaunchConfiguration('run_rl'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'use_mujoco_viewer': LaunchConfiguration('use_mujoco_viewer'),
            'use_cameras': LaunchConfiguration('use_cameras'),
            'use_reward_monitor': LaunchConfiguration('use_reward_monitor'),
            'total_steps': LaunchConfiguration('total_steps'),
            'rollout_steps': LaunchConfiguration('rollout_steps'),
            'ckpt_dir': LaunchConfiguration('ckpt_dir'),
            'load_ckpt': LaunchConfiguration('load_ckpt'),
        }.items(),
    )
    
    # Delay RL launch to let SAM3 server start
    ld.add_action(TimerAction(period=5.0, actions=[rl_launch]))
    
    # ===================== SAM3 GOAL GENERATOR V2 =====================
    
    sam3_goal_node = Node(
        package='senior_project',
        executable='sam3_goal_generator',  # New V2 node
        name='sam3_goal_generator',
        output='screen',
        parameters=[{
            'server_url': 'http://localhost:8100',
            'mono_depth_url': 'http://localhost:8101',
            'target': LaunchConfiguration('target'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'goal_offset': LaunchConfiguration('goal_offset'),
            'auto_publish_goal': LaunchConfiguration('auto_publish_goal'),
            'depth_mode': LaunchConfiguration('depth_mode'),
            'ns': LaunchConfiguration('ns'),
            'rate': 2.0,
            'min_distance': 0.5,
            'max_distance': 5.0,
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
        }],
    )
    
    # Delay goal generator to let everything else start
    ld.add_action(TimerAction(period=25.0, actions=[sam3_goal_node]))
    
    # ===================== INFO =====================
    
    ld.add_action(LogInfo(msg=""))
    ld.add_action(LogInfo(msg="============================================================"))
    ld.add_action(LogInfo(msg="  SAM3 Navigation - Multi-Depth Mode System"))
    ld.add_action(LogInfo(msg="============================================================"))
    ld.add_action(LogInfo(msg=""))
    ld.add_action(LogInfo(msg="  [0s]  SAM3 server starting"))
    ld.add_action(LogInfo(msg="  [0s]  Mono depth server starting (if enabled)"))
    ld.add_action(LogInfo(msg="  [5s]  RL simulation starting"))
    ld.add_action(LogInfo(msg="  [15s] SAM3 prompt being set"))
    ld.add_action(LogInfo(msg="  [25s] Goal generator starting"))
    ld.add_action(LogInfo(msg=""))
    ld.add_action(LogInfo(msg="  Depth Modes:"))
    ld.add_action(LogInfo(msg="    auto  - depth camera -> mono depth -> bbox size"))
    ld.add_action(LogInfo(msg="    depth - depth camera only"))
    ld.add_action(LogInfo(msg="    mono  - monocular depth estimation"))
    ld.add_action(LogInfo(msg="    bbox  - bounding box size heuristic"))
    ld.add_action(LogInfo(msg=""))
    ld.add_action(LogInfo(msg="  Change target: ros2 topic pub --once \\"))
    ld.add_action(LogInfo(msg="    /sam3_goal_generator/set_target std_msgs/String \"data: 'chair'\""))
    ld.add_action(LogInfo(msg=""))
    
    return ld
