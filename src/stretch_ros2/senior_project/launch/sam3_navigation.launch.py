#!/usr/bin/env python3
"""
SAM3 Navigation Launch File
Starts everything needed for SAM3-based object navigation:
  1. SAM3 server (conda environment)
  2. SAM3 goal generator node
  3. RL simulation + learner

Usage:
    ros2 launch senior_project sam3_navigation.launch.py target:="cup"
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
from launch.substitutions import LaunchConfiguration
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
    
    # ===================== RL SIMULATION =====================
    
    # Include the RL launch file
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
    
    # ===================== SAM3 GOAL GENERATOR =====================
    
    sam3_goal_node = Node(
        package='senior_project',
        executable='sam3_goal_generator',
        name='sam3_goal_generator',
        output='screen',
        parameters=[{
            'server_url': 'http://localhost:8100',
            'target': LaunchConfiguration('target'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'goal_offset': LaunchConfiguration('goal_offset'),
            'auto_publish_goal': LaunchConfiguration('auto_publish_goal'),
            'ns': LaunchConfiguration('ns'),
            'rate': 2.0,
            'min_distance': 0.5,
            'max_distance': 5.0,
        }],
    )
    
    # Delay goal generator to let everything else start
    ld.add_action(TimerAction(period=25.0, actions=[sam3_goal_node]))
    
    # ===================== INFO =====================
    
    ld.add_action(LogInfo(msg=""))
    ld.add_action(LogInfo(msg="========================================"))
    ld.add_action(LogInfo(msg="  SAM3 Navigation System Starting..."))
    ld.add_action(LogInfo(msg="========================================"))
    ld.add_action(LogInfo(msg=""))
    ld.add_action(LogInfo(msg="  [0s]  SAM3 server starting"))
    ld.add_action(LogInfo(msg="  [5s]  RL simulation starting"))
    ld.add_action(LogInfo(msg="  [15s] SAM3 prompt being set"))
    ld.add_action(LogInfo(msg="  [25s] Goal generator starting"))
    ld.add_action(LogInfo(msg=""))
    ld.add_action(LogInfo(msg="  Change target: ros2 topic pub --once \\"))
    ld.add_action(LogInfo(msg="    /sam3_goal_generator/set_target std_msgs/String \"data: 'chair'\""))
    ld.add_action(LogInfo(msg=""))
    
    return ld