#!/usr/bin/env python3
"""
SAM3 Launch - starts server and sets prompt globally
"""

import os

from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    ld = LaunchDescription()
    home = os.path.expanduser("~")

    # Arguments
    ld.add_action(DeclareLaunchArgument("prompt", default_value="object"))
    ld.add_action(DeclareLaunchArgument("camera_topic", default_value="/camera/color/image_raw"))
    ld.add_action(DeclareLaunchArgument("rate", default_value="5.0"))
    ld.add_action(DeclareLaunchArgument("confidence_threshold", default_value="0.3"))

    # 1. Start SAM3 server
    sam3_server = ExecuteProcess(
        cmd=['bash', '-c',
            f'source {home}/miniconda3/etc/profile.d/conda.sh && '
            f'conda activate sam3 && '
            f'python {home}/ament_ws/src/stretch_ros2/senior_project/senior_project/sam3_server.py'
        ],
        output='screen',
    )
    ld.add_action(sam3_server)

    # 2. Set the prompt on server (after it starts)
    set_prompt = ExecuteProcess(
        cmd=['bash', '-c',
            'sleep 15 && curl -X POST "http://localhost:8100/prompt/$(echo $PROMPT | sed \'s/ /%20/g\')"'
        ],
        output='screen',
        additional_env={'PROMPT': LaunchConfiguration("prompt")},
    )
    ld.add_action(set_prompt)

    # 3. Start ROS node (delayed)
    sam3_node = Node(
        package="senior_project",
        executable="sam3_ros_node",
        name="sam3_segmentation_node",
        output="screen",
        parameters=[{
            "camera_topic": LaunchConfiguration("camera_topic"),
            "rate": LaunchConfiguration("rate"),
            "confidence_threshold": LaunchConfiguration("confidence_threshold"),
            "server_url": "http://localhost:8100",
            "prompt": "",  # Empty = use server's prompt
            # Fast segmentation defaults (still SAM3 masks)
            "mask_mode": "combined",
            "mask_size": 256,
            "resize_width": 640,
            "jpeg_quality": 70,
            "enabled": True,
        }],
    )
    ld.add_action(TimerAction(period=20.0, actions=[sam3_node]))

    ld.add_action(LogInfo(msg="SAM3 starting... prompt will be set after server loads"))

    return ld
