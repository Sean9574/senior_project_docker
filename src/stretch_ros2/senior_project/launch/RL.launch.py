#!/usr/bin/env python3
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    ld = LaunchDescription()

    # ---- Core arguments ----
    ld.add_action(DeclareLaunchArgument("ns", default_value="stretch"))
    ld.add_action(
        DeclareLaunchArgument("run_rl", default_value="true", choices=["true", "false"])
    )
    ld.add_action(
        DeclareLaunchArgument(
            "use_rviz", default_value="true", choices=["true", "false"]
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "use_mujoco_viewer",
            default_value="false",
            choices=["true", "false"],
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "use_cameras",
            default_value="true",
            choices=["true", "false"],
        )
    )
    
    # ---- NEW: Reward Monitor Argument ----
    ld.add_action(
        DeclareLaunchArgument(
            "use_reward_monitor",
            default_value="true",
            choices=["true", "false"],
            description="Launch real-time reward monitoring web interface"
        )
    )

    # ---- NEW: Hard Reset Arguments ----
    ld.add_action(
        DeclareLaunchArgument(
            "episode_num",
            default_value="1",
            description="Current episode number (set by hard_reset.sh)"
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "models_dir",
            default_value="./models",
            description="Models checkpoint directory (set by hard_reset.sh)"
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "use_hard_reset",
            default_value="false",
            choices=["true", "false"],
            description="Use hard reset (kill/restart) instead of normal training"
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "episode_length",
            default_value="180",
            description="Episode length in seconds (for hard reset mode)"
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "max_episodes",
            default_value="30.0",
            description="Maximum number of episodes (for hard reset mode)"
        )
    )

    # ---- RL-specific args (forwarded into learner_node.py) ----
    ld.add_action(DeclareLaunchArgument("total_steps", default_value="200000"))
    ld.add_action(DeclareLaunchArgument("rollout_steps", default_value="2048"))
    ld.add_action(
        DeclareLaunchArgument(
            "ckpt_dir", default_value=os.path.expanduser("~/rl_checkpoints")
        )
    )
    ld.add_action(DeclareLaunchArgument("load_ckpt", default_value=""))

    ld.add_action(DeclareLaunchArgument("odom_topic", default_value="odom"))
    ld.add_action(DeclareLaunchArgument("cmd_topic", default_value="cmd_vel"))
    ld.add_action(DeclareLaunchArgument("lidar_topic", default_value="scan"))
    ld.add_action(DeclareLaunchArgument("imu_topic", default_value="imu"))
    ld.add_action(
        DeclareLaunchArgument(
            "use_obstacle", default_value="0", choices=["0", "1"]
        )
    )
    # Dynamic goal topic (matches learner_node.py; with ns=stretch this is /stretch/goal)
    ld.add_action(DeclareLaunchArgument("goal_topic", default_value="goal"))

    # --- Evaluation args ---
    ld.add_action(
        DeclareLaunchArgument("eval_every_steps", default_value="3000")
    )
    ld.add_action(
        DeclareLaunchArgument("eval_episodes", default_value="10")
    )

    # --- Sim args ---
    ld.add_action(
        DeclareLaunchArgument(
            "broadcast_odom_tf",
            default_value="True",
            choices=["True", "False"],
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "fail_out_of_range_goal",
            default_value="False",
            choices=["False", "True"],
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "mode",
            default_value="navigation",
            choices=["position", "navigation", "trajectory", "gamepad"],
        )
    )

    # Robocasa args kept for compatibility; driver uses use_robocasa=False
    ld.add_action(
        DeclareLaunchArgument(
            "use_robocasa",
            default_value="true",
            choices=["true", "false"],
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "robocasa_task", default_value="PnPCounterToCab"
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "robocasa_layout",
            default_value="Random",
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "robocasa_style",
            default_value="Random",
        )
    )

    # --- Load URDF and rewrite mesh paths (FIXED) ---
    robot_description_file = Path(
        get_package_share_directory('stretch_description')
    ) / 'urdf' / 'stretch.urdf'

    mesh_root = get_package_share_directory('stretch_description')

    with open(robot_description_file, "r") as f:
        robot_description_content = f.read()

    robot_description_content = robot_description_content.replace(
        "package://stretch_description", f"file://{mesh_root}"
    )

    # Robot state publisher (namespaced; TF frames still global)
    ld.add_action(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            namespace=LaunchConfiguration("ns"),
            output="both",
            parameters=[
                {"robot_description": robot_description_content},
                {"publish_frequency": 30.0},
            ],
            arguments=["--ros-args", "--log-level", "error"],
        )
    )

    # Joint state publisher (namespaced)
    ld.add_action(
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            namespace=LaunchConfiguration("ns"),
            output="log",
            parameters=[
                {"source_list": ["/stretch/joint_states"]},
                {"rate": 30.0},
            ],
            arguments=["--ros-args", "--log-level", "error"],
        )
    )

    # RViz (launched normally) - FIXED: removed hardcoded config path
    ld.add_action(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=[],
            remappings=[
                ("/move_base_simple/goal", "/goal"),
                ("/goal_pose", "/goal")
            ],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    # MuJoCo driver
    driver_params = [
        {
            "rate": 300.0,  # 10x faster simulation
            "timeout": 0.5,
            "broadcast_odom_tf": LaunchConfiguration("broadcast_odom_tf"),
            "fail_out_of_range_goal": LaunchConfiguration(
                "fail_out_of_range_goal"
            ),
            "mode": LaunchConfiguration("mode"),
            "use_mujoco_viewer": LaunchConfiguration("use_mujoco_viewer"),
            "use_cameras": LaunchConfiguration("use_cameras"),

            # Robocasa off
            "use_robocasa": False,
            "robocasa_task": LaunchConfiguration("robocasa_task"),
            "robocasa_layout": LaunchConfiguration("robocasa_layout"),
            "robocasa_style": LaunchConfiguration("robocasa_style"),
        }
    ]
    ld.add_action(
        Node(
            package="stretch_simulation",
            executable="stretch_mujoco_driver",
            emulate_tty=True,
            output="screen",
            remappings=[
                ("cmd_vel", "/stretch/cmd_vel"),
                ("joint_states", "/stretch/joint_states"),
                ("/scan_filtered", "/stretch/scan"),
                ("odom", "/stretch/odom"),
            ],
            parameters=driver_params,
        )
    )

    # --- START THE PPO LEARNER ---
    learner_proc = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "senior_project.learner_node",
            "--ns",
            LaunchConfiguration("ns"),
            "--total-steps",
            LaunchConfiguration("total_steps"),
            "--rollout-steps",
            LaunchConfiguration("rollout_steps"),
            "--ckpt-dir",
            LaunchConfiguration("ckpt_dir"),
            "--load-ckpt",
            LaunchConfiguration("load_ckpt"),
            "--odom-topic",
            LaunchConfiguration("odom_topic"),
            "--cmd-topic",
            LaunchConfiguration("cmd_topic"),
            "--lidar-topic",
            LaunchConfiguration("lidar_topic"),
            "--imu-topic",
            LaunchConfiguration("imu_topic"),
            "--use-obstacle",
            LaunchConfiguration("use_obstacle"),
            "--goal-topic",
            LaunchConfiguration("goal_topic"),
            "--eval-every-steps",
            LaunchConfiguration("eval_every_steps"),
            "--eval-episodes",
            LaunchConfiguration("eval_episodes"),
            # NEW: Pass hard reset arguments
            "--episode-num",
            LaunchConfiguration("episode_num"),
            "--models-dir",
            LaunchConfiguration("models_dir"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_rl")),
    )

    # Delay slightly so sim is alive first
    ld.add_action(TimerAction(period=2.0, actions=[learner_proc]))
    
    # --- START REWARD MONITOR ---
    launch_file_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(launch_file_dir)
    reward_monitor_path = os.path.join(package_dir, 'scripts', 'reward_monitor.py')
    
    reward_monitor_proc = ExecuteProcess(
        cmd=['python3', reward_monitor_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration("use_reward_monitor")),
    )
    
    # Start reward monitor after learner starts
    ld.add_action(TimerAction(period=5.0, actions=[reward_monitor_proc]))

    # ---- NEW: HARD RESET MODE ----
    # Find the bash script (assuming it's in SENIOR_PROJECT/scripts/)
    bash_script_path = os.path.join(package_dir, 'scripts', 'hard_reset.sh')
    
    # Execute hard reset bash script instead of normal launch
    hard_reset_proc = ExecuteProcess(
        cmd=[
            'bash',
            bash_script_path,
            LaunchConfiguration("episode_length"),
            LaunchConfiguration("max_episodes")
        ],
        output='screen',
        shell=False,
        condition=IfCondition(LaunchConfiguration("use_hard_reset")),
    )
    
    ld.add_action(hard_reset_proc)

    ld.add_action(
        LogInfo(
            msg="[RL.launch] Sim + PPO learner + Reward Monitor"
        )
    )
    return ld