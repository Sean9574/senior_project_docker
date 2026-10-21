#!/usr/bin/env python3
"""
Expert Baseline Node — Standalone State Machine Agent

Runs the same ExpertStateMachine from learner_node.py as a standalone agent
in its own simulation instance. Uses the identical environment, reward function,
and episode logic so results are directly comparable to the RL agent.

Publishes the same /reward_breakdown format with training_phase="EXPERT_BASELINE"
so the reward_monitor picks it up alongside the RL sims automatically.

This gives you a fair, continuous baseline: the state machine runs on the same
hardware, same sim, same reward, same clock — the only difference is the controller.

Usage:
    # Run in its own ROS domain alongside RL sims
    # If RL sims use domains 10-13, put the expert on domain 20
    ROS_DOMAIN_ID=20 python expert_baseline_node.py

    # With custom topics (must match your launch file)
    ROS_DOMAIN_ID=20 python expert_baseline_node.py \\
        --ns stretch \\
        --total-episodes 0 \\  # 0 = run forever
        --odom-topic /stretch/odom \\
        --lidar-topic /stretch/scan

    # Integration with parallel_runner.py:
    # Add domain 20 to your reward_monitor scan range:
    #   python reward_monitor.py --scan-start 0 --scan-end 25 \\
    #       --experiment ~/experiments/comparison_run

Launch alongside RL training:
    # Terminal 1: Start the expert sim (same MuJoCo scene)
    ROS_DOMAIN_ID=20 ros2 launch senior_project RL.launch.py run_rl:=false

    # Terminal 2: Run expert controller on that sim
    ROS_DOMAIN_ID=20 python expert_baseline_node.py --ns stretch

    # Terminal 3: Start RL parallel training (domains 10-13)
    python parallel_runner.py --num_sims 4 --sim_base_domain 10

    # Terminal 4: Monitor everything
    python reward_monitor.py --scan-start 0 --scan-end 25 \\
        --experiment ~/experiments/expert_vs_rl
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

# =============================================================================
# Shared constants and classes from learner_node (one source of truth: if
# this import fails, fix the install instead of running drifted copies)
# =============================================================================

from senior_project.learner_node import (
    ROBOT_WIDTH_M, ROBOT_HALF_WIDTH_M, DESIRED_CLEARANCE_M, MIN_SAFE_DISTANCE,
    ZONE_FREE, ZONE_AWARE, ZONE_CAUTION, ZONE_DANGER, ZONE_EMERGENCY,
    R_COLLISION, COLLISION_COOLDOWN_STEPS, R_GOAL, GOAL_RADIUS,
    PROGRESS_SCALE, STEP_COST_GOAL,
    R_NEW_CELL, R_STEP_EXPLORE, R_REVISIT,
    R_PROXIMITY, PROXIMITY_THRESHOLD,
    EPISODE_SECONDS, GRID_SIZE, GRID_RESOLUTION, GRID_MAX_RANGE,
    V_MAX, V_MIN_FORWARD, W_MAX, MIN_TURN_RADIUS,
    CMD_SMOOTHING_ALPHA, VISIT_DECAY, NOVELTY_RADIUS,
    LIDAR_FORWARD_OFFSET_RAD, NUM_LIDAR_BINS, LIDAR_MAX_RANGE,
    OCC_GRID_UPDATE_INTERVAL, FRONTIER_CACHE_INTERVAL,
    DEBUG_EVERY_N, PUBLISH_MAP, PUBLISH_MAP_EVERY_N, PUBLISH_PATH,
    PATH_HISTORY_LENGTH, MAP_FRAME,
    yaw_from_quat, wrap_to_pi,
    NavigationState, DynamicNavigator, EgoOccupancyGrid,
    ExpertStateMachine,
    StretchRosInterface, StretchExploreEnv,
)


# =============================================================================
# Main Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Expert State Machine Baseline — runs the reactive controller "
                    "in its own sim for fair comparison against RL agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ns", type=str, default="",
                        help="ROS namespace (e.g., 'stretch')")
    parser.add_argument("--odom-topic", type=str, default="/stretch/odom")
    parser.add_argument("--lidar-topic", type=str, default="/stretch/scan")
    parser.add_argument("--imu-topic", type=str, default="/imu_mobile_base")
    parser.add_argument("--goal-topic", type=str, default="goal")
    parser.add_argument("--cmd-topic", type=str, default="/stretch/cmd_vel")
    parser.add_argument("--total-episodes", type=int, default=0,
                        help="Episodes to run (0 = run forever)")
    parser.add_argument("--total-steps", type=int, default=0,
                        help="Max steps (0 = no limit)")
    parser.add_argument("--seed", type=int, default=99)

    args = parser.parse_args()

    np.random.seed(args.seed)

    rclpy.init()
    ros = StretchRosInterface(
        ns=args.ns,
        odom_topic=args.odom_topic,
        scan_topic=args.lidar_topic,
        imu_topic=args.imu_topic,
        goal_topic=args.goal_topic,
        cmd_topic=args.cmd_topic,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(ros)

    env = StretchExploreEnv(ros)
    # Override the training phase so the reward monitor knows this is the baseline
    env._current_training_phase = "EXPERT_BASELINE"
    env._expert_avg_return = 0.0

    expert = ExpertStateMachine()

    G = "\033[92m"
    Y = "\033[93m"
    R = "\033[91m"
    B = "\033[94m"
    RST = "\033[0m"

    domain_id = os.environ.get("ROS_DOMAIN_ID", "?")

    ros.get_logger().info(
        f"\n{G}{'='*55}\n"
        f"  EXPERT BASELINE AGENT\n"
        f"{'='*55}{RST}\n"
        f"  Domain:        {domain_id}\n"
        f"  Controller:    ExpertStateMachine (reactive LIDAR)\n"
        f"  Environment:   Same as RL (v5, never-terminate)\n"
        f"  Reward:        Identical to learner_node\n"
        f"  Phase tag:     EXPERT_BASELINE\n"
        f"  Episodes:      {'unlimited' if args.total_episodes == 0 else args.total_episodes}\n"
        f"  Steps:         {'unlimited' if args.total_steps == 0 else args.total_steps}\n"
        f"\n"
        f"  This agent runs forever alongside your RL sims.\n"
        f"  The reward_monitor will pick it up automatically.\n"
        f"{G}{'='*55}{RST}"
    )

    shutdown_requested = False

    def handle_signal(sig, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        ros.get_logger().info("[EXPERT] Shutdown requested...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # --- Main loop ---
    episode_num = 0
    total_steps = 0

    while not shutdown_requested:
        episode_num += 1
        obs, _ = env.reset()
        expert.reset_episode()
        done = False

        while not done and not shutdown_requested:
            # Get navigation state for the expert
            nav_state = env.last_nav_state
            has_goal = ros.last_goal is not None
            goal_angle = env._goal_angle() if has_goal else 0.0
            goal_dist = env._goal_distance() if has_goal else 0.0

            # Expert produces action
            action = expert.act(nav_state, goal_angle, goal_dist, has_goal)

            # Step environment (computes reward, publishes breakdown)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1

            # Track expert stats
            expert.track_reward(reward, success=info.get("success", False))

            # Update the expert avg return so it appears in the breakdown
            env._expert_avg_return = expert.get_avg_return()

        # Episode finished
        expert_avg = expert.get_avg_return()
        success = info.get("success", False) if 'info' in dir() else False
        status = f"{G}GOAL{RST}" if success else f"{Y}TIMEOUT{RST}"

        ros.get_logger().info(
            f"[EXPERT] ep={episode_num} {status} | "
            f"return={expert.episode_return:+.1f} | "
            f"avg={expert_avg:+.1f} | "
            f"steps={expert.episode_steps} | "
            f"goals={expert.total_goals_reached} | "
            f"total_steps={total_steps}"
        )

        # Check limits
        if args.total_episodes > 0 and episode_num >= args.total_episodes:
            ros.get_logger().info(f"[EXPERT] Reached {args.total_episodes} episodes. Stopping.")
            break
        if args.total_steps > 0 and total_steps >= args.total_steps:
            ros.get_logger().info(f"[EXPERT] Reached {args.total_steps} steps. Stopping.")
            break

    # Shutdown
    try:
        ros.send_cmd(0.0, 0.0)
    except Exception:
        pass

    elapsed = total_steps * 0.1  # rough estimate at 10Hz control
    ros.get_logger().info(
        f"\n{G}{'='*55}\n"
        f"  EXPERT BASELINE COMPLETE\n"
        f"{'='*55}{RST}\n"
        f"  Episodes:      {episode_num}\n"
        f"  Total Steps:   {total_steps}\n"
        f"  Avg Return:    {expert.get_avg_return():+.1f}\n"
        f"  Goals Reached: {expert.total_goals_reached}\n"
        f"  Success Rate:  {expert.total_goals_reached / max(episode_num, 1) * 100:.1f}%\n"
        f"{G}{'='*55}{RST}"
    )

    try:
        executor.shutdown()
        ros.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
