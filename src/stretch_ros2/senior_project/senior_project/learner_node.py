#!/usr/bin/env python3
"""
Stretch Robot RL Environment + Learner — v5 (NEVER TERMINATE ON COLLISION)

KEY INSIGHT: termination on collision teaches the agent that NOT MOVING is safest.
The expected value of spinning (no collision risk) exceeds the expected value of
exploring (collision risk → episode ends → lose all future reward).

FIX: collisions NEVER terminate. They're just a -5 bump + cooldown. The agent
bumps a wall, eats -5, and keeps exploring. Proximity penalty teaches gradual
avoidance. Discovery reward (+5/cell) massively dominates all penalties.

ACTION SPACE (context-aware):
- Explore mode: a[0] maps V_MIN_FORWARD → V_MAX (always moving, can't spin)
- Goal mode:    a[0] maps 0 → V_MAX (can stop at target)
- a[1] = steering, coupled to speed via turn radius

REWARDS (exploration-dominant):
- Explore: discovery(+5/cell) + revisit(-2.0) + step(-0.3)  → exploring >> spinning
- Avoid:   proximity(-3.0) + bump(-5.0 with cooldown)       → learn to steer away
- Goal:    ratchet_progress(+400) + step(-2.5)               → shortest path
- NEVER terminate on collision. Only goal_reached and timeout end episodes.

MATH (per step):
- Spinning:              -0.3 - 2.0           = -2.3/step
- Exploring (5 cells):   -0.3 + 0 + 25.0      = +24.7/step
- Wall-stuck:            -0.3 - 2.0 - 3.0 - 0.6 = -5.9/step
- Explore w/ 20% crash:  0.8*(+24.7)+0.2*(-5) = +18.8/step >> spinning
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry_msgs.msg import Point, PointStamped, Twist
from gymnasium import spaces
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

# =============================================================================
# ROBOT PHYSICAL PARAMETERS
# =============================================================================

ROBOT_WIDTH_M = 0.33
ROBOT_HALF_WIDTH_M = 0.165
DESIRED_CLEARANCE_M = 0.25
MIN_SAFE_DISTANCE = ROBOT_HALF_WIDTH_M + DESIRED_CLEARANCE_M

# =============================================================================
# SAFETY ZONE THRESHOLDS
# =============================================================================

ZONE_FREE = 1.2
ZONE_AWARE = 0.90
ZONE_CAUTION = 0.65
ZONE_DANGER = 0.45
ZONE_EMERGENCY = 0.30

# =============================================================================
# REWARD SYSTEM — v5 (NEVER TERMINATE ON COLLISION)
# =============================================================================

# --- Collision (NEVER terminates) ---
R_COLLISION = -5.0
COLLISION_COOLDOWN_STEPS = 8

R_GOAL = 2000.0
R_TIMEOUT = -50.0
GOAL_RADIUS = 0.45

# --- Goal-seeking ---
PROGRESS_SCALE = 400.0
STEP_COST_GOAL = -2.5

# --- Exploration (discovery-dominant) ---
R_NEW_CELL = 5.0
R_STEP_EXPLORE = -0.3
R_REVISIT = -2.0

# --- Obstacle avoidance (continuous, never terminates) ---
R_PROXIMITY = -3.0
PROXIMITY_THRESHOLD = 0.65

# =============================================================================
# GENERAL CONFIG
# =============================================================================

CHECKPOINT_FILENAME = "td3_safe_rl_agent.pt"
AUTO_LOAD_CHECKPOINT = True

EPISODE_SECONDS = 45.0

# Occupancy Grid
GRID_SIZE = 16
GRID_RESOLUTION = 0.75
GRID_MAX_RANGE = 12.0

# Movement Limits — MINIMUM VELOCITY
V_MAX = 1.0
V_MIN_FORWARD = 0.15
W_MAX = 3.0
V_MIN_REVERSE = -0.05
MIN_TURN_RADIUS = 0.40

# Velocity smoothing
CMD_SMOOTHING_ALPHA = 0.75

# PER Config
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPSILON = 1e-6

# Visit tracking
VISIT_DECAY = 0.995
NOVELTY_RADIUS = 1.0

# LiDAR
LIDAR_FORWARD_OFFSET_RAD = math.pi
NUM_LIDAR_BINS = 36
LIDAR_MAX_RANGE = 20.0

# Training defaults
DEFAULT_START_STEPS = 5000
DEFAULT_EXPL_NOISE = 0.3
DEFAULT_UPDATE_EVERY = 4
DEFAULT_BATCH_SIZE = 256
DEFAULT_REPLAY_SIZE = 150_000
DEFAULT_SAVE_EVERY = 25_000

# Expert demonstration phase
DEFAULT_DEMO_STEPS = 10_000  # State machine drives for this many steps first

# Compute caching intervals
OCC_GRID_UPDATE_INTERVAL = 1
FRONTIER_CACHE_INTERVAL = 10

# Debug
DEBUG_EVERY_N = 100

# Visualization
PUBLISH_MAP = True
PUBLISH_MAP_EVERY_N = 100
PUBLISH_PATH = True
PATH_HISTORY_LENGTH = 1000
MAP_FRAME = "odom"


# =============================================================================
# Utils
# =============================================================================

def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    sin_y_cosp = 2.0 * (qw * qz + qx * qy)
    cos_y_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_y_cosp, cos_y_cosp)


def wrap_to_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * np.clip(t, 0.0, 1.0)


# =============================================================================
# DYNAMIC NAVIGATION SYSTEM — observation only, no control override
# =============================================================================

@dataclass
class NavigationState:
    min_distance: float
    min_angle: float
    left_clearance: float
    right_clearance: float
    front_clearance: float
    back_clearance: float
    sector_distances: np.ndarray
    safety_blend: float
    zone: str
    clear_gaps: List[Tuple[float, float, float]]
    best_direction: float
    best_clearance: float


class DynamicNavigator:
    """LIDAR scan analyzer — provides obstacle awareness for observation."""

    def __init__(self, num_sectors: int = 36):
        self.num_sectors = num_sectors
        self.sector_width = 2 * math.pi / num_sectors
        self.sector_angles = np.array([
            wrap_to_pi(-math.pi + (i + 0.5) * self.sector_width)
            for i in range(num_sectors)
        ])
        self.last_nav_state: Optional[NavigationState] = None

    def analyze_scan(self, scan: Optional[LaserScan]) -> NavigationState:
        if scan is None or len(scan.ranges) == 0:
            return self._default_state()

        ranges = np.array(scan.ranges, dtype=np.float32)
        n_rays = len(ranges)

        max_r = min(scan.range_max, LIDAR_MAX_RANGE) if scan.range_max > 0 else LIDAR_MAX_RANGE
        min_r = max(scan.range_min, 0.05)
        invalid = np.isnan(ranges) | np.isinf(ranges) | (ranges < min_r) | (ranges > max_r)
        ranges[invalid] = max_r

        angles = scan.angle_min + np.arange(n_rays) * scan.angle_increment + LIDAR_FORWARD_OFFSET_RAD
        angles = np.array([wrap_to_pi(a) for a in angles])

        min_idx = np.argmin(ranges)
        min_distance = float(ranges[min_idx])
        min_angle = float(angles[min_idx])

        front_mask = np.abs(angles) < math.pi / 4
        back_mask = np.abs(angles) > 3 * math.pi / 4
        left_mask = (angles > math.pi / 4) & (angles < 3 * math.pi / 4)
        right_mask = (angles < -math.pi / 4) & (angles > -3 * math.pi / 4)

        front_clearance = float(np.min(ranges[front_mask])) if np.any(front_mask) else max_r
        back_clearance = float(np.min(ranges[back_mask])) if np.any(back_mask) else max_r
        left_clearance = float(np.min(ranges[left_mask])) if np.any(left_mask) else max_r
        right_clearance = float(np.min(ranges[right_mask])) if np.any(right_mask) else max_r

        sector_distances = np.full(self.num_sectors, max_r, dtype=np.float32)
        for i in range(n_rays):
            sector_idx = int((angles[i] + math.pi) / self.sector_width) % self.num_sectors
            sector_distances[sector_idx] = min(sector_distances[sector_idx], ranges[i])

        safety_blend = self._compute_safety_blend(min_distance)
        zone = self._get_zone_name(min_distance)

        if left_clearance > right_clearance:
            best_direction = math.pi / 2
            best_clearance = left_clearance
        else:
            best_direction = -math.pi / 2
            best_clearance = right_clearance

        state = NavigationState(
            min_distance=min_distance, min_angle=min_angle,
            left_clearance=left_clearance, right_clearance=right_clearance,
            front_clearance=front_clearance, back_clearance=back_clearance,
            sector_distances=sector_distances, safety_blend=safety_blend,
            zone=zone, clear_gaps=[],
            best_direction=best_direction, best_clearance=best_clearance,
        )
        self.last_nav_state = state
        return state

    def _default_state(self) -> NavigationState:
        return NavigationState(
            min_distance=LIDAR_MAX_RANGE, min_angle=0.0,
            left_clearance=LIDAR_MAX_RANGE, right_clearance=LIDAR_MAX_RANGE,
            front_clearance=LIDAR_MAX_RANGE, back_clearance=LIDAR_MAX_RANGE,
            sector_distances=np.full(self.num_sectors, LIDAR_MAX_RANGE, dtype=np.float32),
            safety_blend=0.0, zone="FREE", clear_gaps=[],
            best_direction=0.0, best_clearance=LIDAR_MAX_RANGE,
        )

    def _compute_safety_blend(self, min_distance: float) -> float:
        if min_distance >= ZONE_FREE:
            return 0.0
        if min_distance <= ZONE_EMERGENCY:
            return 0.95
        t = (min_distance - ZONE_EMERGENCY) / (ZONE_FREE - ZONE_EMERGENCY)
        return float(0.95 * (1.0 - t) ** 2)

    def _get_zone_name(self, min_distance: float) -> str:
        if min_distance >= ZONE_FREE: return "FREE"
        elif min_distance >= ZONE_AWARE: return "AWARE"
        elif min_distance >= ZONE_CAUTION: return "CAUTION"
        elif min_distance >= ZONE_DANGER: return "DANGER"
        elif min_distance >= ZONE_EMERGENCY: return "EMERGENCY"
        else: return "CRITICAL"


# =============================================================================
# Expert State Machine — reactive controller for demonstration phase
# =============================================================================

class ExpertStateMachine:
    """
    Simple reactive controller that uses lidar to navigate.
    Produces actions in the same [-1, 1] action space as the RL agent.
    
    States:
      DRIVE_FORWARD  — open space ahead, go straight
      TURN_TO_GOAL   — goal visible, steer toward it
      AVOID_LEFT     — obstacle on right or ahead, turn left
      AVOID_RIGHT    — obstacle on left or ahead, turn right
      BACK_UP        — too close everywhere, reverse briefly
    
    The state machine's rewards are tracked separately so you can compare
    when the RL agent surpasses the expert baseline.
    """

    class State(Enum):
        DRIVE_FORWARD = "DRIVE_FORWARD"
        TURN_TO_GOAL = "TURN_TO_GOAL"
        AVOID_LEFT = "AVOID_LEFT"
        AVOID_RIGHT = "AVOID_RIGHT"
        BACK_UP = "BACK_UP"

    def __init__(self):
        self.state = self.State.DRIVE_FORWARD
        self._backup_steps = 0
        self._backup_duration = 8  # steps to back up
        
        # Performance tracking (for baseline comparison)
        self.episode_return = 0.0
        self.episode_steps = 0
        self.total_episodes = 0
        self.episode_returns: deque = deque(maxlen=100)
        self.total_goals_reached = 0

    def reset_episode(self):
        """Call at episode boundary to track per-episode stats."""
        if self.episode_steps > 0:
            self.episode_returns.append(self.episode_return)
            self.total_episodes += 1
        self.episode_return = 0.0
        self.episode_steps = 0
        self.state = self.State.DRIVE_FORWARD
        self._backup_steps = 0

    def track_reward(self, reward: float, success: bool = False):
        self.episode_return += reward
        self.episode_steps += 1
        if success:
            self.total_goals_reached += 1

    def get_avg_return(self) -> float:
        if len(self.episode_returns) == 0:
            return 0.0
        return float(sum(self.episode_returns) / len(self.episode_returns))

    def act(self, nav_state: Optional[NavigationState],
            goal_angle: float, goal_dist: float,
            has_goal: bool) -> np.ndarray:
        """
        Produce an action in [-1, 1] x [-1, 1] space matching the RL agent.
        
        a[0]: velocity  — maps to V_MIN_FORWARD..V_MAX (explore) or 0..V_MAX (goal)
        a[1]: steering  — maps to -w_limit..+w_limit
        
        Returns normalized action in [-1, 1].
        """
        if nav_state is None:
            # No sensor data — stop
            return np.array([-1.0, 0.0], dtype=np.float32)

        min_d = nav_state.min_distance
        front = nav_state.front_clearance
        left = nav_state.left_clearance
        right = nav_state.right_clearance

        # --- State transitions ---
        if self.state == self.State.BACK_UP:
            self._backup_steps += 1
            if self._backup_steps >= self._backup_duration:
                self._backup_steps = 0
                # After backing up, turn toward the more open side
                if left > right:
                    self.state = self.State.AVOID_LEFT
                else:
                    self.state = self.State.AVOID_RIGHT
        elif min_d < ZONE_EMERGENCY:
            # Too close — back up
            self.state = self.State.BACK_UP
            self._backup_steps = 0
        elif front < ZONE_CAUTION:
            # Something ahead — avoid
            if left > right:
                self.state = self.State.AVOID_LEFT
            else:
                self.state = self.State.AVOID_RIGHT
        elif has_goal and goal_dist > GOAL_RADIUS:
            self.state = self.State.TURN_TO_GOAL
        elif front < ZONE_AWARE:
            # Getting close, gentle avoidance
            if left > right:
                self.state = self.State.AVOID_LEFT
            else:
                self.state = self.State.AVOID_RIGHT
        else:
            self.state = self.State.DRIVE_FORWARD

        # --- Action generation ---
        if self.state == self.State.BACK_UP:
            # Reverse slowly, turn toward open space
            a_vel = -1.0  # maps to minimum/reverse velocity
            turn = 0.5 if left > right else -0.5
            a_steer = np.clip(turn, -1.0, 1.0)

        elif self.state == self.State.AVOID_LEFT:
            # Slow down, turn left (positive steering)
            danger_factor = np.clip(1.0 - (front / ZONE_FREE), 0.0, 1.0)
            a_vel = np.clip(0.3 - danger_factor * 0.6, -1.0, 1.0)
            a_steer = np.clip(0.4 + danger_factor * 0.5, 0.0, 1.0)

        elif self.state == self.State.AVOID_RIGHT:
            # Slow down, turn right (negative steering)
            danger_factor = np.clip(1.0 - (front / ZONE_FREE), 0.0, 1.0)
            a_vel = np.clip(0.3 - danger_factor * 0.6, -1.0, 1.0)
            a_steer = np.clip(-0.4 - danger_factor * 0.5, -1.0, 0.0)

        elif self.state == self.State.TURN_TO_GOAL:
            # Proportional steering toward goal, speed based on alignment
            steer = np.clip(goal_angle / (math.pi / 2), -1.0, 1.0)
            alignment = 1.0 - abs(steer)  # 1.0 when facing goal, 0.0 when perpendicular
            # Go faster when well-aligned, slower when turning hard
            a_vel = np.clip(0.2 + 0.7 * alignment, -1.0, 1.0)
            a_steer = steer

        else:  # DRIVE_FORWARD
            a_vel = 0.8  # Cruise speed
            # Gentle bias toward more open side for exploration
            side_bias = (left - right) / max(left + right, 0.1)
            a_steer = np.clip(side_bias * 0.15, -0.3, 0.3)

        action = np.array([float(a_vel), float(a_steer)], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)


# =============================================================================
# Running Mean/Std for Normalization
# =============================================================================

class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        if x.ndim == 1:
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class RewardNormalizer:
    def __init__(self, clip: float = 10.0):
        self.rms = RunningMeanStd(shape=())
        self.clip = clip

    def normalize(self, reward: float, update: bool = True) -> float:
        if update:
            self.rms.update(np.array([reward]))
        std = np.sqrt(self.rms.var + 1e-8)
        normalized = reward / max(std, 1.0)
        return float(np.clip(normalized, -self.clip, self.clip))


# =============================================================================
# Prioritized Experience Replay (PER)
# =============================================================================

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self._update_count = 0

    def update(self, tree_idx: int, priority: float):
        if np.isnan(priority) or np.isinf(priority) or priority < 0:
            priority = 1e-8
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
        self._update_count += 1
        # Periodically rebuild parent nodes from leaves to correct float drift
        if self._update_count % 10000 == 0:
            self._rebuild_parents()

    def _rebuild_parents(self):
        """Recompute all internal nodes from leaf values to fix accumulated
        floating-point drift in parent sums."""
        for i in range(self.capacity - 2, -1, -1):
            left = 2 * i + 1
            right = left + 1
            left_val = self.tree[left] if left < len(self.tree) else 0.0
            right_val = self.tree[right] if right < len(self.tree) else 0.0
            self.tree[i] = left_val + right_val

    def add(self, priority: float) -> int:
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        return data_idx

    def get(self, value: float) -> Tuple[int, int, float]:
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break
            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, data_idx, self.tree[leaf_idx]

    @property
    def total_priority(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int,
                 device: torch.device, alpha: float = PER_ALPHA):
        self.device = device
        self.size = int(size)
        self.alpha = alpha
        self.ptr = 0
        self.count = 0

        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.size, act_dim), dtype=np.float32)
        self.rews = np.zeros((self.size, 1), dtype=np.float32)
        self.done = np.zeros((self.size, 1), dtype=np.float32)

        self.tree = SumTree(self.size)
        self.max_priority = 1.0

    def _uniform_fallback(self, batch_size: int):
        """Fall back to uniform sampling and rebuild the tree."""
        indices = np.random.randint(0, self.count, size=batch_size).astype(np.int32)
        weights = np.ones(batch_size, dtype=np.float32)
        self.max_priority = 1.0
        self.tree = SumTree(self.size)
        for _ in range(self.count):
            self.tree.add(1.0)
        return (
            torch.as_tensor(self.obs[indices], device=self.device),
            torch.as_tensor(self.acts[indices], device=self.device),
            torch.as_tensor(self.rews[indices], device=self.device),
            torch.as_tensor(self.next_obs[indices], device=self.device),
            torch.as_tensor(self.done[indices], device=self.device),
            torch.as_tensor(weights, device=self.device, dtype=torch.float32),
            indices,
        )

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        if np.isnan(self.max_priority) or np.isinf(self.max_priority) or self.max_priority <= 0:
            self.max_priority = 1.0
        priority = self.max_priority ** self.alpha
        self.tree.add(priority)
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size: int, beta: float = PER_BETA_START) -> Tuple:
        if self.count < batch_size:
            batch_size = self.count
        if self.count == 0:
            raise ValueError("Cannot sample from empty replay buffer")

        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        total = float(self.tree.total_priority)

        # If tree is corrupted OR numerically dangerous, fall back to uniform sampling
        # Keep TOTAL_MAX low enough that total/batch_size * batch_size stays well under
        # float64 max (~1.8e308). With batch_size up to ~4096, 1e100 is very safe.
        TOTAL_MAX = 1e100
        use_uniform = (
            not np.isfinite(total)
            or total <= 0.0
            or total > TOTAL_MAX
        )

        if use_uniform:
            return self._uniform_fallback(batch_size)

        segment = total / batch_size
        if not np.isfinite(segment) or segment <= 0.0:
            return self._uniform_fallback(batch_size)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)

            # Extra guard: if low/high overflow mid-loop, bail out safely
            if not (np.isfinite(low) and np.isfinite(high) and high > low):
                return self._uniform_fallback(batch_size)

            value = np.random.uniform(low, high)
            tree_idx, data_idx, priority = self.tree.get(value)
            indices[i] = data_idx % self.count
            priorities[i] = max(priority, 1e-8)

        probs = priorities / max(self.tree.total_priority, 1e-8)
        weights = (self.count * probs) ** (-beta)
        weights = weights / max(weights.max(), 1e-8)

        return (
            torch.as_tensor(self.obs[indices], device=self.device),
            torch.as_tensor(self.acts[indices], device=self.device),
            torch.as_tensor(self.rews[indices], device=self.device),
            torch.as_tensor(self.next_obs[indices], device=self.device),
            torch.as_tensor(self.done[indices], device=self.device),
            torch.as_tensor(weights, device=self.device, dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        PRIORITY_MAX = 1e6  # cap to keep SumTree totals sane
        for idx, td_error in zip(indices, td_errors):
            if np.isnan(td_error) or np.isinf(td_error):
                td_error = 0.0
            priority = (abs(td_error) + PER_EPSILON) ** self.alpha
            if np.isnan(priority) or np.isinf(priority):
                priority = PRIORITY_MAX
            priority = float(min(priority, PRIORITY_MAX))

            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)
            # FIX: cap max_priority so add() never inserts astronomical priorities
            self.max_priority = min(max(self.max_priority, abs(td_error) + PER_EPSILON), PRIORITY_MAX)

    def save(self, path: str):
        np.savez_compressed(
            path,
            obs=self.obs[:self.count], acts=self.acts[:self.count],
            rews=self.rews[:self.count], next_obs=self.next_obs[:self.count],
            done=self.done[:self.count], ptr=self.ptr, count=self.count,
        )

    def load(self, path: str):
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            count = int(data["count"])
            if count == 0:
                return False
            n = min(count, self.size)
            if data["obs"].shape[1] != self.obs.shape[1]:
                return False
            self.obs[:n] = data["obs"][:n]
            self.acts[:n] = data["acts"][:n]
            self.rews[:n] = data["rews"][:n]
            self.next_obs[:n] = data["next_obs"][:n]
            self.done[:n] = data["done"][:n]
            self.ptr = int(data["ptr"]) % self.size
            self.count = n
            self.tree = SumTree(self.size)
            for _ in range(n):
                self.tree.add(1.0)
            self.max_priority = 1.0
            return True
        except Exception:
            return False


# =============================================================================
# Ego-Centric Occupancy Grid (16x16, coarser resolution)
# =============================================================================

class EgoOccupancyGrid:
    def __init__(self, size: int = GRID_SIZE, resolution: float = GRID_RESOLUTION):
        self.size = size
        self.resolution = resolution
        self.half_size = size // 2
        self.grid = np.zeros((size, size), dtype=np.float32)
        self.world_grid: Dict[Tuple[int, int], float] = {}
        self.visit_counts: Dict[Tuple[int, int], float] = {}
        self.total_cells_discovered = 0
        self.cells_discovered_this_step = 0

    def world_to_grid_key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x / self.resolution), int(y / self.resolution))

    def update_from_scan(self, robot_x: float, robot_y: float, robot_yaw: float,
                         scan: LaserScan) -> int:
        self.cells_discovered_this_step = 0
        self.grid.fill(0.0)

        ranges = np.array(scan.ranges, dtype=np.float32)
        n_rays = len(ranges)
        if n_rays == 0:
            return 0

        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        range_max = min(scan.range_max, GRID_MAX_RANGE)
        range_min = max(scan.range_min, 0.05)
        sensor_max = scan.range_max if scan.range_max > 0 else LIDAR_MAX_RANGE

        robot_key = self.world_to_grid_key(robot_x, robot_y)
        self.visit_counts[robot_key] = self.visit_counts.get(robot_key, 0) + 1

        FREE_DECREMENT = 0.1
        OBS_INCREMENT = 0.35

        for i in range(0, n_rays, 4):
            original_r = ranges[i]
            if np.isnan(original_r) or np.isinf(original_r) or original_r < range_min:
                continue

            r = min(original_r, range_max)
            ray_angle_world = robot_yaw + angle_min + i * angle_inc + LIDAR_FORWARD_OFFSET_RAD

            step = self.resolution * 0.5
            clear_limit = r - self.resolution * 1.0
            for d in np.arange(step, max(clear_limit, step), step):
                wx = robot_x + d * math.cos(ray_angle_world)
                wy = robot_y + d * math.sin(ray_angle_world)
                world_key = self.world_to_grid_key(wx, wy)
                if world_key not in self.world_grid:
                    self.world_grid[world_key] = 0.3
                    self.total_cells_discovered += 1
                    self.cells_discovered_this_step += 1
                else:
                    self.world_grid[world_key] = max(0.0, self.world_grid[world_key] - FREE_DECREMENT)

                ego_x, ego_y = self._world_to_ego(wx, wy, robot_x, robot_y, robot_yaw)
                gx = int(ego_x / self.resolution) + self.half_size
                gy = int(ego_y / self.resolution) + self.half_size
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    self.grid[gy, gx] = 0.5

            is_real_obstacle = (original_r < sensor_max * 0.95) and (original_r < range_max - 0.1)
            if is_real_obstacle:
                wx = robot_x + r * math.cos(ray_angle_world)
                wy = robot_y + r * math.sin(ray_angle_world)
                world_key = self.world_to_grid_key(wx, wy)
                if world_key not in self.world_grid:
                    self.total_cells_discovered += 1
                    self.cells_discovered_this_step += 1
                    self.world_grid[world_key] = OBS_INCREMENT
                else:
                    self.world_grid[world_key] = min(1.0, self.world_grid[world_key] + OBS_INCREMENT)

                ego_x, ego_y = self._world_to_ego(wx, wy, robot_x, robot_y, robot_yaw)
                gx = int(ego_x / self.resolution) + self.half_size
                gy = int(ego_y / self.resolution) + self.half_size
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    self.grid[gy, gx] = 1.0

        return self.cells_discovered_this_step

    def _world_to_ego(self, wx, wy, robot_x, robot_y, robot_yaw):
        dx = wx - robot_x
        dy = wy - robot_y
        angle = -robot_yaw + math.pi / 2
        ego_x = dx * math.cos(angle) - dy * math.sin(angle)
        ego_y = dx * math.sin(angle) + dy * math.cos(angle)
        return ego_x, ego_y

    def get_novelty(self, robot_x: float, robot_y: float) -> float:
        robot_key = self.world_to_grid_key(robot_x, robot_y)
        total_visits = self.visit_counts.get(robot_key, 0)
        cells_checked = 1
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                key = (robot_key[0] + dx, robot_key[1] + dy)
                total_visits += self.visit_counts.get(key, 0)
                cells_checked += 1
        avg_visits = total_visits / cells_checked
        return float(math.exp(-avg_visits * 0.1))

    def get_frontier_direction(self, robot_x: float, robot_y: float, robot_yaw: float) -> float:
        robot_key = self.world_to_grid_key(robot_x, robot_y)
        best_frontier = None
        best_dist = float('inf')
        search_radius = 16

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                key = (robot_key[0] + dx, robot_key[1] + dy)
                if self.world_grid.get(key, 0) > 0.4:
                    continue
                has_unknown_neighbor = False
                for ndx, ndy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor_key = (key[0] + ndx, key[1] + ndy)
                    if neighbor_key not in self.world_grid:
                        has_unknown_neighbor = True
                        break
                if has_unknown_neighbor:
                    dist = math.hypot(dx, dy)
                    if dist < best_dist and dist > 2:
                        best_dist = dist
                        best_frontier = key

        if best_frontier is None:
            return 0.0

        fx = best_frontier[0] * self.resolution
        fy = best_frontier[1] * self.resolution
        rx = robot_key[0] * self.resolution
        ry = robot_key[1] * self.resolution
        angle_world = math.atan2(fy - ry, fx - rx)
        return float(wrap_to_pi(angle_world - robot_yaw))

    def get_flat_grid(self) -> np.ndarray:
        return self.grid.flatten()

    def decay_visits(self):
        for key in self.visit_counts:
            self.visit_counts[key] *= VISIT_DECAY

    def get_stats(self) -> Dict:
        frontier_count = 0
        for key, value in self.world_grid.items():
            if value > 0.4:
                continue
            for ndx, ndy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (key[0] + ndx, key[1] + ndy) not in self.world_grid:
                    frontier_count += 1
                    break
        return {
            'total_discovered': self.total_cells_discovered,
            'new_this_step': self.cells_discovered_this_step,
            'world_grid_size': len(self.world_grid),
            'frontier_count': frontier_count,
            'fully_explored': frontier_count == 0 and self.total_cells_discovered > 50,
        }

    def reset(self):
        self.grid.fill(0.0)
        self.cells_discovered_this_step = 0

    def get_occupancy_grid_msg(self, frame_id: str = "odom") -> OccupancyGrid:
        if not self.world_grid:
            msg = OccupancyGrid()
            msg.header.frame_id = frame_id
            msg.info.resolution = self.resolution
            msg.info.width = 1
            msg.info.height = 1
            msg.data = [-1]
            return msg

        keys = list(self.world_grid.keys())
        min_x = min(k[0] for k in keys)
        max_x = max(k[0] for k in keys)
        min_y = min(k[1] for k in keys)
        max_y = max(k[1] for k in keys)
        padding = 5
        min_x -= padding; max_x += padding
        min_y -= padding; max_y += padding
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        data = []
        for gy in range(min_y, max_y + 1):
            for gx in range(min_x, max_x + 1):
                key = (gx, gy)
                if key not in self.world_grid: data.append(-1)
                elif self.world_grid[key] >= 0.7: data.append(100)
                elif self.world_grid[key] <= 0.4: data.append(0)
                else: data.append(50)

        msg = OccupancyGrid()
        msg.header.frame_id = frame_id
        msg.info.resolution = self.resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = min_x * self.resolution
        msg.info.origin.position.y = min_y * self.resolution
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = data
        return msg

    def get_visit_heatmap_msg(self, frame_id: str = "odom") -> OccupancyGrid:
        if not self.visit_counts:
            msg = OccupancyGrid()
            msg.header.frame_id = frame_id
            msg.info.resolution = self.resolution
            msg.info.width = 1
            msg.info.height = 1
            msg.data = [0]
            return msg

        keys = list(self.visit_counts.keys())
        min_x = min(k[0] for k in keys); max_x = max(k[0] for k in keys)
        min_y = min(k[1] for k in keys); max_y = max(k[1] for k in keys)
        padding = 2
        min_x -= padding; max_x += padding
        min_y -= padding; max_y += padding
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        max_visits = max(self.visit_counts.values()) if self.visit_counts else 1

        data = []
        for gy in range(min_y, max_y + 1):
            for gx in range(min_x, max_x + 1):
                visits = self.visit_counts.get((gx, gy), 0)
                data.append(int(100 * visits / max(max_visits, 1)))

        msg = OccupancyGrid()
        msg.header.frame_id = frame_id
        msg.info.resolution = self.resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = min_x * self.resolution
        msg.info.origin.position.y = min_y * self.resolution
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = data
        return msg


# =============================================================================
# ROS Interface
# =============================================================================

class StretchRosInterface(Node):
    def __init__(self, ns: str = "", odom_topic="/odom", scan_topic="/scan_filtered",
                 imu_topic="/imu_mobile_base", goal_topic="goal", cmd_topic="/stretch/cmd_vel"):
        super().__init__("learner_node")

        self.last_odom: Optional[Odometry] = None
        self.last_scan: Optional[LaserScan] = None
        self.last_goal: Optional[PointStamped] = None
        self._last_goal_time: float = 0.0
        self._goal_persist_timeout: float = 60.0
        self._last_known_target_pos: Optional[Tuple[float, float]] = None
        self._last_known_target_time: float = 0.0
        self._last_known_persist_timeout: float = 120.0
        self._target_was_lost: bool = False
        self._lost_time: float = 0.0
        self.last_imu: Optional[Imu] = None

        def make_topic(topic: str) -> str:
            if topic.startswith("/"): return topic
            elif ns: return f"/{ns}/{topic}"
            else: return f"/{topic}"

        odom_name = make_topic(odom_topic)
        scan_name = make_topic(scan_topic)
        imu_name = make_topic(imu_topic)
        goal_name = make_topic(goal_topic)
        cmd_name = make_topic(cmd_topic)

        self.get_logger().info(f"[TOPICS] odom={odom_name}, scan={scan_name}, cmd={cmd_name}")
        self.get_logger().info(f"[TOPICS] imu={imu_name}, goal={goal_name}")

        self.create_subscription(Odometry, odom_name, self._odom_cb, 10)
        self.create_subscription(LaserScan, scan_name, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, imu_name, self._imu_cb, 10)
        self.create_subscription(PointStamped, goal_name, self._goal_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, cmd_name, 10)
        self.reward_pub = self.create_publisher(Float32, "/reward", 10)
        self.reward_breakdown_pub = self.create_publisher(StringMsg, "/reward_breakdown", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/exploration_map", 10)
        self.heatmap_pub = self.create_publisher(OccupancyGrid, "/visit_heatmap", 10)
        self.path_pub = self.create_publisher(Marker, "/robot_path", 10)
        self.frontier_pub = self.create_publisher(MarkerArray, "/frontiers", 10)
        self.safety_zone_pub = self.create_publisher(Marker, "/safety_zone", 10)
        self.goal_reached_pub = self.create_publisher(PointStamped, f"/{ns}/goal_reached", 10)
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_marker", 10)
        self.last_known_marker_pub = self.create_publisher(Marker, "/last_known_marker", 10)

        self.path_history: deque = deque(maxlen=PATH_HISTORY_LENGTH)
        self.get_logger().info("[VIZ] Publishing: /exploration_map, /visit_heatmap, /robot_path, /safety_zone, /goal_marker")

        self.reset_client = self.create_client(Trigger, '/sim/reset')
        self._sim_reset_available = False
        if self.reset_client.wait_for_service(timeout_sec=2.0):
            self._sim_reset_available = True
            self.get_logger().info('[ROS] /sim/reset service FOUND')
        else:
            self.get_logger().warn('[ROS] /sim/reset service NOT found — soft reset only')

    def add_path_point(self, x: float, y: float):
        self.path_history.append((x, y))

    def publish_path(self, frame_id: str = MAP_FRAME):
        if len(self.path_history) < 2:
            return
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.5; marker.color.a = 0.8
        for x, y in self.path_history:
            p = Point(); p.x = x; p.y = y; p.z = 0.05
            marker.points.append(p)
        self.path_pub.publish(marker)

    def publish_safety_zone(self, robot_x: float, robot_y: float, nav_state: NavigationState, frame_id: str = MAP_FRAME):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safety_zone"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = robot_x
        marker.pose.position.y = robot_y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = nav_state.min_distance * 2
        marker.scale.y = nav_state.min_distance * 2
        marker.scale.z = 0.1
        zone_colors = {
            "FREE": (0.0, 1.0, 0.0), "AWARE": (0.5, 1.0, 0.0),
            "CAUTION": (1.0, 1.0, 0.0), "DANGER": (1.0, 0.5, 0.0),
        }
        r, g, b = zone_colors.get(nav_state.zone, (1.0, 0.0, 0.0))
        marker.color.r = r; marker.color.g = g; marker.color.b = b; marker.color.a = 0.3
        self.safety_zone_pub.publish(marker)

    def _odom_cb(self, msg): self.last_odom = msg
    def _scan_cb(self, msg): self.last_scan = msg
    def _imu_cb(self, msg): self.last_imu = msg
    def _goal_cb(self, msg):
        if self._target_was_lost and self.last_goal is None:
            self._target_was_lost = False
        self.last_goal = msg
        self._last_goal_time = time.time()
        self._last_known_target_pos = (msg.point.x, msg.point.y)
        self._last_known_target_time = time.time()

    def wait_for_sensors(self, timeout: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.last_odom is not None and self.last_scan is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn("[ENV] Timeout waiting for sensors")
        return False

    def send_cmd(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def reset_simulation(self, timeout: float = 5.0) -> bool:
        for _ in range(10):
            self.send_cmd(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.01)

        if not self._sim_reset_available:
            if self.reset_client.wait_for_service(timeout_sec=2.0):
                self._sim_reset_available = True

        if self._sim_reset_available:
            req = Trigger.Request()
            future = self.reset_client.call_async(req)
            start = time.time()
            while not future.done() and (time.time() - start) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                result = future.result()
                if result and result.success:
                    self.get_logger().info('[RESET] Sim reset SUCCESS')
                    self._wait_for_fresh_data()
                    return True

        self.send_cmd(0.0, 0.0)
        time.sleep(0.2)
        self._wait_for_fresh_data()
        return False

    def _wait_for_fresh_data(self, timeout: float = 3.0):
        old_odom = self.last_odom
        old_scan = self.last_scan
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.last_odom is not None and self.last_odom is not old_odom and
                self.last_scan is not None and self.last_scan is not old_scan):
                return True
        return False


# =============================================================================
# Gym Environment — OPTIMIZED
# =============================================================================

class StretchExploreEnv(gym.Env):
    """
    RL environment — NO safety overrides.
    Simplified observation, simplified rewards, cached expensive computations.
    """

    def __init__(self, ros: StretchRosInterface, control_dt: float = 0.1):
        super().__init__()
        self.ros = ros
        self.control_dt = control_dt

        self.navigator = DynamicNavigator(num_sectors=36)
        self.occ_grid = EgoOccupancyGrid()
        self.obs_rms = None
        self.reward_normalizer = RewardNormalizer()

        # State
        self.step_count = 0
        self.total_steps = 0
        self.max_steps = int(EPISODE_SECONDS / control_dt)
        self.episode_index = 1
        self.episode_return = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_goal_dist = 0.0
        self._smooth_v = 0.0
        self._smooth_w = 0.0
        self._last_pos = (0.0, 0.0)
        self._goals_reached_this_episode = 0
        self._collision_cooldown = 0

        # Cached computation results
        self._cached_frontier_angle = 0.0
        self._cached_new_cells = 0

        # Ratchet progress for goal-seeking
        self._best_goal_dist = float('inf')

        # Observation space: lidar(36) + goal(5) + vel(2) + prev_act(2) + has_goal(1) + grid(256) + frontier(2) + novelty(1)
        grid_flat_size = GRID_SIZE * GRID_SIZE  # 256
        obs_dim = NUM_LIDAR_BINS + 5 + 2 + 2 + 1 + grid_flat_size + 2 + 1  # = 305

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.last_nav_state: Optional[NavigationState] = None
        self.last_safety_state = None

        self.ros.get_logger().info("[ENV] Waiting for sensors...")
        self.ros.wait_for_sensors()
        self.ros.get_logger().info(f"[ENV] Observation dim: {obs_dim} (was 664)")
        self.ros.get_logger().info(f"[ENV] v5 NEVER TERMINATE — context-aware velocity")
        self.ros.get_logger().info(f"[ENV] Explore: always forward {V_MIN_FORWARD}-{V_MAX} m/s | Goal: can stop 0-{V_MAX} m/s")
        self.ros.get_logger().info(f"[ENV] Rewards: discovery(+{R_NEW_CELL}) + revisit({R_REVISIT}) + bump({R_COLLISION}) — no termination")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        sim_was_reset = self.ros.reset_simulation()

        self.step_count = 0
        self.episode_return = 0.0
        self.prev_action[:] = 0.0
        self._smooth_v = 0.0
        self._smooth_w = 0.0
        self._collision_cooldown = 0
        self._cached_frontier_angle = 0.0
        self._cached_new_cells = 0
        self._best_goal_dist = float('inf')

        st = self._get_robot_state()
        self._last_pos = (st["x"], st["y"])

        self.occ_grid.reset()
        if not sim_was_reset:
            self.occ_grid.decay_visits()

        self._goals_reached_this_episode = 0
        self.ros.last_goal = None
        self.ros._last_goal_time = 0.0
        self.ros._target_was_lost = False
        self.ros._lost_time = 0.0
        self.ros._last_known_target_pos = None
        self.ros._last_known_target_time = 0.0
        self.ros.path_history.clear()

        self.prev_goal_dist = self._goal_distance()
        self._best_goal_dist = self.prev_goal_dist if self.prev_goal_dist > 0 else float('inf')

        if self.ros.last_scan is not None:
            self.last_nav_state = self.navigator.analyze_scan(self.ros.last_scan)
        else:
            self.last_nav_state = None
        self.last_safety_state = self.last_nav_state

        obs = self._build_observation()
        return obs, {"sim_reset": sim_was_reset}

    def step(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)

        has_goal_for_velocity = self.ros.last_goal is not None
        if has_goal_for_velocity:
            rl_v = (float(a[0]) + 1.0) * 0.5 * V_MAX
        else:
            rl_v = V_MIN_FORWARD + (float(a[0]) + 1.0) * 0.5 * (V_MAX - V_MIN_FORWARD)

        if rl_v > 0.05:
            w_limit = min(rl_v / MIN_TURN_RADIUS, W_MAX)
        else:
            w_limit = 0.5
        rl_w = float(a[1]) * w_limit

        self._smooth_v = CMD_SMOOTHING_ALPHA * rl_v + (1.0 - CMD_SMOOTHING_ALPHA) * self._smooth_v
        self._smooth_w = CMD_SMOOTHING_ALPHA * rl_w + (1.0 - CMD_SMOOTHING_ALPHA) * self._smooth_w

        self.ros.send_cmd(self._smooth_v, self._smooth_w)

        t_end = time.time() + self.control_dt
        while time.time() < t_end:
            rclpy.spin_once(self.ros, timeout_sec=0.01)

        st = self._get_robot_state()
        scan = self.ros.last_scan

        new_cells = 0
        if scan is not None and self.step_count % OCC_GRID_UPDATE_INTERVAL == 0:
            new_cells = self.occ_grid.update_from_scan(st["x"], st["y"], st["yaw"], scan)
            self._cached_new_cells = new_cells
        else:
            new_cells = 0

        nav_state = self.navigator.analyze_scan(self.ros.last_scan)
        self.last_nav_state = nav_state
        self.last_safety_state = nav_state

        if self.step_count % FRONTIER_CACHE_INTERVAL == 0:
            self._cached_frontier_angle = self.occ_grid.get_frontier_direction(
                st["x"], st["y"], st["yaw"]
            )

        obs = self._build_observation()

        terminated = False
        collision = False
        success = False
        reward = 0.0
        reward_terms = {}

        min_dist = nav_state.min_distance if nav_state else LIDAR_MAX_RANGE

        if self._collision_cooldown > 0:
            self._collision_cooldown -= 1

        if min_dist < ZONE_EMERGENCY and self.step_count > 10:
            collision = True
            if self._collision_cooldown <= 0:
                reward += R_COLLISION
                reward_terms["collision"] = R_COLLISION
                self._collision_cooldown = COLLISION_COOLDOWN_STEPS

        has_goal = self.ros.last_goal is not None
        d_goal = self._goal_distance()

        if has_goal and d_goal <= GOAL_RADIUS and not terminated:
            terminated = True
            success = True
            reward += R_GOAL
            reward_terms["goal"] = R_GOAL
            self._goals_reached_this_episode += 1

            reached_msg = PointStamped()
            reached_msg.header.stamp = self.ros.get_clock().now().to_msg()
            reached_msg.header.frame_id = "odom"
            reached_msg.point.x = self.ros.last_goal.point.x
            reached_msg.point.y = self.ros.last_goal.point.y
            reached_msg.point.z = 0.0
            self.ros.goal_reached_pub.publish(reached_msg)

        truncated = self.step_count >= self.max_steps
        if truncated and not terminated:
            reward += R_TIMEOUT
            reward_terms["timeout"] = R_TIMEOUT

        if not terminated and not truncated:
            self._last_pos = (st["x"], st["y"])

            if min_dist < PROXIMITY_THRESHOLD:
                proximity_ratio = 1.0 - (min_dist / PROXIMITY_THRESHOLD)
                r_proximity = R_PROXIMITY * proximity_ratio
                reward += r_proximity
                reward_terms["proximity"] = r_proximity

            if has_goal:
                if d_goal < self._best_goal_dist:
                    ratchet_progress = self._best_goal_dist - d_goal
                    r_progress = PROGRESS_SCALE * ratchet_progress
                    self._best_goal_dist = d_goal
                else:
                    r_progress = 0.0

                reward_terms["progress"] = r_progress
                reward += r_progress
                reward += STEP_COST_GOAL
                reward_terms["step"] = STEP_COST_GOAL
            else:
                r_discovery = R_NEW_CELL * self._cached_new_cells
                reward_terms["discovery"] = r_discovery
                reward += r_discovery

                novelty = self.occ_grid.get_novelty(st["x"], st["y"])
                r_revisit = R_REVISIT * (1.0 - novelty)
                reward += r_revisit
                reward_terms["revisit"] = r_revisit

                reward += R_STEP_EXPLORE
                reward_terms["step"] = R_STEP_EXPLORE

        self.prev_goal_dist = d_goal
        self.total_steps += 1
        self.episode_return += reward
        self.prev_action[:] = a
        self.step_count += 1

        done = terminated or truncated

        info = {
            "collision": collision,
            "success": success,
            "exploring": not has_goal,
            "goal_dist": d_goal,
            "reward_terms": reward_terms,
            "nav_zone": nav_state.zone if nav_state else "UNKNOWN",
            "min_distance": min_dist,
            "executed_v": self._smooth_v,
            "executed_w": self._smooth_w,
            "raw_v": rl_v,
            "raw_w": rl_w,
        }

        if done:
            stats = self.occ_grid.get_stats()
            if success: status = "\033[92mGOAL_REACHED\033[0m"
            elif collision: status = "\033[91mCOLLISION\033[0m"
            else: status = "\033[93mTIMEOUT\033[0m"
            mode = "GOAL" if has_goal else "EXPLORE"
            ret_color = "\033[92m" if self.episode_return >= 0 else "\033[91m"
            self.ros.get_logger().info(
                f"[EP {self.episode_index:04d}] {mode} {status} | "
                f"Return {ret_color}{self.episode_return:+.1f}\033[0m | Steps {self.step_count} | "
                f"Cells {stats['total_discovered']} | Goals {self._goals_reached_this_episode} | "
                f"MinDist {min_dist:.2f}m"
            )
            self.episode_index += 1
        elif collision and self._collision_cooldown == COLLISION_COOLDOWN_STEPS:
            self.ros.get_logger().info(
                f"\033[93m[BUMP]\033[0m step={self.step_count} min_d={min_dist:.2f}m "
                f"penalty={R_COLLISION} (cooldown {COLLISION_COOLDOWN_STEPS} steps)"
            )

        if self.step_count % DEBUG_EVERY_N == 0:
            mode = "GOAL" if has_goal else "EXPLORE"
            goal_info = f"goal_dist={d_goal:.2f}m " if has_goal else ""
            self.ros.get_logger().info(
                f"[{mode}] step={self.step_count} min_d={min_dist:.2f}m "
                f"{goal_info}cmd=[{self._smooth_v:.2f},{self._smooth_w:.2f}] r={reward:.2f}"
            )

        self._publish_reward_breakdown(reward, info, has_goal)

        if PUBLISH_MAP and self.step_count % PUBLISH_MAP_EVERY_N == 0:
            self._publish_visualization(st, nav_state)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _publish_visualization(self, robot_state: Dict, nav_state: NavigationState):
        self.ros.add_path_point(robot_state["x"], robot_state["y"])
        map_msg = self.occ_grid.get_occupancy_grid_msg(MAP_FRAME)
        map_msg.header.stamp = self.ros.get_clock().now().to_msg()
        self.ros.map_pub.publish(map_msg)
        heatmap_msg = self.occ_grid.get_visit_heatmap_msg(MAP_FRAME)
        heatmap_msg.header.stamp = self.ros.get_clock().now().to_msg()
        self.ros.heatmap_pub.publish(heatmap_msg)
        if PUBLISH_PATH:
            self.ros.publish_path()
        self.ros.publish_safety_zone(robot_state["x"], robot_state["y"], nav_state)
        self._publish_goal_markers()

    def _publish_goal_markers(self):
        now = self.ros.get_clock().now().to_msg()
        if self.ros.last_goal is not None:
            m = Marker()
            m.header.stamp = now; m.header.frame_id = "odom"
            m.ns = "goal"; m.id = 0; m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = self.ros.last_goal.point.x
            m.pose.position.y = self.ros.last_goal.point.y
            m.pose.position.z = 0.5; m.pose.orientation.w = 1.0
            m.scale.x = 0.5; m.scale.y = 0.5; m.scale.z = 0.5
            m.color.g = 1.0; m.color.a = 0.8; m.lifetime.sec = 1
            self.ros.goal_marker_pub.publish(m)
        else:
            m = Marker()
            m.header.stamp = now; m.header.frame_id = "odom"
            m.ns = "goal"; m.id = 0; m.action = Marker.DELETE
            self.ros.goal_marker_pub.publish(m)

        if self.ros._last_known_target_pos is not None:
            m = Marker()
            m.header.stamp = now; m.header.frame_id = "odom"
            m.ns = "last_known"; m.id = 0; m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = self.ros._last_known_target_pos[0]
            m.pose.position.y = self.ros._last_known_target_pos[1]
            m.pose.position.z = 0.3; m.pose.orientation.w = 1.0
            m.scale.x = 0.3; m.scale.y = 0.3; m.scale.z = 0.3
            m.color.r = 1.0; m.color.g = 1.0; m.color.a = 0.6; m.lifetime.sec = 1
            self.ros.last_known_marker_pub.publish(m)

    def _build_observation(self) -> np.ndarray:
        st = self._get_robot_state()
        goal = self.ros.last_goal

        lidar_bins = self._get_lidar_bins()
        lidar_norm = np.clip(lidar_bins / LIDAR_MAX_RANGE, 0.0, 1.0)

        if goal is not None:
            dx = goal.point.x - st["x"]
            dy = goal.point.y - st["y"]
            dist = math.hypot(dx, dy)
            ang = self._goal_angle()
            cap = 6.0
            goal_info = np.array([
                np.clip(dx / cap, -1, 1), np.clip(dy / cap, -1, 1),
                np.clip(dist / cap, 0, 1), math.sin(ang), math.cos(ang),
            ], dtype=np.float32)
        else:
            goal_info = np.zeros(5, dtype=np.float32)

        vel = np.array([
            np.clip(st["v_lin"] / V_MAX, -1, 1),
            np.clip(st["v_ang"] / W_MAX, -1, 1),
        ], dtype=np.float32)

        prev_act = self.prev_action.copy()

        has_goal = np.array([1.0 if goal is not None else 0.0], dtype=np.float32)

        grid_flat = self.occ_grid.get_flat_grid()
        grid_norm = (grid_flat - 0.5) * 2.0

        frontier_dir = np.array([
            math.sin(self._cached_frontier_angle),
            math.cos(self._cached_frontier_angle),
        ], dtype=np.float32)

        novelty = np.array([self.occ_grid.get_novelty(st["x"], st["y"])], dtype=np.float32)

        obs = np.concatenate([
            lidar_norm, goal_info, vel, prev_act, has_goal,
            grid_norm, frontier_dir, novelty,
        ], axis=0)

        self.obs_rms.update(obs.reshape(1, -1))
        return obs.astype(np.float32)

    def _get_robot_state(self) -> Dict:
        odom = self.ros.last_odom
        if odom is None:
            return {"x": 0.0, "y": 0.0, "yaw": 0.0, "v_lin": 0.0, "v_ang": 0.0}
        q = odom.pose.pose.orientation
        return {
            "x": float(odom.pose.pose.position.x),
            "y": float(odom.pose.pose.position.y),
            "yaw": float(yaw_from_quat(q.x, q.y, q.z, q.w)),
            "v_lin": float(odom.twist.twist.linear.x),
            "v_ang": float(odom.twist.twist.angular.z),
        }

    def _get_lidar_bins(self) -> np.ndarray:
        scan = self.ros.last_scan
        if scan is None:
            return np.full(NUM_LIDAR_BINS, LIDAR_MAX_RANGE, dtype=np.float32)
        ranges = np.array(scan.ranges, dtype=np.float32)
        max_r = scan.range_max if scan.range_max > 0 else LIDAR_MAX_RANGE
        min_r = max(scan.range_min, 0.01)
        if ranges.size == 0:
            return np.full(NUM_LIDAR_BINS, max_r, dtype=np.float32)
        bad = np.isnan(ranges) | np.isinf(ranges) | (ranges < min_r) | (ranges > max_r)
        ranges[bad] = max_r
        n = ranges.size
        bin_idx = (np.arange(n) * NUM_LIDAR_BINS // n).astype(int)
        bins = np.full(NUM_LIDAR_BINS, max_r, dtype=np.float32)
        for i in range(NUM_LIDAR_BINS):
            m = bin_idx == i
            if np.any(m):
                bins[i] = float(np.min(ranges[m]))
        return bins

    def _goal_distance(self) -> float:
        goal = self.ros.last_goal
        if goal is None: return 0.0
        st = self._get_robot_state()
        return float(math.hypot(goal.point.x - st["x"], goal.point.y - st["y"]))

    def _goal_angle(self) -> float:
        goal = self.ros.last_goal
        if goal is None: return 0.0
        st = self._get_robot_state()
        ang_world = math.atan2(goal.point.y - st["y"], goal.point.x - st["x"])
        return wrap_to_pi(ang_world - st["yaw"])

    def _publish_reward_breakdown(self, reward: float, info: Dict, has_goal: bool):
        self.ros.reward_pub.publish(Float32(data=float(reward)))
        st = self._get_robot_state()
        goal = self.ros.last_goal
        stats = self.occ_grid.get_stats()
        breakdown = {
            "mode": "goal" if has_goal else "explore",
            "reward": float(reward),
            "reward_terms": info.get("reward_terms", {}),
            "collision": info.get("collision", False),
            "success": info.get("success", False),
            "goals_reached": self._goals_reached_this_episode,
            "min_distance": info.get("min_distance", 0.0),
            "nav_zone": info.get("nav_zone", "UNKNOWN"),
            "velocity": {"v": info.get("executed_v", 0.0), "w": info.get("executed_w", 0.0)},
            "state": {
                "x": st["x"], "y": st["y"], "yaw_deg": st["yaw"] * 180 / math.pi,
                "goal_x": goal.point.x if goal else 0, "goal_y": goal.point.y if goal else 0,
                "goal_dist": self._goal_distance(),
            },
            "explore_stats": stats,
            "episode": self.episode_index,
            "step": self.step_count,
        }
        msg = StringMsg()
        msg.data = json.dumps(breakdown)
        self.ros.reward_breakdown_pub.publish(msg)


# =============================================================================
# TD3 Networks (smaller observation space → smaller networks)
# =============================================================================

class GridCNN(nn.Module):
    def __init__(self, grid_size: int = GRID_SIZE, out_features: int = 32):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        conv_out_size = 32 * (grid_size // 4) * (grid_size // 4)
        self.fc = nn.Sequential(nn.Linear(conv_out_size, out_features), nn.ReLU())

    def forward(self, grid_flat: torch.Tensor) -> torch.Tensor:
        batch_size = grid_flat.shape[0]
        grid = grid_flat.view(batch_size, 1, self.grid_size, self.grid_size)
        return self.fc(self.conv(grid))


class ActorWithCNN(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, grid_size: int = GRID_SIZE,
                 grid_features: int = 32, hidden: int = 256):
        super().__init__()
        self.grid_size = grid_size
        grid_flat_size = grid_size * grid_size
        self.non_grid_size = obs_dim - grid_flat_size
        self.grid_start = NUM_LIDAR_BINS + 5 + 2 + 2 + 1  # 46
        self.grid_end = self.grid_start + grid_flat_size
        self.grid_cnn = GridCNN(grid_size, grid_features)
        combined_size = self.non_grid_size + grid_features
        self.mlp = nn.Sequential(
            nn.Linear(combined_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        non_grid_before = obs[:, :self.grid_start]
        grid_flat = obs[:, self.grid_start:self.grid_end]
        non_grid_after = obs[:, self.grid_end:]
        grid_features = self.grid_cnn(grid_flat)
        combined = torch.cat([non_grid_before, non_grid_after, grid_features], dim=-1)
        return self.mlp(combined)


class CriticWithCNN(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, grid_size: int = GRID_SIZE,
                 grid_features: int = 32, hidden: int = 256):
        super().__init__()
        self.grid_size = grid_size
        grid_flat_size = grid_size * grid_size
        self.non_grid_size = obs_dim - grid_flat_size
        self.grid_start = NUM_LIDAR_BINS + 5 + 2 + 2 + 1
        self.grid_end = self.grid_start + grid_flat_size
        self.grid_cnn = GridCNN(grid_size, grid_features)
        combined_size = self.non_grid_size + grid_features + act_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        non_grid_before = obs[:, :self.grid_start]
        grid_flat = obs[:, self.grid_start:self.grid_end]
        non_grid_after = obs[:, self.grid_end:]
        grid_features = self.grid_cnn(grid_flat)
        combined = torch.cat([non_grid_before, non_grid_after, grid_features, act], dim=-1)
        return self.mlp(combined)


# =============================================================================
# TD3 Agent (no RND dependency)
# =============================================================================

class TD3AgentCNN:
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device,
                 gamma: float = 0.99, tau: float = 0.005,
                 actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_delay: int = 2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.actor = ActorWithCNN(obs_dim, act_dim).to(device)
        self.actor_targ = ActorWithCNN(obs_dim, act_dim).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.critic1 = CriticWithCNN(obs_dim, act_dim).to(device)
        self.critic2 = CriticWithCNN(obs_dim, act_dim).to(device)
        self.critic1_targ = CriticWithCNN(obs_dim, act_dim).to(device)
        self.critic2_targ = CriticWithCNN(obs_dim, act_dim).to(device)
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )
        self.total_updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        a = self.actor(o).squeeze(0).cpu().numpy()
        if noise_std > 0:
            a = a + np.random.normal(0, noise_std, size=a.shape).astype(np.float32)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def update(self, replay: PrioritizedReplayBuffer, batch_size: int, beta: float) -> Tuple[float, float]:
        self.total_updates += 1
        obs, act, rew, next_obs, done, weights, indices = replay.sample(batch_size, beta)

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_targ(next_obs) + noise).clamp(-1.0, 1.0)
            q1_t = self.critic1_targ(next_obs, next_act)
            q2_t = self.critic2_targ(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t)
            target = rew + (1.0 - done) * self.gamma * q_t

        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        td_error1 = target - q1
        td_error2 = target - q2
        critic_loss = (weights * (td_error1.pow(2) + td_error2.pow(2))).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic_opt.step()

        td_errors = (td_error1.abs() + td_error2.abs()).detach().cpu().numpy() / 2.0
        replay.update_priorities(indices, td_errors.flatten())

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_updates % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            self._soft_update(self.actor_targ, self.actor)
            self._soft_update(self.critic1_targ, self.critic1)
            self._soft_update(self.critic2_targ, self.critic2)

        return float(critic_loss.item()), float(actor_loss.item())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    def save(self, path: str, extra_state: Optional[Dict] = None):
        payload = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "actor_targ": self.actor_targ.state_dict(),
            "critic1_targ": self.critic1_targ.state_dict(),
            "critic2_targ": self.critic2_targ.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_updates": self.total_updates,
        }
        if extra_state is not None:
            payload["training_state"] = extra_state
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True) -> Optional[Dict]:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"], strict=strict)
        self.critic1.load_state_dict(payload["critic1"], strict=strict)
        self.critic2.load_state_dict(payload["critic2"], strict=strict)
        self.actor_targ.load_state_dict(payload["actor_targ"], strict=strict)
        self.critic1_targ.load_state_dict(payload["critic1_targ"], strict=strict)
        self.critic2_targ.load_state_dict(payload["critic2_targ"], strict=strict)
        try:
            self.actor_opt.load_state_dict(payload["actor_opt"])
            self.critic_opt.load_state_dict(payload["critic_opt"])
            self.total_updates = payload.get("total_updates", 0)
        except Exception:
            pass
        return payload.get("training_state", None)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ns", type=str, default="")
    parser.add_argument("--odom-topic", type=str, default="/stretch/odom")
    parser.add_argument("--lidar-topic", type=str, default="/stretch/scan")
    parser.add_argument("--imu-topic", type=str, default="/imu_mobile_base")
    parser.add_argument("--goal-topic", type=str, default="goal")
    parser.add_argument("--cmd-topic", type=str, default="/stretch/cmd_vel")

    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--start-steps", type=int, default=DEFAULT_START_STEPS)
    parser.add_argument("--update-after", type=int, default=2000)
    parser.add_argument("--update-every", type=int, default=DEFAULT_UPDATE_EVERY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--replay-size", type=int, default=DEFAULT_REPLAY_SIZE)
    parser.add_argument("--expl-noise", type=float, default=DEFAULT_EXPL_NOISE)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--demo-steps", type=int, default=DEFAULT_DEMO_STEPS)

    parser.add_argument("--ckpt-dir", type=str, default=os.path.expanduser("~/parallel_training"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--inference", action="store_true")

    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--load-ckpt", type=str, default="")
    parser.add_argument("--use-obstacle", type=int, default=1)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--episode-num", type=int, default=1)
    parser.add_argument("--models-dir", type=str, default="./models")

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, CHECKPOINT_FILENAME)

    rclpy.init()
    ros = StretchRosInterface(
        ns=args.ns, odom_topic=args.odom_topic, scan_topic=args.lidar_topic,
        imu_topic=args.imu_topic, goal_topic=args.goal_topic, cmd_topic=args.cmd_topic,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(ros)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid_flat_size = GRID_SIZE * GRID_SIZE
    obs_dim = NUM_LIDAR_BINS + 5 + 2 + 2 + 1 + grid_flat_size + 2 + 1
    act_dim = 2

    env = StretchExploreEnv(ros)

    ros.get_logger().info(f"[AGENT] device={device} obs_dim={obs_dim} act_dim={act_dim}")
    ros.get_logger().info(f"[AGENT] v5: explore_velocity={V_MIN_FORWARD}-{V_MAX}, goal_velocity=0-{V_MAX}")
    ros.get_logger().info(f"[AGENT] Expert demo phase: {args.demo_steps} steps, then RL takes over")
    ros.get_logger().info(f"[AGENT] Collisions never terminate (bump={R_COLLISION}, cooldown={COLLISION_COOLDOWN_STEPS})")
    ros.get_logger().info(f"[AGENT] Explore: discovery({R_NEW_CELL}) + revisit({R_REVISIT}) + step({R_STEP_EXPLORE})")
    ros.get_logger().info(f"[AGENT] Goal: progress({PROGRESS_SCALE}) + step({STEP_COST_GOAL}) | Proximity: {R_PROXIMITY} below {PROXIMITY_THRESHOLD}m")

    agent = TD3AgentCNN(obs_dim, act_dim, device=device)
    replay = PrioritizedReplayBuffer(obs_dim, act_dim, size=args.replay_size, device=device)

    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; W = "\033[97m"; RST = "\033[0m"

    resume_step = 0
    replay_path = os.path.join(args.ckpt_dir, "replay_buffer.npz")

    if AUTO_LOAD_CHECKPOINT and os.path.exists(ckpt_path):
        file_size_mb = os.path.getsize(ckpt_path) / 1e6
        ros.get_logger().info(f"[CKPT] Found checkpoint ({file_size_mb:.1f} MB) — loading...")
        try:
            training_state = agent.load(ckpt_path, strict=False)
            ros.get_logger().info(f"[CKPT] {G}✓ Network weights loaded{RST}")

            if training_state is not None:
                resume_step = training_state.get("step", 0)
                env.episode_index = training_state.get("episode_index", 1)
                env.total_steps = training_state.get("total_steps", 0)
                if "obs_rms_mean" in training_state:
                    env.obs_rms.mean = training_state["obs_rms_mean"]
                    env.obs_rms.var = training_state["obs_rms_var"]
                    env.obs_rms.count = training_state["obs_rms_count"]
                ros.get_logger().info(
                    f"[CKPT] {G}✓ Restored:{RST} step={resume_step}, "
                    f"episode={env.episode_index}"
                )
            else:
                ros.get_logger().warn(f"[CKPT] {Y}No training state — starting counters from 0{RST}")

            replay_loaded = False
            for rp in [replay_path, replay_path + ".tmp.npz"]:
                if os.path.exists(rp) and replay.load(rp):
                    ros.get_logger().info(f"[CKPT] {G}✓ Replay: {replay.count} transitions{RST}")
                    if rp != replay_path:
                        try: os.replace(rp, replay_path)
                        except: pass
                    replay_loaded = True
                    break
            if not replay_loaded:
                ros.get_logger().warn(f"[CKPT] {Y}No replay buffer — starting empty{RST}")

        except Exception as e:
            ros.get_logger().error(f"[CKPT] {R}Load failed: {e} — starting fresh{RST}")
    else:
        ros.get_logger().info(f"[CKPT] {Y}No checkpoint found — starting fresh{RST}")

    last_replay_save_step = resume_step

    def save_weights_only(step: int, label: str = "episode"):
        training_state = {
            "step": step,
            "episode_index": env.episode_index,
            "total_steps": env.total_steps,
            "obs_rms_mean": env.obs_rms.mean,
            "obs_rms_var": env.obs_rms.var,
            "obs_rms_count": env.obs_rms.count,
        }
        try:
            agent.save(ckpt_path + ".tmp", extra_state=training_state)
            os.replace(ckpt_path + ".tmp", ckpt_path)
        except Exception as e:
            ros.get_logger().error(f"[CKPT] {R}Agent save failed: {e}{RST}")

    def save_replay_buffer():
        try:
            replay_tmp = replay_path.replace(".npz", "_tmp.npz")
            replay.save(replay_tmp)
            if not os.path.exists(replay_tmp) and os.path.exists(replay_tmp + ".npz"):
                replay_tmp = replay_tmp + ".npz"
            os.replace(replay_tmp, replay_path)
        except Exception as e:
            ros.get_logger().warn(f"[CKPT] {Y}Replay save failed: {e}{RST}")

    def save_full_checkpoint(step: int, label: str = "periodic"):
        nonlocal last_replay_save_step
        save_weights_only(step, label)
        save_replay_buffer()
        last_replay_save_step = step
        ros.get_logger().info(
            f"[CKPT] {G}✓ {label} full save{RST} step={step} ep={env.episode_index} replay={replay.count}"
        )

    def shutdown_and_save(signum=None, frame=None):
        try: ros.send_cmd(0.0, 0.0)
        except: pass
        current_step = getattr(shutdown_and_save, '_current_step', resume_step)
        save_full_checkpoint(current_step, label="shutdown")
        try:
            executor.shutdown()
            ros.destroy_node()
            rclpy.shutdown()
        except: pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_and_save)
    signal.signal(signal.SIGTERM, shutdown_and_save)

    if args.inference:
        ros.get_logger().info("[MODE] INFERENCE")
        while True:
            obs, _ = env.reset()
            done = False
            while not done:
                act = agent.act(obs, noise_std=0.0)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc

    start_step = resume_step + 1
    demo_steps = args.demo_steps
    expert = ExpertStateMachine()

    # Determine if we're resuming past the demo phase
    demo_already_done = (start_step > demo_steps)

    ros.get_logger().info(
        f"\n{G}{'='*50}\n  TRAINING STARTED\n{'='*50}{RST}\n"
        f"  Resume from: step {start_step}\n"
        f"  Replay: {replay.count} transitions\n"
        f"  Target: {args.total_steps} steps\n"
        f"  Expert demo: {'SKIPPED (already done)' if demo_already_done else f'first {demo_steps} steps'}\n"
        f"  RL random exploration: steps {demo_steps+1}–{demo_steps + args.start_steps}\n"
        f"  RL policy active from: step {demo_steps + args.start_steps}\n"
        f"  Update every: {args.update_every} steps\n"
        f"  Batch size: {args.batch_size}\n"
        f"{G}{'='*50}{RST}"
    )

    obs, _ = env.reset()
    if not demo_already_done:
        expert.reset_episode()

    for t in range(start_step, args.total_steps + 1):
        shutdown_and_save._current_step = t

        # === Phase 1: Expert state machine drives (first demo_steps steps) ===
        if t <= demo_steps:
            # Get nav state and goal info for the expert
            nav_state = env.last_nav_state
            has_goal = env.ros.last_goal is not None
            goal_angle = env._goal_angle() if has_goal else 0.0
            goal_dist = env._goal_distance() if has_goal else 0.0

            act = expert.act(nav_state, goal_angle, goal_dist, has_goal)

        # === Phase 2: Random exploration (demo_steps < t < demo_steps + start_steps) ===
        elif t < demo_steps + args.start_steps:
            act = np.array([
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-0.8, 0.8),
            ], dtype=np.float32)

        # === Phase 3: RL policy ===
        else:
            act = agent.act(obs, noise_std=args.expl_noise)

        next_obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        # Track expert rewards for baseline comparison
        if t <= demo_steps:
            expert.track_reward(reward, success=info.get("success", False))

        replay.add(
            obs, act,
            np.array([reward], dtype=np.float32),
            next_obs,
            np.array([1.0 if done else 0.0], dtype=np.float32),
        )
        obs = next_obs

        if done:
            # Log expert vs RL comparison
            if t <= demo_steps:
                expert_avg = expert.get_avg_return()
                ros.get_logger().info(
                    f"[EXPERT] ep={expert.total_episodes} "
                    f"return={expert.episode_return:+.1f} "
                    f"avg_return={expert_avg:+.1f} "
                    f"goals={expert.total_goals_reached} "
                    f"state={expert.state.value}"
                )
                expert.reset_episode()
            elif t > demo_steps and expert.total_episodes > 0:
                # After expert phase, log RL vs expert baseline periodically
                if env.episode_index % 10 == 0:
                    expert_avg = expert.get_avg_return()
                    rl_avg = env.episode_return  # current episode return
                    ros.get_logger().info(
                        f"{B}[COMPARE]{RST} Expert baseline avg: {Y}{expert_avg:+.1f}{RST} | "
                        f"RL episode return: {G if rl_avg > expert_avg else R}{rl_avg:+.1f}{RST} | "
                        f"{'RL WINNING' if rl_avg > expert_avg else 'Expert still better'}"
                    )

            obs, _ = env.reset()
            if t <= demo_steps:
                expert.reset_episode()

            save_weights_only(t, label="episode")
            if t - last_replay_save_step >= args.save_every:
                save_replay_buffer()
                last_replay_save_step = t
                ros.get_logger().info(
                    f"[CKPT] {G}✓ replay buffer saved{RST} step={t} count={replay.count}"
                )

        # Log phase transition
        if t == demo_steps:
            expert_avg = expert.get_avg_return()
            ros.get_logger().info(
                f"\n{G}{'='*50}\n  EXPERT DEMO PHASE COMPLETE\n{'='*50}{RST}\n"
                f"  Expert episodes: {expert.total_episodes}\n"
                f"  Expert avg return: {expert_avg:+.1f}\n"
                f"  Expert goals reached: {expert.total_goals_reached}\n"
                f"  Replay buffer: {replay.count} transitions (all expert)\n"
                f"  Switching to: random exploration for {args.start_steps} steps\n"
                f"{G}{'='*50}{RST}"
            )
        elif t == demo_steps + args.start_steps:
            ros.get_logger().info(
                f"\n{G}{'='*50}\n  RL POLICY NOW ACTIVE\n{'='*50}{RST}\n"
                f"  Expert baseline avg return: {expert.get_avg_return():+.1f}\n"
                f"  Replay buffer: {replay.count} transitions\n"
                f"  RL policy taking over with noise={args.expl_noise}\n"
                f"{G}{'='*50}{RST}"
            )

        # Don't start updating until we have enough data (expert demos count!)
        update_after_step = min(args.update_after, demo_steps)
        if t >= update_after_step and t % args.update_every == 0 and replay.count >= args.batch_size:
            beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * (t / args.total_steps)
            critic_loss, actor_loss = agent.update(replay, args.batch_size, beta)

    shutdown_and_save()


if __name__ == "__main__":
    main()