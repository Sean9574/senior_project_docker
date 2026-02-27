#!/usr/bin/env python3
"""
Stretch Robot RL Environment + Learner

The RL agent has FULL CONTROL — no hard-coded safety overrides.
It learns to navigate, avoid obstacles, explore, and reach goals
purely through reward signals and episode resets.

EPISODE TERMINATION:
- COLLISION: min LIDAR distance < 0.30m → negative reward, reset
- GOAL REACHED: within 0.45m of goal → positive reward, reset
- TIMEOUT: max steps exceeded → small negative, reset

Each reset calls /sim/reset to teleport the robot back to start.
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

ROBOT_WIDTH_M = 0.33  # 13 inches in meters
ROBOT_HALF_WIDTH_M = 0.165
DESIRED_CLEARANCE_M = 0.25  # Moderate - can get close but not touch
MIN_SAFE_DISTANCE = ROBOT_HALF_WIDTH_M + DESIRED_CLEARANCE_M  # ~0.41m

# =============================================================================
# SAFETY ZONE THRESHOLDS (from LIDAR/center of robot)
# REACT EARLY but GENTLY - don't wait until too close
# =============================================================================

ZONE_FREE = 1.2         # Full RL control
ZONE_AWARE = 0.90       # Start monitoring
ZONE_CAUTION = 0.65     # Light guidance begins - early enough to steer
ZONE_DANGER = 0.45      # Stronger guidance - actively avoiding
ZONE_EMERGENCY = 0.30   # Hard override - last resort

# =============================================================================
# REWARD SHAPING PARAMETERS
# =============================================================================

# =============================================================================
# CURRICULUM LEARNING — the key insight:
# The agent must learn WHAT TO DO before learning WHAT NOT TO DO.
#
# Phase 1 (0-30k steps): "Learn to move and explore"
#   - Collisions DON'T end the episode, just a bump penalty
#   - Agent builds positive experience: exploration, goal-seeking
#   - No collision trauma → no degenerate avoidance strategies
#
# Phase 2 (30k-80k steps): "Learn that collisions matter"
#   - Collisions END the episode with moderate penalty
#   - Agent already knows exploration is rewarding
#   - Now refines behavior to avoid obstacles while exploring
#
# Phase 3 (80k+ steps): "Full difficulty"
#   - Full collision penalty
#   - Agent is skilled enough to navigate without fear
# =============================================================================

CURRICULUM_PHASE1_END = 30_000    # Steps: collisions don't end episode
CURRICULUM_PHASE2_END = 80_000    # Steps: moderate collision penalty

R_COLLISION_PHASE1 = -3.0         # Bump penalty (episode continues) — must be < forward bonus so agent still wants to move
R_COLLISION_PHASE2 = -50.0        # Moderate (episode ends)
R_COLLISION_PHASE3 = -100.0       # Full penalty (episode ends)

R_GOAL = 2000.0                   # Reaching goal -> episode ends
R_TIMEOUT = -50.0                 # Episode timeout -> episode ends

GOAL_RADIUS = 0.45                # Distance to count as "reached"

# =============================================================================
# GOAL-SEEKING REWARDS (when goal is visible)
# =============================================================================

PROGRESS_SCALE = 400.0         # Reward for getting closer to goal
ALIGN_SCALE = 10.0             # Reward for facing goal (only when moving forward)
STEP_COST = -1.5               # Per-step cost with goal (was -1.0 — creeping was too cheap)

# =============================================================================
# EXPLORATION REWARDS (when no goal visible)
# =============================================================================

R_NEW_CELL = 3.0               # Per new cell discovered
R_NOVELTY_SCALE = 0.8          # Bonus for unvisited areas
R_FRONTIER_BONUS = 8.0         # Steering toward frontiers (always-on with sign)
R_REVISIT_SCALE = 0.5          # Per extra visit beyond 2, up to -5.0/step
R_STEP_EXPLORE = -0.3          # Step cost during exploration (was -0.1 — no urgency)

# =============================================================================
# SHAPING REWARDS (always active)
# =============================================================================

R_FORWARD_SCALE = 6.0          # Forward bonus scales with speed — MUST dominate collision penalty
                                # At v=0.3:  +1.44/step
                                # At v=0.7:  +3.36/step
                                # At v=1.25: +6.0/step (full bonus)
R_STUCK_PENALTY = -5.0         # Penalty for not covering ground (displacement-based, not velocity)
R_SPIN_PENALTY = -4.0          # Penalty for spinning in place (high ω, low v)
R_PROXIMITY_BONUS = 0.0        # DISABLED — was rewarding wall-hugging
PROXIMITY_SWEET_SPOT = 0.6     # Ideal distance from nearest obstacle (meters)
PROXIMITY_RANGE = 0.3          # Gaussian width

# =============================================================================
# GENERAL CONFIG
# =============================================================================

CHECKPOINT_FILENAME = "td3_safe_rl_agent.pt"
AUTO_LOAD_CHECKPOINT = True

EPISODE_SECONDS = 60.0

# Occupancy Grid
GRID_SIZE = 24
GRID_RESOLUTION = 0.5
GRID_MAX_RANGE = 12.0

# Movement Limits
V_MAX = 1.25
W_MAX = 3.0
V_MIN_REVERSE = -0.05
MIN_TURN_RADIUS = 0.20          # Physical constraint — real robots have turn limits
                                # Prevents 5cm death spirals but allows corner navigation
                                # Agent CAN stop (v=0), but turn radius applies when moving

# Velocity smoothing — prevents wild swings between steps
# EMA: smoothed = α * new + (1-α) * previous
# α=0.6 — still some smoothing but agent can respond within 2-3 steps
CMD_SMOOTHING_ALPHA = 0.6      # 0.0 = no change ever, 1.0 = no smoothing

# RND Config
RND_WEIGHT_INITIAL = 1.0
RND_WEIGHT_DECAY = 0.99995
RND_WEIGHT_MIN = 0.1
RND_FEATURE_DIM = 64
RND_LR = 1e-4

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
NUM_LIDAR_BINS = 60
LIDAR_MAX_RANGE = 20.0

# Number of angular sectors for safety analysis
NUM_SAFETY_SECTORS = 12

# Safety observation size: sectors(12) + blend(1) + danger_direction(2) = 15
SAFETY_OBS_SIZE = NUM_SAFETY_SECTORS + 3

# Training
DEFAULT_START_STEPS = 10000
DEFAULT_EXPL_NOISE = 0.3

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
    """Linear interpolation between a and b by factor t."""
    return a + (b - a) * np.clip(t, 0.0, 1.0)



# =============================================================================
# DYNAMIC NAVIGATION SYSTEM - Simple & Robust Obstacle Avoidance
# =============================================================================

@dataclass
class NavigationState:
    """Current navigation assessment."""
    min_distance: float           # Closest obstacle distance
    min_angle: float              # Angle to closest obstacle (robot frame, 0=front)
    left_clearance: float         # Minimum clearance on left side
    right_clearance: float        # Minimum clearance on right side  
    front_clearance: float        # Minimum clearance in front
    back_clearance: float         # Minimum clearance behind
    sector_distances: np.ndarray  # Distance in each sector (for observation)
    safety_blend: float           # How much to blend safety (0-1)
    zone: str                     # Human-readable zone
    clear_gaps: List[Tuple[float, float, float]]  # Compatibility
    best_direction: float         # Compatibility
    best_clearance: float         # Compatibility


class DynamicNavigator:
    """
    LIDAR scan analyzer — provides obstacle awareness for the RL observation.
    
    This does NOT control the robot. It just processes LIDAR data into:
    - Sector distances (for observation vector)
    - Zone classification (for logging/visualization)
    - Safety blend metric (for observation vector)
    - Clearance in 4 quadrants
    
    The RL agent sees this data and learns to avoid obstacles on its own.
    """
    
    def __init__(self, num_sectors: int = 36):
        self.num_sectors = num_sectors
        self.sector_width = 2 * math.pi / num_sectors
        
        # Sector center angles (robot frame, 0 = forward)
        self.sector_angles = np.array([
            wrap_to_pi(-math.pi + (i + 0.5) * self.sector_width)
            for i in range(num_sectors)
        ])
        
        self.last_nav_state: Optional[NavigationState] = None
    
    def analyze_scan(self, scan: Optional[LaserScan]) -> NavigationState:
        """Analyze LIDAR and compute simple clearance metrics."""
        if scan is None or len(scan.ranges) == 0:
            return self._default_state()
        
        ranges = np.array(scan.ranges, dtype=np.float32)
        n_rays = len(ranges)
        
        # Clean invalid readings
        max_r = min(scan.range_max, LIDAR_MAX_RANGE) if scan.range_max > 0 else LIDAR_MAX_RANGE
        min_r = max(scan.range_min, 0.05)
        invalid = np.isnan(ranges) | np.isinf(ranges) | (ranges < min_r) | (ranges > max_r)
        ranges[invalid] = max_r
        
        # Calculate angles (robot frame, 0 = forward)
        angles = scan.angle_min + np.arange(n_rays) * scan.angle_increment + LIDAR_FORWARD_OFFSET_RAD
        angles = np.array([wrap_to_pi(a) for a in angles])
        
        # Find minimum distance and angle
        min_idx = np.argmin(ranges)
        min_distance = float(ranges[min_idx])
        min_angle = float(angles[min_idx])
        
        # Compute clearance in 4 quadrants (simple and robust)
        front_mask = np.abs(angles) < math.pi / 4  # ±45°
        back_mask = np.abs(angles) > 3 * math.pi / 4  # ±135-180°
        left_mask = (angles > math.pi / 4) & (angles < 3 * math.pi / 4)  # 45-135°
        right_mask = (angles < -math.pi / 4) & (angles > -3 * math.pi / 4)  # -45 to -135°
        
        front_clearance = float(np.min(ranges[front_mask])) if np.any(front_mask) else max_r
        back_clearance = float(np.min(ranges[back_mask])) if np.any(back_mask) else max_r
        left_clearance = float(np.min(ranges[left_mask])) if np.any(left_mask) else max_r
        right_clearance = float(np.min(ranges[right_mask])) if np.any(right_mask) else max_r
        
        # Build sector histogram for observation
        sector_distances = np.full(self.num_sectors, max_r, dtype=np.float32)
        for i in range(n_rays):
            sector_idx = int((angles[i] + math.pi) / self.sector_width) % self.num_sectors
            sector_distances[sector_idx] = min(sector_distances[sector_idx], ranges[i])
        
        # Safety blend and zone
        safety_blend = self._compute_safety_blend(min_distance)
        zone = self._get_zone_name(min_distance)
        
        # Compute best direction for compatibility
        if left_clearance > right_clearance:
            best_direction = math.pi / 2
            best_clearance = left_clearance
        else:
            best_direction = -math.pi / 2
            best_clearance = right_clearance
        
        state = NavigationState(
            min_distance=min_distance,
            min_angle=min_angle,
            left_clearance=left_clearance,
            right_clearance=right_clearance,
            front_clearance=front_clearance,
            back_clearance=back_clearance,
            sector_distances=sector_distances,
            safety_blend=safety_blend,
            zone=zone,
            clear_gaps=[],
            best_direction=best_direction,
            best_clearance=best_clearance
        )
        
        self.last_nav_state = state
        return state
    
    def _default_state(self) -> NavigationState:
        return NavigationState(
            min_distance=LIDAR_MAX_RANGE,
            min_angle=0.0,
            left_clearance=LIDAR_MAX_RANGE,
            right_clearance=LIDAR_MAX_RANGE,
            front_clearance=LIDAR_MAX_RANGE,
            back_clearance=LIDAR_MAX_RANGE,
            sector_distances=np.full(self.num_sectors, LIDAR_MAX_RANGE, dtype=np.float32),
            safety_blend=0.0,
            zone="FREE",
            clear_gaps=[],
            best_direction=0.0,
            best_clearance=LIDAR_MAX_RANGE
        )
    
    def _compute_safety_blend(self, min_distance: float) -> float:
        if min_distance >= ZONE_FREE:
            return 0.0
        if min_distance <= ZONE_EMERGENCY:
            return 0.95
        t = (min_distance - ZONE_EMERGENCY) / (ZONE_FREE - ZONE_EMERGENCY)
        return float(0.95 * (1.0 - t) ** 2)
    
    def _get_zone_name(self, min_distance: float) -> str:
        if min_distance >= ZONE_FREE:
            return "FREE"
        elif min_distance >= ZONE_AWARE:
            return "AWARE"
        elif min_distance >= ZONE_CAUTION:
            return "CAUTION"
        elif min_distance >= ZONE_DANGER:
            return "DANGER"
        elif min_distance >= ZONE_EMERGENCY:
            return "EMERGENCY"
        else:
            return "CRITICAL"
    


# =============================================================================
# Running Mean/Std for Normalization (unchanged)
# =============================================================================

class RunningMeanStd:
    """Tracks running mean and standard deviation for normalization."""
    
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
    """Normalizes rewards to have roughly unit variance."""
    
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
# Random Network Distillation (RND) - unchanged
# =============================================================================

class RNDModule(nn.Module):
    """Random Network Distillation for intrinsic motivation."""
    
    def __init__(self, obs_dim: int, hidden: int = 256, 
                 feature_dim: int = RND_FEATURE_DIM, device: torch.device = None):
        super().__init__()
        
        self.device = device or torch.device("cpu")
        
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim)
        ).to(self.device)
        
        for param in self.target.parameters():
            param.requires_grad = False
        
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=RND_LR)
        
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd(shape=())
    
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(self.obs_rms.normalize(obs), -5.0, 5.0)
    
    @torch.no_grad()
    def compute_intrinsic_reward(self, obs: np.ndarray) -> float:
        self.obs_rms.update(obs.reshape(1, -1))
        
        obs_norm = self.normalize_obs(obs)
        obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.device)
        
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        target_features = self.target(obs_tensor)
        predictor_features = self.predictor(obs_tensor)
        
        intrinsic_reward = (target_features - predictor_features).pow(2).mean().item()
        
        self.reward_rms.update(np.array([intrinsic_reward]))
        normalized_reward = intrinsic_reward / np.sqrt(self.reward_rms.var + 1e-8)
        
        return float(normalized_reward)
    
    def update(self, obs_batch: torch.Tensor) -> float:
        obs_np = obs_batch.cpu().numpy()
        obs_norm = np.clip(
            (obs_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8),
            -5.0, 5.0
        )
        obs_norm_tensor = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            target_features = self.target(obs_norm_tensor)
        
        predictor_features = self.predictor(obs_norm_tensor)
        loss = F.mse_loss(predictor_features, target_features)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()
        
        return float(loss.item())
    
    def state_dict_custom(self) -> Dict:
        return {
            "predictor": self.predictor.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
            "reward_rms_mean": self.reward_rms.mean,
            "reward_rms_var": self.reward_rms.var,
            "reward_rms_count": self.reward_rms.count,
        }
    
    def load_state_dict_custom(self, state_dict: Dict):
        self.predictor.load_state_dict(state_dict["predictor"])
        self.target.load_state_dict(state_dict["target"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.obs_rms.mean = state_dict["obs_rms_mean"]
        self.obs_rms.var = state_dict["obs_rms_var"]
        self.obs_rms.count = state_dict["obs_rms_count"]
        self.reward_rms.mean = state_dict["reward_rms_mean"]
        self.reward_rms.var = state_dict["reward_rms_var"]
        self.reward_rms.count = state_dict["reward_rms_count"]


# =============================================================================
# Prioritized Experience Replay (PER) - unchanged
# =============================================================================

class SumTree:
    """Binary sum tree for efficient priority-based sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
    
    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
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
    """Prioritized Experience Replay buffer."""
    
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
    
    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        
        priority = self.max_priority ** self.alpha
        self.tree.add(priority)
        
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)
    
    def sample(self, batch_size: int, beta: float = PER_BETA_START) -> Tuple:
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            
            tree_idx, data_idx, priority = self.tree.get(value)
            indices[i] = data_idx
            priorities[i] = priority
        
        probs = priorities / self.tree.total_priority
        weights = (self.count * probs) ** (-beta)
        weights = weights / weights.max()
        
        return (
            torch.as_tensor(self.obs[indices], device=self.device),
            torch.as_tensor(self.acts[indices], device=self.device),
            torch.as_tensor(self.rews[indices], device=self.device),
            torch.as_tensor(self.next_obs[indices], device=self.device),
            torch.as_tensor(self.done[indices], device=self.device),
            torch.as_tensor(weights, device=self.device, dtype=torch.float32),
            indices
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + PER_EPSILON) ** self.alpha
            
            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)
            
            self.max_priority = max(self.max_priority, abs(td_error) + PER_EPSILON)
    
    def save(self, path: str):
        """Save replay buffer data to disk."""
        np.savez_compressed(
            path,
            obs=self.obs[:self.count],
            acts=self.acts[:self.count],
            rews=self.rews[:self.count],
            next_obs=self.next_obs[:self.count],
            done=self.done[:self.count],
            ptr=self.ptr,
            count=self.count,
        )
    
    def load(self, path: str):
        """Load replay buffer data from disk."""
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            count = int(data["count"])
            if count == 0:
                return False
            
            # Restore data (handle size mismatch if replay_size changed)
            n = min(count, self.size)
            self.obs[:n] = data["obs"][:n]
            self.acts[:n] = data["acts"][:n]
            self.rews[:n] = data["rews"][:n]
            self.next_obs[:n] = data["next_obs"][:n]
            self.done[:n] = data["done"][:n]
            self.ptr = int(data["ptr"]) % self.size
            self.count = n
            
            # Rebuild sum tree with uniform priorities
            self.tree = SumTree(self.size)
            for i in range(n):
                self.tree.add(self.max_priority ** self.alpha)
            
            return True
        except Exception:
            return False


# =============================================================================
# Ego-Centric Occupancy Grid - unchanged
# =============================================================================

class EgoOccupancyGrid:
    """Ego-centric occupancy grid that rotates with the robot."""
    
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
        
        for i in range(0, n_rays, 3):
            original_r = ranges[i]
            
            if np.isnan(original_r) or np.isinf(original_r) or original_r < range_min:
                continue
            
            # Clamp for grid update, but remember original for obstacle detection
            r = min(original_r, range_max)
            ray_angle_world = robot_yaw + angle_min + i * angle_inc + LIDAR_FORWARD_OFFSET_RAD
            
            # Mark cells along the ray as FREE
            step = self.resolution * 0.4
            for d in np.arange(step, r, step):
                wx = robot_x + d * math.cos(ray_angle_world)
                wy = robot_y + d * math.sin(ray_angle_world)
                
                world_key = self.world_to_grid_key(wx, wy)
                if world_key not in self.world_grid:
                    self.world_grid[world_key] = 0.5
                    self.total_cells_discovered += 1
                    self.cells_discovered_this_step += 1
                
                ego_x, ego_y = self._world_to_ego(wx, wy, robot_x, robot_y, robot_yaw)
                gx = int(ego_x / self.resolution) + self.half_size
                gy = int(ego_y / self.resolution) + self.half_size
                
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    self.grid[gy, gx] = 0.5
            
            # Only mark endpoint as OBSTACLE if:
            # 1. Original reading was well below sensor max (actually hit something)
            # 2. Reading is within our grid range
            is_real_obstacle = (original_r < sensor_max * 0.95) and (original_r < range_max - 0.1)
            
            if is_real_obstacle:
                wx = robot_x + r * math.cos(ray_angle_world)
                wy = robot_y + r * math.sin(ray_angle_world)
                
                world_key = self.world_to_grid_key(wx, wy)
                if world_key not in self.world_grid:
                    self.total_cells_discovered += 1
                    self.cells_discovered_this_step += 1
                self.world_grid[world_key] = 1.0
                
                ego_x, ego_y = self._world_to_ego(wx, wy, robot_x, robot_y, robot_yaw)
                gx = int(ego_x / self.resolution) + self.half_size
                gy = int(ego_y / self.resolution) + self.half_size
                
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    self.grid[gy, gx] = 1.0
        
        return self.cells_discovered_this_step
    
    def _world_to_ego(self, wx: float, wy: float, 
                      robot_x: float, robot_y: float, robot_yaw: float) -> Tuple[float, float]:
        dx = wx - robot_x
        dy = wy - robot_y
        angle = -robot_yaw + math.pi / 2
        ego_x = dx * math.cos(angle) - dy * math.sin(angle)
        ego_y = dx * math.sin(angle) + dy * math.cos(angle)
        return ego_x, ego_y
    
    def get_novelty(self, robot_x: float, robot_y: float) -> float:
        robot_key = self.world_to_grid_key(robot_x, robot_y)
        visit_count = self.visit_counts.get(robot_key, 0)
        
        total_visits = visit_count
        cells_checked = 1
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                key = (robot_key[0] + dx, robot_key[1] + dy)
                total_visits += self.visit_counts.get(key, 0)
                cells_checked += 1
        
        avg_visits = total_visits / cells_checked
        novelty = math.exp(-avg_visits * 0.1)
        return float(novelty)
    
    def get_frontier_direction(self, robot_x: float, robot_y: float, robot_yaw: float) -> float:
        robot_key = self.world_to_grid_key(robot_x, robot_y)
        
        best_frontier = None
        best_dist = float('inf')
        search_radius = 20
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                key = (robot_key[0] + dx, robot_key[1] + dy)
                
                if self.world_grid.get(key, 0) != 0.5:
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
        angle_robot = wrap_to_pi(angle_world - robot_yaw)
        
        return float(angle_robot)
    
    def get_flat_grid(self) -> np.ndarray:
        return self.grid.flatten()
    
    def decay_visits(self):
        for key in self.visit_counts:
            self.visit_counts[key] *= VISIT_DECAY
    
    def get_stats(self) -> Dict:
        # Count frontier cells (free cells with unknown neighbors)
        frontier_count = 0
        for key, value in self.world_grid.items():
            if value != 0.5:  # Not a free cell
                continue
            for ndx, ndy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_key = (key[0] + ndx, key[1] + ndy)
                if neighbor_key not in self.world_grid:
                    frontier_count += 1
                    break  # Only count cell once
        
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
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        data = []
        for gy in range(min_y, max_y + 1):
            for gx in range(min_x, max_x + 1):
                key = (gx, gy)
                if key not in self.world_grid:
                    data.append(-1)
                elif self.world_grid[key] >= 0.8:
                    data.append(100)
                else:
                    data.append(0)
        
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
        min_x = min(k[0] for k in keys)
        max_x = max(k[0] for k in keys)
        min_y = min(k[1] for k in keys)
        max_y = max(k[1] for k in keys)
        
        padding = 2
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        max_visits = max(self.visit_counts.values()) if self.visit_counts else 1
        
        data = []
        for gy in range(min_y, max_y + 1):
            for gx in range(min_x, max_x + 1):
                key = (gx, gy)
                visits = self.visit_counts.get(key, 0)
                normalized = int(100 * visits / max(max_visits, 1))
                data.append(normalized)
        
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
        self._last_goal_time: float = 0.0  # When we last received a goal
        self._goal_persist_timeout: float = 60.0  # Keep goal for 60s after losing sight
        
        # Last known target position (persists even after goal cleared)
        self._last_known_target_pos: Optional[Tuple[float, float]] = None
        self._last_known_target_time: float = 0.0
        self._last_known_persist_timeout: float = 120.0  # Remember location for 2 minutes
        
        # Re-detection tracking
        self._target_was_lost: bool = False
        self._lost_time: float = 0.0
        self.last_imu: Optional[Imu] = None
        
        def make_topic(topic: str) -> str:
            if topic.startswith("/"):
                return topic
            elif ns:
                return f"/{ns}/{topic}"
            else:
                return f"/{topic}"
        
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
        
        # Safety visualization
        self.safety_zone_pub = self.create_publisher(Marker, "/safety_zone", 10)
        
        # Goal reached publisher - signals to goal generator to cycle targets
        self.goal_reached_pub = self.create_publisher(PointStamped, f"/{ns}/goal_reached", 10)
        
        # Goal visualization - shows current goal and last-known position in RViz
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_marker", 10)
        self.last_known_marker_pub = self.create_publisher(Marker, "/last_known_marker", 10)
        
        self.path_history: deque = deque(maxlen=PATH_HISTORY_LENGTH)
        
        self.get_logger().info("[VIZ] Publishing: /exploration_map, /visit_heatmap, /robot_path, /safety_zone, /goal_marker")
        
        # Sim reset service client
        self.reset_client = self.create_client(Trigger, '/sim/reset')
        self._sim_reset_available = False
        if self.reset_client.wait_for_service(timeout_sec=2.0):
            self._sim_reset_available = True
            self.get_logger().info('[ROS] /sim/reset service FOUND — full sim reset enabled')
        else:
            self.get_logger().warn(
                '[ROS] /sim/reset service NOT found — using soft reset only. '
                'Patch your stretch_mujoco_driver to enable full sim reset.'
            )
    
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
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.5
        marker.color.a = 0.8
        
        for x, y in self.path_history:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.05
            marker.points.append(p)
        
        self.path_pub.publish(marker)
    
    def publish_safety_zone(self, robot_x: float, robot_y: float, nav_state: NavigationState, frame_id: str = MAP_FRAME):
        """Publish a cylinder showing the current safety zone."""
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
        
        # Radius shows current minimum distance
        marker.scale.x = nav_state.min_distance * 2
        marker.scale.y = nav_state.min_distance * 2
        marker.scale.z = 0.1
        
        # Color based on zone
        if nav_state.zone == "FREE":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif nav_state.zone == "AWARE":
            marker.color.r, marker.color.g, marker.color.b = 0.5, 1.0, 0.0
        elif nav_state.zone == "CAUTION":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        elif nav_state.zone == "DANGER":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        else:  # EMERGENCY or CRITICAL
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        
        marker.color.a = 0.3
        
        self.safety_zone_pub.publish(marker)
    
    def _odom_cb(self, msg): self.last_odom = msg
    def _scan_cb(self, msg): self.last_scan = msg
    def _imu_cb(self, msg): self.last_imu = msg
    def _goal_cb(self, msg):
        # Track if this is a re-detection (we had lost the target)
        if self._target_was_lost and self.last_goal is None:
            self._target_was_lost = False  # Will trigger re-detection bonus
            
        self.last_goal = msg
        self._last_goal_time = time.time()
        
        # Update last known target position
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
        """
        Reset the MuJoCo simulation to its initial state.
        
        Returns True if the sim was successfully reset (robot teleported back),
        False if using soft reset (robot stays in place).
        """
        # Step 1: Stop the robot immediately
        for _ in range(10):
            self.send_cmd(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # Step 2: If we haven't found the service yet, try again
        if not self._sim_reset_available:
            if self.reset_client.wait_for_service(timeout_sec=2.0):
                self._sim_reset_available = True
                self.get_logger().info('[RESET] /sim/reset service found (late discovery)')
        
        # Step 3: Call the sim reset service
        if self._sim_reset_available:
            req = Trigger.Request()
            future = self.reset_client.call_async(req)
            
            # Spin until the service responds
            start = time.time()
            while not future.done() and (time.time() - start) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
            
            if future.done():
                result = future.result()
                if result and result.success:
                    self.get_logger().info('[RESET] Sim reset SUCCESS — robot teleported to start')
                    self._wait_for_fresh_data()
                    return True
                else:
                    msg = result.message if result else 'No response'
                    self.get_logger().warn(f'[RESET] Sim reset FAILED: {msg}')
            else:
                self.get_logger().warn('[RESET] Sim reset timed out — service did not respond')
        else:
            self.get_logger().error(
                '[RESET] /sim/reset service NOT available! '
                'Make sure you are running the patched stretch_mujoco_driver. '
                'Falling back to soft reset (robot stays in place).'
            )
        
        # Step 4: Soft reset fallback
        self.send_cmd(0.0, 0.0)
        time.sleep(0.2)
        self._wait_for_fresh_data()
        return False
    
    def _wait_for_fresh_data(self, timeout: float = 3.0):
        """Wait for fresh odom and scan data after a reset."""
        old_odom = self.last_odom
        old_scan = self.last_scan
        
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if we got NEW messages (different object identity)
            odom_fresh = (self.last_odom is not None and self.last_odom is not old_odom)
            scan_fresh = (self.last_scan is not None and self.last_scan is not old_scan)
            
            if odom_fresh and scan_fresh:
                return True
        
        self.get_logger().warn('[RESET] Timeout waiting for fresh sensor data')
        return False


# =============================================================================
# Gym Environment with Graduated Safety
# =============================================================================

class StretchExploreEnv(gym.Env):
    """
    RL environment — NO safety overrides.
    
    The RL agent has full control. Episodes terminate on:
    - Collision (min_distance < ZONE_EMERGENCY)
    - Goal reached (distance < GOAL_RADIUS) 
    - Timeout (max_steps)
    
    Each termination triggers a sim reset via /sim/reset service.
    """
    
    def __init__(self, ros: StretchRosInterface, rnd_module: RNDModule,
                 control_dt: float = 0.1):
        super().__init__()
        self.ros = ros
        self.control_dt = control_dt
        self.rnd = rnd_module
        
        # Navigator for scan analysis (observation only, NOT for control)
        self.navigator = DynamicNavigator(num_sectors=36)
        
        # Occupancy grid
        self.occ_grid = EgoOccupancyGrid()
        
        # Normalization
        self.obs_rms = None
        self.reward_normalizer = RewardNormalizer()
        
        # RND weight
        self.rnd_weight = RND_WEIGHT_INITIAL
        
        # State
        self.step_count = 0
        self.total_steps = 0
        self.max_steps = int(EPISODE_SECONDS / control_dt)
        self.episode_index = 1
        self.episode_return = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_goal_dist = 0.0
        
        # Smoothed velocity commands (EMA)
        self._smooth_v = 0.0
        self._smooth_w = 0.0
        
        # Displacement tracking (anti-gaming)
        self._last_pos = (0.0, 0.0)
        self._displacement_window = 0.0
        self._displacement_steps = 0
        
        self._goals_reached_this_episode = 0  # Count of goals reached
        
        # Observation space (added safety sector distances)
        grid_flat_size = GRID_SIZE * GRID_SIZE
        # Breakdown: lidar(60) + goal(5) + vel(2) + prev_act(2) + has_goal(1) + grid(576) + frontier(2) + novelty(1)
        #            + safety_obs(15) = 664
        obs_dim = (NUM_LIDAR_BINS + 5 + 2 + 2 + 1 + grid_flat_size + 2 + 1 
                   + SAFETY_OBS_SIZE)
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize observation normalizer
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        
        # Store last nav state for observation building
        self.last_nav_state: Optional[NavigationState] = None
        
        # Compatibility alias
        self.last_safety_state = None
        
        # Wait for sensors
        self.ros.get_logger().info("[ENV] Waiting for sensors...")
        self.ros.wait_for_sensors()
        
        self.ros.get_logger().info(f"[ENV] Observation dim: {obs_dim}")
        self.ros.get_logger().info(f"[ENV] Collision threshold: {ZONE_EMERGENCY}m, Goal radius: {GOAL_RADIUS}m")
        self.ros.get_logger().info(f"[ENV] NO SAFETY OVERRIDE — RL has full control")
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # ============================================================
        # PHASE 1: Reset the simulation
        # ============================================================
        
        sim_was_reset = self.ros.reset_simulation()
        
        if sim_was_reset:
            self.ros.get_logger().info(
                f'[RESET] Episode {self.episode_index} — full sim reset'
            )
        else:
            self.ros.get_logger().info(
                f'[RESET] Episode {self.episode_index} — soft reset (in-place)'
            )
        
        # ============================================================
        # PHASE 2: Clear ALL episode state
        # ============================================================
        
        # Core counters
        self.step_count = 0
        self.episode_return = 0.0
        self.prev_action[:] = 0.0
        self._smooth_v = 0.0
        self._smooth_w = 0.0
        
        # Displacement tracking — for anti-gaming stuck detection
        st = self._get_robot_state()
        self._last_pos = (st["x"], st["y"])
        self._displacement_window = 0.0  # Rolling displacement over last N steps
        self._displacement_steps = 0
        
        # Occupancy grid — full reset since robot is back at start
        self.occ_grid.reset()
        if not sim_was_reset:
            # Soft reset: keep some map knowledge but decay it
            self.occ_grid.decay_visits()
        
        # Goal tracking
        self._goals_reached_this_episode = 0
        
        # Clear goal state on the ROS interface
        self.ros.last_goal = None
        self.ros._last_goal_time = 0.0
        self.ros._target_was_lost = False
        self.ros._lost_time = 0.0
        self.ros._last_known_target_pos = None
        self.ros._last_known_target_time = 0.0
        
        # Clear last-known distance tracker (used in explore reward)
        if hasattr(self, '_prev_last_known_dist'):
            del self._prev_last_known_dist
        
        # Clear RViz markers
        self.ros.path_history.clear()
        
        # ============================================================
        # PHASE 3: Get fresh state
        # ============================================================
        
        self.prev_goal_dist = self._goal_distance()
        
        # Get initial navigation state from fresh scan
        if self.ros.last_scan is not None:
            self.last_nav_state = self.navigator.analyze_scan(self.ros.last_scan)
        else:
            self.last_nav_state = None
        self.last_safety_state = self.last_nav_state
        
        obs = self._build_observation()
        
        info = {
            "has_goal": False,
            "goal_dist": self.prev_goal_dist,
            "nav_zone": self.last_nav_state.zone if self.last_nav_state else "UNKNOWN",
            "sim_reset": sim_was_reset,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray):
        # Process RL action — agent has FULL CONTROL
        # a[0] in [-1,1] maps to [V_MIN_REVERSE, V_MAX]
        # a[1] in [-1,1] maps to [-w_limit, w_limit] (turn radius enforced)
        a = np.clip(action, -1.0, 1.0)
        
        # Linear velocity: full range including stop
        rl_v = float(a[0]) * V_MAX
        if rl_v < V_MIN_REVERSE:
            rl_v = V_MIN_REVERSE
        
        # Angular velocity: turn radius couples steering to speed
        # When moving fast → can steer sharply
        # When moving slow → gentle arcs only
        # When stopped → can rotate in place slowly (for reorientation)
        if abs(rl_v) > 0.05:
            w_limit = min(abs(rl_v) / MIN_TURN_RADIUS, W_MAX)
        else:
            w_limit = 0.3  # Slow rotation when stopped — just enough to reorient, not to game
        rl_w = float(a[1]) * w_limit
        
        # Smooth velocity commands (EMA) — prevents wild swings
        # Real robots have inertia; this simulates it and produces cleaner motion
        self._smooth_v = CMD_SMOOTHING_ALPHA * rl_v + (1.0 - CMD_SMOOTHING_ALPHA) * self._smooth_v
        self._smooth_w = CMD_SMOOTHING_ALPHA * rl_w + (1.0 - CMD_SMOOTHING_ALPHA) * self._smooth_w
        
        # ============================================================
        # EXECUTE SMOOTHED ACTION
        # ============================================================
        
        self.ros.send_cmd(self._smooth_v, self._smooth_w)
        
        # Wait for control period
        t_end = time.time() + self.control_dt
        while time.time() < t_end:
            rclpy.spin_once(self.ros, timeout_sec=0.01)
        
        # Update occupancy grid
        st = self._get_robot_state()
        scan = self.ros.last_scan
        new_cells = 0
        if scan is not None:
            new_cells = self.occ_grid.update_from_scan(
                st["x"], st["y"], st["yaw"], scan
            )
        
        # Analyze scan for observation (NOT for control override)
        nav_state = self.navigator.analyze_scan(self.ros.last_scan)
        self.last_nav_state = nav_state
        self.last_safety_state = nav_state
        
        # Build observation
        obs = self._build_observation()
        
        # ============================================================
        # CHECK TERMINATION CONDITIONS (curriculum-aware)
        # ============================================================
        
        terminated = False
        collision = False
        success = False
        reward = 0.0
        reward_terms = {}
        
        min_dist = nav_state.min_distance if nav_state else LIDAR_MAX_RANGE
        
        # --- Determine curriculum phase ---
        total = self.total_steps
        if total < CURRICULUM_PHASE1_END:
            phase = 1
        elif total < CURRICULUM_PHASE2_END:
            phase = 2
        else:
            phase = 3
        
        # --- COLLISION CHECK ---
        if min_dist < ZONE_EMERGENCY and self.step_count > 10:
            collision = True
            
            if phase == 1:
                # Phase 1: Bump penalty, episode CONTINUES
                # Agent learns "that hurt" but keeps exploring
                # Builds positive experience without collision trauma
                reward += R_COLLISION_PHASE1
                reward_terms["collision"] = R_COLLISION_PHASE1
                # NOT terminated — episode continues
            elif phase == 2:
                # Phase 2: Moderate penalty, episode ends
                terminated = True
                reward += R_COLLISION_PHASE2
                reward_terms["collision"] = R_COLLISION_PHASE2
            else:
                # Phase 3: Full penalty, episode ends
                terminated = True
                reward += R_COLLISION_PHASE3
                reward_terms["collision"] = R_COLLISION_PHASE3
        
        # --- GOAL REACHED: episode ends ---
        has_goal = self.ros.last_goal is not None
        d_goal = self._goal_distance()
        
        if has_goal and d_goal <= GOAL_RADIUS and not terminated:
            terminated = True
            success = True
            reward += R_GOAL
            reward_terms["goal"] = R_GOAL
            self._goals_reached_this_episode += 1
            
            # Publish goal reached
            reached_msg = PointStamped()
            reached_msg.header.stamp = self.ros.get_clock().now().to_msg()
            reached_msg.header.frame_id = "odom"
            reached_msg.point.x = self.ros.last_goal.point.x
            reached_msg.point.y = self.ros.last_goal.point.y
            reached_msg.point.z = 0.0
            self.ros.goal_reached_pub.publish(reached_msg)
            
            self.ros.get_logger().info(
                f"\033[92m[GOAL REACHED]\033[0m dist={d_goal:.2f}m, reward={R_GOAL:.0f} -> EPISODE ENDS"
            )
        
        # --- TIMEOUT: episode ends ---
        truncated = self.step_count >= self.max_steps
        if truncated and not terminated:
            reward += R_TIMEOUT
            reward_terms["timeout"] = R_TIMEOUT
        
        # ============================================================
        # COMPUTE SHAPING REWARDS (only if episode continues)
        # ============================================================
        
        if not terminated and not truncated:
            if has_goal:
                # Goal-seeking shaping
                progress = self.prev_goal_dist - d_goal
                r_progress = PROGRESS_SCALE * progress
                reward_terms["progress"] = r_progress
                reward += r_progress
                
                # Alignment — only when moving forward
                if rl_v > 0.15:
                    ang = self._goal_angle()
                    r_align = ALIGN_SCALE * math.cos(ang)
                    reward_terms["alignment"] = r_align
                    reward += r_align
                
                reward += STEP_COST
                reward_terms["step"] = STEP_COST
            else:
                # Exploration shaping — maximize coverage efficiency
                
                # Discovery: reward new cells found this step
                r_discovery = R_NEW_CELL * new_cells
                reward_terms["discovery"] = r_discovery
                reward += r_discovery
                
                # Novelty: diminishing reward for visiting known areas
                novelty = self.occ_grid.get_novelty(st["x"], st["y"])
                r_novelty = R_NOVELTY_SCALE * novelty
                reward_terms["novelty"] = r_novelty
                reward += r_novelty
                
                # Frontier steering: ALWAYS active, scales with cos(angle)
                # cos(0) = +1 (heading straight at frontier)
                # cos(π) = -1 (heading away from frontier → NEGATIVE reward)
                # This gives the agent a continuous gradient to steer toward unknowns
                frontier_angle = self.occ_grid.get_frontier_direction(st["x"], st["y"], st["yaw"])
                if frontier_angle != 0.0:  # 0.0 means no frontier found
                    r_frontier = R_FRONTIER_BONUS * math.cos(frontier_angle)
                    reward_terms["frontier"] = r_frontier
                    reward += r_frontier
                
                # Revisit penalty: escalates with visit count
                # First 2 visits are free, then -0.5 per extra visit, capped at -5
                robot_key = self.occ_grid.world_to_grid_key(st["x"], st["y"])
                visit_count = self.occ_grid.visit_counts.get(robot_key, 0)
                if visit_count > 2:
                    r_revisit = -R_REVISIT_SCALE * min(visit_count - 2, 10)
                    reward_terms["revisit"] = r_revisit
                    reward += r_revisit
                
                # RND intrinsic reward
                r_rnd = self.rnd.compute_intrinsic_reward(obs) * self.rnd_weight
                reward_terms["rnd"] = r_rnd
                reward += r_rnd
                
                reward += R_STEP_EXPLORE
                reward_terms["step"] = R_STEP_EXPLORE
            
            # Movement shaping — DISPLACEMENT-BASED (anti-gaming)
            
            # Track actual ground covered (spinning in place = 0 displacement)
            dx = st["x"] - self._last_pos[0]
            dy = st["y"] - self._last_pos[1]
            step_displacement = math.hypot(dx, dy)
            self._last_pos = (st["x"], st["y"])
            
            # Rolling displacement over a window (catches slow creeping too)
            DISPLACEMENT_WINDOW = 20  # steps
            decay = (DISPLACEMENT_WINDOW - 1) / DISPLACEMENT_WINDOW
            self._displacement_window = self._displacement_window * decay + step_displacement
            self._displacement_steps = min(self._displacement_steps + 1, DISPLACEMENT_WINDOW)
            avg_displacement = self._displacement_window / max(self._displacement_steps, 1)
            
            # Speed-scaled forward bonus — faster = more reward
            # Only counts actual forward movement, not spinning
            if st["v_lin"] > 0.05:
                r_forward = R_FORWARD_SCALE * (st["v_lin"] / V_MAX)
                reward += r_forward
                reward_terms["forward"] = r_forward
            
            # SPIN PENALTY — high angular velocity with low linear velocity
            # This is the #1 degenerate strategy: spin in place to avoid everything
            if abs(st["v_ang"]) > 0.2 and abs(st["v_lin"]) < 0.15:
                spin_ratio = abs(st["v_ang"]) / max(abs(st["v_lin"]) + 0.01, 0.01)
                r_spin = R_SPIN_PENALTY * min(spin_ratio / 5.0, 1.0)
                reward += r_spin
                reward_terms["spin"] = r_spin
            
            # STUCK PENALTY — based on displacement, NOT velocity
            # Spinning in place, creeping, or oscillating all get caught
            if self._displacement_steps >= 10 and avg_displacement < 0.02:
                reward += R_STUCK_PENALTY
                reward_terms["stuck"] = R_STUCK_PENALTY
            
            # Proximity reward — navigating NEAR obstacles without hitting them
            # Only active when actually MOVING (no free reward for sitting near walls)
            if min_dist < 2.0 and st["v_lin"] > 0.2 and R_PROXIMITY_BONUS > 0:
                dist_from_sweet = (min_dist - PROXIMITY_SWEET_SPOT) / PROXIMITY_RANGE
                prox_r = R_PROXIMITY_BONUS * math.exp(-0.5 * dist_from_sweet ** 2)
                reward += prox_r
                reward_terms["proximity"] = prox_r
        
        # ============================================================
        # BOOKKEEPING
        # ============================================================
        
        self.prev_goal_dist = d_goal
        self.total_steps += 1
        self.rnd_weight = max(
            RND_WEIGHT_MIN,
            RND_WEIGHT_INITIAL * (RND_WEIGHT_DECAY ** self.total_steps)
        )
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
            "curriculum_phase": phase,
        }
        
        # Logging
        if done:
            stats = self.occ_grid.get_stats()
            if success:
                status = "\033[92mGOAL_REACHED\033[0m"
            elif collision:
                status = "\033[91mCOLLISION\033[0m"
            else:
                status = "\033[93mTIMEOUT\033[0m"
            mode = "GOAL" if has_goal else "EXPLORE"
            ret_color = "\033[92m" if self.episode_return >= 0 else "\033[91m"
            phase_str = f"\033[94mP{phase}\033[0m"
            self.ros.get_logger().info(
                f"[EP {self.episode_index:04d}] {phase_str} {mode} {status} | "
                f"Return {ret_color}{self.episode_return:+.1f}\033[0m | Steps {self.step_count} | "
                f"Cells {stats['total_discovered']} | "
                f"Goals {self._goals_reached_this_episode} | "
                f"MinDist {min_dist:.2f}m"
            )
            self.episode_index += 1
        elif collision:
            # Phase 1: collision didn't end episode — log the bump
            self.ros.get_logger().info(
                f"\033[93m[BUMP]\033[0m step={self.step_count} min_d={min_dist:.2f}m "
                f"penalty={R_COLLISION_PHASE1} (Phase 1 — episode continues)"
            )
        
        # Debug logging
        if self.step_count % DEBUG_EVERY_N == 0:
            mode = "GOAL" if has_goal else "EXPLORE"
            goal_info = f"goal_dist={d_goal:.2f}m " if has_goal else ""
            self.ros.get_logger().info(
                f"[{mode}] step={self.step_count} min_d={min_dist:.2f}m "
                f"{goal_info}"
                f"cmd=[{self._smooth_v:.2f},{self._smooth_w:.2f}] "
                f"r={reward:.2f}"
            )
        
        # Publish reward
        self._publish_reward_breakdown(reward, info, has_goal)
        
        # Publish visualization
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
        
        # Publish safety zone visualization
        self.ros.publish_safety_zone(robot_state["x"], robot_state["y"], nav_state)
        
        # Publish goal markers
        self._publish_goal_markers()
        
        # Publish navigation direction marker
        self._publish_nav_direction(robot_state, nav_state)
    
    def _publish_nav_direction(self, robot_state: Dict, nav_state: NavigationState):
        """Publish arrow showing best navigation direction."""
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = self.ros.get_clock().now().to_msg()
        marker.ns = "nav_direction"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Arrow from robot pointing in best direction
        world_angle = robot_state["yaw"] + nav_state.best_direction
        length = 0.8
        
        start = Point()
        start.x = robot_state["x"]
        start.y = robot_state["y"]
        start.z = 0.3
        
        end = Point()
        end.x = robot_state["x"] + length * math.cos(world_angle)
        end.y = robot_state["y"] + length * math.sin(world_angle)
        end.z = 0.3
        
        marker.points = [start, end]
        marker.scale.x = 0.08  # Shaft diameter
        marker.scale.y = 0.15  # Head diameter
        marker.scale.z = 0.15  # Head length
        
        # Cyan color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.9
        
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 200000000  # 200ms
        
        self.ros.safety_zone_pub.publish(marker)  # Reuse publisher
    
    def _publish_goal_markers(self):
        """Publish markers for current goal and last-known position."""
        now = self.ros.get_clock().now().to_msg()
        
        # Current goal marker (green sphere)
        if self.ros.last_goal is not None:
            goal_marker = Marker()
            goal_marker.header.stamp = now
            goal_marker.header.frame_id = "odom"
            goal_marker.ns = "goal"
            goal_marker.id = 0
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = self.ros.last_goal.point.x
            goal_marker.pose.position.y = self.ros.last_goal.point.y
            goal_marker.pose.position.z = 0.5
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = 0.5
            goal_marker.scale.y = 0.5
            goal_marker.scale.z = 0.5
            goal_marker.color.r = 0.0
            goal_marker.color.g = 1.0
            goal_marker.color.b = 0.0
            goal_marker.color.a = 0.8
            goal_marker.lifetime.sec = 1
            self.ros.goal_marker_pub.publish(goal_marker)
        else:
            # Delete marker if no goal
            delete_marker = Marker()
            delete_marker.header.stamp = now
            delete_marker.header.frame_id = "odom"
            delete_marker.ns = "goal"
            delete_marker.id = 0
            delete_marker.action = Marker.DELETE
            self.ros.goal_marker_pub.publish(delete_marker)
        
        # Last-known position marker (yellow sphere, smaller)
        if self.ros._last_known_target_pos is not None:
            lk_marker = Marker()
            lk_marker.header.stamp = now
            lk_marker.header.frame_id = "odom"
            lk_marker.ns = "last_known"
            lk_marker.id = 0
            lk_marker.type = Marker.SPHERE
            lk_marker.action = Marker.ADD
            lk_marker.pose.position.x = self.ros._last_known_target_pos[0]
            lk_marker.pose.position.y = self.ros._last_known_target_pos[1]
            lk_marker.pose.position.z = 0.3
            lk_marker.pose.orientation.w = 1.0
            lk_marker.scale.x = 0.3
            lk_marker.scale.y = 0.3
            lk_marker.scale.z = 0.3
            lk_marker.color.r = 1.0
            lk_marker.color.g = 1.0
            lk_marker.color.b = 0.0
            lk_marker.color.a = 0.6
            lk_marker.lifetime.sec = 1
            self.ros.last_known_marker_pub.publish(lk_marker)
    
    def _build_observation(self) -> np.ndarray:
        st = self._get_robot_state()
        goal = self.ros.last_goal
        
        # LiDAR
        lidar_bins = self._get_lidar_bins()
        lidar_norm = np.clip(lidar_bins / LIDAR_MAX_RANGE, 0.0, 1.0)
        
        # Goal info
        if goal is not None:
            dx = goal.point.x - st["x"]
            dy = goal.point.y - st["y"]
            dist = math.hypot(dx, dy)
            ang = self._goal_angle()
            
            cap = 6.0
            goal_info = np.array([
                np.clip(dx / cap, -1, 1),
                np.clip(dy / cap, -1, 1),
                np.clip(dist / cap, 0, 1),
                math.sin(ang),
                math.cos(ang)
            ], dtype=np.float32)
        else:
            goal_info = np.zeros(5, dtype=np.float32)
        
        # Velocity
        vel = np.array([
            np.clip(st["v_lin"] / V_MAX, -1, 1),
            np.clip(st["v_ang"] / W_MAX, -1, 1)
        ], dtype=np.float32)
        
        # Previous action
        prev_act = self.prev_action.copy()
        
        # Has goal flag
        has_goal = np.array([1.0 if goal is not None else 0.0], dtype=np.float32)
        
        # Occupancy grid
        grid_flat = self.occ_grid.get_flat_grid()
        grid_norm = (grid_flat - 0.5) * 2.0
        
        # Frontier direction
        frontier_ang = self.occ_grid.get_frontier_direction(st["x"], st["y"], st["yaw"])
        frontier_dir = np.array([math.sin(frontier_ang), math.cos(frontier_ang)], dtype=np.float32)
        
        # Novelty
        novelty = np.array([self.occ_grid.get_novelty(st["x"], st["y"])], dtype=np.float32)
        
        # Safety information (helps RL learn about obstacles)
        if self.last_nav_state is not None:
            # Normalized sector distances (0=far, 1=close)
            # Downsample from 36 sectors to NUM_SAFETY_SECTORS (12)
            nav_sectors = self.last_nav_state.sector_distances
            ratio = len(nav_sectors) // NUM_SAFETY_SECTORS
            sector_obs_raw = np.array([
                np.min(nav_sectors[i*ratio:(i+1)*ratio]) 
                for i in range(NUM_SAFETY_SECTORS)
            ])
            sector_obs = 1.0 - np.clip(sector_obs_raw / ZONE_FREE, 0.0, 1.0)
            
            # Safety blend (how much navigation is intervening)
            safety_blend = np.array([self.last_nav_state.safety_blend], dtype=np.float32)
            
            # Danger direction (encoded as sin/cos) - using min_angle
            danger_dir = np.array([
                math.sin(self.last_nav_state.min_angle),
                math.cos(self.last_nav_state.min_angle)
            ], dtype=np.float32)
            
            # Combine
            safety_obs = np.concatenate([sector_obs, safety_blend, danger_dir])
        else:
            safety_obs = np.zeros(SAFETY_OBS_SIZE, dtype=np.float32)
        
        # Concatenate all
        obs = np.concatenate([
            lidar_norm,
            goal_info,
            vel,
            prev_act,
            has_goal,
            grid_norm,
            frontier_dir,
            novelty,
            safety_obs,  # NEW
        ], axis=0)
        
        # Update running statistics
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
        if goal is None:
            return 0.0
        st = self._get_robot_state()
        dx = goal.point.x - st["x"]
        dy = goal.point.y - st["y"]
        return float(math.hypot(dx, dy))
    
    def _goal_angle(self) -> float:
        goal = self.ros.last_goal
        if goal is None:
            return 0.0
        st = self._get_robot_state()
        dx = goal.point.x - st["x"]
        dy = goal.point.y - st["y"]
        ang_world = math.atan2(dy, dx)
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
            "velocity": {
                "v": info.get("executed_v", 0.0),
                "w": info.get("executed_w", 0.0),
            },
            "state": {
                "x": st["x"],
                "y": st["y"],
                "yaw_deg": st["yaw"] * 180 / math.pi,
                "goal_x": goal.point.x if goal else 0,
                "goal_y": goal.point.y if goal else 0,
                "goal_dist": self._goal_distance(),
            },
            "explore_stats": stats,
            "rnd_weight": self.rnd_weight,
            "curriculum_phase": info.get("curriculum_phase", 0),
            "episode": self.episode_index,
            "step": self.step_count,
        }
        
        msg = StringMsg()
        msg.data = json.dumps(breakdown)
        self.ros.reward_breakdown_pub.publish(msg)


# =============================================================================
# TD3 Networks (adjusted for new observation size)
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
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, out_features),
            nn.ReLU(),
        )
    
    def forward(self, grid_flat: torch.Tensor) -> torch.Tensor:
        batch_size = grid_flat.shape[0]
        grid = grid_flat.view(batch_size, 1, self.grid_size, self.grid_size)
        
        x = self.conv(grid)
        x = self.fc(x)
        return x


class ActorWithCNN(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, grid_size: int = GRID_SIZE,
                 grid_features: int = 32, hidden: int = 256):
        super().__init__()
        
        self.grid_size = grid_size
        grid_flat_size = grid_size * grid_size
        
        self.non_grid_size = obs_dim - grid_flat_size
        self.grid_start = NUM_LIDAR_BINS + 5 + 2 + 2 + 1
        self.grid_end = self.grid_start + grid_flat_size
        
        self.grid_cnn = GridCNN(grid_size, grid_features)
        
        combined_size = self.non_grid_size + grid_features
        
        self.mlp = nn.Sequential(
            nn.Linear(combined_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
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
            nn.Linear(combined_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
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
# TD3 Agent
# =============================================================================

class TD3AgentCNN:
    """TD3 Agent with CNN and RND integration."""
    
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device,
                 rnd_module: RNDModule,
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
        self.rnd = rnd_module
        
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
            lr=critic_lr
        )
        
        self.total_updates = 0
    
    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        a = self.actor(o).squeeze(0).cpu().numpy()
        if noise_std > 0:
            a = a + np.random.normal(0, noise_std, size=a.shape).astype(np.float32)
        return np.clip(a, -1.0, 1.0).astype(np.float32)
    
    def update(self, replay: PrioritizedReplayBuffer, batch_size: int, beta: float) -> Tuple[float, float, float]:
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
        
        rnd_loss = self.rnd.update(next_obs)
        
        return float(critic_loss.item()), float(actor_loss.item()), rnd_loss
    
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
            "rnd": self.rnd.state_dict_custom(),
        }
        if extra_state is not None:
            payload["training_state"] = extra_state
        torch.save(payload, path)
    
    def load(self, path: str, strict: bool = True) -> Optional[Dict]:
        """Load checkpoint. Returns training_state dict if present, else None."""
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
            if "rnd" in payload:
                self.rnd.load_state_dict_custom(payload["rnd"])
        except Exception:
            pass
        return payload.get("training_state", None)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # Topics
    parser.add_argument("--ns", type=str, default="")
    parser.add_argument("--odom-topic", type=str, default="/stretch/odom")
    parser.add_argument("--lidar-topic", type=str, default="/stretch/scan")
    parser.add_argument("--imu-topic", type=str, default="/imu_mobile_base")
    parser.add_argument("--goal-topic", type=str, default="goal")
    parser.add_argument("--cmd-topic", type=str, default="/stretch/cmd_vel")
    
    # Training
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--start-steps", type=int, default=DEFAULT_START_STEPS)
    parser.add_argument("--update-after", type=int, default=2000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=300_000)
    parser.add_argument("--expl-noise", type=float, default=DEFAULT_EXPL_NOISE)
    parser.add_argument("--save-every", type=int, default=10_000)
    
    # Checkpoint
    parser.add_argument("--ckpt-dir", type=str, default=os.path.expanduser("~/rl_checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    
    # Mode
    parser.add_argument("--inference", action="store_true")
    
    # Compatibility args
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
        ns=args.ns,
        odom_topic=args.odom_topic,
        scan_topic=args.lidar_topic,
        imu_topic=args.imu_topic,
        goal_topic=args.goal_topic,
        cmd_topic=args.cmd_topic,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(ros)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Observation dimension (with safety features)
    grid_flat_size = GRID_SIZE * GRID_SIZE
    obs_dim = (NUM_LIDAR_BINS + 5 + 2 + 2 + 1 + grid_flat_size + 2 + 1 
               + SAFETY_OBS_SIZE)
    act_dim = 2
    
    # Create RND module
    rnd_module = RNDModule(obs_dim, device=device)
    
    # Create environment
    env = StretchExploreEnv(ros, rnd_module)
    
    ros.get_logger().info(f"[AGENT] device={device} obs_dim={obs_dim} act_dim={act_dim}")
    ros.get_logger().info(f"[AGENT] NO SAFETY OVERRIDE — RL learns from raw consequences")
    ros.get_logger().info(f"[AGENT] Episodes end on: COLLISION / GOAL_REACHED / TIMEOUT")
    
    # Create agent
    agent = TD3AgentCNN(obs_dim, act_dim, device=device, rnd_module=rnd_module)
    
    # Replay buffer
    replay = PrioritizedReplayBuffer(obs_dim, act_dim, size=args.replay_size, device=device)
    
    # =====================================================================
    # LOAD CHECKPOINT — restores weights + training state + replay buffer
    # =====================================================================
    
    # ANSI colors for terminal
    G = "\033[92m"  # Green
    R = "\033[91m"  # Red
    Y = "\033[93m"  # Yellow
    B = "\033[94m"  # Blue
    W = "\033[97m"  # White bold
    RST = "\033[0m" # Reset
    
    resume_step = 0  # Which training step to resume from
    replay_path = os.path.join(args.ckpt_dir, "replay_buffer.npz")
    
    ros.get_logger().info(f"[CKPT] Checkpoint dir: {B}{args.ckpt_dir}{RST}")
    ros.get_logger().info(f"[CKPT] Looking for: {B}{ckpt_path}{RST}")
    
    if AUTO_LOAD_CHECKPOINT and os.path.exists(ckpt_path):
        file_size_mb = os.path.getsize(ckpt_path) / 1e6
        ros.get_logger().info(f"[CKPT] Found checkpoint ({file_size_mb:.1f} MB) — loading...")
        try:
            training_state = agent.load(ckpt_path, strict=False)
            ros.get_logger().info(f"[CKPT] {G}✓ Network weights loaded successfully{RST}")
            
            if training_state is not None:
                resume_step = training_state.get("step", 0)
                env.episode_index = training_state.get("episode_index", 1)
                env.total_steps = training_state.get("total_steps", 0)
                env.rnd_weight = training_state.get("rnd_weight", RND_WEIGHT_INITIAL)
                
                # Restore observation normalizer
                if "obs_rms_mean" in training_state:
                    env.obs_rms.mean = training_state["obs_rms_mean"]
                    env.obs_rms.var = training_state["obs_rms_var"]
                    env.obs_rms.count = training_state["obs_rms_count"]
                    ros.get_logger().info(f"[CKPT] {G}✓ Observation normalizer restored{RST} (count={env.obs_rms.count:.0f})")
                else:
                    ros.get_logger().warn(f"[CKPT] {Y}✗ No obs_rms in training state — normalizer starts fresh{RST}")
                
                ros.get_logger().info(
                    f"[CKPT] {G}✓ Training state restored:{RST} step={W}{resume_step}{RST}, "
                    f"episode={W}{env.episode_index}{RST}, total_steps={W}{env.total_steps}{RST}, "
                    f"rnd_weight={env.rnd_weight:.4f}"
                )
            else:
                ros.get_logger().warn(
                    f"[CKPT] {Y}✗ No training state in checkpoint (old format){RST} — "
                    "weights loaded but step/episode/replay start from 0"
                )
            
            # Load replay buffer (check for broken temp files from old bug)
            replay_loaded = False
            search_paths = [
                replay_path,
                replay_path + ".tmp.npz",
                replay_path.replace(".npz", ".npz.tmp.npz"),
            ]
            ros.get_logger().info(f"[CKPT] Looking for replay buffer...")
            for rp in search_paths:
                if os.path.exists(rp):
                    rp_size_mb = os.path.getsize(rp) / 1e6
                    ros.get_logger().info(f"[CKPT] Found: {B}{rp}{RST} ({rp_size_mb:.1f} MB)")
                    if replay.load(rp):
                        ros.get_logger().info(f"[CKPT] {G}✓ Replay buffer loaded: {replay.count} transitions{RST}")
                        # Rename to correct path if loaded from broken name
                        if rp != replay_path:
                            try:
                                os.replace(rp, replay_path)
                                ros.get_logger().info(f"[CKPT] {G}Renamed{RST} {rp} → {replay_path}")
                            except Exception as e:
                                ros.get_logger().warn(f"[CKPT] {Y}Could not rename: {e}{RST}")
                        replay_loaded = True
                        break
                    else:
                        ros.get_logger().warn(f"[CKPT] {R}✗ Replay buffer file exists but failed to load{RST}")
            if not replay_loaded:
                ros.get_logger().warn(f"[CKPT] {Y}✗ No replay buffer found — starting empty (will need to refill){RST}")
            
            # Summary
            skip_explore = resume_step >= args.start_steps
            ros.get_logger().info(
                f"\n{W}{'='*50}\n"
                f"  CHECKPOINT LOAD SUMMARY\n"
                f"{'='*50}{RST}\n"
                f"  Weights:       {G}✓ loaded{RST}\n"
                f"  Training step: {W}{resume_step}{RST}\n"
                f"  Episode:       {W}{env.episode_index}{RST}\n"
                f"  Replay buffer: {G if replay.count > 0 else Y}{replay.count} transitions{RST}\n"
                f"  Updates done:  {W}{agent.total_updates}{RST}\n"
                f"  Skip random:   {G + 'YES' if skip_explore else Y + 'NO (need ' + str(args.start_steps - resume_step) + ' more random steps)'}{RST}\n"
                f"{W}{'='*50}{RST}"
            )
            
        except Exception as e:
            ros.get_logger().error(
                f"\n{R}{'='*50}\n"
                f"  CHECKPOINT LOAD FAILED\n"
                f"{'='*50}\n"
                f"  Error: {e}\n"
                f"  Starting from scratch with random weights\n"
                f"{'='*50}{RST}"
            )
    else:
        if not os.path.exists(ckpt_path):
            ros.get_logger().info(f"[CKPT] {Y}No checkpoint found at {ckpt_path} — starting fresh{RST}")
        else:
            ros.get_logger().info(f"[CKPT] {Y}AUTO_LOAD_CHECKPOINT disabled — starting fresh{RST}")
    
    # =====================================================================
    # SAVE HELPER — saves everything needed to resume training
    # =====================================================================
    
    def save_full_checkpoint(step: int, label: str = "periodic"):
        """Save agent weights + training state + replay buffer."""
        training_state = {
            "step": step,
            "episode_index": env.episode_index,
            "total_steps": env.total_steps,
            "rnd_weight": env.rnd_weight,
            "obs_rms_mean": env.obs_rms.mean,
            "obs_rms_var": env.obs_rms.var,
            "obs_rms_count": env.obs_rms.count,
        }
        
        try:
            agent.save(ckpt_path + ".tmp", extra_state=training_state)
            os.replace(ckpt_path + ".tmp", ckpt_path)
        except Exception as e:
            ros.get_logger().error(f"[CKPT] {R}✗ Agent save failed: {e}{RST}")
            return
        
        try:
            replay_tmp = replay_path.replace(".npz", "_tmp.npz")
            replay.save(replay_tmp)
            # np.savez_compressed may add .npz if not present
            if not os.path.exists(replay_tmp) and os.path.exists(replay_tmp + ".npz"):
                replay_tmp = replay_tmp + ".npz"
            os.replace(replay_tmp, replay_path)
        except Exception as e:
            ros.get_logger().warn(f"[CKPT] {Y}✗ Replay save failed: {e}{RST}")
        
        ros.get_logger().info(
            f"[CKPT] {G}✓ {label} save{RST} at step={W}{step}{RST} | "
            f"episode={W}{env.episode_index}{RST} | replay={W}{replay.count}{RST}"
        )
    
    # Shutdown handler
    def shutdown_and_save(signum=None, frame=None):
        nonlocal resume_step
        try:
            ros.send_cmd(0.0, 0.0)
        except:
            pass
        
        # Use the current step from the training loop
        current_step = getattr(shutdown_and_save, '_current_step', resume_step)
        save_full_checkpoint(current_step, label="shutdown")
        
        stats = env.occ_grid.get_stats()
        ros.get_logger().info(f"[FINAL] Cells discovered: {stats['total_discovered']}")
        
        try:
            executor.shutdown()
            ros.destroy_node()
            rclpy.shutdown()
        except:
            pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_and_save)
    signal.signal(signal.SIGTERM, shutdown_and_save)
    
    # Inference mode
    if args.inference:
        ros.get_logger().info("[MODE] INFERENCE (no safety override)")
        while True:
            obs, _ = env.reset()
            done = False
            while not done:
                act = agent.act(obs, noise_std=0.0)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc
    
    # =====================================================================
    # TRAINING LOOP — resumes from saved step
    # =====================================================================
    
    start_step = resume_step + 1
    ros.get_logger().info(
        f"\n{G}{'='*50}\n"
        f"  TRAINING STARTED\n"
        f"{'='*50}{RST}\n"
        f"  Mode:          {W}TRAINING (no safety override){RST}\n"
        f"  Resuming from: {W}step {start_step}{RST}\n"
        f"  Replay buffer: {W}{replay.count} transitions{RST}\n"
        f"  Target steps:  {W}{args.total_steps}{RST}\n"
        f"{G}{'='*50}{RST}"
    )
    
    obs, _ = env.reset()
    last_save = resume_step
    last_phase = 0  # Track curriculum phase transitions
    
    for t in range(start_step, args.total_steps + 1):
        # Track current step for shutdown handler
        shutdown_and_save._current_step = t
        
        # Curriculum phase transition logging
        if env.total_steps < CURRICULUM_PHASE1_END:
            current_phase = 1
        elif env.total_steps < CURRICULUM_PHASE2_END:
            current_phase = 2
        else:
            current_phase = 3
        
        if current_phase != last_phase:
            phase_names = {
                1: f"{G}Phase 1: LEARN TO EXPLORE{RST} (collisions = bump, episode continues)",
                2: f"{Y}Phase 2: LEARN TO AVOID{RST} (collisions end episode, moderate penalty)",
                3: f"{W}Phase 3: FULL DIFFICULTY{RST} (collisions end episode, full penalty)",
            }
            ros.get_logger().info(
                f"\n{W}{'='*50}\n"
                f"  CURRICULUM TRANSITION → {phase_names[current_phase]}\n"
                f"  Total steps: {env.total_steps}\n"
                f"{'='*50}{RST}"
            )
            last_phase = current_phase
        
        # Random actions at start (skipped if resuming past start_steps)
        # a[0] in [-1,1]: negative = reverse/stop, positive = forward
        # a[1] in [-1,1]: steering (turn radius limited by step())
        if t < args.start_steps:
            act = np.array([
                np.random.uniform(0.0, 1.0),     # Bias forward during exploration
                np.random.uniform(-0.8, 0.8),     # Moderate steering
            ], dtype=np.float32)
        else:
            act = agent.act(obs, noise_std=args.expl_noise)
        
        next_obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        
        replay.add(
            obs, act,
            np.array([reward], dtype=np.float32),
            next_obs,
            np.array([1.0 if done else 0.0], dtype=np.float32)
        )
        
        obs = next_obs
        
        if done:
            obs, _ = env.reset()
        
        # Update
        if t >= args.update_after and t % args.update_every == 0 and replay.count >= args.batch_size:
            beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * (t / args.total_steps)
            critic_loss, actor_loss, rnd_loss = agent.update(replay, args.batch_size, beta)
        
        # Save
        if t - last_save >= args.save_every:
            last_save = t
            save_full_checkpoint(t, label="periodic")
    
    shutdown_and_save()


if __name__ == "__main__":
    main()