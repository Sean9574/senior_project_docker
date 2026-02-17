#!/usr/bin/env python3
"""
Stretch Robot RL Environment + Learner with GRADUATED SAFETY SYSTEM

KEY FEATURES:
1. Graduated Safety Blender - smooth transition from RL control to safety override
2. Potential Field Avoidance - continuous repulsive forces from obstacles
3. Shaped Rewards - RL learns to avoid, not just react
4. Safety-Aware Training - RL experiences consequences without destruction

SAFETY ZONES (based on 13" robot width + 2ft desired clearance):
- FREE:      > 1.5m   - Full RL control
- AWARE:    1.5-1.0m  - Gentle safety influence (10%)
- CAUTION:  1.0-0.6m  - Blended control (40% safety)
- DANGER:   0.6-0.35m - Safety dominant (70%)  
- EMERGENCY: < 0.35m  - Hard override (100% safety)

The RL agent still learns because:
1. It experiences the "resistance" from safety as environment dynamics
2. Shaped rewards penalize getting into danger zones
3. Near-misses still happen, teaching avoidance
4. Reward includes penalty for safety intervention amount
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
from visualization_msgs.msg import Marker, MarkerArray

# =============================================================================
# ROBOT PHYSICAL PARAMETERS
# =============================================================================

ROBOT_WIDTH_M = 0.33  # 13 inches in meters
ROBOT_HALF_WIDTH_M = 0.165
DESIRED_CLEARANCE_M = 0.61  # 2 feet in meters
MIN_SAFE_DISTANCE = ROBOT_HALF_WIDTH_M + DESIRED_CLEARANCE_M  # ~0.77m

# =============================================================================
# SAFETY ZONE THRESHOLDS (from LIDAR/center of robot)
# =============================================================================

ZONE_FREE = 1.5         # Full RL control
ZONE_AWARE = 1.0        # Gentle safety influence begins
ZONE_CAUTION = 0.6      # Blended control (around 2ft clearance)
ZONE_DANGER = 0.35      # Safety takes priority
ZONE_EMERGENCY = 0.20   # Hard override (robot half-width + small buffer)

# Safety blend ratios at each zone boundary
BLEND_FREE = 0.0        # 0% safety
BLEND_AWARE = 0.10      # 10% safety
BLEND_CAUTION = 0.40    # 40% safety  
BLEND_DANGER = 0.70     # 70% safety
BLEND_EMERGENCY = 1.0   # 100% safety

# =============================================================================
# POTENTIAL FIELD PARAMETERS
# =============================================================================

REPULSIVE_GAIN = 2.0           # Strength of obstacle repulsion
REPULSIVE_INFLUENCE = 1.5      # Distance at which repulsion starts
ATTRACTIVE_GAIN = 0.5          # Strength of goal attraction (when goal exists)

# =============================================================================
# REWARD SHAPING PARAMETERS
# =============================================================================

# Proximity penalties (graduated)
R_PROXIMITY_SCALE = -50.0      # Max penalty per step when very close
R_CLEARANCE_BONUS = 0.5        # Bonus for maintaining good clearance
R_SAFETY_INTERVENTION = -10.0  # Penalty scaled by how much safety overrode RL

# =============================================================================
# EXISTING CONFIG (preserved from original)
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
W_MAX = 7.0
V_MIN_REVERSE = -0.05

# Goal Seeking Rewards
R_GOAL = 2500.0
R_COLLISION = -2000.0
R_TIMEOUT = -100.0
GOAL_RADIUS = 0.45

PROGRESS_SCALE = 500.0
ALIGN_SCALE = 3.0
STEP_COST = -0.05

# Exploration Rewards
R_NEW_CELL = 2.0
R_NOVELTY_SCALE = 0.5
R_FRONTIER_BONUS = 5.0
R_COLLISION_EXPLORE = -500.0
R_STEP_EXPLORE = -0.02

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
# SAFETY SYSTEM: Graduated Safety Blender + Potential Fields
# =============================================================================

@dataclass
class SafetyState:
    """Current safety assessment."""
    min_distance: float
    zone: str
    safety_blend: float  # 0.0 = full RL, 1.0 = full safety
    repulsive_v: float   # Suggested linear velocity from potential field
    repulsive_w: float   # Suggested angular velocity from potential field
    sector_distances: np.ndarray  # Distance to obstacle in each sector
    danger_direction: float  # Angle to closest obstacle (radians)


class GraduatedSafetyBlender:
    """
    Blends RL actions with safety actions based on proximity to obstacles.
    
    Key features:
    - Smooth, graduated response (no sudden jumps)
    - Potential field for natural obstacle avoidance
    - RL still experiences consequences through blended dynamics
    - Shaped rewards based on safety state
    """
    
    def __init__(self, num_sectors: int = NUM_SAFETY_SECTORS):
        self.num_sectors = num_sectors
        self.sector_angles = np.linspace(-np.pi, np.pi, num_sectors + 1)[:-1]
        self.sector_width = 2 * np.pi / num_sectors
        
        # Logging
        self.last_safety_state: Optional[SafetyState] = None
        self.intervention_history = deque(maxlen=100)
    
    def analyze_scan(self, scan: LaserScan) -> SafetyState:
        """
        Analyze LIDAR scan and return safety state.
        """
        if scan is None:
            return SafetyState(
                min_distance=LIDAR_MAX_RANGE,
                zone="FREE",
                safety_blend=0.0,
                repulsive_v=0.0,
                repulsive_w=0.0,
                sector_distances=np.full(self.num_sectors, LIDAR_MAX_RANGE),
                danger_direction=0.0
            )
        
        ranges = np.array(scan.ranges, dtype=np.float32)
        n_rays = len(ranges)
        
        if n_rays == 0:
            return SafetyState(
                min_distance=LIDAR_MAX_RANGE,
                zone="FREE",
                safety_blend=0.0,
                repulsive_v=0.0,
                repulsive_w=0.0,
                sector_distances=np.full(self.num_sectors, LIDAR_MAX_RANGE),
                danger_direction=0.0
            )
        
        # Clean up invalid readings
        max_r = min(scan.range_max, LIDAR_MAX_RANGE) if scan.range_max > 0 else LIDAR_MAX_RANGE
        min_r = max(scan.range_min, 0.01)
        
        invalid = np.isnan(ranges) | np.isinf(ranges) | (ranges < min_r) | (ranges > max_r)
        ranges[invalid] = max_r
        
        # Calculate angle for each ray
        angles = scan.angle_min + np.arange(n_rays) * scan.angle_increment + LIDAR_FORWARD_OFFSET_RAD
        angles = np.array([wrap_to_pi(a) for a in angles])
        
        # Find minimum distance and its direction
        min_idx = np.argmin(ranges)
        min_distance = float(ranges[min_idx])
        danger_direction = float(angles[min_idx])
        
        # Compute sector distances (minimum in each sector)
        sector_distances = np.full(self.num_sectors, max_r, dtype=np.float32)
        
        for i in range(n_rays):
            angle = angles[i]
            # Find which sector this ray belongs to
            sector_idx = int((angle + np.pi) / self.sector_width) % self.num_sectors
            sector_distances[sector_idx] = min(sector_distances[sector_idx], ranges[i])
        
        # Determine zone and blend factor
        zone, safety_blend = self._compute_zone_and_blend(min_distance)
        
        # Compute potential field repulsive velocities
        repulsive_v, repulsive_w = self._compute_repulsive_velocity(
            ranges, angles, min_distance
        )
        
        state = SafetyState(
            min_distance=min_distance,
            zone=zone,
            safety_blend=safety_blend,
            repulsive_v=repulsive_v,
            repulsive_w=repulsive_w,
            sector_distances=sector_distances,
            danger_direction=danger_direction
        )
        
        self.last_safety_state = state
        return state
    
    def _compute_zone_and_blend(self, min_distance: float) -> Tuple[str, float]:
        """
        Determine safety zone and compute smooth blend factor.
        Uses linear interpolation within each zone for smooth transitions.
        """
        if min_distance >= ZONE_FREE:
            return "FREE", 0.0
        
        elif min_distance >= ZONE_AWARE:
            # Interpolate between FREE and AWARE
            t = (ZONE_FREE - min_distance) / (ZONE_FREE - ZONE_AWARE)
            blend = lerp(BLEND_FREE, BLEND_AWARE, t)
            return "AWARE", blend
        
        elif min_distance >= ZONE_CAUTION:
            # Interpolate between AWARE and CAUTION
            t = (ZONE_AWARE - min_distance) / (ZONE_AWARE - ZONE_CAUTION)
            blend = lerp(BLEND_AWARE, BLEND_CAUTION, t)
            return "CAUTION", blend
        
        elif min_distance >= ZONE_DANGER:
            # Interpolate between CAUTION and DANGER
            t = (ZONE_CAUTION - min_distance) / (ZONE_CAUTION - ZONE_DANGER)
            blend = lerp(BLEND_CAUTION, BLEND_DANGER, t)
            return "DANGER", blend
        
        elif min_distance >= ZONE_EMERGENCY:
            # Interpolate between DANGER and EMERGENCY
            t = (ZONE_DANGER - min_distance) / (ZONE_DANGER - ZONE_EMERGENCY)
            blend = lerp(BLEND_DANGER, BLEND_EMERGENCY, t)
            return "EMERGENCY", blend
        
        else:
            # Below emergency threshold - full safety
            return "CRITICAL", 1.0
    
    def _compute_repulsive_velocity(
        self, 
        ranges: np.ndarray, 
        angles: np.ndarray,
        min_distance: float
    ) -> Tuple[float, float]:
        """
        Compute repulsive velocity using potential field approach.
        Returns (linear_velocity, angular_velocity) that pushes away from obstacles.
        """
        # Only compute if within influence range
        if min_distance >= REPULSIVE_INFLUENCE:
            return 0.0, 0.0
        
        # Accumulate repulsive force from all nearby obstacles
        force_x = 0.0  # Forward/backward
        force_y = 0.0  # Left/right
        
        for i, (r, angle) in enumerate(zip(ranges, angles)):
            if r >= REPULSIVE_INFLUENCE:
                continue
            
            # Repulsive magnitude (inverse square, clamped)
            # Higher when closer
            magnitude = REPULSIVE_GAIN * (1.0 / r - 1.0 / REPULSIVE_INFLUENCE) ** 2
            magnitude = min(magnitude, 10.0)  # Clamp to prevent explosion
            
            # Direction away from obstacle (in robot frame)
            # angle=0 means obstacle is ahead
            force_x += -magnitude * math.cos(angle)  # Push backward if obstacle ahead
            force_y += -magnitude * math.sin(angle)  # Push away laterally
        
        # Convert forces to velocity commands
        # force_x negative means "go backward", positive means "go forward"
        # force_y negative means "turn right", positive means "turn left"
        
        # Scale to velocity limits
        repulsive_v = np.clip(force_x * 0.3, -V_MAX * 0.5, V_MAX * 0.3)
        repulsive_w = np.clip(force_y * 2.0, -W_MAX * 0.7, W_MAX * 0.7)
        
        return float(repulsive_v), float(repulsive_w)
    
    def blend_action(
        self, 
        rl_v: float, 
        rl_w: float, 
        safety_state: SafetyState,
        goal_angle: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Blend RL action with safety action based on current safety state.
        
        Returns:
            (blended_v, blended_w, intervention_amount)
        
        intervention_amount: How much safety overrode RL (0-1)
        """
        blend = safety_state.safety_blend
        
        # If no safety needed, return RL action directly
        if blend < 0.01:
            return rl_v, rl_w, 0.0
        
        # Compute safety action
        safety_v = safety_state.repulsive_v
        safety_w = safety_state.repulsive_w
        
        # If we have a goal and we're not in emergency, bias toward goal
        if goal_angle is not None and safety_state.zone not in ["EMERGENCY", "CRITICAL"]:
            # Add attractive component toward goal
            goal_w = ATTRACTIVE_GAIN * math.sin(goal_angle) * W_MAX
            safety_w = safety_w + goal_w * (1.0 - blend)
        
        # In emergency/critical, if RL wants to go forward toward danger, override
        if safety_state.zone in ["EMERGENCY", "CRITICAL"]:
            # Check if RL is trying to move toward the obstacle
            danger_ahead = abs(safety_state.danger_direction) < math.pi / 3
            
            if danger_ahead and rl_v > 0:
                # Force backward motion
                safety_v = -V_MAX * 0.3
                blend = 1.0
            
            # Turn away from danger
            if safety_state.danger_direction > 0:
                safety_w = -W_MAX * 0.5  # Turn right
            else:
                safety_w = W_MAX * 0.5   # Turn left
        
        # Blend RL and safety actions
        blended_v = lerp(rl_v, safety_v, blend)
        blended_w = lerp(rl_w, safety_w, blend)
        
        # Apply velocity limits
        blended_v = np.clip(blended_v, V_MIN_REVERSE, V_MAX)
        blended_w = np.clip(blended_w, -W_MAX, W_MAX)
        
        # Calculate how much we intervened
        original_magnitude = math.sqrt(rl_v**2 + rl_w**2)
        diff_magnitude = math.sqrt((rl_v - blended_v)**2 + (rl_w - blended_w)**2)
        
        if original_magnitude > 0.01:
            intervention = diff_magnitude / (original_magnitude + 0.01)
        else:
            intervention = blend
        
        intervention = np.clip(intervention, 0.0, 1.0)
        
        # Track intervention
        self.intervention_history.append(intervention)
        
        return float(blended_v), float(blended_w), float(intervention)
    
    def compute_proximity_reward(self, safety_state: SafetyState) -> float:
        """
        Compute shaped reward based on proximity to obstacles.
        This teaches RL to maintain safe distance proactively.
        """
        min_dist = safety_state.min_distance
        
        # Good clearance bonus
        if min_dist >= MIN_SAFE_DISTANCE:
            # Bonus for maintaining desired clearance
            return R_CLEARANCE_BONUS
        
        # Graduated penalty as we get closer
        # Penalty increases smoothly as distance decreases
        if min_dist >= ZONE_DANGER:
            # Mild penalty in aware/caution zones
            penalty_factor = 1.0 - (min_dist - ZONE_DANGER) / (MIN_SAFE_DISTANCE - ZONE_DANGER)
            return R_PROXIMITY_SCALE * 0.2 * penalty_factor
        
        elif min_dist >= ZONE_EMERGENCY:
            # Stronger penalty in danger zone
            penalty_factor = 1.0 - (min_dist - ZONE_EMERGENCY) / (ZONE_DANGER - ZONE_EMERGENCY)
            return R_PROXIMITY_SCALE * 0.5 * (0.5 + 0.5 * penalty_factor)
        
        else:
            # Maximum penalty in emergency/critical
            return R_PROXIMITY_SCALE
    
    def compute_intervention_penalty(self, intervention_amount: float) -> float:
        """
        Penalize RL for requiring safety intervention.
        This teaches it to avoid situations where safety takes over.
        """
        return R_SAFETY_INTERVENTION * intervention_amount
    
    def get_average_intervention(self) -> float:
        """Get recent average intervention rate."""
        if len(self.intervention_history) == 0:
            return 0.0
        return float(np.mean(self.intervention_history))


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
        
        robot_key = self.world_to_grid_key(robot_x, robot_y)
        self.visit_counts[robot_key] = self.visit_counts.get(robot_key, 0) + 1
        
        for i in range(0, n_rays, 3):
            r = ranges[i]
            
            if np.isnan(r) or np.isinf(r) or r < range_min:
                continue
            
            r = min(r, range_max)
            ray_angle_world = robot_yaw + angle_min + i * angle_inc + LIDAR_FORWARD_OFFSET_RAD
            
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
            
            if r < range_max - 0.5:
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
        return {
            'total_discovered': self.total_cells_discovered,
            'new_this_step': self.cells_discovered_this_step,
            'world_grid_size': len(self.world_grid),
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
                 imu_topic="/imu_mobile_base", goal_topic="/goal", cmd_topic="/stretch/cmd_vel"):
        super().__init__("learner_node")
        
        self.last_odom: Optional[Odometry] = None
        self.last_scan: Optional[LaserScan] = None
        self.last_goal: Optional[PointStamped] = None
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
        
        self.path_history: deque = deque(maxlen=PATH_HISTORY_LENGTH)
        
        self.get_logger().info("[VIZ] Publishing: /exploration_map, /visit_heatmap, /robot_path, /safety_zone")
    
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
    
    def publish_safety_zone(self, robot_x: float, robot_y: float, safety_state: SafetyState, frame_id: str = MAP_FRAME):
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
        marker.scale.x = safety_state.min_distance * 2
        marker.scale.y = safety_state.min_distance * 2
        marker.scale.z = 0.1
        
        # Color based on zone
        if safety_state.zone == "FREE":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif safety_state.zone == "AWARE":
            marker.color.r, marker.color.g, marker.color.b = 0.5, 1.0, 0.0
        elif safety_state.zone == "CAUTION":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        elif safety_state.zone == "DANGER":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        else:  # EMERGENCY or CRITICAL
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        
        marker.color.a = 0.3
        
        self.safety_zone_pub.publish(marker)
    
    def _odom_cb(self, msg): self.last_odom = msg
    def _scan_cb(self, msg): self.last_scan = msg
    def _imu_cb(self, msg): self.last_imu = msg
    def _goal_cb(self, msg): self.last_goal = msg
    
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


# =============================================================================
# Gym Environment with Graduated Safety
# =============================================================================

class StretchExploreEnv(gym.Env):
    """
    Environment with graduated safety system.
    
    Key changes from original:
    1. Safety blender processes all actions before execution
    2. Shaped rewards based on proximity and safety intervention
    3. RL experiences blended dynamics (learns from safety influence)
    4. Additional observation features for safety awareness
    """
    
    def __init__(self, ros: StretchRosInterface, rnd_module: RNDModule,
                 control_dt: float = 0.1):
        super().__init__()
        self.ros = ros
        self.control_dt = control_dt
        self.rnd = rnd_module
        
        # Safety system
        self.safety_blender = GraduatedSafetyBlender()
        
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
        
        # Goal tracking to prevent instant re-success after reaching goal
        self._goal_just_reached = False
        self._goal_cooldown_pos = None  # Position when goal was reached
        self._min_move_from_goal = 1.0  # Must move 1m away before new goal accepted
        
        # Track safety interventions for logging
        self.episode_interventions = []
        
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
        
        # Store last safety state for observation building
        self.last_safety_state: Optional[SafetyState] = None
        
        # Wait for sensors
        self.ros.get_logger().info("[ENV] Waiting for sensors...")
        self.ros.wait_for_sensors()
        
        self.ros.get_logger().info(f"[ENV] Observation dim: {obs_dim}")
        self.ros.get_logger().info(f"[ENV] Robot width: {ROBOT_WIDTH_M:.2f}m, Clearance: {DESIRED_CLEARANCE_M:.2f}m")
        self.ros.get_logger().info(f"[ENV] Safety zones: FREE>{ZONE_FREE}m, AWARE>{ZONE_AWARE}m, "
                                    f"CAUTION>{ZONE_CAUTION}m, DANGER>{ZONE_DANGER}m, EMERGENCY>{ZONE_EMERGENCY}m")
        self.ros.get_logger().info(f"[ENV] GRADUATED SAFETY BLENDING ENABLED")
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_return = 0.0
        self.prev_action[:] = 0.0
        self.occ_grid.reset()
        self.episode_interventions = []
        
        self.occ_grid.decay_visits()
        
        # CRITICAL FIX: Clear goal after episode ends
        # This prevents instant re-success when robot is still at goal position
        # The goal generator will re-publish when it detects the target again
        if hasattr(self, '_goal_just_reached') and self._goal_just_reached:
            self.ros.last_goal = None
            self._goal_just_reached = False
            self._goal_cooldown_pos = self._get_robot_state()
        
        self.prev_goal_dist = self._goal_distance()
        
        # Get initial safety state
        self.last_safety_state = self.safety_blender.analyze_scan(self.ros.last_scan)
        
        obs = self._build_observation()
        info = {
            "has_goal": self.ros.last_goal is not None,
            "goal_dist": self.prev_goal_dist,
            "safety_zone": self.last_safety_state.zone if self.last_safety_state else "UNKNOWN"
        }
        
        return obs, info
    
    def step(self, action: np.ndarray):
        # Process RL action
        a = np.clip(action, -1.0, 1.0)
        rl_v = float(a[0]) * V_MAX
        rl_w = float(a[1]) * W_MAX
        
        if rl_v < V_MIN_REVERSE:
            rl_v = V_MIN_REVERSE
        
        # ============================================================
        # GRADUATED SAFETY BLENDING
        # ============================================================
        
        # Analyze current scan for safety
        safety_state = self.safety_blender.analyze_scan(self.ros.last_scan)
        self.last_safety_state = safety_state
        
        # Get goal angle for attraction (if goal exists)
        goal_angle = self._goal_angle() if self.ros.last_goal is not None else None
        
        # Blend RL action with safety
        safe_v, safe_w, intervention = self.safety_blender.blend_action(
            rl_v, rl_w, safety_state, goal_angle
        )
        
        # Track intervention
        self.episode_interventions.append(intervention)
        
        # ============================================================
        # EXECUTE SAFE ACTION
        # ============================================================
        
        self.ros.send_cmd(safe_v, safe_w)
        
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
        
        # Update safety state after movement
        safety_state = self.safety_blender.analyze_scan(self.ros.last_scan)
        self.last_safety_state = safety_state
        
        # Build observation
        obs = self._build_observation()
        
        # ============================================================
        # COMPUTE REWARD WITH SAFETY SHAPING
        # ============================================================
        
        has_goal = self.ros.last_goal is not None
        
        # Check if we're in cooldown after reaching a goal
        # Robot must move away before accepting a new goal
        if has_goal and self._goal_cooldown_pos is not None:
            dx = st["x"] - self._goal_cooldown_pos["x"]
            dy = st["y"] - self._goal_cooldown_pos["y"]
            dist_from_success = math.hypot(dx, dy)
            
            if dist_from_success < self._min_move_from_goal:
                # Still too close to last success position - ignore goal
                has_goal = False
            else:
                # Moved far enough, clear cooldown
                self._goal_cooldown_pos = None
        
        if has_goal:
            reward, terminated, info = self._compute_goal_reward(safe_v, safe_w)
        else:
            reward, terminated, info = self._compute_explore_reward(new_cells, st, obs)
        
        # Add safety-shaped rewards
        proximity_reward = self.safety_blender.compute_proximity_reward(safety_state)
        intervention_penalty = self.safety_blender.compute_intervention_penalty(intervention)
        
        reward += proximity_reward + intervention_penalty
        
        info["reward_terms"]["proximity"] = proximity_reward
        info["reward_terms"]["intervention_penalty"] = intervention_penalty
        info["safety_zone"] = safety_state.zone
        info["safety_blend"] = safety_state.safety_blend
        info["intervention"] = intervention
        info["executed_v"] = safe_v
        info["executed_w"] = safe_w
        info["rl_v"] = rl_v
        info["rl_w"] = rl_w
        
        # Update RND weight
        self.total_steps += 1
        self.rnd_weight = max(
            RND_WEIGHT_MIN,
            RND_WEIGHT_INITIAL * (RND_WEIGHT_DECAY ** self.total_steps)
        )
        
        self.episode_return += reward
        self.prev_action[:] = a
        self.step_count += 1
        
        truncated = self.step_count >= self.max_steps
        
        # Logging
        if terminated or truncated:
            mode = "GOAL" if has_goal else "EXPLORE"
            status = "SUCCESS" if info.get("success", False) else (
                "COLLISION" if info.get("collision", False) else "TIMEOUT"
            )
            stats = self.occ_grid.get_stats()
            avg_intervention = np.mean(self.episode_interventions) if self.episode_interventions else 0.0
            self.ros.get_logger().info(
                f"[EP {self.episode_index:04d}] {mode} {status} | "
                f"Return {self.episode_return:+.1f} | Steps {self.step_count} | "
                f"Discovered {stats['total_discovered']} cells | "
                f"Avg Intervention {avg_intervention:.1%}"
            )
            self.episode_index += 1
        
        # Debug logging
        if self.step_count % DEBUG_EVERY_N == 0:
            mode = "GOAL" if has_goal else "EXPLORE"
            self.ros.get_logger().info(
                f"[{mode}] step={self.step_count} zone={safety_state.zone} "
                f"blend={safety_state.safety_blend:.0%} min_d={safety_state.min_distance:.2f}m "
                f"rl=[{rl_v:.2f},{rl_w:.2f}] safe=[{safe_v:.2f},{safe_w:.2f}]"
            )
        
        # Publish reward breakdown
        self._publish_reward_breakdown(reward, info, has_goal)
        
        # Publish visualization
        if PUBLISH_MAP and self.step_count % PUBLISH_MAP_EVERY_N == 0:
            self._publish_visualization(st, safety_state)
        
        return obs, float(reward), bool(terminated), bool(truncated), info
    
    def _publish_visualization(self, robot_state: Dict, safety_state: SafetyState):
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
        self.ros.publish_safety_zone(robot_state["x"], robot_state["y"], safety_state)
    
    def _compute_explore_reward(self, new_cells: int, st: Dict, obs: np.ndarray) -> Tuple[float, bool, Dict]:
        """Compute reward for exploration mode."""
        reward = 0.0
        terminated = False
        collision = False
        
        r_discovery = R_NEW_CELL * new_cells
        
        novelty = self.occ_grid.get_novelty(st["x"], st["y"])
        r_novelty = R_NOVELTY_SCALE * novelty
        
        frontier_angle = self.occ_grid.get_frontier_direction(st["x"], st["y"], st["yaw"])
        if abs(frontier_angle) < math.pi / 4 and st["v_lin"] > 0.1:
            r_frontier = R_FRONTIER_BONUS * math.cos(frontier_angle) * 0.1
        else:
            r_frontier = 0.0
        
        r_step = R_STEP_EXPLORE
        
        # RND intrinsic reward
        r_rnd = self.rnd.compute_intrinsic_reward(obs) * self.rnd_weight
        
        # Collision - now based on ZONE_EMERGENCY (hard safety should prevent this)
        if self.last_safety_state and self.last_safety_state.min_distance < ZONE_EMERGENCY:
            if self.step_count > 10:
                terminated = True
                collision = True
                r_collision = R_COLLISION_EXPLORE
        else:
            r_collision = 0.0
        
        reward = r_discovery + r_novelty + r_frontier + r_step + r_collision + r_rnd
        
        info = {
            "collision": collision,
            "success": False,
            "exploring": True,
            "reward_terms": {
                "discovery": r_discovery,
                "novelty": r_novelty,
                "frontier": r_frontier,
                "step": r_step,
                "collision": r_collision,
                "rnd": r_rnd,
            }
        }
        
        return float(reward), terminated, info
    
    def _compute_goal_reward(self, v_cmd: float, w_cmd: float) -> Tuple[float, bool, Dict]:
        """Compute reward for goal-seeking mode."""
        reward = 0.0
        terminated = False
        collision = False
        success = False
        
        d_goal = self._goal_distance()
        ang = self._goal_angle()
        
        progress = self.prev_goal_dist - d_goal
        r_progress = PROGRESS_SCALE * progress
        
        r_align = ALIGN_SCALE * math.cos(ang)
        
        r_step = STEP_COST
        
        r_goal = 0.0
        if d_goal <= GOAL_RADIUS:
            terminated = True
            success = True
            r_goal = R_GOAL
            # Mark that goal was reached - reset() will clear the goal
            self._goal_just_reached = True
        
        # Collision based on safety zone
        r_collision = 0.0
        if self.last_safety_state and self.last_safety_state.min_distance < ZONE_EMERGENCY:
            if self.step_count > 10:
                terminated = True
                collision = True
                r_collision = R_COLLISION
        
        r_timeout = 0.0
        if not terminated and self.step_count + 1 >= self.max_steps:
            terminated = True
            r_timeout = R_TIMEOUT
        
        self.prev_goal_dist = d_goal
        
        reward = r_progress + r_align + r_step + r_goal + r_collision + r_timeout
        
        info = {
            "collision": collision,
            "success": success,
            "exploring": False,
            "goal_dist": d_goal,
            "reward_terms": {
                "progress": r_progress,
                "alignment": r_align,
                "step": r_step,
                "goal": r_goal,
                "collision": r_collision,
                "timeout": r_timeout,
            }
        }
        
        return float(reward), terminated, info
    
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
        
        # Safety information (NEW - helps RL learn about obstacles)
        if self.last_safety_state is not None:
            # Normalized sector distances (0=far, 1=close)
            sector_obs = 1.0 - np.clip(
                self.last_safety_state.sector_distances / ZONE_FREE, 0.0, 1.0
            )
            
            # Safety blend (how much safety is intervening)
            safety_blend = np.array([self.last_safety_state.safety_blend], dtype=np.float32)
            
            # Danger direction (encoded as sin/cos)
            danger_dir = np.array([
                math.sin(self.last_safety_state.danger_direction),
                math.cos(self.last_safety_state.danger_direction)
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
            "safety": {
                "zone": info.get("safety_zone", "UNKNOWN"),
                "blend": info.get("safety_blend", 0.0),
                "intervention": info.get("intervention", 0.0),
                "rl_v": info.get("rl_v", 0.0),
                "rl_w": info.get("rl_w", 0.0),
                "executed_v": info.get("executed_v", 0.0),
                "executed_w": info.get("executed_w", 0.0),
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
    
    def save(self, path: str):
        torch.save({
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
        }, path)
    
    def load(self, path: str, strict: bool = True):
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
    parser.add_argument("--goal-topic", type=str, default="/goal")
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
    ros.get_logger().info(f"[AGENT] GRADUATED SAFETY SYSTEM ACTIVE")
    ros.get_logger().info(f"[AGENT] Robot: {ROBOT_WIDTH_M:.2f}m wide, {DESIRED_CLEARANCE_M:.2f}m clearance")
    
    # Create agent
    agent = TD3AgentCNN(obs_dim, act_dim, device=device, rnd_module=rnd_module)
    
    # Replay buffer
    replay = PrioritizedReplayBuffer(obs_dim, act_dim, size=args.replay_size, device=device)
    
    # Load checkpoint
    if AUTO_LOAD_CHECKPOINT and os.path.exists(ckpt_path):
        ros.get_logger().info(f"[CKPT] Loading from {ckpt_path}")
        try:
            agent.load(ckpt_path, strict=False)
            ros.get_logger().info("[CKPT] Load SUCCESS")
        except Exception as e:
            ros.get_logger().warn(f"[CKPT] Load FAILED (model architecture changed?): {e}")
    
    # Shutdown handler
    def shutdown_and_save(signum=None, frame=None):
        try:
            ros.send_cmd(0.0, 0.0)
        except:
            pass
        
        ros.get_logger().info(f"[CKPT] Saving to {ckpt_path}")
        try:
            agent.save(ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            ros.get_logger().info("[CKPT] Save SUCCESS")
        except Exception as e:
            ros.get_logger().error(f"[CKPT] Save FAILED: {e}")
        
        stats = env.occ_grid.get_stats()
        avg_intervention = env.safety_blender.get_average_intervention()
        ros.get_logger().info(f"[FINAL] Cells discovered: {stats['total_discovered']}, "
                               f"Avg safety intervention: {avg_intervention:.1%}")
        
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
        ros.get_logger().info("[MODE] INFERENCE (with safety)")
        while True:
            obs, _ = env.reset()
            done = False
            while not done:
                act = agent.act(obs, noise_std=0.0)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc
    
    # Training mode
    ros.get_logger().info("[MODE] TRAINING (with graduated safety)")
    obs, _ = env.reset()
    last_save = 0
    
    for t in range(1, args.total_steps + 1):
        # Random actions at start
        if t < args.start_steps:
            v = np.random.uniform(0.2, 1.0)
            w = np.random.uniform(-0.5, 0.5)
            act = np.array([v, w], dtype=np.float32)
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
            stats = env.occ_grid.get_stats()
            avg_intervention = env.safety_blender.get_average_intervention()
            ros.get_logger().info(
                f"[CKPT] step={t} cells={stats['total_discovered']} "
                f"intervention={avg_intervention:.1%}"
            )
            try:
                agent.save(ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
            except Exception as e:
                ros.get_logger().warn(f"[CKPT] Save failed: {e}")
    
    shutdown_and_save()


if __name__ == "__main__":
    main()