#!/usr/bin/env python3
"""
Stretch Robot RL Environment + Learner (IMPROVED VERSION)

IMPROVEMENTS OVER ORIGINAL:
1. Prioritized Experience Replay (PER) - focuses learning on important transitions
2. Random Network Distillation (RND) - learned intrinsic motivation replaces heuristics
3. Observation normalization - stabilizes training
4. Reward normalization - handles reward scale mismatch between modes
5. SMART COLLISION AVOIDANCE - navigates around obstacles instead of stopping (NEW!)

Original behaviors preserved:
- Agent receives ego-centric occupancy grid as part of observation
- When no goal: rewarded for discovering new areas (now with RND bonus)
- When goal set: rewarded for reaching goal (progress, alignment)
- Same network learns both behaviors via "has_goal" flag in observation
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
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry_msgs.msg import Point, PointStamped, Twist
from gymnasium import spaces
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import String as StringMsg
from visualization_msgs.msg import Marker, MarkerArray

# =============================================================================
# CONFIG
# =============================================================================

CHECKPOINT_FILENAME = "td3_explore_agent_v2.pt"
AUTO_LOAD_CHECKPOINT = True

EPISODE_SECONDS = 60.0

# --- Occupancy Grid ---
GRID_SIZE = 24
GRID_RESOLUTION = 0.5
GRID_MAX_RANGE = 12.0

# --- Movement Limits ---
V_MAX = 1.25
W_MAX = 7.0
V_MIN_REVERSE = -0.05

# --- Collision/Safety --- UPDATED VALUES
COLLISION_DIST = 0.25  # Reduced from 0.40 - closer threshold with smart avoidance
SAFE_DIST = 0.80

# --- Goal Seeking Rewards (when goal is set) ---
R_GOAL = 2500.0
R_COLLISION = -2000.0
R_TIMEOUT = -100.0
GOAL_RADIUS = 0.45

PROGRESS_SCALE = 500.0
ALIGN_SCALE = 3.0
STEP_COST = -0.05

# --- Exploration Rewards (when NO goal) ---
R_NEW_CELL = 2.0
R_NOVELTY_SCALE = 0.5
R_FRONTIER_BONUS = 5.0
R_COLLISION_EXPLORE = -500.0
R_STEP_EXPLORE = -0.02

# --- RND Config ---
RND_WEIGHT_INITIAL = 1.0
RND_WEIGHT_DECAY = 0.99995
RND_WEIGHT_MIN = 0.1
RND_FEATURE_DIM = 64
RND_LR = 1e-4

# --- PER Config ---
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPSILON = 1e-6

# Visit tracking
VISIT_DECAY = 0.995
NOVELTY_RADIUS = 1.0

# --- LiDAR ---
LIDAR_FORWARD_OFFSET_RAD = math.pi
NUM_LIDAR_BINS = 60
LIDAR_MAX_RANGE = 20.0

# --- Training ---
DEFAULT_START_STEPS = 10000
DEFAULT_EXPL_NOISE = 0.3

# --- Debug ---
DEBUG_EVERY_N = 50

# --- Visualization ---
PUBLISH_MAP = True
PUBLISH_MAP_EVERY_N = 10
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


# =============================================================================
# Running Mean/Std for Normalization
# =============================================================================

class RunningMeanStd:
    """Tracks running mean and standard deviation for normalization."""
    
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update statistics with a batch of data."""
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
        """Normalize input using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class RewardNormalizer:
    """Normalizes rewards to have roughly unit variance."""
    
    def __init__(self, clip: float = 10.0):
        self.rms = RunningMeanStd(shape=())
        self.clip = clip
    
    def normalize(self, reward: float, update: bool = True) -> float:
        """Normalize reward and optionally update statistics."""
        if update:
            self.rms.update(np.array([reward]))
        
        std = np.sqrt(self.rms.var + 1e-8)
        normalized = reward / max(std, 1.0)
        return float(np.clip(normalized, -self.clip, self.clip))


# =============================================================================
# Random Network Distillation (RND)
# =============================================================================

class RNDModule(nn.Module):
    """Random Network Distillation for intrinsic motivation."""
    
    def __init__(self, obs_dim: int, hidden: int = 256, 
                 feature_dim: int = RND_FEATURE_DIM, device: torch.device = None):
        super().__init__()
        
        self.device = device or torch.device("cpu")
        
        # Fixed random target network (NEVER updated)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim)
        ).to(self.device)
        
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False
        
        # Predictor network (trained to match target)
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
        """Normalize observation for RND."""
        return np.clip(self.obs_rms.normalize(obs), -5.0, 5.0)
    
    @torch.no_grad()
    def compute_intrinsic_reward(self, obs: np.ndarray) -> float:
        """Compute intrinsic reward as prediction error."""
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
        """Update predictor network."""
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
        """Get state dict including running statistics."""
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
        """Load state dict including running statistics."""
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
# Prioritized Experience Replay (PER)
# =============================================================================

class SumTree:
    """Binary sum tree for efficient priority-based sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
    
    def update(self, tree_idx: int, priority: float):
        """Update priority at tree index."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def add(self, priority: float) -> int:
        """Add new priority, return data index."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        
        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        return data_idx
    
    def get(self, value: float) -> Tuple[int, int, float]:
        """Get leaf index, data index, and priority for a value."""
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
        """Add transition with max priority."""
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
        """Sample batch with importance sampling weights."""
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
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + PER_EPSILON) ** self.alpha
            
            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)
            
            self.max_priority = max(self.max_priority, abs(td_error) + PER_EPSILON)


# =============================================================================
# Ego-Centric Occupancy Grid (unchanged)
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
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        
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
# ROS Interface (unchanged)
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
        
        self.path_history: deque = deque(maxlen=PATH_HISTORY_LENGTH)
        
        self.get_logger().info("[VIZ] Publishing: /exploration_map, /visit_heatmap, /robot_path")
    
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
# Gym Environment (MODIFIED: Smart Collision Avoidance)
# =============================================================================

class StretchExploreEnv(gym.Env):
    """
    Environment that supports both exploration and goal-seeking.
    
    IMPROVEMENTS:
    - RND module provides learned intrinsic reward
    - Observation normalization for stable training
    - Reward normalization for consistent scales
    - SMART collision avoidance - navigates around obstacles instead of stopping!
    """
    
    def __init__(self, ros: StretchRosInterface, rnd_module: RNDModule,
                 control_dt: float = 0.1):
        super().__init__()
        self.ros = ros
        self.control_dt = control_dt
        self.rnd = rnd_module
        
        # Occupancy grid
        self.occ_grid = EgoOccupancyGrid()
        
        # Normalization
        self.obs_rms = None
        self.reward_normalizer = RewardNormalizer()
        
        # RND weight (decays over time)
        self.rnd_weight = RND_WEIGHT_INITIAL
        
        # State
        self.step_count = 0
        self.total_steps = 0
        self.max_steps = int(EPISODE_SECONDS / control_dt)
        self.episode_index = 1
        self.episode_return = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_goal_dist = 0.0
        
        # Observation space
        grid_flat_size = GRID_SIZE * GRID_SIZE
        obs_dim = NUM_LIDAR_BINS + 5 + 2 + 2 + 1 + grid_flat_size + 2 + 1
        
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
        
        # Wait for sensors
        self.ros.get_logger().info("[ENV] Waiting for sensors...")
        self.ros.wait_for_sensors()
        
        self.ros.get_logger().info(
            f"[ENV] Observation dim: {obs_dim} (lidar={NUM_LIDAR_BINS}, "
            f"grid={GRID_SIZE}x{GRID_SIZE}={grid_flat_size})"
        )
        self.ros.get_logger().info(
            f"[ENV] IMPROVED: Using RND + PER + Smart Collision Avoidance"
        )
        self.ros.get_logger().info(
            f"[ENV] Collision threshold: {COLLISION_DIST}m (navigates around instead of stopping)"
        )
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_return = 0.0
        self.prev_action[:] = 0.0
        self.occ_grid.reset()
        
        self.occ_grid.decay_visits()
        
        self.prev_goal_dist = self._goal_distance()
        
        obs = self._build_observation()
        info = {
            "has_goal": self.ros.last_goal is not None,
            "goal_dist": self.prev_goal_dist
        }
        
        return obs, info
    
    def step(self, action: np.ndarray):
        # Process action
        a = np.clip(action, -1.0, 1.0)
        v_cmd = float(a[0]) * V_MAX
        w_cmd = float(a[1]) * W_MAX
        
        if v_cmd < V_MIN_REVERSE:
            v_cmd = V_MIN_REVERSE
        
        # ============================================================
        # NEW: SMART COLLISION AVOIDANCE
        # ============================================================
        min_lidar = self._min_lidar()
        obstacle_angle = self._get_obstacle_angle()
        
        if min_lidar < COLLISION_DIST:
            # Calculate danger factor (0 = at safe distance, 1 = at collision)
            danger_factor = 1.0 - (min_lidar / COLLISION_DIST)
            
            # Strategy 1: Obstacle is to the side - turn away from it
            if abs(obstacle_angle) > 0.1:  # Obstacle not directly ahead
                # Turn away from obstacle
                turn_away = -np.sign(obstacle_angle) * W_MAX * danger_factor
                w_cmd = turn_away
                
                # Reduce forward velocity but don't stop completely
                v_cmd = v_cmd * (1.0 - danger_factor * 0.7)
                
            # Strategy 2: Obstacle directly ahead and moving forward - back up and turn
            elif v_cmd > 0:
                # Move backwards proportional to danger
                v_cmd = -V_MAX * 0.5 * danger_factor
                
                # Turn toward more open space
                w_cmd = w_cmd + self._get_escape_direction() * W_MAX * 0.5
            
            # Strategy 3: Already reversing - allow it but guide the turn
            else:
                # Keep backing up, adjust turn toward open space
                w_cmd = w_cmd + self._get_escape_direction() * W_MAX * 0.3
        
        # ============================================================
        # END SMART COLLISION AVOIDANCE
        # ============================================================
        
        # Send command
        self.ros.send_cmd(v_cmd, w_cmd)
        
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
        
        # Build observation
        obs = self._build_observation()
        
        # Compute reward based on mode
        has_goal = self.ros.last_goal is not None
        
        if has_goal:
            reward, terminated, info = self._compute_goal_reward(v_cmd, w_cmd)
        else:
            reward, terminated, info = self._compute_explore_reward(new_cells, st, obs)
        
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
            self.ros.get_logger().info(
                f"[EP {self.episode_index:04d}] {mode} {status} | "
                f"Return {self.episode_return:+.1f} | Steps {self.step_count} | "
                f"Discovered {stats['total_discovered']} cells | "
                f"RND_w={self.rnd_weight:.3f}"
            )
            self.episode_index += 1
        
        # Debug logging
        if self.step_count % DEBUG_EVERY_N == 0:
            mode = "GOAL" if has_goal else "EXPLORE"
            stats = self.occ_grid.get_stats()
            novelty = self.occ_grid.get_novelty(st["x"], st["y"])
            rnd_r = info.get("reward_terms", {}).get("rnd", 0.0)
            self.ros.get_logger().info(
                f"[{mode}] step={self.step_count} r={reward:+.2f} "
                f"new_cells={new_cells} total={stats['total_discovered']} "
                f"novelty={novelty:.2f} rnd_r={rnd_r:.2f} min_lidar={min_lidar:.2f}"
            )
        
        # Publish reward breakdown
        self._publish_reward_breakdown(reward, info, has_goal)
        
        # Publish visualization
        if PUBLISH_MAP and self.step_count % PUBLISH_MAP_EVERY_N == 0:
            self._publish_visualization(st)
        
        return obs, float(reward), bool(terminated), bool(truncated), info
    
    def _publish_visualization(self, robot_state: Dict):
        self.ros.add_path_point(robot_state["x"], robot_state["y"])
        
        map_msg = self.occ_grid.get_occupancy_grid_msg(MAP_FRAME)
        map_msg.header.stamp = self.ros.get_clock().now().to_msg()
        self.ros.map_pub.publish(map_msg)
        
        heatmap_msg = self.occ_grid.get_visit_heatmap_msg(MAP_FRAME)
        heatmap_msg.header.stamp = self.ros.get_clock().now().to_msg()
        self.ros.heatmap_pub.publish(heatmap_msg)
        
        if PUBLISH_PATH:
            self.ros.publish_path()
    
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
        
        # Collision
        min_lidar = self._min_lidar()
        if min_lidar < COLLISION_DIST:
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
        min_lidar = self._min_lidar()
        
        progress = self.prev_goal_dist - d_goal
        r_progress = PROGRESS_SCALE * progress
        
        r_align = ALIGN_SCALE * math.cos(ang)
        
        r_step = STEP_COST
        
        r_goal = 0.0
        if d_goal <= GOAL_RADIUS:
            terminated = True
            success = True
            r_goal = R_GOAL
        
        r_collision = 0.0
        if min_lidar < COLLISION_DIST:
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
    
    # ============================================================
    # NEW HELPER FUNCTIONS FOR SMART COLLISION AVOIDANCE
    # ============================================================
    
    def _get_obstacle_angle(self) -> float:
        """
        Find the angle to the closest obstacle.
        Returns angle in radians (-pi to pi) where obstacle is located.
        0 = directly ahead, positive = left, negative = right
        """
        scan = self.ros.last_scan
        if scan is None:
            return 0.0
        
        ranges = np.array(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return 0.0
        
        # Filter invalid readings
        max_r = scan.range_max if scan.range_max > 0 else LIDAR_MAX_RANGE
        min_r = max(scan.range_min, 0.01)
        valid = ~(np.isnan(ranges) | np.isinf(ranges) | (ranges < min_r) | (ranges > max_r))
        
        if not np.any(valid):
            return 0.0
        
        # Find closest point
        ranges[~valid] = max_r
        closest_idx = np.argmin(ranges)
        
        # Convert to angle (accounting for LIDAR_FORWARD_OFFSET)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        obstacle_angle = angle_min + closest_idx * angle_inc
        
        # Normalize to robot frame (0 = forward)
        obstacle_angle = wrap_to_pi(obstacle_angle + LIDAR_FORWARD_OFFSET_RAD)
        
        return float(obstacle_angle)
    
    def _get_escape_direction(self) -> float:
        """
        Determine which direction has more free space.
        Returns -1 (turn right) or +1 (turn left).
        """
        scan = self.ros.last_scan
        if scan is None:
            return 1.0
        
        ranges = np.array(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return 1.0
        
        # Filter valid readings
        max_r = scan.range_max if scan.range_max > 0 else LIDAR_MAX_RANGE
        min_r = max(scan.range_min, 0.01)
        valid = ~(np.isnan(ranges) | np.isinf(ranges) | (ranges < min_r) | (ranges > max_r))
        ranges[~valid] = 0.0
        
        # Split into left and right halves
        mid = len(ranges) // 2
        left_space = np.mean(ranges[:mid])
        right_space = np.mean(ranges[mid:])
        
        # Turn toward more open space
        return 1.0 if left_space > right_space else -1.0
    
    # ============================================================
    # END NEW HELPER FUNCTIONS
    # ============================================================
    
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
    
    def _min_lidar(self) -> float:
        bins = self._get_lidar_bins()
        return float(max(0.01, np.min(bins)))
    
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
# TD3 Networks (unchanged)
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
# TD3 Agent (uses PER and RND)
# =============================================================================

class TD3AgentCNN:
    """TD3 Agent with CNN. Improved with PER and RND integration."""
    
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
        
        # Networks
        self.actor = ActorWithCNN(obs_dim, act_dim).to(device)
        self.actor_targ = ActorWithCNN(obs_dim, act_dim).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())
        
        self.critic1 = CriticWithCNN(obs_dim, act_dim).to(device)
        self.critic2 = CriticWithCNN(obs_dim, act_dim).to(device)
        self.critic1_targ = CriticWithCNN(obs_dim, act_dim).to(device)
        self.critic2_targ = CriticWithCNN(obs_dim, act_dim).to(device)
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
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
        """Update networks using PER. Returns (critic_loss, actor_loss, rnd_loss)."""
        self.total_updates += 1
        
        # Sample with priorities
        obs, act, rew, next_obs, done, weights, indices = replay.sample(batch_size, beta)
        
        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_targ(next_obs) + noise).clamp(-1.0, 1.0)
            
            q1_t = self.critic1_targ(next_obs, next_act)
            q2_t = self.critic2_targ(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t)
            target = rew + (1.0 - done) * self.gamma * q_t
        
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        
        # Weight losses by importance sampling weights
        td_error1 = target - q1
        td_error2 = target - q2
        
        critic_loss = (weights * (td_error1.pow(2) + td_error2.pow(2))).mean()
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic_opt.step()
        
        # Update priorities based on TD error
        td_errors = (td_error1.abs() + td_error2.abs()).detach().cpu().numpy() / 2.0
        replay.update_priorities(indices, td_errors.flatten())
        
        # Actor update (delayed)
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
        
        # Update RND predictor
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
    parser.add_argument("--update-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--replay-size", type=int, default=300_000)
    parser.add_argument("--expl-noise", type=float, default=DEFAULT_EXPL_NOISE)
    parser.add_argument("--save-every", type=int, default=10_000)
    
    # Checkpoint
    parser.add_argument("--ckpt-dir", type=str, default=os.path.expanduser("~/rl_checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    
    # Mode
    parser.add_argument("--inference", action="store_true")
    
    # Compatibility args (ignored)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--load-ckpt", type=str, default="")
    parser.add_argument("--use-obstacle", type=int, default=1)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--episode-num", type=int, default=1)
    parser.add_argument("--models-dir", type=str, default="./models")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Setup checkpoint path
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, CHECKPOINT_FILENAME)
    
    # Initialize ROS
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
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate observation dimension
    grid_flat_size = GRID_SIZE * GRID_SIZE
    obs_dim = NUM_LIDAR_BINS + 5 + 2 + 2 + 1 + grid_flat_size + 2 + 1
    act_dim = 2
    
    # Create RND module
    rnd_module = RNDModule(obs_dim, device=device)
    
    # Create environment
    env = StretchExploreEnv(ros, rnd_module)
    
    ros.get_logger().info(f"[AGENT] device={device} obs_dim={obs_dim} act_dim={act_dim}")
    ros.get_logger().info(f"[AGENT] Grid: {GRID_SIZE}x{GRID_SIZE} @ {GRID_RESOLUTION}m/cell")
    ros.get_logger().info(f"[AGENT] IMPROVED: PER + RND + Smart Collision Avoidance")
    
    # Create agent
    agent = TD3AgentCNN(obs_dim, act_dim, device=device, rnd_module=rnd_module)
    
    # Prioritized replay buffer
    replay = PrioritizedReplayBuffer(obs_dim, act_dim, size=args.replay_size, device=device)
    
    # Load checkpoint if exists
    if AUTO_LOAD_CHECKPOINT and os.path.exists(ckpt_path):
        ros.get_logger().info(f"[CKPT] Loading from {ckpt_path}")
        try:
            agent.load(ckpt_path, strict=False)
            ros.get_logger().info("[CKPT] Load SUCCESS")
        except Exception as e:
            ros.get_logger().warn(f"[CKPT] Load FAILED: {e}")
    
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
        ros.get_logger().info(f"[FINAL] Total cells discovered: {stats['total_discovered']}")
        
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
        ros.get_logger().info("[MODE] INFERENCE")
        while True:
            obs, _ = env.reset()
            done = False
            while not done:
                act = agent.act(obs, noise_std=0.0)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc
    
    # Training mode
    ros.get_logger().info("[MODE] TRAINING (IMPROVED)")
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
        if t >= args.update_after and replay.count >= args.batch_size:
            # Anneal beta for importance sampling
            beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * (t / args.total_steps)
            
            for _ in range(args.update_every):
                critic_loss, actor_loss, rnd_loss = agent.update(replay, args.batch_size, beta)
        
        # Save
        if t - last_save >= args.save_every:
            last_save = t
            stats = env.occ_grid.get_stats()
            ros.get_logger().info(
                f"[CKPT] step={t} cells_discovered={stats['total_discovered']} "
                f"rnd_weight={env.rnd_weight:.3f}"
            )
            try:
                agent.save(ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
            except Exception as e:
                ros.get_logger().warn(f"[CKPT] Save failed: {e}")
    
    shutdown_and_save()


if __name__ == "__main__":
    main()