#!/usr/bin/env python3
"""
Stretch Robot RL Environment + Learner with VFH+ NAVIGATION SYSTEM

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

IMPORTANT FIXES IN THIS VERSION (AVOIDANCE):
1) VFH polar histogram scaling + thresholds fixed (your old TAU_HIGH was unreachable)
2) Goal-aware threat/blending: avoids early *when goal direction is obstructed*,
   but allows close passing if a goal-aligned gap exists.
3) Safer defaults for LiDAR forward offset (set to 0.0; change if your scan frame differs)
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
DESIRED_CLEARANCE_M = 0.25  # Moderate - can get close but not touch
MIN_SAFE_DISTANCE = ROBOT_HALF_WIDTH_M + DESIRED_CLEARANCE_M  # ~0.41m

# =============================================================================
# SAFETY ZONE THRESHOLDS (from LIDAR/center of robot)
# =============================================================================

ZONE_FREE = 1.2
ZONE_AWARE = 0.90
ZONE_CAUTION = 0.65
ZONE_DANGER = 0.45
ZONE_EMERGENCY = 0.30

BLEND_FREE = 0.0
BLEND_AWARE = 0.0
BLEND_CAUTION = 0.15
BLEND_DANGER = 0.40
BLEND_EMERGENCY = 0.85

# =============================================================================
# POTENTIAL FIELD PARAMETERS
# =============================================================================

REPULSIVE_GAIN = 0.8
REPULSIVE_INFLUENCE = 0.8
ATTRACTIVE_GAIN = 0.8

# =============================================================================
# REWARD SHAPING PARAMETERS
# =============================================================================

R_PROXIMITY_SCALE = -25.0
R_CLEARANCE_BONUS = 0.2
R_SAFETY_INTERVENTION = -3.0

# =============================================================================
# GOAL-SEEKING REWARDS
# =============================================================================

R_GOAL = 2000.0
R_COLLISION = -800.0
R_TIMEOUT = -50.0
R_REDETECTION = 300.0

PROGRESS_SCALE = 500.0
ALIGN_SCALE = 15.0
STEP_COST = -0.5
GOAL_RADIUS = 0.45

R_GOAL_PURSUIT = 2.0

LAST_KNOWN_PROGRESS_SCALE = 150.0
LAST_KNOWN_ALIGN_SCALE = 5.0

# =============================================================================
# EXPLORATION REWARDS
# =============================================================================

R_NEW_CELL = 3.0
R_NOVELTY_SCALE = 0.8
R_FRONTIER_BONUS = 5.0
R_COLLISION_EXPLORE = -300.0
R_STEP_EXPLORE = -0.01

# =============================================================================
# MISSION REWARDS
# =============================================================================

R_MISSION_COMPLETE = 5000.0

# =============================================================================
# GENERAL CONFIG
# =============================================================================

CHECKPOINT_FILENAME = "td3_safe_rl_agent.pt"
AUTO_LOAD_CHECKPOINT = True
EPISODE_SECONDS = 60.0

GRID_SIZE = 24
GRID_RESOLUTION = 0.5
GRID_MAX_RANGE = 12.0

V_MAX = 1.25
W_MAX = 7.0
V_MIN_REVERSE = -0.05

RND_WEIGHT_INITIAL = 1.0
RND_WEIGHT_DECAY = 0.99995
RND_WEIGHT_MIN = 0.1
RND_FEATURE_DIM = 64
RND_LR = 1e-4

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPSILON = 1e-6

VISIT_DECAY = 0.995
NOVELTY_RADIUS = 1.0

# LiDAR
# IMPORTANT: In your original file this was math.pi (180deg). If your scan already uses 0=forward,
# that flips front/back logic and can make avoidance look "broken".
LIDAR_FORWARD_OFFSET_RAD = -math.pi  # <-- FIXED DEFAULT (change only if your scan frame needs it)

NUM_LIDAR_BINS = 60
LIDAR_MAX_RANGE = 20.0

# =============================================================================
# VFH+ NAVIGATION PARAMETERS
# =============================================================================

VFH_GRID_SIZE_CELLS = 80
VFH_GRID_RESOLUTION_M = 0.10
VFH_GRID_DECAY_RATE = 0.95
VFH_GRID_MAX_CERTAINTY = 15.0
VFH_GRID_INCREMENT = 2.0

NUM_POLAR_SECTORS = 72
VFH_ACTIVE_RADIUS = 3.0

# --- VFH Polar Histogram ---
# FIX: your old A=5 with TAU_HIGH=8 meant sectors could NEVER become blocked.
# This mapping produces values ~0..20 (close obstacles become large), matching thresholds below.
VFH_A_COEFF = 20.0
VFH_B_COEFF = 8.0

# --- VFH Binary Histogram (threshold hysteresis) ---
VFH_TAU_HIGH = 8.0
VFH_TAU_LOW = 3.0

# --- VFH Masked Histogram (robot geometry) ---
VFH_SAFETY_DIST = 0.12  # slightly less conservative so robot can pass closer if needed

# --- VFH Gap Finding ---
VFH_MIN_GAP_WIDTH = 3
VFH_WIDE_GAP_THRESHOLD = 12

# --- VFH Cost Function Weights ---
VFH_MU_TARGET = 5.0
VFH_MU_CURRENT = 2.0
VFH_MU_PREVIOUS = 2.0

# --- VFH Speed Control ---
# Scale for "density -> speed". With the polar scale above, 15 is a reasonable normalization.
VFH_MAX_DENSITY_FOR_SPEED = 15.0
VFH_TURN_SLOWDOWN = 0.6
VFH_MIN_SPEED_FRACTION = 0.1

# --- VFH Emergency Thresholds ---
VFH_EMERGENCY_DIST = 0.25
VFH_DANGER_DIST = 0.40
VFH_CAUTION_DIST = 0.65

# --- VFH Blending ---
# Encourage *early gentle* avoidance when goal direction is blocked.
VFH_BLEND_THREAT_LOW = 0.05
VFH_BLEND_THREAT_HIGH = 0.60

# --- VFH Observation Space ---
NUM_VFH_OBS_SECTORS = 16
SAFETY_OBS_SIZE = NUM_VFH_OBS_SECTORS + 8

DEFAULT_START_STEPS = 10000
DEFAULT_EXPL_NOISE = 0.3

DEBUG_EVERY_N = 100

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


def angle_diff(a: float, b: float) -> float:
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def sector_to_angle(sector: int, num_sectors: int) -> float:
    return wrap_to_pi(-math.pi + (sector + 0.5) * (2 * math.pi / num_sectors))


def angle_to_sector(angle: float, num_sectors: int) -> int:
    a = wrap_to_pi(angle)
    sector = int((a + math.pi) / (2 * math.pi / num_sectors))
    return sector % num_sectors


# =============================================================================
# VFH+ NAVIGATION SYSTEM
# =============================================================================

@dataclass
class Gap:
    start_sector: int
    end_sector: int
    width_sectors: int
    center_angle: float
    near_edge_angle: float
    far_edge_angle: float
    min_distance: float


@dataclass
class NavigationState:
    min_distance: float
    min_angle: float
    threat_level: float
    selected_direction: float
    selected_speed: float
    num_gaps: int
    best_gap_width: float
    best_gap_center: float
    blend_ratio: float

    polar_histogram: np.ndarray
    sector_distances: np.ndarray
    gaps: List[Gap]

    front_clearance: float
    back_clearance: float
    left_clearance: float
    right_clearance: float

    safety_blend: float
    zone: str
    clear_gaps: List[Tuple[float, float, float]]
    best_direction: float
    best_clearance: float


class HistogramGrid:
    def __init__(
        self,
        size: int = VFH_GRID_SIZE_CELLS,
        resolution: float = VFH_GRID_RESOLUTION_M,
        decay_rate: float = VFH_GRID_DECAY_RATE,
    ):
        self.size = size
        self.resolution = resolution
        self.decay_rate = decay_rate
        self.center = size // 2
        self.grid = np.zeros((size, size), dtype=np.float32)
        self._origin_x = 0.0
        self._origin_y = 0.0

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int((wx - self._origin_x) / self.resolution) + self.center
        gy = int((wy - self._origin_y) / self.resolution) + self.center
        return gx, gy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.size and 0 <= gy < self.size

    def update(self, ranges: np.ndarray, angles: np.ndarray,
               robot_x: float, robot_y: float, robot_yaw: float,
               range_max: float = 20.0, range_min: float = 0.05):
        self.grid *= self.decay_rate

        gx_robot, gy_robot = self.world_to_grid(robot_x, robot_y)
        if not (self.size // 4 < gx_robot < 3 * self.size // 4 and
                self.size // 4 < gy_robot < 3 * self.size // 4):
            self._recenter(robot_x, robot_y)

        n_rays = len(ranges)
        for i in range(0, n_rays, 3):
            r = ranges[i]
            a = angles[i]
            if np.isnan(r) or np.isinf(r) or r < range_min:
                continue

            r_clamped = min(r, VFH_ACTIVE_RADIUS + 0.5)
            ray_angle = robot_yaw + a
            ex = robot_x + r_clamped * math.cos(ray_angle)
            ey = robot_y + r_clamped * math.sin(ray_angle)

            if r < range_max * 0.95:
                egx, egy = self.world_to_grid(ex, ey)
                if self.in_bounds(egx, egy):
                    self.grid[egy, egx] = min(
                        self.grid[egy, egx] + VFH_GRID_INCREMENT,
                        VFH_GRID_MAX_CERTAINTY
                    )

            step = self.resolution * 0.6
            for d in np.arange(step, r_clamped - self.resolution, step):
                fx = robot_x + d * math.cos(ray_angle)
                fy = robot_y + d * math.sin(ray_angle)
                fgx, fgy = self.world_to_grid(fx, fy)
                if self.in_bounds(fgx, fgy):
                    self.grid[fgy, fgx] = max(self.grid[fgy, fgx] - 0.5, 0.0)

    def _recenter(self, robot_x: float, robot_y: float):
        old_gx, old_gy = self.world_to_grid(robot_x, robot_y)
        shift_x = old_gx - self.center
        shift_y = old_gy - self.center
        self.grid = np.roll(self.grid, -shift_y, axis=0)
        self.grid = np.roll(self.grid, -shift_x, axis=1)
        if shift_y > 0:
            self.grid[-shift_y:, :] = 0
        elif shift_y < 0:
            self.grid[:-shift_y, :] = 0
        if shift_x > 0:
            self.grid[:, -shift_x:] = 0
        elif shift_x < 0:
            self.grid[:, :-shift_x] = 0
        self._origin_x += shift_x * self.resolution
        self._origin_y += shift_y * self.resolution

    def reset(self):
        self.grid.fill(0.0)


class DynamicNavigator:
    def __init__(self, num_sectors: int = NUM_POLAR_SECTORS,
                 robot_radius: float = ROBOT_HALF_WIDTH_M):
        self.num_sectors = num_sectors
        self.sector_width = 2 * math.pi / num_sectors
        self.robot_radius = robot_radius
        self.safety_distance = VFH_SAFETY_DIST

        self.sector_angles = np.array([
            wrap_to_pi(-math.pi + (i + 0.5) * self.sector_width)
            for i in range(num_sectors)
        ])

        self.histogram_grid = HistogramGrid()
        self.prev_binary = np.zeros(num_sectors, dtype=np.float32)
        self.prev_selected_direction = 0.0

        self.intervention_history = deque(maxlen=100)
        self.last_nav_state: Optional[NavigationState] = None

        # -----------------------------
        # Turn-commit / latch state
        # -----------------------------
        self._turn_commit_active: bool = False
        self._turn_commit_dir: float = 0.0  # +1 left, -1 right
        self._turn_commit_until: float = 0.0

        # Hysteresis thresholds (tuned to your existing VFH distances)
        self._turn_commit_enter_dist = max(VFH_EMERGENCY_DIST, 0.30) + 0.02  # ~0.27-0.35 region -> commit
        self._turn_commit_exit_front = VFH_CAUTION_DIST + 0.15              # must clear front to exit
        self._turn_commit_min_time = 0.35                                   # seconds: prevents flapping
        self._turn_commit_w = 0.75 * W_MAX                                   # turning speed during commit
        self._turn_commit_v = 0.1                                           # no backup bounce

    def analyze_scan(self, scan: Optional[LaserScan]) -> NavigationState:
        if scan is None or len(scan.ranges) == 0:
            return self._default_state()

        ranges = np.array(scan.ranges, dtype=np.float32)
        n_rays = len(ranges)

        max_r = min(scan.range_max, LIDAR_MAX_RANGE) if scan.range_max > 0 else LIDAR_MAX_RANGE
        min_r = max(scan.range_min, 0.05)

        invalid = np.isnan(ranges) | np.isinf(ranges) | (ranges < min_r) | (ranges > max_r)
        ranges[invalid] = max_r

        # IMPORTANT: LIDAR_FORWARD_OFFSET_RAD must match TF base_link->laser yaw (yours is ~ -pi)
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
            s = angle_to_sector(angles[i], self.num_sectors)
            sector_distances[s] = min(sector_distances[s], ranges[i])

        polar = self._build_polar_histogram(sector_distances, max_r)
        binary = self._build_binary_histogram(polar)
        masked = self._build_masked_histogram(binary, sector_distances)
        gaps = self._find_gaps(masked, sector_distances)

        threat_level = 0.0  # computed goal-aware later
        best_gap = gaps[0] if gaps else None
        best_gap_width = best_gap.width_sectors * self.sector_width if best_gap else 0.0
        best_gap_center = best_gap.center_angle if best_gap else 0.0

        zone = self._get_zone_name(min_distance)
        clear_gaps = [
            (g.center_angle, g.width_sectors * self.sector_width, g.min_distance)
            for g in gaps
        ]

        state = NavigationState(
            min_distance=min_distance,
            min_angle=min_angle,
            threat_level=threat_level,
            selected_direction=best_gap_center,
            selected_speed=V_MAX,
            num_gaps=len(gaps),
            best_gap_width=best_gap_width,
            best_gap_center=best_gap_center,
            blend_ratio=0.0,
            polar_histogram=polar,
            sector_distances=sector_distances,
            gaps=gaps,
            front_clearance=front_clearance,
            back_clearance=back_clearance,
            left_clearance=left_clearance,
            right_clearance=right_clearance,
            safety_blend=0.0,
            zone=zone,
            clear_gaps=clear_gaps,
            best_direction=best_gap_center,
            best_clearance=max(left_clearance, right_clearance),
        )

        self.last_nav_state = state
        return state

    def compute_safe_velocity(
        self, rl_v: float, rl_w: float,
        nav_state: NavigationState,
        goal_angle: Optional[float] = None,
    ) -> Tuple[float, float, float]:

        now = time.time()
        min_dist = nav_state.min_distance

        # -----------------------------
        # 1) Turn-commit latch (prevents oscillation)
        # -----------------------------
        # Enter commit if we're close AND front is the problem (not just a side)
        front_is_threat = (abs(nav_state.min_angle) < (math.pi / 3)) or (nav_state.front_clearance < VFH_CAUTION_DIST)

        if (not self._turn_commit_active
                and front_is_threat
                and min_dist <= self._turn_commit_enter_dist):
            # choose a consistent turn direction toward the more open side / best gap
            if nav_state.gaps:
                best_gap = nav_state.gaps[0]
                self._turn_commit_dir = 1.0 if best_gap.center_angle > 0 else -1.0
            else:
                # fall back to side clearances
                self._turn_commit_dir = 1.0 if nav_state.left_clearance >= nav_state.right_clearance else -1.0

            self._turn_commit_active = True
            self._turn_commit_until = now + self._turn_commit_min_time

        # While committed: do NOT reverse; just rotate until front is clear enough + min time satisfied
        if self._turn_commit_active:
            v = self._turn_commit_v
            w = float(self._turn_commit_dir * self._turn_commit_w)
            if nav_state.min_distance <= (VFH_EMERGENCY_DIST + 0.03):
                v = 0.0  # stop if extremely close, to prevent bounce
            
            if nav_state.front_clearance <= (VFH_EMERGENCY_DIST + 0.05):
                v = 0.0# exit condition: (a) min time passed, (b) front is cleared enough OR we’re no longer near
             # ---------------- CHANGE 4 GOES HERE ----------------
            # Track best front clearance during this commit
            if not hasattr(self, "_commit_best_front"):
                self._commit_best_front = nav_state.front_clearance

            self._commit_best_front = max(self._commit_best_front, nav_state.front_clearance)

            # If after ~0.6s we haven't improved front clearance by ~10cm, flip turn direction
            # (this helps corners / parallel-to-wall cases)
            if (now >= (self._turn_commit_until - 0.35)) and (self._commit_best_front < (nav_state.front_clearance + 0.10)):
                self._turn_commit_dir *= -1.0
                self._turn_commit_until = now + 0.35
                self._commit_best_front = nav_state.front_clearance
            # ----------------------------------------------------

            # exit condition: (a) min time passed, (b) front is cleared enough OR we’re no longer near
            if (now >= self._turn_commit_until) and (
                nav_state.front_clearance >= self._turn_commit_exit_front
                or min_dist >= VFH_CAUTION_DIST
            ):
                self._turn_commit_active = False
                if hasattr(self, "_commit_best_front"):
                    del self._commit_best_front

            # publish high intervention so your logs show we’re taking over
            blend = 0.95
            self.intervention_history.append(blend)
            nav_state.blend_ratio = blend
            nav_state.safety_blend = blend
            nav_state.threat_level = 1.0
            nav_state.selected_direction = float(np.clip(w / 3.0, -math.pi, math.pi))
            nav_state.best_direction = nav_state.selected_direction
            return v, w, blend

        # -----------------------------
        # 2) EMERGENCY behavior: NO backup-bounce
        # -----------------------------
        # Instead of reversing, we enter turn-commit immediately
        if min_dist < VFH_EMERGENCY_DIST and front_is_threat:
            # force commit right now
            if nav_state.gaps:
                best_gap = nav_state.gaps[0]
                self._turn_commit_dir = 1.0 if best_gap.center_angle > 0 else -1.0
            else:
                self._turn_commit_dir = 1.0 if nav_state.left_clearance >= nav_state.right_clearance else -1.0

            self._turn_commit_active = True
            self._turn_commit_until = now + max(self._turn_commit_min_time, 0.45)

            v = 0.0
            w = float(self._turn_commit_dir * self._turn_commit_w)

            blend = 0.95
            self.intervention_history.append(blend)
            nav_state.blend_ratio = blend
            nav_state.safety_blend = blend
            nav_state.threat_level = 1.0
            return v, w, blend

        # -----------------------------
        # 3) Normal VFH blending (goal-aware), but prevent reverse near obstacles
        # -----------------------------
        vfh_direction = self._select_direction(nav_state.gaps, goal_angle, 0.0)
        self.prev_selected_direction = vfh_direction
        nav_state.selected_direction = vfh_direction
        nav_state.best_direction = vfh_direction

        threat = self._goal_aware_threat(nav_state, goal_angle)
        nav_state.threat_level = threat

        vfh_speed = self._compute_speed(vfh_direction, nav_state.polar_histogram, min_dist)
        vfh_w = float(np.clip(3.0 * vfh_direction, -W_MAX, W_MAX))

        if threat < VFH_BLEND_THREAT_LOW:
            blend = 0.0
        elif threat > VFH_BLEND_THREAT_HIGH:
            blend = 0.9
        else:
            t = (threat - VFH_BLEND_THREAT_LOW) / (VFH_BLEND_THREAT_HIGH - VFH_BLEND_THREAT_LOW)
            blend = 0.9 * t * t

        if min_dist < VFH_DANGER_DIST:
            danger_blend = 1.0 - (min_dist - VFH_EMERGENCY_DIST) / (VFH_DANGER_DIST - VFH_EMERGENCY_DIST)
            blend = max(blend, 0.7 * float(np.clip(danger_blend, 0.0, 1.0)))

        safe_v = (1.0 - blend) * rl_v + blend * vfh_speed
        safe_w = (1.0 - blend) * rl_w + blend * vfh_w

        # -----------------------------
        # 4) Front speed limiting: do NOT allow negative here (prevents bounce)
        # -----------------------------
        if min_dist < VFH_CAUTION_DIST and abs(nav_state.min_angle) < math.pi / 3:
            speed_limit = V_MAX * (min_dist - VFH_EMERGENCY_DIST) / (VFH_CAUTION_DIST - VFH_EMERGENCY_DIST)
            speed_limit = float(np.clip(speed_limit, 0.0, V_MAX))  # <- changed: never negative
            safe_v = min(safe_v, speed_limit)

        # If close to obstacles, refuse reverse unless you explicitly want it (you said you don't)
        if min_dist < VFH_CAUTION_DIST:
            safe_v = max(0.0, safe_v)

        safe_v = float(np.clip(safe_v, 0.0, V_MAX))
        safe_w = float(np.clip(safe_w, -W_MAX, W_MAX))

        self.intervention_history.append(blend)
        nav_state.blend_ratio = blend
        nav_state.safety_blend = blend
        return safe_v, safe_w, float(blend)

    # ---- Everything below here is unchanged from your original DynamicNavigator ----

    def _build_polar_histogram(self, sector_distances: np.ndarray,
                               max_range: float) -> np.ndarray:
        polar = np.zeros(self.num_sectors, dtype=np.float32)
        for i in range(self.num_sectors):
            d = sector_distances[i]
            if d < max_range * 0.95:
                magnitude = max(0.0, VFH_A_COEFF - VFH_B_COEFF * d)
                polar[i] = magnitude

        kernel = np.array([0.2, 0.6, 0.2])
        polar_smoothed = np.convolve(
            np.concatenate([polar[-1:], polar, polar[:1]]),
            kernel, mode='valid'
        )
        return polar_smoothed

    def _build_binary_histogram(self, polar: np.ndarray) -> np.ndarray:
        binary = np.zeros(self.num_sectors, dtype=np.float32)
        for i in range(self.num_sectors):
            if polar[i] > VFH_TAU_HIGH:
                binary[i] = 1.0
            elif polar[i] < VFH_TAU_LOW:
                binary[i] = 0.0
            else:
                binary[i] = self.prev_binary[i]
        self.prev_binary = binary.copy()
        return binary

    def _build_masked_histogram(self, binary: np.ndarray,
                                sector_distances: np.ndarray) -> np.ndarray:
        masked = binary.copy()
        enlarge_radius = self.robot_radius + self.safety_distance

        for i in range(self.num_sectors):
            if binary[i] > 0.5:
                dist = sector_distances[i]
                if dist > 0.01:
                    ratio = enlarge_radius / dist
                    if ratio >= 1.0:
                        enlarge_angle = math.pi / 2
                    else:
                        enlarge_angle = math.asin(ratio)
                    sectors_to_block = int(enlarge_angle / self.sector_width) + 1
                    for j in range(-sectors_to_block, sectors_to_block + 1):
                        idx = (i + j) % self.num_sectors
                        masked[idx] = 1.0
        return masked

    def _find_gaps(self, masked: np.ndarray,
                   sector_distances: np.ndarray) -> List[Gap]:
        n = self.num_sectors
        free = (masked < 0.5)

        if not np.any(free):
            return []
        if np.all(free):
            return [Gap(
                start_sector=0, end_sector=n - 1, width_sectors=n,
                center_angle=0.0, near_edge_angle=-math.pi,
                far_edge_angle=math.pi,
                min_distance=float(np.min(sector_distances)),
            )]

        in_gap = False
        start = 0
        runs = []
        for i in range(n):
            if free[i] and not in_gap:
                start = i
                in_gap = True
            elif not free[i] and in_gap:
                runs.append((start, i - 1))
                in_gap = False
        if in_gap:
            runs.append((start, n - 1))

        if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n - 1:
            merged_start = runs[-1][0]
            merged_end = runs[0][1]
            runs = runs[1:-1]
            runs.append((merged_start, merged_end))

        gaps = []
        for (s, e) in runs:
            width = (e - s + 1) if s <= e else (n - s) + (e + 1)
            if width < VFH_MIN_GAP_WIDTH:
                continue
            center_sector = (s + width // 2) % n if s <= e else (s + width // 2) % n
            center_angle = sector_to_angle(center_sector, n)
            start_angle = sector_to_angle(s, n)
            end_angle = sector_to_angle(e, n)
            edge_dist = min(
                sector_distances[s], sector_distances[e],
                sector_distances[center_sector % n]
            )
            gaps.append(Gap(
                start_sector=s, end_sector=e, width_sectors=width,
                center_angle=center_angle, near_edge_angle=start_angle,
                far_edge_angle=end_angle, min_distance=float(edge_dist),
            ))

        gaps.sort(key=lambda g: g.width_sectors, reverse=True)
        return gaps

    def _select_direction(self, gaps: List[Gap],
                          target_angle: Optional[float],
                          current_heading: float = 0.0) -> float:
        if not gaps:
            return current_heading

        target = target_angle if target_angle is not None else 0.0
        prev = self.prev_selected_direction
        best_cost = float('inf')
        best_dir = current_heading

        for gap in gaps:
            if gap.width_sectors > VFH_WIDE_GAP_THRESHOLD:
                candidates = [gap.center_angle, gap.near_edge_angle, gap.far_edge_angle]
            else:
                candidates = [gap.center_angle]

            for c in candidates:
                cost = (
                    VFH_MU_TARGET * abs(angle_diff(c, target)) +
                    VFH_MU_CURRENT * abs(angle_diff(c, current_heading)) +
                    VFH_MU_PREVIOUS * abs(angle_diff(c, prev))
                )
                if cost < best_cost:
                    best_cost = cost
                    best_dir = c
        return best_dir

    def _goal_aware_threat(self, nav_state: NavigationState, goal_angle: Optional[float]) -> float:
        polar = nav_state.polar_histogram
        n = self.num_sectors

        front_sectors = [
            angle_to_sector(a, n)
            for a in np.linspace(-math.pi / 3, math.pi / 3, 20)
        ]
        front_density = float(np.mean([polar[s] for s in front_sectors]))

        goal_density = 0.0
        goal_blocked_bonus = 0.0
        if goal_angle is not None:
            gs = angle_to_sector(goal_angle, n)
            goal_density = float(polar[gs])

            if nav_state.gaps:
                best_dir = self._select_direction(nav_state.gaps, goal_angle, 0.0)
                misalign = abs(angle_diff(best_dir, goal_angle))
                goal_blocked_bonus = float(np.clip(misalign / (math.pi / 2), 0.0, 1.0)) * (0.5 * VFH_MAX_DENSITY_FOR_SPEED)

        density = max(front_density, goal_density + goal_blocked_bonus)
        threat = float(np.clip(density / VFH_MAX_DENSITY_FOR_SPEED, 0.0, 1.0))

        if nav_state.min_distance < VFH_DANGER_DIST:
            threat = max(threat, 1.0 - (nav_state.min_distance / VFH_DANGER_DIST))

        return threat

    def _compute_speed(self, direction: float, polar: np.ndarray,
                       min_distance: float) -> float:
        sector = angle_to_sector(direction, self.num_sectors)
        density = polar[sector]

        density_factor = 1.0 - np.clip(density / VFH_MAX_DENSITY_FOR_SPEED, 0.0, 0.85)
        turn_factor = 1.0 - VFH_TURN_SLOWDOWN * (abs(direction) / math.pi)
        prox_factor = min_distance / VFH_CAUTION_DIST if min_distance < VFH_CAUTION_DIST else 1.0

        speed = V_MAX * density_factor * turn_factor * prox_factor
        return float(max(speed, V_MAX * VFH_MIN_SPEED_FRACTION))

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

    def _default_state(self) -> NavigationState:
        return NavigationState(
            min_distance=LIDAR_MAX_RANGE, min_angle=0.0,
            threat_level=0.0, selected_direction=0.0,
            selected_speed=V_MAX, num_gaps=1,
            best_gap_width=2 * math.pi, best_gap_center=0.0,
            blend_ratio=0.0,
            polar_histogram=np.zeros(self.num_sectors, dtype=np.float32),
            sector_distances=np.full(self.num_sectors, LIDAR_MAX_RANGE, dtype=np.float32),
            gaps=[],
            front_clearance=LIDAR_MAX_RANGE, back_clearance=LIDAR_MAX_RANGE,
            left_clearance=LIDAR_MAX_RANGE, right_clearance=LIDAR_MAX_RANGE,
            safety_blend=0.0, zone="FREE", clear_gaps=[],
            best_direction=0.0, best_clearance=LIDAR_MAX_RANGE,
        )

    def compute_proximity_reward(self, nav_state: NavigationState) -> float:
        min_dist = nav_state.min_distance
        if min_dist >= MIN_SAFE_DISTANCE:
            return R_CLEARANCE_BONUS
        if min_dist >= VFH_DANGER_DIST:
            t = 1.0 - (min_dist - VFH_DANGER_DIST) / (MIN_SAFE_DISTANCE - VFH_DANGER_DIST)
            return R_PROXIMITY_SCALE * 0.2 * t
        elif min_dist >= VFH_EMERGENCY_DIST:
            t = 1.0 - (min_dist - VFH_EMERGENCY_DIST) / (VFH_DANGER_DIST - VFH_EMERGENCY_DIST)
            return R_PROXIMITY_SCALE * 0.5 * (0.5 + 0.5 * t)
        else:
            return R_PROXIMITY_SCALE

    def compute_intervention_penalty(self, intervention: float) -> float:
        return R_SAFETY_INTERVENTION * intervention

    def get_average_intervention(self) -> float:
        if len(self.intervention_history) == 0:
            return 0.0
        return float(np.mean(self.intervention_history))

def build_vfh_observation(nav_state: NavigationState) -> np.ndarray:
    polar = nav_state.polar_histogram
    ratio = max(1, len(polar) // NUM_VFH_OBS_SECTORS)
    sector_obs_raw = np.array([
        np.max(polar[i * ratio:(i + 1) * ratio])
        for i in range(NUM_VFH_OBS_SECTORS)
    ])
    sector_obs = np.clip(sector_obs_raw / VFH_MAX_DENSITY_FOR_SPEED, 0.0, 1.0)

    obs = np.concatenate([
        sector_obs,
        [nav_state.threat_level],
        [nav_state.best_gap_width / math.pi],
        [math.sin(nav_state.best_gap_center), math.cos(nav_state.best_gap_center)],
        [min(nav_state.num_gaps, 10) / 10.0],
        [nav_state.blend_ratio],
        [math.sin(nav_state.min_angle), math.cos(nav_state.min_angle)],
    ])
    return obs.astype(np.float32)


SafetyState = NavigationState
GraduatedSafetyBlender = DynamicNavigator

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
# Random Network Distillation (RND)
# =============================================================================

class RNDModule(nn.Module):
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
# Prioritized Experience Replay (PER)
# =============================================================================

class SumTree:
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
# Ego-Centric Occupancy Grid (UNCHANGED)
# =============================================================================
# NOTE: This class is long; it is unchanged from your original other than being included here.
# (kept exactly as you had it)

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

        for i in range(0, n_rays, 3):
            original_r = ranges[i]
            if np.isnan(original_r) or np.isinf(original_r) or original_r < range_min:
                continue

            r = min(original_r, range_max)
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
        frontier_count = 0
        for key, value in self.world_grid.items():
            if value != 0.5:
                continue
            for ndx, ndy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_key = (key[0] + ndx, key[1] + ndy)
                if neighbor_key not in self.world_grid:
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
        self._last_goal_time: float = 0.0
        self._goal_persist_timeout: float = 60.0

        self._last_known_target_pos: Optional[Tuple[float, float]] = None
        self._last_known_target_time: float = 0.0
        self._last_known_persist_timeout: float = 120.0

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

        self.safety_zone_pub = self.create_publisher(Marker, "/safety_zone", 10)
        self.goal_reached_pub = self.create_publisher(PointStamped, f"/{ns}/goal_reached", 10)

        self.goal_marker_pub = self.create_publisher(Marker, "/goal_marker", 10)
        self.last_known_marker_pub = self.create_publisher(Marker, "/last_known_marker", 10)

        self.path_history: deque = deque(maxlen=PATH_HISTORY_LENGTH)

        self.get_logger().info("[VIZ] Publishing: /exploration_map, /visit_heatmap, /robot_path, /safety_zone, /goal_marker")

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

        if nav_state.zone == "FREE":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif nav_state.zone == "AWARE":
            marker.color.r, marker.color.g, marker.color.b = 0.5, 1.0, 0.0
        elif nav_state.zone == "CAUTION":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        elif nav_state.zone == "DANGER":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0

        marker.color.a = 0.3
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


# =============================================================================
# Gym Environment with Graduated Safety (MOSTLY UNCHANGED)
# =============================================================================
# NOTE: The environment logic is unchanged except it benefits from the navigator fixes above.

class StretchExploreEnv(gym.Env):
    def __init__(self, ros: StretchRosInterface, rnd_module: RNDModule,
                 control_dt: float = 0.1):
        super().__init__()
        self.ros = ros
        self.control_dt = control_dt
        self.rnd = rnd_module

        self.navigator = DynamicNavigator(num_sectors=NUM_POLAR_SECTORS, robot_radius=ROBOT_HALF_WIDTH_M)
        self.occ_grid = EgoOccupancyGrid()

        self.obs_rms = None
        self.reward_normalizer = RewardNormalizer()
        self.rnd_weight = RND_WEIGHT_INITIAL

        self.step_count = 0
        self.total_steps = 0
        self.max_steps = int(EPISODE_SECONDS / control_dt)
        self.episode_index = 1
        self.episode_return = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_goal_dist = 0.0

        self._last_goal_reached_pos = None
        self._min_goal_separation = 1.5
        self._goals_reached_this_episode = 0

        self.episode_interventions = []

        grid_flat_size = GRID_SIZE * GRID_SIZE
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

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))

        self.last_nav_state: Optional[NavigationState] = None
        self.last_safety_state = None

        self.ros.get_logger().info("[ENV] Waiting for sensors...")
        self.ros.wait_for_sensors()

        self.ros.get_logger().info(f"[ENV] Observation dim: {obs_dim}")
        self.ros.get_logger().info(f"[ENV] Robot width: {ROBOT_WIDTH_M:.2f}m, Clearance: {DESIRED_CLEARANCE_M:.2f}m")
        self.ros.get_logger().info(f"[ENV] Safety zones: FREE>{ZONE_FREE}m, AWARE>{ZONE_AWARE}m, "
                                   f"CAUTION>{ZONE_CAUTION}m, DANGER>{ZONE_DANGER}m, EMERGENCY>{ZONE_EMERGENCY}m")
        self.ros.get_logger().info(f"[ENV] VFH+ NAVIGATION ENABLED (sectors={NUM_POLAR_SECTORS}, robot_r={ROBOT_HALF_WIDTH_M:.3f}m)")
        self.ros.get_logger().info(f"[ENV] VFH FIX: A={VFH_A_COEFF:.1f}, B={VFH_B_COEFF:.1f}, TAU_HIGH={VFH_TAU_HIGH:.1f}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.episode_return = 0.0
        self.prev_action[:] = 0.0
        self.occ_grid.reset()
        self.episode_interventions = []
        self._goals_reached_this_episode = 0
        self._had_goal_last_step = False

        self.occ_grid.decay_visits()
        self.prev_goal_dist = self._goal_distance()

        self.last_nav_state = self.navigator.analyze_scan(self.ros.last_scan)
        self.last_safety_state = self.last_nav_state

        obs = self._build_observation()
        info = {
            "has_goal": self.ros.last_goal is not None,
            "goal_dist": self.prev_goal_dist,
            "nav_zone": self.last_nav_state.zone if self.last_nav_state else "UNKNOWN"
        }
        return obs, info

    def step(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)
        rl_v = float(a[0]) * V_MAX
        rl_w = float(a[1]) * W_MAX

        if rl_v < V_MIN_REVERSE:
            rl_v = V_MIN_REVERSE

        nav_state = self.navigator.analyze_scan(self.ros.last_scan)
        self.last_nav_state = nav_state
        self.last_safety_state = nav_state

        goal_angle = self._goal_angle() if self.ros.last_goal is not None else None

        safe_v, safe_w, intervention = self.navigator.compute_safe_velocity(
            rl_v, rl_w, nav_state, goal_angle
        )

        self.episode_interventions.append(intervention)

        self.ros.send_cmd(safe_v, safe_w)

        t_end = time.time() + self.control_dt
        while time.time() < t_end:
            rclpy.spin_once(self.ros, timeout_sec=0.01)

        st = self._get_robot_state()
        scan = self.ros.last_scan
        new_cells = 0
        if scan is not None:
            new_cells = self.occ_grid.update_from_scan(
                st["x"], st["y"], st["yaw"], scan
            )

        nav_state = self.navigator.analyze_scan(self.ros.last_scan)
        self.last_nav_state = nav_state
        self.last_safety_state = nav_state

        obs = self._build_observation()

        has_goal = self.ros.last_goal is not None

        if has_goal:
            goal_age = time.time() - self.ros._last_goal_time
            if goal_age > self.ros._goal_persist_timeout:
                self.ros.last_goal = None
                has_goal = False
                self.ros._target_was_lost = True
                self.ros._lost_time = time.time()
                if self.step_count % 100 == 0:
                    self.ros.get_logger().info(f'[GOAL] Cleared stale goal (age: {goal_age:.1f}s > {self.ros._goal_persist_timeout:.0f}s)')

        if self.ros._last_known_target_pos is not None:
            last_known_age = time.time() - self.ros._last_known_target_time
            if last_known_age > self.ros._last_known_persist_timeout:
                self.ros._last_known_target_pos = None

        if has_goal and self._last_goal_reached_pos is not None:
            goal_x = self.ros.last_goal.point.x
            goal_y = self.ros.last_goal.point.y
            last_x, last_y = self._last_goal_reached_pos
            dist_from_last_goal = math.hypot(goal_x - last_x, goal_y - last_y)
            if dist_from_last_goal < self._min_goal_separation:
                has_goal = False
            else:
                self._last_goal_reached_pos = None

        r_redetection = 0.0
        if has_goal and self.ros._target_was_lost:
            r_redetection = R_REDETECTION
            self.ros._target_was_lost = False
            self.ros.get_logger().info(f'[REDETECT] Target re-acquired! Bonus +{R_REDETECTION:.0f}')

        if has_goal and not self._had_goal_last_step:
            d_goal = self._goal_distance()
            self.ros.get_logger().info(f'[MODE CHANGE] EXPLORE -> GOAL | Target acquired at {d_goal:.2f}m')
        elif not has_goal and self._had_goal_last_step:
            self.ros.get_logger().info(f'[MODE CHANGE] GOAL -> EXPLORE | Target lost')
        self._had_goal_last_step = has_goal

        if has_goal:
            reward, terminated, info = self._compute_goal_reward(safe_v, safe_w, new_cells)
            reward += r_redetection
            if r_redetection > 0:
                info["reward_terms"]["redetection"] = r_redetection
        else:
            reward, terminated, info = self._compute_explore_reward(new_cells, st, obs)

        proximity_reward = self.navigator.compute_proximity_reward(nav_state)
        intervention_penalty = self.navigator.compute_intervention_penalty(intervention)
        reward += proximity_reward + intervention_penalty

        info["reward_terms"]["proximity"] = proximity_reward
        info["reward_terms"]["intervention_penalty"] = intervention_penalty
        info["nav_zone"] = nav_state.zone
        info["safety_blend"] = nav_state.safety_blend
        info["intervention"] = intervention
        info["executed_v"] = safe_v
        info["executed_w"] = safe_w
        info["rl_v"] = rl_v
        info["rl_w"] = rl_w
        info["best_direction"] = nav_state.best_direction
        info["num_gaps"] = len(nav_state.clear_gaps)
        info["threat_level"] = nav_state.threat_level

        self.total_steps += 1
        self.rnd_weight = max(
            RND_WEIGHT_MIN,
            RND_WEIGHT_INITIAL * (RND_WEIGHT_DECAY ** self.total_steps)
        )

        self.episode_return += reward
        self.prev_action[:] = a
        self.step_count += 1

        truncated = self.step_count >= self.max_steps

        stats = self.occ_grid.get_stats()
        mission_complete = False
        if stats['fully_explored'] and self._goals_reached_this_episode > 0:
            terminated = True
            mission_complete = True
            info["mission_complete"] = True
            reward += R_MISSION_COMPLETE
            self.episode_return += R_MISSION_COMPLETE
            info["reward_terms"]["mission_complete"] = R_MISSION_COMPLETE
            self.ros.get_logger().info(
                f"[MISSION COMPLETE] Explored {stats['total_discovered']} cells, "
                f"reached {self._goals_reached_this_episode} goal(s), bonus +{R_MISSION_COMPLETE:.0f}"
            )

        if terminated or truncated:
            mode = "GOAL" if has_goal else "EXPLORE"
            if mission_complete:
                status = "MISSION_COMPLETE"
            elif info.get("success", False):
                status = "SUCCESS"
            elif info.get("collision", False):
                status = "COLLISION"
            else:
                status = "TIMEOUT"
            avg_intervention = np.mean(self.episode_interventions) if self.episode_interventions else 0.0
            self.ros.get_logger().info(
                f"[EP {self.episode_index:04d}] {mode} {status} | "
                f"Return {self.episode_return:+.1f} | Steps {self.step_count} | "
                f"Cells {stats['total_discovered']} | Frontiers {stats['frontier_count']} | "
                f"Goals {self._goals_reached_this_episode} | "
                f"Intervention {avg_intervention:.1%}"
            )
            self.episode_index += 1

        if self.step_count % DEBUG_EVERY_N == 0:
            mode = "GOAL" if has_goal else "EXPLORE"
            goal_info = ""
            if has_goal:
                d_goal = self._goal_distance()
                goal_info = f"goal_dist={d_goal:.2f}m "
            self.ros.get_logger().info(
                f"[{mode}] step={self.step_count} zone={nav_state.zone} "
                f"blend={nav_state.safety_blend:.0%} threat={nav_state.threat_level:.2f} min_d={nav_state.min_distance:.2f}m "
                f"gaps={nav_state.num_gaps} best_dir={math.degrees(nav_state.best_direction):.0f}° "
                f"{goal_info}"
                f"rl=[{rl_v:.2f},{rl_w:.2f}] safe=[{safe_v:.2f},{safe_w:.2f}] "
                f"frontiers={stats['frontier_count']} goals={self._goals_reached_this_episode}"
            )

        self._publish_reward_breakdown(reward, info, has_goal)

        if PUBLISH_MAP and self.step_count % PUBLISH_MAP_EVERY_N == 0:
            self._publish_visualization(st, nav_state)

        return obs, float(reward), bool(terminated), bool(truncated), info

    # --- visualization + reward + goal/explore reward code ---
    # These are unchanged from your original, so kept as-is below.

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
        self._publish_nav_direction(robot_state, nav_state)

    def _publish_nav_direction(self, robot_state: Dict, nav_state: NavigationState):
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = self.ros.get_clock().now().to_msg()
        marker.ns = "nav_direction"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

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
        marker.scale.x = 0.08
        marker.scale.y = 0.15
        marker.scale.z = 0.15

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.9

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 200000000

        self.ros.safety_zone_pub.publish(marker)

    def _publish_goal_markers(self):
        now = self.ros.get_clock().now().to_msg()

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
            delete_marker = Marker()
            delete_marker.header.stamp = now
            delete_marker.header.frame_id = "odom"
            delete_marker.ns = "goal"
            delete_marker.id = 0
            delete_marker.action = Marker.DELETE
            self.ros.goal_marker_pub.publish(delete_marker)

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

    # --- reward functions are unchanged from your original ---
    # (they are long; kept identical to preserve your training behavior)

    def _compute_explore_reward(self, new_cells: int, st: Dict, obs: np.ndarray) -> Tuple[float, bool, Dict]:
        reward = 0.0
        terminated = False
        collision = False

        r_discovery = R_NEW_CELL * new_cells
        novelty = self.occ_grid.get_novelty(st["x"], st["y"])
        r_novelty = R_NOVELTY_SCALE * novelty

        frontier_angle = self.occ_grid.get_frontier_direction(st["x"], st["y"], st["yaw"])
        if abs(frontier_angle) < math.pi / 3 and st["v_lin"] > 0.05:
            r_frontier = R_FRONTIER_BONUS * math.cos(frontier_angle)
        else:
            r_frontier = 0.0

        r_step = R_STEP_EXPLORE
        r_rnd = self.rnd.compute_intrinsic_reward(obs) * self.rnd_weight

        r_movement = 0.0
        if st["v_lin"] > 0.2:
            r_movement = 1.0
        elif st["v_lin"] < 0.05 and abs(st["v_ang"]) < 0.1:
            r_movement = -0.5
        elif abs(st["v_ang"]) > 0.5 and st["v_lin"] < 0.1:
            r_movement = -0.3

        r_spin_penalty = 0.0
        if st["v_lin"] < 0.1 and abs(st["v_ang"]) > 1.5:
            r_spin_penalty = -3.0

        r_last_known_progress = 0.0
        r_last_known_align = 0.0
        r_last_known_forward = 0.0
        if self.ros._last_known_target_pos is not None:
            lk_x, lk_y = self.ros._last_known_target_pos
            robot_x, robot_y = st["x"], st["y"]

            dist_to_last_known = math.hypot(lk_x - robot_x, lk_y - robot_y)

            if not hasattr(self, '_prev_last_known_dist'):
                self._prev_last_known_dist = dist_to_last_known

            progress = self._prev_last_known_dist - dist_to_last_known
            r_last_known_progress = LAST_KNOWN_PROGRESS_SCALE * progress * 2.0
            self._prev_last_known_dist = dist_to_last_known

            angle_to_last_known = math.atan2(lk_y - robot_y, lk_x - robot_x) - st["yaw"]
            while angle_to_last_known > math.pi:
                angle_to_last_known -= 2 * math.pi
            while angle_to_last_known < -math.pi:
                angle_to_last_known += 2 * math.pi

            if st["v_lin"] > 0.1:
                r_last_known_align = LAST_KNOWN_ALIGN_SCALE * math.cos(angle_to_last_known) * 2.0

            if st["v_lin"] > 0.2 and abs(angle_to_last_known) < math.pi / 2:
                r_last_known_forward = 2.0

        r_collision = 0.0
        if self.last_safety_state and self.last_safety_state.min_distance < ZONE_EMERGENCY:
            if self.step_count > 10:
                terminated = True
                collision = True
                r_collision = R_COLLISION_EXPLORE

        reward = (r_discovery + r_novelty + r_frontier + r_step + r_collision + r_rnd +
                  r_movement + r_spin_penalty + r_last_known_progress + r_last_known_align + r_last_known_forward)

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
                "movement": r_movement,
                "spin_penalty": r_spin_penalty,
                "last_known_progress": r_last_known_progress,
                "last_known_align": r_last_known_align,
                "last_known_forward": r_last_known_forward,
            }
        }

        return float(reward), terminated, info

    def _compute_goal_reward(self, v_cmd: float, w_cmd: float, new_cells: int) -> Tuple[float, bool, Dict]:
        reward = 0.0
        terminated = False
        collision = False
        success = False

        d_goal = self._goal_distance()
        ang = self._goal_angle()

        progress = self.prev_goal_dist - d_goal
        r_progress = PROGRESS_SCALE * progress

        r_align = 0.0
        if v_cmd > 0.15:
            r_align = ALIGN_SCALE * math.cos(ang)

        r_step = STEP_COST

        r_spin_penalty = 0.0
        if v_cmd < 0.1 and abs(w_cmd) > 1.0:
            r_spin_penalty = -8.0

        r_forward = 0.0
        if v_cmd > 0.2:
            r_forward = 3.0
        elif v_cmd < 0.05 and abs(w_cmd) < 0.5:
            r_forward = -2.0

        r_stall = 0.0
        if abs(progress) < 0.01 and v_cmd < 0.15:
            r_stall = -3.0

        r_pursuit = 0.0
        if progress > 0.01 and v_cmd > 0.1:
            r_pursuit = R_GOAL_PURSUIT * 2
        elif progress < -0.01:
            r_pursuit = -R_GOAL_PURSUIT * 3

        r_goal = 0.0
        if d_goal <= GOAL_RADIUS:
            success = True
            r_goal = R_GOAL

            self._goals_reached_this_episode += 1
            self._last_goal_reached_pos = (self.ros.last_goal.point.x, self.ros.last_goal.point.y)

            reached_msg = PointStamped()
            reached_msg.header.stamp = self.ros.get_clock().now().to_msg()
            reached_msg.header.frame_id = "odom"
            reached_msg.point.x = self.ros.last_goal.point.x
            reached_msg.point.y = self.ros.last_goal.point.y
            reached_msg.point.z = 0.0
            self.ros.goal_reached_pub.publish(reached_msg)

            self.ros.last_goal = None

            self.ros.get_logger().info(
                f"[GOAL REACHED #{self._goals_reached_this_episode}] dist={d_goal:.2f}m, reward={r_goal:.0f}, continuing exploration"
            )

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

        reward = (r_progress + r_align + r_step + r_pursuit + r_goal +
                  r_collision + r_timeout + r_spin_penalty + r_forward + r_stall)

        info = {
            "collision": collision,
            "success": success,
            "exploring": False,
            "goal_dist": d_goal,
            "reward_terms": {
                "progress": r_progress,
                "alignment": r_align,
                "step": r_step,
                "pursuit": r_pursuit,
                "goal": r_goal,
                "collision": r_collision,
                "timeout": r_timeout,
                "spin_penalty": r_spin_penalty,
                "forward_bonus": r_forward,
                "stall_penalty": r_stall,
            }
        }

        return float(reward), terminated, info

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
                np.clip(dx / cap, -1, 1),
                np.clip(dy / cap, -1, 1),
                np.clip(dist / cap, 0, 1),
                math.sin(ang),
                math.cos(ang)
            ], dtype=np.float32)
        else:
            goal_info = np.zeros(5, dtype=np.float32)

        vel = np.array([
            np.clip(st["v_lin"] / V_MAX, -1, 1),
            np.clip(st["v_ang"] / W_MAX, -1, 1)
        ], dtype=np.float32)

        prev_act = self.prev_action.copy()
        has_goal = np.array([1.0 if goal is not None else 0.0], dtype=np.float32)

        grid_flat = self.occ_grid.get_flat_grid()
        grid_norm = (grid_flat - 0.5) * 2.0

        frontier_ang = self.occ_grid.get_frontier_direction(st["x"], st["y"], st["yaw"])
        frontier_dir = np.array([math.sin(frontier_ang), math.cos(frontier_ang)], dtype=np.float32)

        novelty = np.array([self.occ_grid.get_novelty(st["x"], st["y"])], dtype=np.float32)

        if self.last_nav_state is not None:
            safety_obs = build_vfh_observation(self.last_nav_state)
        else:
            safety_obs = np.zeros(SAFETY_OBS_SIZE, dtype=np.float32)

        obs = np.concatenate([
            lidar_norm,
            goal_info,
            vel,
            prev_act,
            has_goal,
            grid_norm,
            frontier_dir,
            novelty,
            safety_obs,
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
            "mission_complete": info.get("mission_complete", False),
            "goals_reached": self._goals_reached_this_episode,
            "navigation": {
                "zone": info.get("nav_zone", "UNKNOWN"),
                "blend": info.get("safety_blend", 0.0),
                "intervention": info.get("intervention", 0.0),
                "threat": info.get("threat_level", 0.0),
                "rl_v": info.get("rl_v", 0.0),
                "rl_w": info.get("rl_w", 0.0),
                "executed_v": info.get("executed_v", 0.0),
                "executed_w": info.get("executed_w", 0.0),
                "best_direction": info.get("best_direction", 0.0),
                "num_gaps": info.get("num_gaps", 0),
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
# TD3 Networks + Agent (UNCHANGED)
# =============================================================================
# NOTE: These parts are exactly your original (included below for completeness).

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


class TD3AgentCNN:
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

    parser.add_argument("--ns", type=str, default="")
    parser.add_argument("--odom-topic", type=str, default="/stretch/odom")
    parser.add_argument("--lidar-topic", type=str, default="/stretch/scan")
    parser.add_argument("--imu-topic", type=str, default="/imu_mobile_base")
    parser.add_argument("--goal-topic", type=str, default="goal")
    parser.add_argument("--cmd-topic", type=str, default="/stretch/cmd_vel")

    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--start-steps", type=int, default=DEFAULT_START_STEPS)
    parser.add_argument("--update-after", type=int, default=2000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=300_000)
    parser.add_argument("--expl-noise", type=float, default=DEFAULT_EXPL_NOISE)
    parser.add_argument("--save-every", type=int, default=10_000)

    parser.add_argument("--ckpt-dir", type=str, default=os.path.expanduser("~/rl_checkpoints"))
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

    grid_flat_size = GRID_SIZE * GRID_SIZE
    obs_dim = (NUM_LIDAR_BINS + 5 + 2 + 2 + 1 + grid_flat_size + 2 + 1
               + SAFETY_OBS_SIZE)
    act_dim = 2

    rnd_module = RNDModule(obs_dim, device=device)
    env = StretchExploreEnv(ros, rnd_module)

    ros.get_logger().info(f"[AGENT] device={device} obs_dim={obs_dim} act_dim={act_dim}")
    ros.get_logger().info(f"[AGENT] VFH+ NAVIGATION ACTIVE")
    ros.get_logger().info(f"[AGENT] Robot: {ROBOT_WIDTH_M:.2f}m wide, {DESIRED_CLEARANCE_M:.2f}m clearance")
    ros.get_logger().info(f"[AGENT] LIDAR_FORWARD_OFFSET_RAD={LIDAR_FORWARD_OFFSET_RAD:.2f} rad")

    agent = TD3AgentCNN(obs_dim, act_dim, device=device, rnd_module=rnd_module)
    replay = PrioritizedReplayBuffer(obs_dim, act_dim, size=args.replay_size, device=device)

    if AUTO_LOAD_CHECKPOINT and os.path.exists(ckpt_path):
        ros.get_logger().info(f"[CKPT] Loading from {ckpt_path}")
        try:
            agent.load(ckpt_path, strict=False)
            ros.get_logger().info("[CKPT] Load SUCCESS")
        except Exception as e:
            ros.get_logger().warn(f"[CKPT] Load FAILED (model architecture changed?): {e}")

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
        avg_intervention = env.navigator.get_average_intervention()
        ros.get_logger().info(f"[FINAL] Cells discovered: {stats['total_discovered']}, "
                              f"Avg navigation intervention: {avg_intervention:.1%}")

        try:
            executor.shutdown()
            ros.destroy_node()
            rclpy.shutdown()
        except:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_and_save)
    signal.signal(signal.SIGTERM, shutdown_and_save)

    if args.inference:
        ros.get_logger().info("[MODE] INFERENCE (with VFH navigation)")
        while True:
            obs, _ = env.reset()
            done = False
            while not done:
                act = agent.act(obs, noise_std=0.0)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc

    ros.get_logger().info("[MODE] TRAINING (with VFH navigation)")
    obs, _ = env.reset()
    last_save = 0

    for t in range(1, args.total_steps + 1):
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

        if t >= args.update_after and t % args.update_every == 0 and replay.count >= args.batch_size:
            beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * (t / args.total_steps)
            critic_loss, actor_loss, rnd_loss = agent.update(replay, args.batch_size, beta)

        if t - last_save >= args.save_every:
            last_save = t
            stats = env.occ_grid.get_stats()
            avg_intervention = env.navigator.get_average_intervention()
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