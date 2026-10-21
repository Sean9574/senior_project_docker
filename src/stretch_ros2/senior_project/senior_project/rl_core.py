#!/usr/bin/env python3
"""
Pure learning logic for the Stretch RL learner (numpy only, no ROS or torch).

Kept separate from learner_node.py so the parts that decide whether the agent
can learn are unit-tested without a simulator: see test/test_rl_core.py.
"""

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

# =============================================================================
# Motion limits (real Stretch base: Nav2 config caps 0.26 m/s, 1.0 rad/s)
# =============================================================================

V_MAX = 0.3
W_MAX = 1.0
V_MIN_FORWARD = 0.1      # explore mode always moves (anti-spin)
MIN_TURN_RADIUS = 0.2    # explore mode: w_limit = v / MIN_TURN_RADIUS

# Goal sampling
GOAL_LOS_HALF_WIDTH_RAD = math.radians(10.0)
GOAL_FALLBACK_BEARINGS = 72  # every 5 degrees


def wrap_to_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


# =============================================================================
# LiDAR cleaning
# =============================================================================

def compute_self_mask(ranges: np.ndarray, range_min: float) -> np.ndarray:
    """Beams that hit closer than range_min while the robot is in open space
    are hitting the robot's own body. MuJoCo reports a miss as -1, so only
    0 <= r < range_min counts."""
    r = np.asarray(ranges, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(r) & (r >= 0.0) & (r < range_min)


def clean_ranges(ranges: np.ndarray, range_min: float, range_max: float,
                 self_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return ranges in [range_min, range_max]:
    miss (r < 0), NaN, inf, beyond range_max, or self-hit -> range_max;
    real contact (0 <= r < range_min) -> range_min."""
    r = np.array(ranges, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        no_hit = ~np.isfinite(r) | (r < 0.0) | (r > range_max)
        too_close = ~no_hit & (r < range_min)
    r[too_close] = range_min
    r[no_hit] = range_max
    if self_mask is not None and len(self_mask) == len(r):
        r[np.asarray(self_mask, dtype=bool)] = range_max
    return r


# =============================================================================
# Action mapping
# =============================================================================

def map_action(a: np.ndarray, has_goal: bool) -> Tuple[float, float]:
    """Map a normalized action in [-1, 1]^2 to (v m/s, w rad/s).

    Goal mode: v in [0, V_MAX] (can stop), w in [-W_MAX, W_MAX] (can pivot).
    Explore mode: v in [V_MIN_FORWARD, V_MAX], turn rate coupled to speed.
    """
    a0 = float(np.clip(a[0], -1.0, 1.0))
    a1 = float(np.clip(a[1], -1.0, 1.0))
    if has_goal:
        return (a0 + 1.0) * 0.5 * V_MAX, a1 * W_MAX
    v = V_MIN_FORWARD + (a0 + 1.0) * 0.5 * (V_MAX - V_MIN_FORWARD)
    w_limit = min(v / MIN_TURN_RADIUS, W_MAX)
    return v, a1 * w_limit


# =============================================================================
# Reward
# =============================================================================

def goal_progress_reward(prev_d: float, d: float, scale: float) -> float:
    """Potential-based progress: moving away costs what moving closer pays."""
    if not (math.isfinite(prev_d) and math.isfinite(d)):
        return 0.0
    return scale * (prev_d - d)


# =============================================================================
# Goal sampling + curriculum
# =============================================================================

def _clearance(ranges: np.ndarray, beam_angles: np.ndarray, bearing: float) -> float:
    near = np.abs(wrap_to_pi(beam_angles - bearing)) <= GOAL_LOS_HALF_WIDTH_RAD
    if not np.any(near):
        return 0.0
    return float(np.min(ranges[near]))


def sample_goal(rng: np.random.Generator, ranges: np.ndarray,
                angle_min: float, angle_inc: float, angle_offset: float,
                robot_x: float, robot_y: float, robot_yaw: float,
                d_min: float, d_max: float, margin: float = 0.5,
                tries: int = 50) -> Tuple[float, float, float]:
    """Sample a goal the robot can see: (gx, gy, dist) in the odom frame.

    A bearing (robot frame) and distance are drawn uniformly; the goal is kept
    if every beam within +/-10 deg of that bearing is clear past dist + margin.
    If no draw passes, use the most open bearing at the largest distance that
    fits, clipped to [d_min, d_max].
    """
    r = np.nan_to_num(np.asarray(ranges, dtype=np.float64), nan=0.0, posinf=1e6)
    beam_angles = wrap_to_pi(angle_min + np.arange(len(r)) * angle_inc + angle_offset)

    bearing, dist = None, None
    for _ in range(tries):
        b = float(rng.uniform(-math.pi, math.pi))
        d = float(rng.uniform(d_min, d_max))
        if _clearance(r, beam_angles, b) >= d + margin:
            bearing, dist = b, d
            break

    if bearing is None:
        candidates = np.linspace(-math.pi, math.pi, GOAL_FALLBACK_BEARINGS, endpoint=False)
        clear = [_clearance(r, beam_angles, b) for b in candidates]
        best = int(np.argmax(clear))
        bearing = float(candidates[best])
        dist = float(np.clip(clear[best] - margin, d_min, d_max))

    heading = robot_yaw + bearing
    return robot_x + dist * math.cos(heading), robot_y + dist * math.sin(heading), dist


class GoalCurriculum:
    """Start with near goals; push them out once the agent reaches them.

    d_max rises by `step` (up to `d_cap`) when the last `window` episodes since
    the previous raise have a success rate >= `threshold`.
    """

    def __init__(self, d_min: float = 1.0, d_start: float = 1.5, d_cap: float = 4.0,
                 step: float = 0.5, window: int = 20, threshold: float = 0.6):
        self.d_min = d_min
        self.d_max = d_start
        self.d_cap = d_cap
        self.step = step
        self.window = window
        self.threshold = threshold
        self._recent: deque = deque(maxlen=window)

    def record(self, success: bool) -> None:
        self._recent.append(bool(success))
        if len(self._recent) < self.window or self.d_max >= self.d_cap:
            return
        if sum(self._recent) >= self.threshold * self.window - 1e-9:
            self.d_max = min(self.d_max + self.step, self.d_cap)
            self._recent.clear()

    def state(self) -> dict:
        return {"d_max": self.d_max, "recent": list(self._recent)}

    def load(self, state: dict) -> None:
        self.d_max = float(state.get("d_max", self.d_max))
        self._recent = deque((bool(x) for x in state.get("recent", [])), maxlen=self.window)


# =============================================================================
# Training phase
# =============================================================================

# Episodes that move the goal curriculum: the controller being measured.
# Expert-demo and random-exploration episodes must not push goals out before
# the policy acts; eval runs pin the curriculum.
CURRICULUM_PHASES = ("RL_POLICY", "EXPERT_BASELINE")


def counts_for_curriculum(phase: str) -> bool:
    return phase in CURRICULUM_PHASES


def phase_for_step(t: int, demo_steps: int, start_steps: int) -> str:
    """Phase label for global step t (also correct for resumed runs)."""
    if t <= demo_steps:
        return "EXPERT_DEMO"
    if t < demo_steps + start_steps:
        return "RANDOM_EXPLORE"
    return "RL_POLICY"
