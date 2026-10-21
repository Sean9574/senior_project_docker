#!/usr/bin/env python3
"""
Pure learning logic for the Stretch RL learner (numpy only, no ROS or torch).

Kept separate from learner_node.py so the parts that decide whether the agent
can learn are unit-tested without a simulator: see test/test_rl_core.py.
"""

import math
from typing import Optional, Tuple

import numpy as np

# =============================================================================
# Motion limits (real Stretch base: Nav2 config caps 0.26 m/s, 1.0 rad/s)
# =============================================================================

V_MAX = 0.3
W_MAX = 1.0
V_MIN_FORWARD = 0.1      # explore mode always moves (anti-spin)
MIN_TURN_RADIUS = 0.2    # explore mode: w_limit = v / MIN_TURN_RADIUS

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
# Training phase
# =============================================================================

def phase_for_step(t: int, demo_steps: int, start_steps: int) -> str:
    """Phase label for global step t (also correct for resumed runs)."""
    if t <= demo_steps:
        return "EXPERT_DEMO"
    if t < demo_steps + start_steps:
        return "RANDOM_EXPLORE"
    return "RL_POLICY"


# =============================================================================
# Vectorised training (grid arena: N robots, one agent, one replay buffer)
# =============================================================================

def vector_update_count(n_new: int, update_every: int, carry: int) -> Tuple[int, int]:
    """Gradient updates owed for n_new transitions at one update per
    `update_every` transitions; leftovers carry to the next step."""
    total = carry + n_new
    return total // update_every, total % update_every


def vector_transitions(prev_obs: list, actions: list, results: list, reset_fn) -> Tuple[list, list, list]:
    """Turn one vector step into replay transitions.

    results[i] = (next_obs, reward, terminated, truncated, info) of robot i.
    Transition i = (prev_obs[i], actions[i], reward, next_obs, terminated):
    only a true terminal state stops bootstrapping, a timeout does not.
    Robots whose episode ended are reset with reset_fn(i) -> new first obs.
    Returns (transitions, observations for the next step, finished indices).
    """
    transitions, next_obs, finished = [], [], []
    for i, (obs, reward, terminated, truncated, _info) in enumerate(results):
        transitions.append((prev_obs[i], actions[i], reward, obs, bool(terminated)))
        if terminated or truncated:
            finished.append(i)
            next_obs.append(reset_fn(i))
        else:
            next_obs.append(obs)
    return transitions, next_obs, finished
