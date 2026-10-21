"""
Random collision-free start pose for the Stretch base (numpy only).

Obstacles are 2D footprints (rotated boxes and circles) taken from the MuJoCo
world by the driver; this module only does the geometry, so it is unit-tested
without MuJoCo or ROS (test/test_spawn.py).
"""

import math
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np


class Box(NamedTuple):
    cx: float
    cy: float
    hx: float   # half extent along the box's own x axis
    hy: float   # half extent along the box's own y axis
    yaw: float  # box rotation about z (rad)


class Circle(NamedTuple):
    cx: float
    cy: float
    r: float


Obstacle = Union[Box, Circle]


def distance_to_obstacle(x: float, y: float, obs: Obstacle) -> float:
    """2D distance from (x, y) to the obstacle footprint; 0 when inside it."""
    dx, dy = x - obs.cx, y - obs.cy
    if isinstance(obs, Circle):
        return max(math.hypot(dx, dy) - obs.r, 0.0)
    c, s = math.cos(obs.yaw), math.sin(obs.yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    qx = abs(local_x) - obs.hx
    qy = abs(local_y) - obs.hy
    return math.hypot(max(qx, 0.0), max(qy, 0.0))


def sample_spawn_pose(rng: np.random.Generator, obstacles: Sequence[Obstacle],
                      bounds: Tuple[float, float, float, float], clearance: float = 1.0,
                      tries: int = 500) -> Optional[Tuple[float, float, float]]:
    """Uniform (x, y, yaw) inside bounds with every obstacle at least
    `clearance` away. None if no such pose is found in `tries` draws."""
    xmin, xmax, ymin, ymax = bounds
    if xmax - xmin <= 2 * clearance or ymax - ymin <= 2 * clearance:
        return None
    for _ in range(tries):
        x = float(rng.uniform(xmin + clearance, xmax - clearance))
        y = float(rng.uniform(ymin + clearance, ymax - clearance))
        if all(distance_to_obstacle(x, y, o) >= clearance for o in obstacles):
            return x, y, float(rng.uniform(-math.pi, math.pi))
    return None


def yaw_to_quat(yaw: float) -> List[float]:
    """Rotation about z as a MuJoCo quaternion [w, x, y, z]."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
