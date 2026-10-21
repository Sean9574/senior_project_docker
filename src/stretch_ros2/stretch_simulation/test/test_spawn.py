"""Random spawn pose sampler (numpy only, no MuJoCo or ROS needed).

Run: python3 -m pytest src/stretch_ros2/stretch_simulation/test/test_spawn.py
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stretch_mujoco_driver import spawn  # noqa: E402
from stretch_mujoco_driver.spawn import Box, Circle  # noqa: E402

# 10 x 10 m arena (like config/scene.xml): walls 0.1 m thick at +/-5 m
WALLS = [Box(0, 5, 5, 0.1, 0), Box(0, -5, 5, 0.1, 0), Box(5, 0, 0.1, 5, 0), Box(-5, 0, 0.1, 5, 0)]
BOUNDS = (-5.0, 5.0, -5.0, 5.0)


def test_distance_inside_box_is_zero():
    assert spawn.distance_to_obstacle(0.0, 0.0, Box(0, 0, 1, 1, 0)) == 0.0


def test_distance_to_box_edge_and_corner():
    assert spawn.distance_to_obstacle(3.0, 0.0, Box(0, 0, 1, 1, 0)) == pytest.approx(2.0)
    assert spawn.distance_to_obstacle(4.0, 5.0, Box(0, 0, 1, 1, 0)) == pytest.approx(5.0)  # 3-4-5 from corner (1,1)


def test_distance_to_rotated_box():
    # 4 m x 0.2 m box rotated 90 deg: its long side now runs along y
    assert spawn.distance_to_obstacle(0.0, 1.5, Box(0, 0, 2, 0.1, math.pi / 2)) == 0.0
    assert spawn.distance_to_obstacle(1.0, 0.0, Box(0, 0, 2, 0.1, math.pi / 2)) == pytest.approx(0.9)


def test_distance_to_circle():
    assert spawn.distance_to_obstacle(3.0, 4.0, Circle(0, 0, 1)) == pytest.approx(4.0)
    assert spawn.distance_to_obstacle(0.5, 0.0, Circle(0, 0, 1)) == 0.0


def test_spawn_keeps_clearance_and_stays_inside():
    obstacles = WALLS + [Circle(2, 2, 0.5), Box(-2, -1, 1.0, 0.5, 0.3), Circle(4, 0, 0.35)]
    rng = np.random.default_rng(0)
    for _ in range(500):
        pose = spawn.sample_spawn_pose(rng, obstacles, BOUNDS, clearance=1.0)
        assert pose is not None
        x, y, yaw = pose
        assert -5 < x < 5 and -5 < y < 5
        assert min(spawn.distance_to_obstacle(x, y, o) for o in obstacles) >= 1.0
        assert -math.pi <= yaw < math.pi


def test_spawn_poses_are_spread_out():
    rng = np.random.default_rng(1)
    xs = [spawn.sample_spawn_pose(rng, WALLS, BOUNDS, clearance=1.0)[0] for _ in range(200)]
    assert max(xs) - min(xs) > 4.0  # not stuck at one spot


def test_spawn_returns_none_when_no_free_space():
    rng = np.random.default_rng(2)
    blocked = WALLS + [Box(0, 0, 5, 5, 0)]
    assert spawn.sample_spawn_pose(rng, blocked, BOUNDS, clearance=1.0, tries=200) is None


def test_yaw_to_quat_is_mujoco_wxyz():
    assert spawn.yaw_to_quat(0.0) == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert spawn.yaw_to_quat(math.pi / 2) == pytest.approx([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
