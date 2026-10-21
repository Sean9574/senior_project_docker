"""Unit tests for senior_project.rl_core (numpy only, no ROS or torch needed).

Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_rl_core.py
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from senior_project import rl_core  # noqa: E402


# ---------------------------------------------------------------- scan cleaning

def test_no_hit_reads_max_range():
    out = rl_core.clean_ranges(np.array([-1.0, 5.0]), 0.2, 20.0)
    assert out.tolist() == [20.0, 5.0]


def test_nan_inf_and_beyond_max_read_max_range():
    out = rl_core.clean_ranges(np.array([np.nan, np.inf, 25.0]), 0.2, 20.0)
    assert out.tolist() == [20.0, 20.0, 20.0]


def test_contact_clamps_to_range_min():
    out = rl_core.clean_ranges(np.array([0.05, 0.0]), 0.2, 20.0)
    assert out.tolist() == pytest.approx([0.2, 0.2])


def test_clean_ranges_does_not_mutate_input():
    raw = np.array([-1.0, 0.05], dtype=np.float32)
    rl_core.clean_ranges(raw, 0.2, 20.0)
    assert raw.tolist() == pytest.approx([-1.0, 0.05])


def test_self_mask_marks_only_short_real_hits():
    mask = rl_core.compute_self_mask(np.array([0.1, 3.0, -1.0, 0.0]), 0.2)
    assert mask.tolist() == [True, False, False, True]


def test_self_mask_beams_ignored():
    raw = np.array([0.1, 3.0])
    mask = rl_core.compute_self_mask(raw, 0.2)
    out = rl_core.clean_ranges(raw, 0.2, 20.0, mask)
    assert out.tolist() == [20.0, 3.0]


# ---------------------------------------------------------------- action mapping

def test_goal_action_can_stop_and_pivot():
    assert rl_core.map_action(np.array([-1.0, 1.0]), True) == pytest.approx((0.0, 1.0))


def test_goal_action_full_speed_is_real_stretch_limit():
    assert rl_core.map_action(np.array([1.0, -1.0]), True) == pytest.approx((0.3, -1.0))


def test_action_outside_range_is_clipped():
    assert rl_core.map_action(np.array([5.0, -5.0]), True) == pytest.approx((0.3, -1.0))


def test_explore_action_always_moves_forward():
    assert rl_core.map_action(np.array([-1.0, 0.0]), False) == pytest.approx((0.1, 0.0))


def test_explore_turn_rate_coupled_to_speed():
    # v = 0.1 m/s -> w_limit = 0.1 / 0.2 = 0.5 rad/s
    assert rl_core.map_action(np.array([-1.0, 1.0]), False) == pytest.approx((0.1, 0.5))
    # v = 0.3 m/s -> 0.3 / 0.2 = 1.5, capped at 1.0 rad/s
    assert rl_core.map_action(np.array([1.0, 1.0]), False) == pytest.approx((0.3, 1.0))


# ---------------------------------------------------------------- reward

def test_progress_reward_signed_and_finite():
    assert rl_core.goal_progress_reward(2.0, 1.5, 1.0) == pytest.approx(0.5)
    assert rl_core.goal_progress_reward(1.5, 2.0, 1.0) == pytest.approx(-0.5)
    assert rl_core.goal_progress_reward(float("inf"), 1.0, 1.0) == 0.0
    assert rl_core.goal_progress_reward(1.0, float("nan"), 1.0) == 0.0


# ---------------------------------------------------------------- goal sampling

def _scan(n=360, value=20.0):
    return np.full(n, value, dtype=np.float32)


def _gap_scan(center_deg, half_width_deg, wall=1.2, n=360):
    """Wall at `wall` metres on every beam except a gap around center_deg (raw beam angle)."""
    ranges = np.full(n, wall, dtype=np.float32)
    for i in range(n):
        diff = (i - center_deg + 180) % 360 - 180
        if abs(diff) <= half_width_deg:
            ranges[i] = 20.0
    return ranges


INC = 2 * math.pi / 360


def test_goal_never_inside_radius():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        gx, gy, dist = rl_core.sample_goal(
            rng, _scan(), 0.0, INC, 0.0, 0.0, 0.0, 0.0, d_min=1.0, d_max=1.5)
        assert 1.0 <= dist <= 1.5
        assert math.hypot(gx, gy) == pytest.approx(dist, abs=1e-6)
        assert math.hypot(gx, gy) > 0.45


def test_goal_respects_line_of_sight():
    rng = np.random.default_rng(1)
    ranges = _gap_scan(center_deg=0, half_width_deg=20)
    for _ in range(200):
        gx, gy, dist = rl_core.sample_goal(
            rng, ranges, 0.0, INC, 0.0, 0.0, 0.0, 0.0, d_min=1.0, d_max=3.0)
        bearing = math.degrees(math.atan2(gy, gx))
        assert abs(bearing) <= 20.0


def test_goal_in_odom_frame_uses_robot_pose():
    # Gap straight ahead in the robot frame; robot at (2, 3) facing +y.
    rng = np.random.default_rng(2)
    ranges = _gap_scan(center_deg=0, half_width_deg=20)
    for _ in range(100):
        gx, gy, dist = rl_core.sample_goal(
            rng, ranges, 0.0, INC, 0.0, 2.0, 3.0, math.pi / 2, d_min=1.0, d_max=3.0)
        assert gy > 3.0
        assert abs(gx - 2.0) <= dist * math.sin(math.radians(20)) + 1e-6


def test_goal_applies_lidar_mount_offset():
    # Learner mounts the lidar backwards (offset pi): raw beam 180 deg points forward.
    rng = np.random.default_rng(3)
    ranges = _gap_scan(center_deg=180, half_width_deg=20)
    for _ in range(100):
        gx, gy, _ = rl_core.sample_goal(
            rng, ranges, 0.0, INC, math.pi, 0.0, 0.0, 0.0, d_min=1.0, d_max=3.0)
        assert gx > 0.0


def test_goal_fallback_when_everything_blocked_stays_in_range():
    rng = np.random.default_rng(4)
    ranges = _scan(value=1.0)  # wall at 1 m everywhere: no bearing passes the LOS test
    gx, gy, dist = rl_core.sample_goal(
        rng, ranges, 0.0, INC, 0.0, 0.0, 0.0, 0.0, d_min=1.0, d_max=3.0)
    assert dist == pytest.approx(1.0)
    assert math.hypot(gx, gy) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- curriculum

def test_curriculum_raises_after_window():
    c = rl_core.GoalCurriculum(d_min=1.0, d_start=1.5, d_cap=4.0, step=0.5, window=20, threshold=0.6)
    for _ in range(19):
        c.record(True)
    assert c.d_max == pytest.approx(1.5)
    c.record(True)
    assert c.d_max == pytest.approx(2.0)
    for _ in range(20):
        c.record(False)
    assert c.d_max == pytest.approx(2.0)


def test_curriculum_needs_a_fresh_window_after_each_raise():
    c = rl_core.GoalCurriculum(d_min=1.0, d_start=1.5, d_cap=4.0, step=0.5, window=20, threshold=0.6)
    for _ in range(20):
        c.record(True)
    for _ in range(19):
        c.record(True)
    assert c.d_max == pytest.approx(2.0)
    c.record(True)
    assert c.d_max == pytest.approx(2.5)


def test_curriculum_threshold_is_inclusive_and_capped():
    c = rl_core.GoalCurriculum(d_min=1.0, d_start=1.5, d_cap=2.0, step=0.5, window=10, threshold=0.6)
    for ok in [True] * 6 + [False] * 4:  # exactly 60 %
        c.record(ok)
    assert c.d_max == pytest.approx(2.0)
    for _ in range(50):
        c.record(True)
    assert c.d_max == pytest.approx(2.0)


def test_curriculum_below_threshold_does_not_raise():
    c = rl_core.GoalCurriculum(d_min=1.0, d_start=1.5, d_cap=4.0, step=0.5, window=10, threshold=0.6)
    for ok in [True] * 5 + [False] * 5:
        c.record(ok)
    assert c.d_max == pytest.approx(1.5)


def test_curriculum_state_round_trip():
    a = rl_core.GoalCurriculum(window=5)
    for _ in range(5):
        a.record(True)
    a.record(False)
    b = rl_core.GoalCurriculum(window=5)
    b.load(a.state())
    assert b.d_max == pytest.approx(a.d_max)
    for _ in range(3):
        b.record(True)
    assert b.d_max == pytest.approx(a.d_max)  # F,T,T,T: window of 5 not full yet
    b.record(True)  # F,T,T,T,T = 80 %: raises only if the False survived the reload
    assert b.d_max > a.d_max


# ---------------------------------------------------------------- training phase

@pytest.mark.parametrize("t, want", [
    (1, "EXPERT_DEMO"),
    (10_000, "EXPERT_DEMO"),
    (10_001, "RANDOM_EXPLORE"),
    (14_999, "RANDOM_EXPLORE"),
    (15_000, "RL_POLICY"),
    (50_000, "RL_POLICY"),
])
def test_phase_for_step(t, want):
    assert rl_core.phase_for_step(t, 10_000, 5_000) == want


def test_phase_with_no_demo_or_random_steps_is_rl():
    assert rl_core.phase_for_step(1, 0, 0) == "RL_POLICY"
