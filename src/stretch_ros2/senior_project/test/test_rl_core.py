"""Unit tests for senior_project.rl_core (numpy only, no ROS or torch needed).

Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_rl_core.py
"""

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



# ---------------------------------------------------------------- vector loop

@pytest.mark.parametrize("n_new, every, carry, want", [
    (8, 4, 0, (2, 0)),   # 8 robots, 1 update per 4 transitions
    (3, 4, 0, (0, 3)),
    (3, 4, 3, (1, 2)),   # leftover transitions carry over
    (1, 1, 0, (1, 0)),
])
def test_vector_update_count_keeps_per_transition_ratio(n_new, every, carry, want):
    assert rl_core.vector_update_count(n_new, every, carry) == want


def test_one_robot_finishing_does_not_disturb_the_others():
    prev = ["o0", "o1", "o2"]
    acts = ["a0", "a1", "a2"]
    results = [("n0", 0.1, False, False, {}),
               ("n1", 10.0, True, False, {}),     # robot 1 reached its goal
               ("n2", -0.1, False, True, {})]     # robot 2 timed out
    resets = []

    def reset(i):
        resets.append(i)
        return f"reset{i}"

    transitions, next_obs, finished = rl_core.vector_transitions(prev, acts, results, reset)
    assert transitions == [("o0", "a0", 0.1, "n0", False),
                           ("o1", "a1", 10.0, "n1", True),
                           ("o2", "a2", -0.1, "n2", False)]  # timeout is not terminal
    assert next_obs == ["n0", "reset1", "reset2"]
    assert finished == [1, 2] and resets == [1, 2]
