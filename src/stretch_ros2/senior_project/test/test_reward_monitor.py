"""Episode accounting + dashboard path safety in scripts/reward_monitor.py.

ROS, Flask and OpenCV are stubbed: DomainStats and the path helper don't use
them, and the stubs keep these tests runnable without a ROS install.
Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_reward_monitor.py
"""

import csv
import os
import sys
from unittest import mock

import pytest

for _name in ["cv2", "rclpy", "rclpy.context", "rclpy.executors", "rclpy.qos",
              "cv_bridge", "flask", "flask_cors", "flask_socketio",
              "sensor_msgs", "sensor_msgs.msg", "std_msgs", "std_msgs.msg"]:
    sys.modules.setdefault(_name, mock.MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import reward_monitor as rm  # noqa: E402


def _stats():
    return rm.DomainStats(domain_id=90, namespace="stretch", display_name="D90:stretch")


def _msg(episode, reward, done=False, success=False, phase="RL_POLICY"):
    return {
        "episode": episode, "reward": reward, "done": done, "success": success,
        "outcome": ("success" if success else "timeout") if done else "",
        "training_phase": phase, "reward_terms": {}, "min_distance": 1.0,
        "velocity": {"v": 0.2, "w": 0.0}, "state": {}, "explore_stats": {},
    }


def _episode_rows(tmp_path):
    with open(tmp_path / "sim_90" / "episodes.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_success_counted_on_terminal_step():
    ds = _stats()
    ds.add_breakdown(_msg(1, 0.1))
    ds.add_breakdown(_msg(1, 10.0, done=True, success=True))
    ds.add_breakdown(_msg(2, 0.1))
    assert (ds.successes, ds.timeouts, ds.episode_count) == (1, 0, 1)
    assert list(ds.episode_returns) == pytest.approx([10.1])


def test_timeout_counted_on_terminal_step():
    ds = _stats()
    ds.add_breakdown(_msg(1, -0.01))
    ds.add_breakdown(_msg(1, -0.01, done=True, success=False))
    assert (ds.successes, ds.timeouts, ds.episode_count) == (0, 1, 1)


def test_reward_topic_not_double_counted():
    ds = _stats()
    ds.add_reward(0.1, 0.0)
    ds.add_breakdown(_msg(1, 0.1))
    ds.add_reward(10.0, 0.1)
    ds.add_breakdown(_msg(1, 10.0, done=True, success=True))
    assert list(ds.episode_returns) == pytest.approx([10.1])


def test_episode_csv_row_belongs_to_the_episode_that_ended(tmp_path):
    ds = _stats()
    ds.init_csv_logging(str(tmp_path), 90, 0.0)
    ds.add_breakdown(_msg(7, 0.1))
    ds.add_breakdown(_msg(7, 10.0, done=True, success=True))
    ds.close_csv()
    rows = _episode_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["episode"] == "7"
    assert rows[0]["outcome"] == "goal_reached"
    assert rows[0]["episode_steps"] == "2"
    assert float(rows[0]["episode_return"]) == pytest.approx(10.1)


def test_unfinished_episode_is_incomplete_not_timeout(tmp_path):
    ds = _stats()
    ds.init_csv_logging(str(tmp_path), 90, 0.0)
    ds.add_breakdown(_msg(3, 0.1))
    ds.close_csv()
    rows = _episode_rows(tmp_path)
    assert [r["outcome"] for r in rows] == ["incomplete"]


def test_restarting_monitor_appends_instead_of_truncating(tmp_path):
    first = _stats()
    first.init_csv_logging(str(tmp_path), 90, 0.0)
    first.add_breakdown(_msg(1, 1.0, done=True, success=True))
    first.close_csv()
    second = _stats()
    second.init_csv_logging(str(tmp_path), 90, 0.0)
    second.add_breakdown(_msg(2, 1.0, done=True, success=False))
    second.close_csv()
    assert [r["episode"] for r in _episode_rows(tmp_path)] == ["1", "2"]


def test_safe_subpath_stays_inside_root(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    assert rm.safe_subpath(str(root), "graphs") == os.path.realpath(root / "graphs")
    assert rm.safe_subpath(str(root), str(root / "a" / "b")) == os.path.realpath(root / "a" / "b")
    assert rm.safe_subpath(str(root), "/etc") is None
    assert rm.safe_subpath(str(root), "../outside") is None
    assert rm.safe_subpath("", "graphs") is None
