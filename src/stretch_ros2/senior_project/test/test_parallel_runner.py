"""parallel_runner launch commands (stdlib only, nothing is launched).

Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_parallel_runner.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from senior_project.parallel_runner import ParallelRunner  # noqa: E402


def _args(cmd):
    return dict(a.split(":=", 1) for a in cmd if ":=" in a)


def test_random_goals_train_through_rl_launch_without_sam3(tmp_path):
    r = ParallelRunner(num_sims=2, base_ckpt_dir=str(tmp_path))
    cmd = r._build_launch_command(1)
    args = _args(cmd)
    assert cmd[:4] == ["ros2", "launch", "senior_project", "RL.launch.py"]
    assert args["goal_source"] == "random"
    assert args["use_rviz"] == "false"
    assert args["ckpt_dir"] == os.path.join(str(tmp_path), "sim_1")
    assert "sam3_server_port" not in args


def test_only_sim_zero_runs_the_dashboard(tmp_path):
    r = ParallelRunner(num_sims=3, base_ckpt_dir=str(tmp_path))
    assert _args(r._build_launch_command(0))["use_reward_monitor"] == "true"
    assert _args(r._build_launch_command(2))["use_reward_monitor"] == "false"


def test_topic_goals_share_one_sam3_server_by_default(tmp_path):
    r = ParallelRunner(num_sims=3, base_ckpt_dir=str(tmp_path), goal_source="topic")
    args = _args(r._build_launch_command(2))
    assert args["goal_source"] == "topic"
    assert args["sam3_server_port"] == "8100"


def test_sims_use_docker_domains(tmp_path):
    r = ParallelRunner(num_sims=4, base_ckpt_dir=str(tmp_path))
    assert [r._get_sim_domain(i) for i in range(4)] == [90, 91, 92, 93]


def test_stop_all_twice_is_safe(tmp_path, capsys):
    r = ParallelRunner(num_sims=1, base_ckpt_dir=str(tmp_path))
    r.stop_all()
    capsys.readouterr()
    r.stop_all()
    assert capsys.readouterr().out == ""
