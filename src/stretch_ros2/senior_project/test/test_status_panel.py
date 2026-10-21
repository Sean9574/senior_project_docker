"""Terminal status panel: state + rendering (stdlib only, nothing is drawn).

Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_status_panel.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from senior_project.status_panel_core import PanelState, render, summary_line  # noqa: E402


def _state():
    return PanelState(total_steps=200_000, target="chair", domain="90", log_dir="/logs", started_at=0.0)


def _bd(**kw):
    d = {"episode": 1, "step": 10, "global_step": 100, "training_phase": "RL_POLICY",
         "mode": "explore", "done": False, "success": False, "goal_seen": False,
         "episode_return": 0.0, "state": {"goal_dist": 0.0}, "min_distance": 1.0, "dt": 0.1,
         "last_reset_s": 3.2, "last_ckpt_wall": None}
    d.update(kw)
    return d


def _text(state, now, width=80):
    return "\n".join(render(state, now, width))


def test_waiting_before_the_learner_reports():
    assert "waiting for the learner" in _text(_state(), 1.0)


def test_success_and_seen_rates_over_finished_episodes():
    s = _state()
    for i, (ok, seen) in enumerate([(True, True), (False, True), (True, False), (True, True)]):
        s.on_breakdown(_bd(episode=i + 1, done=True, success=ok, goal_seen=seen, episode_return=float(i)), 10.0)
    out = _text(s, 11.0)
    assert "+-++" in out
    assert "success 75%" in out
    assert "seen 75%" in out
    assert "best +3.0" in out


def test_progress_rate_and_eta():
    s = _state()
    s.on_breakdown(_bd(global_step=1_000), 0.0)
    s.on_breakdown(_bd(global_step=2_100), 100.0)
    out = _text(s, 100.0)
    assert "2,100 / 200,000" in out
    assert "~11 steps/s" in out
    assert "ETA 4h59m" in out


def test_search_vs_goal_seek():
    s = _state()
    s.on_breakdown(_bd(mode="explore"), 1.0)
    assert "SEARCHING" in _text(s, 1.0)
    s.on_breakdown(_bd(mode="goal", state={"goal_dist": 1.8}), 2.0)
    out = _text(s, 2.0)
    assert "GOAL-SEEK" in out and "goal 1.8m" in out


def test_silent_learner_is_flagged():
    s = _state()
    s.on_breakdown(_bd(), 0.0)
    assert "learner silent" not in _text(s, 5.0)
    assert "learner silent" in _text(s, 40.0)


def test_unconfirmed_sam3_prompt_is_flagged():
    s = _state()
    s.on_generator_status({"target": "chair", "prompt_confirmed": False}, 1.0)
    assert "SAM3 NOT CONFIRMED" in _text(s, 1.0)
    s.on_generator_status({"target": "chair", "prompt_confirmed": True}, 2.0)
    assert 'SAM3 ok "chair"' in _text(s, 2.0)


def test_alerts_dedupe_ignore_info_and_keep_last_three():
    s = _state()
    s.on_log(20, "learner", "just info", 1.0)
    for t in range(3):
        s.on_log(30, "learner", "replay save failed", 2.0 + t)
    out = _text(s, 10.0)
    assert "just info" not in out
    assert out.count("replay save failed") == 1 and "(x3)" in out
    for i, msg in enumerate(["first", "second", "third", "fourth"]):
        s.on_log(40, "driver", msg, 20.0 + i)
    out = _text(s, 30.0)
    assert "first" not in out and "fourth" in out and "ERROR driver" in out


def test_every_line_fits_the_terminal():
    s = _state()
    s.on_breakdown(_bd(), 0.0)
    s.on_log(40, "sam3_goal_generator", "x" * 300, 1.0)
    assert all(len(line) <= 40 for line in render(s, 2.0, 40))


def test_summary_is_a_single_line():
    s = _state()
    s.on_breakdown(_bd(done=True, success=True, goal_seen=True), 1.0)
    line = summary_line(s, 2.0)
    assert "\n" not in line and "RL_POLICY" in line and "success" in line


def test_colour_codes_in_node_logs_do_not_reach_the_panel():
    s = _state()
    s.on_log(30, "learner", "\x1b[93mNo replay buffer\x1b[0m", 1.0)
    out = _text(s, 2.0)
    assert "\x1b" not in out and "No replay buffer" in out


def test_learner_that_never_reports_gets_a_hint():
    s = _state()  # started_at=0
    assert "crashed" not in _text(s, 30.0)
    out = _text(s, 120.0)
    assert "2m00s" in out and "crashed" in out
