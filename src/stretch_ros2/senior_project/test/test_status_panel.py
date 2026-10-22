"""Terminal status panel: state + rendering (stdlib only, nothing is drawn).

Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_status_panel.py
"""

import fcntl
import os
import pty
import struct
import sys
import termios
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from senior_project.status_panel_core import (  # noqa: E402
    MIN_WIDTH, PanelState, frame, render, summary_line, terminal_width)


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


def test_eta_is_unknown_while_the_learner_is_silent():
    s = _state()
    s.on_breakdown(_bd(global_step=1_000), 0.0)
    s.on_breakdown(_bd(global_step=2_100), 100.0)
    out = _text(s, 140.0)
    assert "ETA --" in out and "ETA 4h59m" not in out


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


def test_no_camera_frames_is_not_reported_as_sam3_waiting():
    s = _state()
    s.on_generator_status({"target": "chair", "prompt_confirmed": True, "frame_age_s": 15.0}, 1.0)
    out = _text(s, 1.0)
    assert "no camera" in out and "SAM3 waiting" not in out and "SAM3 ok" not in out
    s.on_generator_status({"target": "chair", "prompt_confirmed": True, "frame_age_s": 0.5}, 2.0)
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


@contextmanager
def _terminal(columns):
    """A real pseudo-terminal `columns` wide; yields its device path."""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, columns, 0, 0))
    try:
        yield os.ttyname(slave)
    finally:
        os.close(master)
        os.close(slave)


def test_width_is_the_real_terminal_width():
    with _terminal(132) as tty:
        assert terminal_width(tty) == 132


def test_width_never_drops_below_the_minimum():
    with _terminal(10) as tty:
        assert terminal_width(tty) == MIN_WIDTH


def test_width_falls_back_to_80_without_a_sized_terminal(tmp_path):
    with _terminal(0) as tty:  # what launch's emulate_tty hands the panel
        assert terminal_width(tty) == 80
    assert terminal_width(str(tmp_path / "no-tty")) == 80


def test_redraw_overwrites_in_place_instead_of_clearing_the_screen():
    # cursor home, each line then clear-to-end-of-line, clear below the last one
    assert frame(["a", "b"]) == "\x1b[Ha\x1b[K\nb\x1b[K\x1b[J\n"


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


# ---------------------------------------------------------------- grid arena

def test_arena_progress_sums_every_robot():
    s = _state()
    s.on_breakdown(_bd(robot=0, global_step=100), 1.0)
    s.on_breakdown(_bd(robot=1, global_step=200), 1.0)
    assert "300 / 200,000" in _text(s, 1.0)


def test_arena_shows_one_row_per_robot():
    s = _state()
    s.on_breakdown(_bd(robot=0, episode=4, mode="explore"), 1.0)
    s.on_breakdown(_bd(robot=1, episode=7, mode="goal", state={"goal_dist": 2.5}), 1.0)
    out = _text(s, 1.0)
    assert " r0 " in out and " r1 " in out
    assert "#4" in out and "#7" in out
    assert "SEARCHING" in out and "GOAL-SEEK" in out


def test_arena_pools_finished_episodes_across_robots():
    s = _state()
    s.on_breakdown(_bd(robot=0, done=True, success=True, goal_seen=True), 1.0)
    s.on_breakdown(_bd(robot=1, done=True, success=False, goal_seen=True), 1.0)
    assert "success 50%" in _text(s, 2.0)


def test_arena_sam3_confirmation_per_robot():
    s = _state()
    s.on_generator_status({"target": "chair", "prompt_confirmed": True}, 1.0, robot=0)
    s.on_generator_status({"target": "chair", "prompt_confirmed": False}, 1.0, robot=1)
    assert "SAM3 NOT CONFIRMED: r1" in _text(s, 1.0)
    s.on_generator_status({"target": "chair", "prompt_confirmed": True}, 2.0, robot=1)
    assert 'SAM3 ok 2/2 "chair"' in _text(s, 2.0)


def test_arena_names_robots_without_camera_frames():
    s = _state()
    s.on_generator_status({"target": "chair", "prompt_confirmed": True, "frame_age_s": 0.5}, 1.0, robot=0)
    s.on_generator_status({"target": "chair", "prompt_confirmed": True, "frame_age_s": 20.0}, 1.0, robot=1)
    assert "no camera: r1" in _text(s, 1.0)
