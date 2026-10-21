#!/usr/bin/env python3
"""
State and rendering for the terminal status panel (stdlib only).

status_panel.py feeds it ROS messages; this module only keeps the numbers and
turns them into text, so it is unit-tested without ROS (test/test_status_panel.py).
"""

import time
from collections import deque
from typing import Deque, List, Optional, Tuple

STALE_AFTER_S = 30.0      # learner silent this long -> flagged
RATE_WINDOW_S = 60.0      # steps/s measured over this window
WINDOW_EPISODES = 20      # success / seen / return statistics
MAX_ALERTS = 3

LEVEL_NAMES = {30: "WARN", 40: "ERROR", 50: "FATAL"}  # rcl_interfaces/Log levels


def _duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _clock(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class PanelState:
    """Latest training numbers, updated from ROS callbacks."""

    def __init__(self, total_steps: int = 0, target: str = "", domain: str = "",
                 log_dir: str = "", started_at: Optional[float] = None):
        self.total_steps = total_steps
        self.target = target
        self.domain = domain
        self.log_dir = log_dir
        self.started_at = time.time() if started_at is None else started_at

        self.last_breakdown_t: Optional[float] = None
        self.phase = ""
        self.episode = 0
        self.episode_step = 0
        self.global_step = 0
        self.mode = ""
        self.goal_dist = 0.0
        self.min_distance = 0.0
        self.dt = 0.0
        self.last_reset_s: Optional[float] = None
        self.last_ckpt_wall: Optional[float] = None
        self.episode_started_t: Optional[float] = None

        self.outcomes: Deque[Tuple[bool, bool]] = deque(maxlen=WINDOW_EPISODES)
        self.returns: Deque[float] = deque(maxlen=WINDOW_EPISODES)
        self.best_return: Optional[float] = None
        self._rate: Deque[Tuple[float, int]] = deque()

        self.sam3_confirmed: Optional[bool] = None
        self.sam3_target = ""

        # [time, level, node, msg, count], newest last
        self.alerts: List[list] = []

    # ------------------------------------------------------------ updates
    def on_breakdown(self, d: dict, now: float) -> None:
        episode = int(d.get("episode", 0))
        if episode != self.episode:
            self.episode_started_t = now
        self.episode = episode
        self.last_breakdown_t = now
        self.phase = d.get("training_phase", self.phase)
        self.episode_step = int(d.get("step", 0))
        self.global_step = int(d.get("global_step", self.global_step))
        self.mode = d.get("mode", "")
        self.goal_dist = float((d.get("state") or {}).get("goal_dist", 0.0))
        self.min_distance = float(d.get("min_distance", 0.0))
        self.dt = float(d.get("dt", 0.0))
        if d.get("last_reset_s") is not None:
            self.last_reset_s = float(d["last_reset_s"])
        if d.get("last_ckpt_wall") is not None:
            self.last_ckpt_wall = float(d["last_ckpt_wall"])

        self._rate.append((now, self.global_step))
        while len(self._rate) > 2 and now - self._rate[0][0] > RATE_WINDOW_S:
            self._rate.popleft()

        if d.get("done"):
            self.outcomes.append((bool(d.get("success")), bool(d.get("goal_seen"))))
            ret = float(d.get("episode_return", 0.0))
            self.returns.append(ret)
            self.best_return = ret if self.best_return is None else max(self.best_return, ret)

    def on_generator_status(self, d: dict, now: float) -> None:
        if "prompt_confirmed" in d:
            self.sam3_confirmed = bool(d["prompt_confirmed"])
        self.sam3_target = d.get("target", self.sam3_target) or self.sam3_target

    def on_log(self, level: int, node: str, msg: str, now: float) -> None:
        if level < 30:
            return
        for alert in self.alerts:
            if alert[2] == node and alert[3] == msg:
                alert[0], alert[4] = now, alert[4] + 1
                self.alerts.remove(alert)
                self.alerts.append(alert)
                return
        self.alerts.append([now, level, node, msg, 1])
        del self.alerts[:-MAX_ALERTS]

    # ------------------------------------------------------------ derived
    def steps_per_sec(self) -> float:
        if len(self._rate) < 2:
            return 0.0
        (t0, s0), (t1, s1) = self._rate[0], self._rate[-1]
        return (s1 - s0) / (t1 - t0) if t1 > t0 else 0.0

    def silent_for(self, now: float) -> Optional[float]:
        if self.last_breakdown_t is None:
            return None
        gap = now - self.last_breakdown_t
        return gap if gap > STALE_AFTER_S else None


# ---------------------------------------------------------------- rendering
def _progress(state: PanelState) -> str:
    if state.total_steps <= 0:
        return f"{state.global_step:,} steps"
    frac = min(state.global_step / state.total_steps, 1.0)
    bar = "#" * int(frac * 10) + "." * (10 - int(frac * 10))
    return f"{state.global_step:,} / {state.total_steps:,}  [{bar}] {int(frac * 100)}%"


def _eta(state: PanelState) -> str:
    rate = state.steps_per_sec()
    if rate <= 0 or state.total_steps <= 0:
        return "ETA --"
    return f"ETA {_duration((state.total_steps - state.global_step) / rate)}   ~{rate:.0f} steps/s"


def _sam3(state: PanelState) -> str:
    target = state.sam3_target or state.target
    if state.sam3_confirmed is None:
        return f'SAM3 waiting for "{target}"'
    if not state.sam3_confirmed:
        return f'SAM3 NOT CONFIRMED "{target}"'
    return f'SAM3 ok "{target}"'


def render(state: PanelState, now: float, width: int = 80) -> List[str]:
    """Panel lines, each at most `width` characters."""
    rule = " " + "-" * max(min(width - 2, 54), 1)
    lines = [f' Stretch TD3  target "{state.target}"  domain {state.domain}  '
             f"up {_clock(now - state.started_at)}", rule]

    if state.last_breakdown_t is None:
        lines.append(" Phase    waiting for the learner to start...")
    else:
        lines.append(f" Phase    {state.phase or '?':<11} {_progress(state)}")
        lines.append(f"          {_eta(state)}")
        seeking = state.mode == "goal"
        ep_time = (now - state.episode_started_t) if state.episode_started_t is not None else 0.0
        where = f"goal {state.goal_dist:.1f}m" if seeking else "object not seen yet"
        lines.append(f" Episode  #{state.episode}  {'GOAL-SEEK' if seeking else 'SEARCHING'}  "
                     f"{ep_time:.1f}s  {where}  clr {state.min_distance:.2f}m")

    if state.outcomes:
        marks = "".join("+" if ok else "-" for ok, _ in state.outcomes)
        n = len(state.outcomes)
        success = sum(ok for ok, _ in state.outcomes) / n
        seen = sum(s for _, s in state.outcomes) / n
        lines.append(f" Last {n:<3} {marks}   success {success:.0%}  seen {seen:.0%}")
        avg = sum(state.returns) / len(state.returns)
        lines.append(f" Return   avg{n} {avg:+.1f}   best {state.best_return:+.1f}")
    else:
        lines.append(" Last 20  no finished episodes yet")

    silent = state.silent_for(now)
    health = [_sam3(state)]
    if state.dt:
        health.append(f"step {state.dt:.2f}s")
    if state.last_reset_s is not None:
        health.append(f"reset {state.last_reset_s:.1f}s")
    lines.append(" Health   " + " | ".join(health))
    ckpt = ("no checkpoint yet" if state.last_ckpt_wall is None
            else f"ckpt saved {_duration(time.time() - state.last_ckpt_wall)} ago")
    lines.append(f"          {ckpt}")
    if silent is not None:
        lines.append(f" !!       learner silent for {_duration(silent)} -- check the logs")

    if state.alerts:
        for i, (t, level, node, msg, count) in enumerate(state.alerts):
            label = " Alerts   " if i == 0 else "          "
            stamp = time.strftime("%H:%M:%S", time.localtime(t))
            more = f" (x{count})" if count > 1 else ""
            lines.append(f"{label}{stamp} {LEVEL_NAMES.get(level, level)} {node}: {msg}{more}")
    else:
        lines.append(" Alerts   none")

    lines.append("")
    lines.append(f" logs: {state.log_dir}   (Ctrl+C stops training)")
    return [line[:width] for line in lines]


def summary_line(state: PanelState, now: float) -> str:
    """One line for non-terminal output (docker logs, nohup)."""
    parts = [f"[status] {state.phase or 'waiting'}", _progress(state), f"ep {state.episode}"]
    if state.outcomes:
        n = len(state.outcomes)
        parts.append(f"success{n} {sum(ok for ok, _ in state.outcomes) / n:.0%}")
        parts.append(f"seen{n} {sum(s for _, s in state.outcomes) / n:.0%}")
    parts.append(_sam3(state))
    if state.silent_for(now) is not None:
        parts.append("LEARNER SILENT")
    if state.alerts:
        parts.append(f"alerts {len(state.alerts)}")
    return " | ".join(parts)
