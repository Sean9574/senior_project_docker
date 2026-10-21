#!/usr/bin/env python3
"""
Ask the user what the robot should go to (the SAM3 prompt), before anything
starts. Used by the launch files and parallel_runner; stdlib only so it is
unit-tested without ROS (test/test_prompt.py).
"""

import time
import urllib.parse
from typing import Callable, List, Optional

QUESTION = "What should the robot go to? (e.g. chair, potted plant, blue cylinder): "


def target_from_argv(argv: List[str]) -> Optional[str]:
    """The value of a `target:=...` launch argument, if one was given."""
    for arg in argv:
        if arg.startswith("target:="):
            value = arg[len("target:="):].strip().strip("'\"").strip()
            return value or None
    return None


def ask_target(argv: List[str], isatty: bool, input_fn: Callable[[str], str] = input) -> str:
    """Return the prompt: `target:=` from argv if given, otherwise ask in the
    terminal until a non-empty answer. Never blocks a non-interactive process."""
    if "--show-args" in argv or "-s" in argv:
        return ""
    given = target_from_argv(argv)
    if given:
        return given
    if not isatty:
        raise SystemExit(
            "No target given and no terminal to ask in. "
            "Pass one, e.g.  target:=chair"
        )
    while True:
        try:
            answer = input_fn(QUESTION).strip()
        except EOFError:
            raise SystemExit("No target given. Pass one, e.g.  target:=chair")
        if answer:
            return answer


def count_from_argv(argv: List[str], key: str = "num_robots") -> Optional[int]:
    """The integer value of a `num_robots:=N` launch argument, if valid."""
    for arg in argv:
        if arg.startswith(key + ":="):
            try:
                return int(arg[len(key) + 2:].strip())
            except ValueError:
                return None
    return None


def ask_count(argv: List[str], isatty: bool, input_fn: Callable[[str], str] = input,
              default: int = 8, key: str = "num_robots", lo: int = 1, hi: int = 10) -> int:
    """How many robots: `num_robots:=` from argv, else ask (Enter = default,
    invalid answers are asked again). Non-interactive starts use the default."""
    if "--show-args" in argv or "-s" in argv:
        return default
    given = count_from_argv(argv, key)
    if given is not None:
        if not lo <= given <= hi:
            raise SystemExit(f"{key} must be between {lo} and {hi} (got {given})")
        return given
    if not isatty:
        return default
    while True:
        try:
            answer = input_fn(f"How many robots? [{default}]: ").strip()
        except EOFError:
            return default
        if not answer:
            return default
        if answer.isdigit() and lo <= int(answer) <= hi:
            return int(answer)


def send_prompt(post_fn: Callable[[str], object], server_url: str, target: str,
                timeout_s: float, sleep_fn: Callable[[float], None] = time.sleep,
                now_fn: Callable[[], float] = time.monotonic, retry_s: float = 2.0) -> bool:
    """POST the prompt to the SAM3 server until it answers 200 and echoes the
    same prompt (it may still be loading the model). False after timeout_s, so
    callers can say so instead of silently searching for the server's default."""
    url = f"{server_url}/prompt/{urllib.parse.quote(target, safe='')}"
    deadline = now_fn() + timeout_s
    while True:
        try:
            r = post_fn(url)
            if r.status_code == 200 and r.json().get("prompt") == target:
                return True
        except Exception:
            pass
        if now_fn() + retry_s > deadline:
            return False
        sleep_fn(retry_s)
