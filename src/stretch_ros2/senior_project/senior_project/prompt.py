#!/usr/bin/env python3
"""
Ask the user what the robot should go to (the SAM3 prompt), before anything
starts. Used by the launch files and parallel_runner; stdlib only so it is
unit-tested without ROS (test/test_prompt.py).
"""

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
