"""Asking the user what the robot should go to (stdlib only).

Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_prompt.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from senior_project import prompt  # noqa: E402


def _never_called(_):
    raise AssertionError("must not ask")


def test_target_from_launch_args():
    assert prompt.target_from_argv(["ros2", "launch", "pkg", "f.launch.py", "target:=chair"]) == "chair"
    assert prompt.target_from_argv(["target:=blue cylinder"]) == "blue cylinder"
    assert prompt.target_from_argv(["use_rviz:=false"]) is None
    assert prompt.target_from_argv(["target:="]) is None


def test_given_target_is_used_without_asking():
    assert prompt.ask_target(["target:=chair"], isatty=True, input_fn=_never_called) == "chair"


def test_asks_until_it_gets_an_answer():
    answers = iter(["", "   ", " potted plant "])
    asked = []

    def fake_input(text):
        asked.append(text)
        return next(answers)

    assert prompt.ask_target([], isatty=True, input_fn=fake_input) == "potted plant"
    assert len(asked) == 3
    assert "go to" in asked[0]


def test_non_interactive_start_without_target_exits_with_help():
    with pytest.raises(SystemExit) as exc:
        prompt.ask_target([], isatty=False, input_fn=_never_called)
    assert "target:=" in str(exc.value)


def test_closed_input_exits_instead_of_hanging():
    def eof(_):
        raise EOFError

    with pytest.raises(SystemExit):
        prompt.ask_target([], isatty=True, input_fn=eof)


def test_show_args_never_asks():
    assert prompt.ask_target(["ros2", "launch", "pkg", "f.launch.py", "--show-args"],
                             isatty=True, input_fn=_never_called) == ""
