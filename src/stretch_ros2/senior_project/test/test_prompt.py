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


class _Resp:
    def __init__(self, status, prompt):
        self.status_code = status
        self._prompt = prompt

    def json(self):
        return {"prompt": self._prompt}


class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_send_prompt_url_encodes_multi_word_targets():
    urls = []

    def post(url):
        urls.append(url)
        return _Resp(200, "potted plant")

    clock = _Clock()
    assert prompt.send_prompt(post, "http://localhost:8100", "potted plant", 10, clock.sleep, clock.now)
    assert urls == ["http://localhost:8100/prompt/potted%20plant"]


def test_send_prompt_retries_until_server_is_up():
    replies = iter([ConnectionError("down"), _Resp(503, "object"), _Resp(200, "chair")])

    def post(url):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    clock = _Clock()
    assert prompt.send_prompt(post, "http://s", "chair", 60, clock.sleep, clock.now) is True


def test_send_prompt_rejects_a_server_that_echoes_another_prompt():
    clock = _Clock()
    assert prompt.send_prompt(lambda url: _Resp(200, "object"), "http://s", "chair",
                              6, clock.sleep, clock.now) is False


def test_send_prompt_gives_up_after_the_deadline():
    clock = _Clock()

    def post(url):
        raise ConnectionError("down")

    assert prompt.send_prompt(post, "http://s", "chair", 10, clock.sleep, clock.now) is False
    assert clock.t >= 10
