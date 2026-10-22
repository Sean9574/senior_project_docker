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
    assert prompt.target_from_argv(["target:=soda can"]) == "soda can"
    assert prompt.target_from_argv(["use_rviz:=false"]) is None
    assert prompt.target_from_argv(["target:="]) is None


def test_given_target_is_used_without_asking():
    assert prompt.ask_target(["target:=chair"], isatty=True, input_fn=_never_called) == "chair"


def test_typed_answer_is_the_target():
    asked = []

    def fake_input(text):
        asked.append(text)
        return " potted plant "

    assert prompt.ask_target([], isatty=True, input_fn=fake_input) == "potted plant"
    assert "go to" in asked[0] and "[soda can]" in asked[0]


def test_enter_picks_the_soda_can_on_the_table():
    answers = iter(["   "])

    def enter_once(_):
        return next(answers, None) or _never_called(_)  # must not ask twice

    assert prompt.ask_target([], isatty=True, input_fn=enter_once) == "soda can"


def test_non_interactive_start_uses_the_soda_can():
    assert prompt.ask_target([], isatty=False, input_fn=_never_called) == "soda can"


def test_closed_input_uses_the_soda_can_instead_of_hanging():
    def eof(_):
        raise EOFError

    assert prompt.ask_target([], isatty=True, input_fn=eof) == "soda can"


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


# ---------------------------------------------------------------- robot count

def _answers(*items):
    it = iter(items)
    asked = []

    def fake(text):
        asked.append(text)
        return next(it)
    return fake, asked


def test_count_from_launch_args():
    assert prompt.count_from_argv(["ros2", "launch", "p", "f.py", "num_robots:=4"]) == 4
    assert prompt.count_from_argv(["num_robots:=abc"]) is None
    assert prompt.count_from_argv(["target:=chair"]) is None


def test_enter_gives_the_default_of_eight():
    fake, asked = _answers("")
    assert prompt.ask_count([], isatty=True, input_fn=fake) == 8
    assert "[8]" in asked[0]


def test_invalid_or_out_of_range_answers_are_asked_again():
    fake, asked = _answers("lots", "0", "11", " 6 ")
    assert prompt.ask_count([], isatty=True, input_fn=fake) == 6
    assert len(asked) == 4


def test_launch_arg_skips_the_count_question():
    assert prompt.ask_count(["num_robots:=3"], isatty=True, input_fn=_never_called) == 3


def test_non_interactive_start_uses_the_default():
    assert prompt.ask_count([], isatty=False, input_fn=_never_called) == 8


def test_out_of_range_launch_arg_is_rejected():
    with pytest.raises(SystemExit):
        prompt.ask_count(["num_robots:=50"], isatty=True, input_fn=_never_called)
