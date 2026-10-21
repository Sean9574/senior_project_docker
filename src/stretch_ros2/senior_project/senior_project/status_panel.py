#!/usr/bin/env python3
"""
Live terminal status panel for training.

Started by RL.launch.py (status_panel:=true, the default). Listens to the
learner's /reward_breakdown, the SAM3 goal generator's status and /rosout
(warnings and errors from every node) and redraws one panel in place, so the
launch terminal shows progress instead of interleaved node logs.

With --plain (stdout is not a terminal: docker logs, nohup) it prints one
summary line every 30 s instead of redrawing.
"""

import argparse
import json
import os
import shutil
import sys
import time

import rclpy
from rcl_interfaces.msg import Log
from rclpy.node import Node
from std_msgs.msg import String

from senior_project.status_panel_core import PanelState, render, summary_line

try:
    from rclpy.executors import ExternalShutdownException
except ImportError:  # older rclpy
    class ExternalShutdownException(Exception):
        pass

CLEAR = "\x1b[H\x1b[2J"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"


class StatusPanelNode(Node):
    def __init__(self, state: PanelState, interactive: bool):
        super().__init__("status_panel")
        self.state = state
        self.interactive = interactive
        self.create_subscription(String, "/reward_breakdown", self._on_breakdown, 10)
        self.create_subscription(String, "/sam3_goal_generator/status", self._on_generator_status, 10)
        self.create_subscription(Log, "/rosout", self._on_log, 50)
        self.create_timer(0.5 if interactive else 30.0, self._draw)

    def _on_breakdown(self, msg: String):
        try:
            self.state.on_breakdown(json.loads(msg.data), time.time())
        except (ValueError, TypeError):
            pass

    def _on_generator_status(self, msg: String):
        try:
            self.state.on_generator_status(json.loads(msg.data), time.time())
        except (ValueError, TypeError):
            pass

    def _on_log(self, msg: Log):
        if msg.name != self.get_name():
            self.state.on_log(int(msg.level), msg.name, msg.msg, time.time())

    def _draw(self):
        now = time.time()
        if self.interactive:
            width = shutil.get_terminal_size((80, 24)).columns
            sys.stdout.write(CLEAR + "\n".join(render(self.state, now, width)) + "\n")
        else:
            sys.stdout.write(summary_line(self.state, now) + "\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Live training status in the terminal")
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--plain", action="store_true",
                        help="one summary line every 30 s instead of a redrawn panel")
    args, _ = parser.parse_known_args()

    log_dir = os.environ.get("ROS_LOG_DIR") or os.path.join(
        os.environ.get("ROS_HOME", os.path.expanduser("~/.ros")), "log")
    state = PanelState(total_steps=args.total_steps, target=args.target,
                       domain=os.environ.get("ROS_DOMAIN_ID", "0"), log_dir=log_dir)
    interactive = not args.plain

    rclpy.init()
    node = StatusPanelNode(state, interactive)
    if interactive:
        sys.stdout.write(HIDE_CURSOR)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if interactive:
            sys.stdout.write(SHOW_CURSOR + "\n")
            sys.stdout.flush()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
