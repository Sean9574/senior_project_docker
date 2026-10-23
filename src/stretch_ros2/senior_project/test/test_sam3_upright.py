"""Sending SAM3 upright frames from the sideways head camera (numpy only).

Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_sam3_upright.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from senior_project import sam3_upright  # noqa: E402

RAW_H, RAW_W = 240, 424  # head camera frame as published (sideways)


def _raw_with_object():
    """Raw frame with one object covering x 100..139, y 50..79 (box 100,50,140,80)."""
    raw = np.zeros((RAW_H, RAW_W, 3), np.uint8)
    raw[50:80, 100:140] = 255
    return raw


def _box_of(mask):
    ys, xs = np.nonzero(mask)
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def test_upright_frame_is_the_raw_frame_turned_clockwise():
    raw = _raw_with_object()
    up = sam3_upright.to_upright(raw)
    assert up.shape == (RAW_W, RAW_H, 3)
    # top-left of the raw frame ends up top-right when turned clockwise
    raw[0, 0] = (1, 2, 3)
    assert tuple(sam3_upright.to_upright(raw)[0, RAW_H - 1]) == (1, 2, 3)


def test_box_found_in_the_upright_frame_maps_back_to_the_raw_frame():
    up = sam3_upright.to_upright(_raw_with_object())
    up_box = _box_of(up[:, :, 0])
    assert sam3_upright.box_to_raw(up_box, upright_w=RAW_H) == [100, 50, 140, 80]


def test_mask_found_in_the_upright_frame_maps_back_to_the_raw_frame():
    raw = _raw_with_object()[:, :, 0]
    up_mask = sam3_upright.to_upright(raw)
    back = sam3_upright.mask_to_raw(up_mask)
    assert back.shape == raw.shape
    assert np.array_equal(back, raw)
