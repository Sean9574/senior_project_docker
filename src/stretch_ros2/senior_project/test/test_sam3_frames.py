"""What the SAM3 server does with a frame outside the model (numpy only): it
shrinks frames over its input cap, and what it finds must come back in the
pixels of the frame the client sent; its embedding cache must never mix up
two different frames.

Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_sam3_frames.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from senior_project import sam3_frames  # noqa: E402

SENT_W, SENT_H = 720, 1280      # upright 1280x720 head-camera frame
OBJECT_BOX = [100, 300, 200, 500]


def test_frames_within_the_cap_go_to_sam3_unchanged():
    assert sam3_frames.capped_size(SENT_W, SENT_H, cap=1280) == (SENT_W, SENT_H)
    assert sam3_frames.capped_size(240, 424, cap=1280) == (240, 424)


def test_bigger_frames_shrink_to_the_cap_keeping_their_shape():
    assert sam3_frames.capped_size(1440, 2560, cap=1280) == (720, 1280)


def test_box_found_in_the_shrunk_frame_maps_back_to_the_sent_frame():
    resized = sam3_frames.capped_size(SENT_W, SENT_H, cap=512)
    assert resized == (288, 512)
    shrunk_box = [v * 0.4 for v in OBJECT_BOX]
    back = sam3_frames.boxes_to_sent([shrunk_box], resized, (SENT_W, SENT_H))
    assert np.allclose(back, [OBJECT_BOX])


def test_mask_found_in_the_shrunk_frame_maps_back_to_the_sent_frame():
    x1, y1, x2, y2 = OBJECT_BOX
    shrunk = np.zeros((512, 288), np.uint8)
    shrunk[int(y1 * 0.4):int(y2 * 0.4), int(x1 * 0.4):int(x2 * 0.4)] = 255
    back = sam3_frames.mask_to_sent(shrunk, (SENT_W, SENT_H))
    expected = np.zeros((SENT_H, SENT_W), np.uint8)
    expected[y1:y2, x1:x2] = 255
    assert back.shape == expected.shape
    assert np.array_equal(back, expected)


def test_frames_that_differ_only_in_the_middle_get_different_cache_keys():
    # A 1280x720 frame is ~26k base64 chars; the can sits mid-image, and a
    # frame with it and one without once matched in their first/last 8k chars
    with_can = "A" * 13000 + "can" + "B" * 13000
    without = "A" * 13000 + "---" + "B" * 13000
    assert sam3_frames.cache_key(with_can) != sam3_frames.cache_key(without)
    assert sam3_frames.cache_key(with_can) == sam3_frames.cache_key("A" * 13000 + "can" + "B" * 13000)
