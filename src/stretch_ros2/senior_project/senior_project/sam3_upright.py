"""
The Stretch head camera is mounted turned 90 degrees, so its frames arrive
sideways. SAM3 finds objects far less often in sideways images (measured in
the arena: "red can" found out to 6 m upright, only to 3 m sideways; "table"
not at all sideways at 1.5-2 m), so the goal generator sends it upright frames
and maps the boxes and masks back to the camera's own pixels, where the depth
image and the camera TF live.

numpy only, so it is unit-tested without ROS (test/test_sam3_upright.py).
"""

from typing import List, Sequence

import numpy as np


def to_upright(img: np.ndarray) -> np.ndarray:
    """Raw (sideways) frame turned 90 degrees clockwise."""
    return np.ascontiguousarray(np.rot90(img, k=-1))


def box_to_raw(box: Sequence[float], upright_w: int) -> List[float]:
    """[x1, y1, x2, y2] in the upright frame -> the same box in the raw frame.
    upright_w is the upright frame's width (= the raw frame's height)."""
    x1, y1, x2, y2 = box
    return [y1, upright_w - x2, y2, upright_w - x1]


def mask_to_raw(mask: np.ndarray) -> np.ndarray:
    """Mask of the upright frame turned back to the raw frame."""
    return np.ascontiguousarray(np.rot90(mask, k=1))
