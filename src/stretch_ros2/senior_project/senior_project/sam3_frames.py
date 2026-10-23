"""
What the SAM3 server does with an incoming frame, outside the model.

It shrinks frames whose long side is over its input cap before the model sees
them. SAM3 answers in the pixels of the image it was given, so the server maps
boxes and masks back to the frame the client sent: clients never need to know
the cap. It also skips the encoder when the same frame comes twice, keyed on
the whole payload (a key on its first and last 8k chars once matched a frame
with the soda can to one without it).

numpy only, so it is unit-tested without torch (test/test_sam3_frames.py).
"""

import hashlib
from typing import List, Sequence, Tuple

import numpy as np


def cache_key(base64_str: str) -> str:
    """Key of a frame for the server's embedding cache (whole payload)."""
    return hashlib.md5(base64_str.encode(), usedforsecurity=False).hexdigest()


def capped_size(w: int, h: int, cap: int) -> Tuple[int, int]:
    """(w, h) shrunk so the long side is at most cap, shape kept."""
    if max(w, h) <= cap:
        return w, h
    scale = cap / max(w, h)
    return max(1, int(w * scale)), max(1, int(h * scale))


def boxes_to_sent(boxes: Sequence[Sequence[float]], resized_wh: Tuple[int, int],
                  sent_wh: Tuple[int, int]) -> List[List[float]]:
    """[x1, y1, x2, y2] boxes in the resized frame -> the sent frame."""
    sx, sy = sent_wh[0] / resized_wh[0], sent_wh[1] / resized_wh[1]
    return [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes]


def mask_to_sent(mask: np.ndarray, sent_wh: Tuple[int, int]) -> np.ndarray:
    """Mask of the resized frame -> the sent frame's size (nearest pixel)."""
    h, w = mask.shape[:2]
    rows = np.minimum(((np.arange(sent_wh[1]) + 0.5) * h / sent_wh[1]).astype(int), h - 1)
    cols = np.minimum(((np.arange(sent_wh[0]) + 0.5) * w / sent_wh[0]).astype(int), w - 1)
    return np.ascontiguousarray(mask[rows[:, None], cols[None, :]])
