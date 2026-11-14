"""Deterministic RGB clips for regression tests.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import numpy as np


def _rgb(v):
    v = v.astype(np.int64) % 256
    return np.stack([v, (v + 85) % 256, (v + 170) % 256], axis=-1).astype(np.uint8)


def regression_clips():
    """Return integer-built cut and flash clips as uint8 [2, 64, 112, 112, 3]."""
    t = np.arange(64)[:, None, None]
    yy, xx = np.indices((112, 112))
    ramp = _rgb(xx[None] + yy[None] + 4 * t)
    board = _rgb(((xx[None] + t) // 14 + yy[None] // 14) % 2 * 150 + 50 + 0 * t)
    cut = np.where((np.arange(64) < 32)[:, None, None, None], ramp, board)
    flash = _rgb(2 * yy[None] + 3 * t + 0 * xx[None])
    flash[30:32] = 255
    return np.stack([cut, flash])
