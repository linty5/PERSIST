"""Shared validation for inference parameters.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import math
from numbers import Integral, Real


def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def probability_threshold(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"threshold must be a finite number in (0, 1], got {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"threshold must be a finite number in (0, 1], got {value!r}")
    return value


def window_settings(clip_len, stride, keep, edges):
    clip_len = positive_int(clip_len, "clip_len")
    stride = positive_int(stride, "stride")
    keep = positive_int(keep, "center_keep_len")
    if keep > clip_len:
        raise ValueError("center_keep_len must not exceed clip_len")
    if edges not in ("full", "paper", "pad"):
        raise ValueError(f"edges must be 'full', 'paper' or 'pad', got {edges!r}")
    if edges in ("full", "pad") and stride > keep:
        raise ValueError(f"stride must not exceed center_keep_len when edges={edges!r} to ensure full-frame coverage")
    return clip_len, stride, keep
