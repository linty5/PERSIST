"""Sliding-window inference and conversion of scores to transitions and shots.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import math
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

from ._validation import positive_int, window_settings

GATES = {"change": "trcs_change", "transient": "trcs_innov", "return": "trcs_return", "joint": "trcs_gate"}

# The training cut label spans six frames; longer response spans are labelled gradual.
CUT_MAX_FRAMES = 6
# Merge brief below-threshold gaps within a transition.
MERGE_GAP_FRAMES = 1
# Discard boundaries leaving an edge shot shorter than this.
MIN_EDGE_SHOT_FRAMES = 6


def _windows(frames: Iterable[np.ndarray], clip_len: int, stride: int, keep: int,
             edges: str) -> Iterator[Tuple[int, np.ndarray, int, int]]:
    """Yield (start, RGB clip, lo, hi); retain clip positions [lo, hi) using a rolling buffer."""
    centre = (clip_len - keep) // 2
    head_real = edges == "full"
    start = -stride * math.ceil(centre / stride) if edges == "pad" else 0
    back = stride if edges == "full" else 0  # retain context for the end-aligned window
    buf: List[np.ndarray] = []
    base = 0  # absolute index of buf[0]
    first = None
    n = 0
    kept_to = None  # "full": first frame not yet kept
    it = iter(frames)
    done = False
    while True:
        while not done and n < start + clip_len:
            try:
                f = next(it)
            except StopIteration:
                done = True
                break
            if first is None:
                first = f
            buf.append(f)
            n += 1
        if n == 0:
            return
        if done and start + clip_len > n:
            if edges == "paper":
                return
            if head_real and kept_to is None:
                yield from _windows(iter(buf), clip_len, stride, keep, "pad")
                return
            if edges == "full":
                if kept_to < n:
                    e = n - clip_len
                    yield e, np.stack(buf[e - base:n - base]), kept_to - e, clip_len
                return
            if start + centre > n - 1:
                return
        idx = [min(max(i, 0), n - 1) for i in range(start, start + clip_len)]
        clip = np.stack([first if i < base else buf[i - base] for i in idx])
        lo = 0 if (head_real and start == 0) else centre
        yield start, clip, lo, centre + keep
        kept_to = start + centre + keep
        start += stride
        drop = max(0, min(start - back, n - 1) - base)  # keep the last frame for right padding
        if drop:
            del buf[:drop]
            base += drop


def run(model: torch.nn.Module, frames: Iterable[np.ndarray], device: torch.device, clip_len: int = 64,
        stride: int = 16, keep: int = 16, edges: str = "paper", batch_size: int = 4,
        on_window: Optional[Callable[[int], None]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Return frame probabilities, persistence gates and a coverage mask."""
    clip_len, stride, keep = window_settings(clip_len, stride, keep, edges)
    batch_size = positive_int(batch_size, "batch_size")
    names = ["probs"] + list(GATES.values())
    sums = {k: np.zeros(0, np.float64) for k in names}
    count = np.zeros(0, np.int64)
    n_seen = 0

    def flush(batch: List[Tuple[int, np.ndarray, int, int]]) -> None:
        nonlocal count
        clips = torch.from_numpy(np.stack([w[1] for w in batch])).permute(0, 4, 1, 2, 3).contiguous().to(device)
        with torch.no_grad():
            out = model(clips)
        missing = [k for k in names if k not in out]
        if missing:
            raise KeyError(f"the model does not output {missing}; persist_sbd needs the persistence gates")
        vals = {k: out[k].float().reshape(len(batch), -1).cpu().numpy() for k in names}
        for b, (s, _, klo, khi) in enumerate(batch):
            lo, hi = s + klo, s + khi
            if hi > len(count):
                grow = max(hi, 2 * len(count))
                count = np.concatenate([count, np.zeros(grow - len(count), np.int64)])
                for k in names:
                    sums[k] = np.concatenate([sums[k], np.zeros(grow - len(sums[k]))])
            a = max(lo, 0)
            for k in names:
                sums[k][a:hi] += vals[k][b, klo + (a - lo):khi]
            count[a:hi] += 1
            if on_window:
                on_window(s)

    def counted(fr: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        nonlocal n_seen
        for f in fr:
            n_seen += 1
            yield f

    centre = (clip_len - keep) // 2
    batch: List[Tuple[int, np.ndarray, int, int]] = []
    for w in _windows(counted(frames), clip_len, stride, keep, edges):
        if edges == "full" and w[2] > centre and batch:
            # Keep grid batches identical to paper mode; run the end-aligned window separately.
            flush(batch)
            batch = []
        batch.append(w)
        if len(batch) == batch_size:
            flush(batch)
            batch = []
    if batch:
        flush(batch)

    n = n_seen
    count = np.concatenate([count, np.zeros(max(0, n - len(count)), np.int64)])[:n]
    covered = count > 0
    avg = {}
    for k in names:
        s = np.concatenate([sums[k], np.zeros(max(0, n - len(sums[k])))])[:n]
        v = np.zeros(n, np.float32)
        v[covered] = (s[covered] / count[covered]).astype(np.float32)
        avg[k] = v
    probs = avg.pop("probs")
    gates = {name: avg[key] for name, key in GATES.items()}
    return probs, gates, covered


def transitions(probs: np.ndarray, threshold: float) -> List[Dict]:
    """Merge thresholded runs and locate boundaries at their midpoints.

    Spans of at most CUT_MAX_FRAMES are cuts; longer spans are gradual.
    The boundary is the previous shot's last frame. Frame spans are inclusive.
    """
    on = probs >= threshold
    runs, i, n = [], 0, len(probs)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1]:
            j += 1
        if runs and i - runs[-1][1] - 1 <= MERGE_GAP_FRAMES:
            runs[-1][1] = j
        else:
            runs.append([i, j])
        i = j + 1
    runs = [(i, j, (i + j) // 2) for i, j in runs]
    runs = [r for r in runs if r[2] < n - 1]
    while runs and runs[0][2] + 1 < MIN_EDGE_SHOT_FRAMES:
        runs.pop(0)
    while runs and n - 1 - runs[-1][2] < MIN_EDGE_SHOT_FRAMES:
        runs.pop()
    out = []
    for i, j, boundary in runs:
        cut = j - i + 1 <= CUT_MAX_FRAMES
        out.append(dict(start=boundary if cut else i, end=boundary + 1 if cut else j,
                        boundary=boundary, kind="cut" if cut else "gradual",
                        peak=round(float(probs[i:j + 1].max()), 6), run_start=i, run_end=j))
    return out


def shots(probs: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """Return consecutive inclusive shot intervals covering the video."""
    n = len(probs)
    if n == 0:
        return []
    out, start = [], 0
    for t in transitions(probs, threshold):
        out.append((start, t["boundary"]))
        start = t["boundary"] + 1
    out.append((start, n - 1))
    return out
