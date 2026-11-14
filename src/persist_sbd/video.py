"""Video decoding and RGB frame preprocessing.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Real
from typing import Iterator, List, Optional

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    n_frames: Optional[int]  # from the container; the decoded count is authoritative
    # Decoded presentation times; None when unavailable.
    times: List[Optional[float]] = field(default_factory=list)
    # Duration-based estimate for progress display only.
    n_frames_estimate: Optional[int] = None


def _seconds(value) -> Optional[float]:
    """A finite timestamp in seconds, or None."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def preprocess(frame: np.ndarray, short_side: int = 128, crop: int = 112) -> np.ndarray:
    """Resize the short side, then centre-crop RGB input with the paper's rounding rules."""
    h, w = frame.shape[:2]
    scale = short_side / min(h, w)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    if (new_w, new_h) != (w, h):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, left = (new_h - crop) // 2, (new_w - crop) // 2
    return np.ascontiguousarray(frame[top:top + crop, left:left + crop])


@contextmanager
def _pyav(path: str):
    import av
    container = av.open(path)
    iterator = None
    try:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        info = VideoInfo(path, float(stream.average_rate or stream.guessed_rate or 0.0), stream.codec_context.width,
                         stream.codec_context.height, stream.frames or None)
        if not info.n_frames:
            try:
                if getattr(stream, "duration", None) and getattr(stream, "time_base", None):
                    seconds = float(stream.duration * stream.time_base)
                else:
                    seconds = float(container.duration) / 1e6
                if seconds > 0 and info.fps > 0:
                    info.n_frames_estimate = int(round(seconds * info.fps))
            except (TypeError, ValueError, AttributeError):
                pass

        def frames() -> Iterator[np.ndarray]:
            for frame in container.decode(stream):
                info.times.append(_seconds(getattr(frame, "time", None)))
                yield frame.to_ndarray(format="rgb24")

        iterator = frames()
        yield info, iterator
    finally:
        if iterator is not None:
            iterator.close()
        container.close()


_DECORD_BATCH_BYTES = 64 * 1024 * 1024


@contextmanager
def _decord(path: str):
    import decord
    vr = decord.VideoReader(path, num_threads=1)
    iterator = None
    try:
        n_frames = len(vr)
        if n_frames == 0:
            raise ValueError(f"no video frames decoded from {path}")
        h, w = vr[0].shape[:2]
        info = VideoInfo(path, float(vr.get_avg_fps()), w, h, n_frames)
        # Bound raw RGB batches; a single frame and decoder buffers can exceed this budget.
        batch_size = max(1, min(64, _DECORD_BATCH_BYTES // max(1, h * w * 3)))

        def frames() -> Iterator[np.ndarray]:
            for start in range(0, n_frames, batch_size):
                indices = list(range(start, min(start + batch_size, n_frames)))
                batch = vr.get_batch(indices).asnumpy()
                try:
                    stamps = [_seconds(t) for t in np.asarray(vr.get_frame_timestamp(indices))[:, 0]]
                except Exception:  # fall back to frame-rate timing
                    stamps = [None] * len(indices)
                for frame, stamp in zip(batch, stamps):
                    info.times.append(stamp)
                    yield frame
                del batch

        iterator = frames()
        yield info, iterator
    finally:
        if iterator is not None:
            iterator.close()
        # VideoReader has no close(); clear its reference after closing the generator.
        vr = None


@contextmanager
def _opencv(path: str):
    cap = cv2.VideoCapture(path)
    iterator = None
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open {path}")
        info = VideoInfo(path, float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None)

        def frames() -> Iterator[np.ndarray]:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                info.times.append(None)  # do not use unreliable OpenCV frame timestamps
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        iterator = frames()
        yield info, iterator
    finally:
        if iterator is not None:
            iterator.close()
        cap.release()


DECODERS = {"pyav": _pyav, "decord": _decord, "opencv": _opencv}


@contextmanager
def open_video(path: str, decoder: str = "pyav", short_side: int = 128, crop: int = 112):
    """Yield (VideoInfo, preprocessed RGB frame iterator) and close decoder resources.

    Empty streams raise ValueError when consumed.
    """
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {sorted(DECODERS)}, got {decoder!r}")
    with DECODERS[decoder](path) as (info, frames):
        def processed() -> Iterator[np.ndarray]:
            decoded = False
            for frame in frames:
                decoded = True
                yield preprocess(frame, short_side, crop)
            if not decoded:
                raise ValueError(f"no video frames decoded from {path}")

        iterator = processed()
        try:
            yield info, iterator
        finally:
            iterator.close()
