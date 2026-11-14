"""Installation self-test using a generated video with one hard cut.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import os
import tempfile
import time

import numpy as np

W, H, N, CUT = 320, 180, 128, 64


def _texture(kind: str, t: int) -> np.ndarray:
    yy, xx = np.indices((H, W)).astype(np.float64)
    if kind == "a":
        v = 128 + 60 * np.sin(2 * np.pi * (xx / 40 + t / 50)) + 40 * np.sin(2 * np.pi * yy / 30)
        rgb = np.stack([v, 0.8 * v + 30, 255 - v], axis=-1)
    else:
        v = 128 + 70 * np.sin(2 * np.pi * (xx / 17 - t / 60) + yy / 23) + 30 * np.cos(2 * np.pi * yy / 11)
        rgb = np.stack([0.6 * v + 60, 255 - 0.9 * v, 0.7 * v + 20], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def write_video(path: str) -> str:
    """Write the test video: texture "b" for frames 0..63, texture "a" from frame 64 on."""
    import av
    with av.open(path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.bit_rate = 8_000_000
        for t in range(N):
            frame = _texture("b" if t < CUT else "a", t)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def run(detector, out=print, decoder: str = "pyav", batch_size=None) -> bool:
    """Report detections and return whether the expected cut was found."""
    from .api import _software
    software = _software()
    source = detector.source
    out(f"persist_sbd {software.get('persist_sbd', '?')}, torch {software.get('torch', '?')}, "
        f"PyAV {software.get('av', '?')}, OpenCV {software.get('opencv', '?')}, device {detector.device.type}")
    out(f"model {source.get('model', '?')}" + (f" @ {source['revision'][:12]}" if source.get("revision") else "")
        + (f", sha256 {source['sha256'][:16]}" if source.get("sha256") else "")
        + f", threshold {detector.threshold}")
    with tempfile.TemporaryDirectory() as directory:
        try:
            path = write_video(os.path.join(directory, "selftest.mp4"))
        except Exception as exc:
            out(f"selftest FAILED: cannot write the test video ({type(exc).__name__}: {exc}); "
                "PyAV with FFmpeg's mpeg4 encoder is needed for the test")
            return False
        try:
            t0 = time.time()
            result = detector.detect(path, decoder=decoder, batch_size=batch_size)
            seconds = time.time() - t0
        except Exception as exc:
            hint = "" if detector.device.type == "cpu" else "; try --device cpu"
            out(f"selftest FAILED: detection raised {type(exc).__name__}: {exc}{hint}")
            return False
    found = [(t["kind"], t["boundary"]) for t in result.transitions]
    ok = len(found) == 1 and found[0][0] == "cut" and found[0][1] in (CUT - 2, CUT - 1, CUT)
    out(f"test video: {result.n_frames} frames in {seconds:.1f} s ({result.n_frames / max(seconds, 1e-9):.0f} frames/s); "
        f"expected one cut after frame {CUT - 1}, found {found}")
    out("selftest passed" if ok else "selftest FAILED: unexpected detections")
    return ok
