"""Synthetic videos for decoder and detection tests.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import numpy as np

W, H, N = 320, 180, 128


def _texture(kind, t):
    yy, xx = np.indices((H, W)).astype(np.float64)
    if kind == "a":
        v = 128 + 60 * np.sin(2 * np.pi * (xx / 40 + t / 50)) + 40 * np.sin(2 * np.pi * yy / 30)
        rgb = np.stack([v, 0.8 * v + 30, 255 - v], axis=-1)
    else:
        v = 128 + 70 * np.sin(2 * np.pi * (xx / 17 - t / 60) + yy / 23) + 30 * np.cos(2 * np.pi * yy / 11)
        rgb = np.stack([0.6 * v + 60, 255 - 0.9 * v, 0.7 * v + 20], axis=-1)
    return np.clip(rgb, 0, 255)


def frames(kind):
    """Return a cut, dissolve or flash sequence.

    Cuts end on texture a to avoid texture b's elevated end-window response.
    """
    out = []
    for t in range(N):
        a, b = _texture("a", t), _texture("b", t)
        if kind == "cut":
            f = b if t < 64 else a
        elif kind == "dissolve":
            w = min(max((t - 51) / 25.0, 0.0), 1.0)
            f = (1 - w) * a + w * b
        else:
            f = a if t not in (62, 63) else 0.1 * a + 0.9 * 255
        out.append(np.round(f).astype(np.uint8))
    return out


def write(path, kind):
    import av
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.bit_rate = 8_000_000
        for f in frames(kind):
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
