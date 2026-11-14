"""Model loading, video detection and result serialization.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import csv
import dataclasses
import errno
import hashlib
import io
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import torch

from . import _download, infer
from .model import PERSIST
from .video import open_video, preprocess
from ._validation import positive_int, probability_threshold, window_settings

DEFAULT_REPO = "linty5/PERSIST"
# Pin the default model to the code release.
DEFAULT_REVISION = "v1.1.0"
DEFAULT_THRESHOLD = 0.5
RESULT_FORMAT, RESULT_FORMAT_VERSION = "persist_sbd.result", 1
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

Progress = Callable[[int, Optional[int]], None]  # (frames read so far, total frames if known)


def _device(device: Optional[str]) -> torch.device:
    """Select CUDA or CPU by default; MPS requires an explicit request."""
    if not device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        chosen = torch.device(device)
    except (RuntimeError, ValueError):
        raise ValueError(f"unknown device {device!r}: use cuda, cpu or mps") from None
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("this PyTorch installation cannot use a CUDA GPU here; use --device cpu "
                         "(or install a CUDA build of PyTorch)")
    if chosen.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise ValueError("Apple MPS is not available here; use --device cpu")
    return chosen


def _hub_error(model: str, exc: Exception) -> str:
    name = type(exc).__name__
    if name == "LocalEntryNotFoundError" or "offline" in str(exc).lower():
        offline = _download.offline_mode()
        return (f"the model {model!r} is not in the local Hugging Face cache and cannot be downloaded now. "
                + ("HF_HUB_OFFLINE is set: unset it and run once with network access (persist_sbd --download), "
                   if offline else "Run once with network access (persist_sbd --download), ")
                + "or pass a local model folder with --model.")
    if name == "RepositoryNotFoundError":
        return f"{model!r} is neither a local model folder nor a Hugging Face repository"
    if name in ("RevisionNotFoundError", "EntryNotFoundError"):
        return f"the requested model files were not found in {model!r} ({exc})"
    return f"could not get the model {model!r} from Hugging Face ({name}: {exc})"


def _files(model: str, variant: Optional[str], revision: Optional[str],
           cache_dir: Optional[str] = None, local_files_only: bool = False) -> Tuple[str, str]:
    """Resolve config and weights from a local folder or the Hub."""
    sub = f"{variant}/" if variant else ""
    if os.path.isdir(model):
        files = os.path.join(model, sub, "config.json"), os.path.join(model, sub, "model.safetensors")
        missing = [os.path.basename(f) for f in files if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(f"the model folder {os.path.join(model, sub)} has no {' and no '.join(missing)}")
        return files
    if os.path.isfile(model):
        raise FileNotFoundError(f"--model needs the folder that holds config.json and model.safetensors, "
                                f"not a file: try {os.path.dirname(os.path.abspath(model))}")
    if not _REPO_ID.match(model) or model.startswith((".", "~")):
        raise FileNotFoundError(f"model folder not found: {model}")
    from huggingface_hub import constants, hf_hub_download
    options = dict(revision=revision, cache_dir=cache_dir,
                   local_files_only=_download.offline_mode(local_files_only or constants.HF_HUB_OFFLINE))
    try:
        cfg = hf_hub_download(model, f"{sub}config.json", **options)
        return cfg, hf_hub_download(model, f"{sub}model.safetensors", **options)
    except (AssertionError, KeyboardInterrupt):
        raise
    except Exception as exc:  # the Hub client raises different classes across versions
        raise RuntimeError(_hub_error(model, exc)) from exc


def _sha256(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _software() -> Dict[str, str]:
    import cv2
    from . import __version__
    versions = {"persist_sbd": __version__, "torch": torch.__version__, "numpy": np.__version__,
                "opencv": cv2.__version__}
    for module in ("av", "decord"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            pass
    return versions


def _postprocess() -> Dict:
    """Postprocessing settings recorded with each result."""
    from . import __version__
    return dict(persist_sbd=__version__, cut_max_frames=infer.CUT_MAX_FRAMES,
                merge_gap_frames=infer.MERGE_GAP_FRAMES, min_edge_shot_frames=infer.MIN_EDGE_SHOT_FRAMES)


def _quantized(values: np.ndarray) -> np.ndarray:
    """Round scores before thresholding so saved results preserve decisions."""
    return np.round(np.asarray(values, np.float64), 6).astype(np.float32)


def _public_name(name: str) -> str:
    """Remove URL credentials, query and fragment from result metadata."""
    if "://" not in name:
        return name
    parts = urlsplit(name)
    return urlunsplit((parts.scheme, parts.hostname or "", parts.path, "", "")) if parts.hostname else name


def _write(path: str, text: str, overwrite: bool) -> None:
    """Stage output before publication; refuse existing paths unless overwrite=True.

    Without hard links, exclusive creation is non-atomic; cleanup checks file identity.
    """
    path = os.fspath(path)
    tmp = os.path.join(os.path.dirname(os.path.abspath(path)),
                       f".{os.path.basename(path)}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8", newline="") as fh:
            fh.write(text)
        if overwrite:
            os.replace(tmp, path)
        elif os.path.lexists(path):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
        else:
            try:
                os.link(tmp, path)  # atomic publication without replacing concurrent output
            except FileExistsError:
                raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path) from None
            except OSError:  # file systems without hard links
                created = None
                try:
                    with open(path, "x", encoding="utf-8", newline="") as fh:
                        created = os.fstat(fh.fileno())
                        fh.write(text)
                except BaseException:
                    if created is not None:
                        try:
                            current = os.stat(path, follow_symlinks=False)
                            if os.path.samestat(created, current):
                                os.unlink(path)
                        except OSError:
                            pass  # leave a replaced path alone; preserve the write error
                    raise
    finally:
        if os.path.lexists(tmp):
            os.unlink(tmp)


def _relative_times(times: List[Optional[float]], n: int) -> Tuple[Optional[np.ndarray], str]:
    """Return relative timestamps or mark frame-rate timing as the fallback."""
    if n == 0 or len(times) != n or any(t is None for t in times):
        return None, "frame_rate"
    t = np.asarray(times, np.float64)
    if not np.all(np.isfinite(t)) or (n > 1 and np.any(np.diff(t) <= 0)):
        return None, "frame_rate"
    return t - t[0], "timestamps"


def _counted(frames: Iterable[np.ndarray], progress: Optional[Progress], total: Optional[int]) -> Iterator[np.ndarray]:
    done = 0
    for frame in frames:
        done += 1
        if progress is not None and done % 32 == 0:
            progress(done, total)
        yield frame
    if progress is not None:
        progress(done, total)


@dataclass
class Result:
    """Video detections with zero-based frames and seconds relative to the first frame."""
    video: str
    fps: float
    n_frames: int
    threshold: float
    edges: str
    scores: np.ndarray  # [n_frames] boundary probability per frame
    gates: Dict[str, np.ndarray] = field(repr=False)  # change, transient, return, joint (per frame)
    covered: np.ndarray = field(repr=False)  # frames that received a score
    seconds: float = 0.0
    times: Optional[np.ndarray] = field(default=None, repr=False)  # relative frame timestamps, when available
    info: Dict = field(default_factory=dict, repr=False)

    @property
    def transitions(self) -> List[Dict]:
        """Transitions with frame spans, cut/gradual labels and peak scores."""
        return infer.transitions(self.scores, self.threshold)

    @property
    def shots(self) -> List[Tuple[int, int]]:
        """Consecutive shots (first frame, last frame) covering the whole video."""
        return infer.shots(self.scores, self.threshold)

    @property
    def scored_frames(self) -> int:
        return int(np.count_nonzero(self.covered))

    @property
    def scored_ranges(self) -> List[Tuple[int, int]]:
        """Inclusive, zero-based ranges whose scores came from the model."""
        changes = np.diff(np.pad(np.asarray(self.covered, dtype=np.int8), (1, 1)))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    def time(self, frame: int) -> Optional[float]:
        """Start time of a frame in seconds (frame == n_frames: the end of the video), or None if unknown."""
        rate = 1.0 / self.fps if math.isfinite(self.fps) and self.fps > 0 else None
        if self.times is not None and len(self.times) == self.n_frames > 0:
            if frame < self.n_frames:
                return float(self.times[frame])
            step = float(self.times[-1] - self.times[-2]) if self.n_frames > 1 else rate
            return None if step is None else float(self.times[-1]) + step
        return frame * rate if rate is not None else None

    def with_threshold(self, threshold: float) -> "Result":
        """The same scores read at another threshold (no new inference)."""
        return dataclasses.replace(self, threshold=probability_threshold(threshold),
                                   info=dict(self.info, postprocess=_postprocess()))

    def to_dict(self, per_frame: bool = False) -> Dict:
        def t(frame):
            value = self.time(frame)
            return None if value is None else round(value, 6)

        fps = self.fps if math.isfinite(self.fps) and self.fps > 0 else None
        d = dict(format=RESULT_FORMAT, format_version=RESULT_FORMAT_VERSION, video=self.video, fps=fps,
                 n_frames=self.n_frames, threshold=self.threshold, edges=self.edges,
                 time_source=self.info.get("time_source", "frame_rate"),
                 scored_frames=self.scored_frames,
                 scored_ranges=[dict(start=s, end=e) for s, e in self.scored_ranges],
                 shots=[dict(start=s, end=e, frames=e - s + 1, start_time=t(s), end_time=t(e + 1))
                        for s, e in self.shots],
                 transitions=[dict(x, time=t(x["boundary"] + 1), start_time=t(x["start"]), end_time=t(x["end"] + 1))
                              for x in self.transitions])
        for key in ("model", "decoder", "device", "software", "postprocess"):
            if key in self.info:
                d[key] = self.info[key]
        if per_frame:
            d["scores"] = np.round(np.asarray(self.scores, np.float64), 6).tolist()
            d["gates"] = {k: np.round(np.asarray(v, np.float64), 6).tolist() for k, v in self.gates.items()}
            d["covered"] = np.asarray(self.covered, bool).tolist()
            if self.times is not None:
                d["times"] = np.round(np.asarray(self.times, np.float64), 6).tolist()
        return d

    def save_json(self, path: str, per_frame: bool = False, *, overwrite: bool = True) -> None:
        """Save JSON; per_frame=True retains scores for re-thresholding.

        overwrite=False refuses existing files; see _write for publication semantics.
        """
        d = self.to_dict(per_frame)
        arrays = {k: d.pop(k) for k in ("scores", "gates", "covered", "times") if k in d}
        text = json.dumps(d, indent=1, allow_nan=False)
        if arrays:
            compact = [f" {json.dumps(k)}: {json.dumps(v, separators=(',', ':'), allow_nan=False)}"
                       for k, v in arrays.items()]
            text = text[:-2] + ",\n" + ",\n".join(compact) + "\n}"
        _write(path, text, overwrite)

    def save_csv(self, path: str, *, overwrite: bool = True) -> None:
        """Save inclusive shot intervals and times; paper mode adds scored-frame counts."""
        def seconds(value):
            return "" if value is None else f"{value:.3f}"

        header = ["shot", "first_frame", "last_frame", "start_time", "end_time", "frames", "duration"]
        paper = self.edges == "paper"
        buffer = io.StringIO()
        w = csv.writer(buffer)
        w.writerow(header + (["scored_frames"] if paper else []))
        for i, (s, e) in enumerate(self.shots, 1):
            start, end = self.time(s), self.time(e + 1)
            duration = end - start if start is not None and end is not None else None
            row = [i, s, e, seconds(start), seconds(end), e - s + 1, seconds(duration)]
            if paper:
                row.append(int(np.count_nonzero(self.covered[s:e + 1])))
            w.writerow(row)
        _write(path, buffer.getvalue(), overwrite)

    @classmethod
    def load_json(cls, path: str) -> "Result":
        """Read a result written with per_frame=True (CLI --per-frame)."""
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if not isinstance(d, dict) or d.get("format", RESULT_FORMAT) != RESULT_FORMAT or (
                "format" not in d and not {"n_frames", "shots", "transitions", "threshold"} <= set(d)):
            raise ValueError(f"{path} is not a persist_sbd result")
        if int(d.get("format_version", 1)) > RESULT_FORMAT_VERSION:
            raise ValueError(f"{path} was written by a newer persist_sbd; please upgrade to read it")
        if "scores" not in d:
            raise ValueError(f"{path} has no per-frame scores; write results with --per-frame "
                             "(per_frame=True) to apply another threshold later")
        scores = np.asarray(d["scores"], np.float32)
        n = len(scores)
        fps = d.get("fps")
        times = d.get("times")
        return cls(video=d.get("video", os.fspath(path)), fps=float(fps) if fps else float("nan"), n_frames=n,
                   threshold=probability_threshold(d["threshold"]), edges=d.get("edges", "full"), scores=scores,
                   gates={k: np.asarray(v, np.float32) for k, v in d.get("gates", {}).items()},
                   covered=np.asarray(d.get("covered", [True] * n), bool),
                   times=None if times is None else np.asarray(times, np.float64),
                   info={k: d[k] for k in ("time_source", "model", "decoder", "device", "software", "postprocess")
                         if k in d})


class Detector:
    """A loaded PERSIST model. Use persist_sbd.load() to create one."""

    def __init__(self, model: PERSIST, config: Dict, device: torch.device, source: Optional[Dict] = None) -> None:
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.source = dict(source or {})
        self.threshold = probability_threshold(config.get("threshold", DEFAULT_THRESHOLD))

    def _settings(self, threshold, edges, batch_size):
        threshold = probability_threshold(self.threshold if threshold is None else threshold)
        bs = (8 if self.device.type == "cuda" else 2) if batch_size is None else positive_int(batch_size, "batch_size")
        p = self.config["preprocess"]
        return threshold, bs, window_settings(p["clip_len"], p["clip_stride"], p["center_keep_len"], edges)

    def _result(self, name, fps, run, threshold, edges, decoder, times, t0) -> Result:
        scores, gates, covered = run
        rel, source = _relative_times(times, len(scores))
        info = dict(time_source=source, model=self.source, decoder=decoder, device=self.device.type,
                    software=_software(), postprocess=_postprocess())
        return Result(video=_public_name(name), fps=fps, n_frames=len(scores), threshold=threshold, edges=edges,
                      scores=_quantized(scores), gates={k: _quantized(v) for k, v in gates.items()},
                      covered=covered, seconds=time.time() - t0, times=rel, info=info)

    def detect(self, video: str, threshold: Optional[float] = None, edges: str = "full", decoder: str = "pyav",
               batch_size: Optional[int] = None, progress: Optional[Progress] = None) -> Result:
        """Detect boundaries in a video using PyAV and full-frame coverage by default.

        Paper mode leaves uncovered frames at zero with covered=False.
        progress receives (frames_read, total_or_None).
        """
        threshold, bs, (clip_len, stride, keep) = self._settings(threshold, edges, batch_size)
        p = self.config["preprocess"]
        t0 = time.time()
        video = os.fspath(video)
        with open_video(video, decoder, p["resize_short_side"], p["center_crop"]) as (info, frames):
            total = getattr(info, "n_frames", None) or getattr(info, "n_frames_estimate", None)
            run = infer.run(self.model, _counted(frames, progress, total), self.device, clip_len, stride, keep,
                            edges, bs)
            times = list(getattr(info, "times", []))
        return self._result(video, info.fps, run, threshold, edges, decoder, times, t0)

    def detect_frames(self, frames: Iterable[np.ndarray], fps: Optional[float] = None,
                      threshold: Optional[float] = None, edges: str = "full", batch_size: Optional[int] = None,
                      progress: Optional[Progress] = None, name: str = "frames") -> Result:
        """Detect from ordered RGB uint8 frames [H, W, 3]; convert OpenCV BGR input first."""
        threshold, bs, (clip_len, stride, keep) = self._settings(threshold, edges, batch_size)
        p = self.config["preprocess"]

        def prepared() -> Iterator[np.ndarray]:
            for frame in frames:
                frame = np.asarray(frame)
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise ValueError(f"frames must be RGB uint8 arrays [H, W, 3], got {frame.dtype} {frame.shape}")
                yield preprocess(frame, p["resize_short_side"], p["center_crop"])

        t0 = time.time()
        run = infer.run(self.model, _counted(prepared(), progress, None), self.device, clip_len, stride, keep,
                        edges, bs)
        if not len(run[0]):
            raise ValueError("no frames were given")
        return self._result(name, float(fps) if fps else float("nan"), run, threshold, edges, None, [], t0)


def load(model: str = DEFAULT_REPO, variant: Optional[str] = None, device: Optional[str] = None,
         revision: Optional[str] = None, cache_dir: Optional[str] = None,
         local_files_only: bool = False) -> Detector:
    """Load a Hub repository or local folder as a Detector.

    The default uses a fixed HF revision, with a GitHub fallback only when model,
    revision and variant are left at their defaults. local_files_only forbids networking.
    """
    from safetensors.torch import load_file
    model = os.fspath(model)
    target = _device(device)  # validate before downloading
    fallback = model == DEFAULT_REPO and revision is None and variant is None and not os.path.isdir(model)
    if revision is None and model == DEFAULT_REPO and not os.path.isdir(model):
        revision = DEFAULT_REVISION
    github = False
    try:
        cfg_path, weights_path = _files(model, variant, revision, cache_dir, local_files_only)
    except RuntimeError:
        if not fallback:
            raise
        from huggingface_hub import constants
        root = constants.HF_HUB_CACHE if cache_dir is None else cache_dir
        cfg_path, weights_path = _download.release_files(
            root, local_files_only=bool(local_files_only or constants.HF_HUB_OFFLINE))
        github = True
    try:
        with open(cfg_path) as fh:
            config = json.load(fh)
    except ValueError as exc:
        raise ValueError(f"{cfg_path} is not a valid JSON file ({exc})") from None
    if not isinstance(config, dict) or not {"model", "preprocess"} <= set(config):
        raise ValueError(f"{cfg_path} is not a persist_sbd model config")
    try:
        weights = load_file(weights_path)
    except Exception as exc:  # safetensors raises its own error class
        if os.path.getsize(weights_path) < 4096:
            raise ValueError(f"{weights_path} is not a weights file but a small placeholder: probably a Git LFS "
                             "pointer or an incomplete download; download the model again") from None
        raise ValueError(f"{weights_path} cannot be read as safetensors weights ({exc})") from None
    net = PERSIST(config["model"])
    net.load_state_dict(weights, strict=True)
    snapshot = re.search(r"[/\\]snapshots[/\\]([0-9a-f]{40})[/\\]", os.fspath(weights_path))
    local = os.path.isdir(model)
    # Keep local paths out of shared result metadata.
    source = dict(model=os.path.basename(os.path.abspath(model)) if local else model, variant=variant,
                  revision=None if local else (snapshot.group(1) if snapshot else revision),
                  sha256=_sha256(weights_path), seed=config.get("seed"), epoch=config.get("epoch"))
    if github:
        source.update(model="https://github.com/" + _download.RELEASE_REPO + "/releases/tag/"
                      + _download.DEFAULT_RELEASE, revision=_download.DEFAULT_RELEASE, provider="github-release")
    return Detector(net, config, target, source)
