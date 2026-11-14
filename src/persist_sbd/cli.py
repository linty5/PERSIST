"""Command-line video detection and result export.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import argparse
import collections
import glob
import os
import sys
import tempfile
from urllib.parse import urlsplit

from . import __version__
from .api import DEFAULT_REPO, DEFAULT_REVISION, Result, _device, load
from ._validation import positive_int, probability_threshold

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".qt", ".mkv", ".avi", ".divx", ".webm", ".mpg", ".mpeg", ".m2v",
                    ".mts", ".m2ts", ".ts", ".mod", ".tod", ".wmv", ".asf", ".flv", ".f4v", ".3gp", ".ogv", ".mxf",
                    ".dv", ".vob", ".rm", ".rmvb"}
RESULT_SUFFIX = ".persist.json"
FORMATS = ("json", "csv")
DECODER_PACKAGES = {"pyav": ("av", "av"), "decord": ("decord", "decord"), "opencv": ("cv2", "opencv-python")}
CASE_INSENSITIVE = sys.platform in ("darwin", "win32")


def _formats(text):
    values = [v.strip().lower() for v in text.split(",") if v.strip()]
    if not values or any(v not in FORMATS for v in values):
        raise argparse.ArgumentTypeError(f"formats are json and csv, comma-separated (e.g. json,csv), got {text!r}")
    return values


def _is_result(path: str) -> bool:
    return path.lower().endswith(".json")


def _stem(path: str) -> str:
    base = os.path.basename(urlsplit(path).path if "://" in path else path) or "video"
    return base[:-len(RESULT_SUFFIX)] if base.lower().endswith(RESULT_SUFFIX) else os.path.splitext(base)[0]


def _scan(folder: str, recursive: bool, problems: list):
    """Find videos, or saved results if no videos exist; skip hidden entries."""
    videos, results = [], []

    def take(root, name):
        path = os.path.join(root, name)
        if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
            videos.append(path)
        elif name.lower().endswith(RESULT_SUFFIX):
            results.append(path)

    if recursive:
        def unreadable(exc):
            problems.append(f"cannot read the folder {exc.filename}: {exc.strerror or exc}")
        for root, dirs, files in os.walk(folder, onerror=unreadable):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                if not name.startswith("."):
                    take(root, name)
    else:
        for name in sorted(os.listdir(folder)):
            if not name.startswith(".") and os.path.isfile(os.path.join(folder, name)):
                take(folder, name)
    return [(p, os.path.relpath(p, folder)) for p in videos or results]


def _inputs(paths, recursive, problems):
    """(input, output name relative to the output folder, folder it was found in or None)."""
    items = []
    for path in paths:
        if os.path.isdir(path):
            for found, rel in _scan(path, recursive, problems):
                rel = rel[:-len(RESULT_SUFFIX)] if rel.lower().endswith(RESULT_SUFFIX) else os.path.splitext(rel)[0]
                items.append((found, rel, path))
        else:
            items.append((path, _stem(path), None))
    return items


def _unique_names(items, out):
    """Resolve name collisions using extensions, folder names and numeric suffixes."""
    names = [stem for _, stem, _ in items]

    def key(name):
        k = os.path.normpath(os.path.join(out, name))
        return k.casefold() if CASE_INSENSITIVE else k

    def clashes():
        groups = collections.defaultdict(list)
        for i, name in enumerate(names):
            groups[key(name)].append(i)
        return [g for g in groups.values() if len(g) > 1]

    for group in clashes():
        for i in group:
            src = items[i][0]
            ext = os.path.splitext(urlsplit(src).path if "://" in src else src)[1]
            if ext and not _is_result(src):
                names[i] += ext
    for group in clashes():
        for i in group:
            src, _, folder = items[i]
            where = folder or os.path.dirname(os.path.abspath(src))
            names[i] = os.path.join(os.path.basename(os.path.normpath(where)) or "input", names[i])
    for group in clashes():
        for n, i in enumerate(group, 1):
            names[i] += f"-{n}"
    return names


def _describe(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    if "out of memory" in text.lower():
        return "out of GPU memory; try --batch-size 1 or --device cpu"
    if type(exc).__module__.split(".")[0] == "av":
        return f"cannot decode the video ({getattr(exc, 'strerror', None) or text})"
    if isinstance(exc, ModuleNotFoundError):
        return f"{text}; install the missing package or choose another --decoder"
    return text if isinstance(exc, (OSError, ValueError, RuntimeError)) else f"{type(exc).__name__}: {text}"


def _progress_bar(enabled: bool, label: str):
    """A callback drawing a progress bar on stderr, or None."""
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    state = {}

    def update(done, total):
        bar = state.get("bar")
        if bar is None:
            bar = state["bar"] = tqdm(total=total or None, unit="frame", desc=label, leave=False, file=sys.stderr,
                                      dynamic_ncols=True)
        bar.update(done - bar.n)

    update.close = lambda: state["bar"].close() if "bar" in state else None
    return update


def _load(a):
    if a.device is None:
        try:
            import torch
            if not torch.cuda.is_available() and torch.backends.mps.is_available():
                print("Note: running on the CPU. An Apple GPU is available: try --device mps.", file=sys.stderr)
        except Exception:
            pass
    return load(a.model, a.variant, a.device, revision=a.revision, cache_dir=a.cache_dir,
                local_files_only=a.local_files_only)


def _parser():
    ap = argparse.ArgumentParser(
        prog="persist_sbd", description="Detect shot boundaries with PERSIST.",
        epilog="Inputs can be video files, folders of videos, or result files written with --per-frame "
               "(to apply another --threshold without running the model again; a folder holding only such "
               "results is read as results).")
    ap.add_argument("inputs", nargs="*", metavar="VIDEO", help="video files, folders, or saved .json results")
    ap.add_argument("-o", "--out", default=".", help="output folder (default: current folder)")
    ap.add_argument("-f", "--format", type=_formats, action="extend", default=None, metavar="FORMATS",
                    help="json (default), csv, or json,csv: json has shots and transitions, csv the shot list")
    ap.add_argument("-r", "--recursive", action="store_true", help="also process videos in subfolders of folders")
    ap.add_argument("--per-frame", action="store_true",
                    help="also write per-frame scores, gates and times to the JSON (needed to re-threshold later)")
    ap.add_argument("--model", default=DEFAULT_REPO,
                    help="model repository or local folder (default: fetch the official checkpoint automatically)")
    ap.add_argument("--variant", default=None, help="optional checkpoint subfolder (default: root model)")
    ap.add_argument("--device", default=None, help="cuda, cpu or mps (default: CUDA when available, otherwise CPU)")
    ap.add_argument("--revision", default=None,
                    help=f"Hugging Face revision (default: {DEFAULT_REVISION} for {DEFAULT_REPO}, "
                         "main for other repositories; use --revision main to follow the latest files)")
    ap.add_argument("--cache-dir", default=None, help="model cache directory")
    ap.add_argument("--local-files-only", action="store_true", help="load only cached/local model files without networking")
    ap.add_argument("--threshold", type=float, default=None, help="decision threshold (default: the model's, 0.5)")
    ap.add_argument("--edges", default="full", choices=["full", "paper", "pad"],
                    help="full: score every frame (default); paper: the paper's evaluation coverage, without the "
                         "first 24 and the last frames; pad: score every frame by repeating the end frames")
    ap.add_argument("--decoder", default="pyav", choices=["pyav", "decord", "opencv"])
    ap.add_argument("--batch-size", type=int, default=None)
    existing = ap.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true", help="replace existing output files")
    existing.add_argument("--skip-existing", action="store_true",
                          help="skip inputs whose output files all exist (to resume an interrupted run)")
    ap.add_argument("-q", "--quiet", action="store_true", help="no progress bars")
    ap.add_argument("--selftest", action="store_true", help="check the installation on a generated test video")
    ap.add_argument("--download", action="store_true", help="download the model for later offline use")
    ap.add_argument("--version", action="version", version=f"persist_sbd {__version__}")
    return ap


def main(argv=None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(errors="backslashreplace")  # tolerate filenames outside the console encoding
        except Exception:
            pass
    ap = _parser()
    a = ap.parse_intermixed_args(argv)
    try:
        return _run(ap, a)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def _run(ap, a) -> int:
    try:
        if a.batch_size is not None:
            positive_int(a.batch_size, "batch_size")
        if a.threshold is not None:
            probability_threshold(a.threshold)
        _device(a.device)
    except ValueError as exc:
        ap.error(str(exc))
    if not a.inputs and not (a.selftest or a.download):
        ap.error("give at least one video, folder or result file (or --selftest / --download)")
    formats = list(dict.fromkeys(a.format or ["json"]))

    paths = []
    for path in a.inputs:
        if "://" in path or os.path.exists(path):  # URLs are left to the decoder
            paths.append(path)
        elif any(c in path for c in "*?["):
            matches = sorted(glob.glob(path, recursive=True))
            if not matches:
                ap.error(f"no files match {path}")
            paths += matches
        else:
            ap.error(f"input not found: {path}")
    problems = []
    try:
        items = _inputs(paths, a.recursive, problems)
    except OSError as exc:
        ap.error(f"cannot read {exc.filename}: {exc.strerror or exc}")
    if paths and not items:
        ap.error("no videos or saved results found in the given folders" + ("" if a.recursive else
                                                                             " (use -r to include subfolders)"))
    if any(not _is_result(src) for src, _, _ in items) or a.selftest:
        module, package = DECODER_PACKAGES[a.decoder]
        try:
            __import__(module)
        except ImportError:
            ap.error(f"--decoder {a.decoder} needs the Python package {package}: pip install {package}")

    # Check outputs before loading the model.
    if os.path.exists(a.out) and not os.path.isdir(a.out):
        ap.error(f"-o {a.out} is a file; give a folder for the outputs")
    names = _unique_names(items, a.out)
    plan = []
    for (src, stem, _), name in zip(items, names):
        if name != stem:
            print(f"note: {src} -> {name}.persist.* (another input has the same name)", file=sys.stderr)
        paths_out = {fmt: os.path.join(a.out, name + ".persist." + fmt) for fmt in formats}
        if _is_result(src) and any(os.path.realpath(p) == os.path.realpath(src) for p in paths_out.values()) \
                and not a.overwrite:
            ap.error(f"the output for {src} is the input file itself; use -o another folder (or --overwrite)")
        present = {p for p in paths_out.values() if os.path.lexists(p)}
        if present and not (a.overwrite or a.skip_existing):
            ap.error(f"output already exists: {sorted(present)[0]}; use --overwrite to replace it "
                     "or --skip-existing to skip it")
        plan.append((src, paths_out, present))
    if plan:
        try:
            for _, paths_out, _ in plan:
                for p in paths_out.values():
                    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with tempfile.TemporaryFile(dir=a.out):
                pass
        except OSError as exc:
            ap.error(f"cannot write to the output folder {a.out}: {exc.strerror or exc}")

    detector = None
    needs_model = a.selftest or a.download or any(
        not _is_result(src) and not (a.skip_existing and len(present) == len(p))
        for src, p, present in plan)
    if needs_model:
        try:
            detector = _load(a)
        except Exception as exc:
            print(f"persist_sbd: error: cannot load the model: {_describe(exc)}", file=sys.stderr)
            return 2
    if a.download:
        src = detector.source
        print(f"model ready: {src.get('model')}" + (f" @ {src['revision']}" if src.get("revision") else "")
              + (f" (sha256 {src['sha256']})" if src.get("sha256") else ""), flush=True)
    if a.selftest:
        from . import selftest
        if not selftest.run(detector, decoder=a.decoder, batch_size=a.batch_size):
            return 1
    if not plan:
        return 0

    bars = not a.quiet and sys.stderr.isatty()
    done = skipped = 0
    failed = []
    try:
        for k, (src, paths_out, present) in enumerate(plan, 1):
            label = f"[{k}/{len(plan)}] {os.path.basename(src)}" if len(plan) > 1 else os.path.basename(src)
            now = {p for p in paths_out.values() if os.path.lexists(p)}
            if a.skip_existing and len(now) == len(paths_out):
                skipped += 1
                print(f"{src}: skipped, outputs exist", flush=True)
                continue
            progress = None
            try:
                appeared = now - present
                if appeared and not a.overwrite:
                    raise FileExistsError(2, "File exists", sorted(appeared)[0])
                if _is_result(src):
                    r = Result.load_json(src)
                    if a.threshold is not None:
                        r = r.with_threshold(a.threshold)
                else:
                    progress = _progress_bar(bars, label)
                    r = detector.detect(src, threshold=a.threshold, edges=a.edges, decoder=a.decoder,
                                        batch_size=a.batch_size, progress=progress)
                for fmt, path in paths_out.items():
                    replace = a.overwrite or path in present  # partial results found when planning
                    if fmt == "json":
                        r.save_json(path, a.per_frame or _is_result(src), overwrite=replace)
                    else:
                        r.save_csv(path, overwrite=replace)
                cuts = sum(t["kind"] == "cut" for t in r.transitions)
                took = f" at threshold {r.threshold}" if _is_result(src) else f" in {r.seconds:.1f} s on {detector.device.type}"
                print(f"{src}: {r.n_frames} frames, {len(r.shots)} shots ({cuts} cuts, "
                      f"{len(r.transitions) - cuts} gradual){took}"
                      + (f"; {r.scored_frames} frames scored" if r.edges == "paper" else ""), flush=True)
                if not r.scored_frames:
                    print("No frames received a prediction: the video is shorter than a complete window. "
                          "Use --edges full to enable predictions for short videos.", file=sys.stderr)
            except FileExistsError as exc:
                reason = f"output already exists: {exc.filename}; use --overwrite to replace it"
                failed.append((src, reason))
                print(f"error: {src}: {reason}", file=sys.stderr)
                continue
            except Exception as exc:
                failed.append((src, _describe(exc)))
                print(f"error: {src}: {_describe(exc)}", file=sys.stderr)
                continue
            finally:
                if progress is not None:
                    progress.close()
            done += 1
    except KeyboardInterrupt:
        left = len(plan) - done - skipped - len(failed)
        print(f"\ninterrupted: {done} done, {skipped} skipped, {len(failed)} failed, {left} not finished; "
              "run the same command with --skip-existing to continue", file=sys.stderr)
        return 130
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    if len(plan) > 1 or failed:
        print(f"{done} done, {skipped} skipped, {len(failed)} failed", file=sys.stderr)
        if len(failed) > 1:
            for src, reason in failed:
                print(f"  failed: {src}: {reason}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
