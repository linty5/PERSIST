"""Tests for batch processing, result files and timing.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from persist_sbd import api, cli, infer, video

import _videos


class FrameValueModel(torch.nn.Module):
    """Return each frame's mean intensity at its window position."""

    def forward(self, clips):
        values = clips.float().mean(dim=(1, 3, 4)) / 255.0
        return {key: values for key in ["probs", *infer.GATES.values()]}


CONFIG = {"threshold": 0.5, "preprocess": dict(clip_len=64, clip_stride=16, center_keep_len=16,
                                                resize_short_side=128, center_crop=112)}


def detector(model=None):
    return api.Detector(model or FrameValueModel(), dict(CONFIG), torch.device("cpu"),
                        dict(model="stand-in", sha256="0" * 64))


def result(n=200, runs=((50, 55, 0.9), (120, 140, 0.6)), times=None, edges="full"):
    scores = np.zeros(n, np.float32)
    for first, last, value in runs:
        scores[first:last + 1] = value
    return api.Result("clip.mp4", 25.0, n, 0.5, edges, scores, {k: scores.copy() for k in infer.GATES},
                      np.ones(n, bool), times=times,
                      info=dict(time_source="frame_rate" if times is None else "timestamps", decoder="pyav",
                                device="cpu", model=dict(model="stand-in"), software=dict(persist_sbd="test")))


class ResultFileTests(unittest.TestCase):
    def test_threshold_can_be_changed_from_a_saved_result(self):
        r = result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.persist.json"
            r.save_json(path, per_frame=True)
            loaded = api.Result.load_json(path)
        self.assertEqual(loaded.transitions, r.transitions)
        self.assertEqual(loaded.shots, r.shots)
        self.assertEqual(len(loaded.with_threshold(0.7).transitions), 1)
        self.assertEqual(r.threshold, 0.5)  # with_threshold returns a copy
        with self.assertRaises(ValueError):
            r.with_threshold(0)

    def test_saved_result_without_scores_explains_how_to_keep_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.persist.json"
            result().save_json(path)
            with self.assertRaisesRegex(ValueError, "--per-frame"):
                api.Result.load_json(path)

    def test_json_records_how_the_result_was_made(self):
        d = result().to_dict()
        self.assertEqual((d["format"], d["format_version"]), (api.RESULT_FORMAT, api.RESULT_FORMAT_VERSION))
        for key in ("model", "decoder", "device", "software", "time_source"):
            self.assertIn(key, d)
        cut = d["transitions"][0]
        self.assertEqual((cut["kind"], cut["start"], cut["end"], cut["boundary"]), ("cut", 52, 53, 52))
        self.assertEqual(cut["time"], round(53 / 25, 6))  # the first frame of the next shot
        self.assertEqual(d["shots"][0], dict(start=0, end=52, frames=53, start_time=0.0, end_time=round(53 / 25, 6)))

    def test_times_follow_the_video_timestamps(self):
        n = 100
        times = np.concatenate([np.arange(50) * 0.04, 2.0 + np.arange(50) * 0.02])  # 25 fps, then 50 fps
        r = result(n=n, runs=((60, 62, 0.9),), times=times)
        self.assertAlmostEqual(r.time(10), 0.4)
        self.assertAlmostEqual(r.time(70), 2.4)
        self.assertAlmostEqual(r.time(n), 2.0 + 50 * 0.02)  # end of the last frame
        self.assertAlmostEqual(r.to_dict()["transitions"][0]["time"], 2.0 + 12 * 0.02)
        self.assertAlmostEqual(result(n=n).time(70), 70 / 25)  # without timestamps: frame / fps

    def test_unusable_timestamps_fall_back_to_the_frame_rate(self):
        for times in ([0.0, None, 0.08], [0.0, 0.04, 0.04], [0.0, float("nan"), 0.08], [0.0, 0.04]):
            with self.subTest(times=times):
                self.assertEqual(api._relative_times(times, 3), (None, "frame_rate"))
        rel, source = api._relative_times([5.0, 5.04, 5.08], 3)
        self.assertEqual(source, "timestamps")
        np.testing.assert_allclose(rel, [0.0, 0.04, 0.08])


class DetectFramesTests(unittest.TestCase):
    def test_frames_are_preprocessed_like_a_video(self):
        frames = _videos.frames("cut")
        d = detector()
        r = d.detect_frames(iter(frames), fps=25)
        prepared = [video.preprocess(f) for f in frames]
        scores, _, _ = infer.run(d.model, iter(prepared), d.device, edges="full", batch_size=2)
        np.testing.assert_array_equal(r.scores, api._quantized(scores))
        self.assertEqual((r.n_frames, r.fps, r.info["time_source"]), (len(frames), 25.0, "frame_rate"))

    def test_bad_frames_are_rejected(self):
        d = detector()
        for frames in ([np.zeros((64, 64), np.uint8)], [np.zeros((64, 64, 3), np.float32)]):
            with self.subTest(shape=frames[0].shape, dtype=frames[0].dtype), self.assertRaises(ValueError):
                d.detect_frames(frames)
        with self.assertRaisesRegex(ValueError, "no frames"):
            d.detect_frames([])

    def test_progress_reports_every_frame_read(self):
        calls = []
        detector().detect_frames(iter(_videos.frames("cut")), progress=lambda done, total: calls.append(done))
        self.assertEqual(calls[-1], _videos.N)
        self.assertEqual(calls, sorted(calls))


class ModelLoadingTests(unittest.TestCase):
    def test_cli_and_api_use_the_same_hub_revision_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "config.json"
            cfg.write_text(json.dumps({"model": {}, "preprocess": {}, "threshold": 0.5}))
            cases = (([], api.DEFAULT_REPO, api.DEFAULT_REVISION),
                     (["--revision", "main"], api.DEFAULT_REPO, "main"),
                     (["--model", "example/persist"], "example/persist", None),
                     (["--model", "example/persist", "--revision", "custom-tag"], "example/persist", "custom-tag"),
                     (["--model", directory], directory, None))
            for options, model, expected in cases:
                with self.subTest(options=options), \
                        patch.object(api, "_files", return_value=(str(cfg), "model.safetensors")) as files, \
                        patch("safetensors.torch.load_file", return_value={}), patch.object(api, "PERSIST"):
                    cli._load(cli._parser().parse_args(["--device", "cpu", *options]))
                files.assert_called_once_with(model, None, expected, None, False)

    def test_missing_local_folder_is_named(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "model folder not found"):
                api._files(os.path.join(directory, "nope", "model"), None, None)

    def test_hub_errors_become_one_readable_line(self):
        class RepositoryNotFoundError(Exception):
            pass

        with patch("huggingface_hub.hf_hub_download", side_effect=RepositoryNotFoundError("404")):
            with self.assertRaisesRegex(RuntimeError, "neither a local model folder nor a Hugging Face repository"):
                api._files("someone/missing", None, None)

    def test_apple_gpu_is_not_chosen_automatically(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.backends.mps, "is_available", return_value=True):
            self.assertEqual(api._device(None).type, "cpu")
        self.assertEqual(api._device("cpu").type, "cpu")


class BatchCLITests(unittest.TestCase):
    def run_cli(self, argv, d=None, **detect):
        d = d or detector()
        out, err = io.StringIO(), io.StringIO()
        with patch.object(cli, "load", return_value=d) as load, patch.object(d, "detect", **detect) as det, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue(), load, det

    @staticmethod
    def tree(root):
        for name in ("a.mp4", "B.MOV", "notes.txt", "sub/c.mkv", "sub/deeper/d.mp4"):
            path = Path(root) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    def test_folders_are_processed_and_mirrored(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            self.tree(src)
            code, *_ = self.run_cli([src, "-o", out, "-q"], return_value=result())
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in Path(out).rglob("*.json")), ["B.persist.json", "a.persist.json"])
            code, *_ = self.run_cli([src, "-r", "-o", out, "-q", "--skip-existing"], return_value=result())
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.relative_to(out).as_posix() for p in Path(out).rglob("*.json")),
                             ["B.persist.json", "a.persist.json", "sub/c.persist.json", "sub/deeper/d.persist.json"])

    def test_existing_results_are_skipped_on_request(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            self.tree(src)
            self.run_cli([src, "-o", out, "-q"], return_value=result())
            code, stdout, _, _, det = self.run_cli([src, "-o", out, "-q", "--skip-existing"], return_value=result())
            self.assertEqual(code, 0)
            det.assert_not_called()
            self.assertEqual(stdout.count("skipped"), 2)

    def test_one_failure_does_not_stop_the_batch(self):
        def detect(path, **kwargs):
            if path.endswith("a.mp4"):
                raise RuntimeError("corrupt file")
            return result()

        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            self.tree(src)
            code, _, stderr, _, _ = self.run_cli([src, "-o", out, "-q"], side_effect=detect)
            self.assertEqual(code, 1)
            self.assertIn("corrupt file", stderr)
            self.assertIn("1 done, 0 skipped, 1 failed", stderr)
            self.assertTrue((Path(out) / "B.persist.json").exists())
            self.assertNotIn("Traceback", stderr)

    def test_saved_results_are_re_thresholded_without_the_model(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "clip.persist.json"
            result().save_json(saved, per_frame=True)
            out = Path(directory) / "t07"
            with patch.object(cli, "load", side_effect=AssertionError("the model must not be loaded")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([str(saved), "--threshold", "0.7", "-o", str(out), "-f", "json,csv"]), 0)
            data = json.loads((out / "clip.persist.json").read_text())
            self.assertEqual(data["threshold"], 0.7)
            self.assertEqual(len(data["transitions"]), 1)
            self.assertIn("scores", data)  # still re-thresholdable
            self.assertTrue((out / "clip.persist.csv").exists())

    def test_model_errors_are_one_line(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "clip.mp4"
            clip.write_bytes(b"")
            err = io.StringIO()
            with patch.object(cli, "load", side_effect=RuntimeError("not in the local cache")), \
                    contextlib.redirect_stderr(err):
                self.assertEqual(cli.main([str(clip), "-o", directory]), 2)
            self.assertIn("cannot load the model: not in the local cache", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())


class OutputAndBatchTests(unittest.TestCase):

    def run_cli(self, argv, **detect):
        d = detector()
        out, err = io.StringIO(), io.StringIO()
        with patch.object(cli, "load", return_value=d) as load, patch.object(d, "detect", **detect) as det, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main(argv)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue(), load, det

    def test_failed_write_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "clip.persist.json"
            target.write_text("previous complete result")
            with patch("os.replace", side_effect=OSError(28, "No space left on device")):
                with self.assertRaises(OSError):
                    result().save_json(target, per_frame=True)
            self.assertEqual(target.read_text(), "previous complete result")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["clip.persist.json"])

    @contextlib.contextmanager
    def failing_fallback(self, target, replace=None):
        actual_open = open

        @contextlib.contextmanager
        def opened(path, mode, **options):
            with actual_open(path, mode, **options) as handle:
                if os.fspath(path) != os.fspath(target):
                    yield handle
                    return

                class InterruptedFile:
                    def fileno(self):
                        return handle.fileno()

                    def write(self, text):
                        handle.write(text[:12])
                        handle.flush()
                        if replace is not None:
                            replace()
                        raise OSError(28, "No space left on device")

                yield InterruptedFile()

        with patch("os.link", side_effect=OSError("hard links unavailable")), \
                patch.object(api, "open", side_effect=opened, create=True):
            yield

    def test_hardlink_fallback_removes_its_partial_file_after_a_write_error(self):
        for fmt in ("json", "csv"):
            with self.subTest(format=fmt), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / ("clip.persist." + fmt)
                with self.failing_fallback(target), self.assertRaisesRegex(OSError, "No space"):
                    getattr(result(), "save_" + fmt)(target, overwrite=False)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_hardlink_fallback_does_not_remove_a_concurrent_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            target, other = Path(directory) / "clip.persist.json", Path(directory) / "replacement"
            other.write_text("another process finished first")
            with self.failing_fallback(target, replace=lambda: os.replace(other, target)), \
                    self.assertRaisesRegex(OSError, "No space"):
                result().save_json(target, overwrite=False)
            self.assertEqual(target.read_text(), "another process finished first")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_hardlink_fallback_refuses_a_file_created_before_exclusive_open(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "clip.persist.json"

            def no_link(*args):
                target.write_text("another process finished first")
                raise OSError("hard links unavailable")

            with patch("os.link", side_effect=no_link), self.assertRaises(FileExistsError):
                result().save_json(target, overwrite=False)
            self.assertEqual(target.read_text(), "another process finished first")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_hardlink_fallback_writes_complete_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            original = result()
            json_path, csv_path = Path(directory) / "clip.persist.json", Path(directory) / "clip.persist.csv"
            with patch("os.link", side_effect=OSError("hard links unavailable")):
                original.save_json(json_path, per_frame=True, overwrite=False)
                original.save_csv(csv_path, overwrite=False)
            self.assertEqual(api.Result.load_json(json_path).transitions, original.transitions)
            self.assertTrue(csv_path.read_text().startswith("shot,first_frame,last_frame,"))
            self.assertEqual(sorted(Path(directory).iterdir()), sorted([json_path, csv_path]))

    def test_saved_per_frame_json_is_compact_and_reads_back_identically(self):
        r = result(n=3000, runs=((100, 105, 0.5000004), (2000, 2030, 0.71)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.persist.json"
            r.save_json(path, per_frame=True)
            text = path.read_text()
            loaded = api.Result.load_json(path)
        self.assertLess(text.count("\n"), 400)  # arrays on one line each
        self.assertEqual(loaded.transitions, r.transitions)

    def test_other_json_files_are_not_taken_for_results(self):
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory) / "metadata.json"
            other.write_text(json.dumps({"scores": [0.1, 0.9], "threshold": 0.5}))
            with self.assertRaisesRegex(ValueError, "not a persist_sbd result"):
                api.Result.load_json(other)
            newer = Path(directory) / "newer.persist.json"
            newer.write_text(json.dumps(dict(result().to_dict(per_frame=True), format_version=99)))
            with self.assertRaisesRegex(ValueError, "newer persist_sbd"):
                api.Result.load_json(newer)
            old = Path(directory) / "old.persist.json"  # optional format metadata omitted
            d = result().to_dict(per_frame=True)
            for key in ("format", "format_version", "time_source", "model", "decoder", "device", "software",
                        "postprocess", "times"):
                d.pop(key, None)
            old.write_text(json.dumps(d))
            self.assertEqual(len(api.Result.load_json(old).with_threshold(0.7).transitions), 1)

    def test_output_folder_problems_are_found_before_the_model_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "clip.mp4"
            clip.write_bytes(b"")
            not_a_folder = Path(directory) / "results.csv"
            not_a_folder.write_text("x")
            code, _, err, load, det = self.run_cli([str(clip), "-o", str(not_a_folder)], return_value=result())
            self.assertEqual(code, 2)
            self.assertIn("is a file", err)
            load.assert_not_called()

    def test_hidden_files_are_skipped_and_same_names_are_kept_apart(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            for name in ("reel1.mov", "reel1.mp4", "._reel1.mp4", ".hidden/x.mp4", "sub/.y.mp4"):
                path = Path(src) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
            code, _, err, _, det = self.run_cli([src, "-r", "-o", out, "-q"], return_value=result())
            self.assertEqual(code, 0, err)
            self.assertEqual(det.call_count, 2)
            self.assertEqual(sorted(p.name for p in Path(out).rglob("*.json")),
                             ["reel1.mov.persist.json", "reel1.mp4.persist.json"])

    def test_a_folder_of_saved_results_can_be_re_thresholded(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "saved"
            (saved / "sub").mkdir(parents=True)
            result().save_json(saved / "a.persist.json", per_frame=True)
            result().save_json(saved / "sub" / "b.persist.json", per_frame=True)
            out = Path(directory) / "t07"
            code, _, err, load, _ = self.run_cli([str(saved), "-r", "--threshold", "0.7", "-o", str(out)])
            self.assertEqual(code, 0, err)
            load.assert_not_called()
            self.assertEqual(sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json")),
                             ["a.persist.json", "sub/b.persist.json"])
            code, _, err, _, _ = self.run_cli([str(saved / "a.persist.json"), "-o", str(saved), "--skip-existing"])
            self.assertEqual(code, 2)
            self.assertIn("is the input file itself", err)

    def test_unexpanded_wildcards_are_expanded(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("a.mp4", "b.mp4"):
                (Path(directory) / name).write_bytes(b"")
            out = Path(directory) / "out"
            code, _, err, _, det = self.run_cli([os.path.join(directory, "*.mp4"), "-o", str(out), "-q"],
                                                return_value=result())
            self.assertEqual((code, det.call_count), (0, 2), err)

    def test_interrupting_a_batch_says_how_to_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("a.mp4", "b.mp4"):
                (Path(directory) / name).write_bytes(b"")
            code, _, err, _, _ = self.run_cli([directory, "-o", os.path.join(directory, "out"), "-q"],
                                              side_effect=[result(), KeyboardInterrupt()])
            self.assertEqual(code, 130)
            self.assertIn("1 done", err)
            self.assertIn("--skip-existing", err)
            self.assertNotIn("Traceback", err)

    def test_names_the_console_cannot_show_do_not_stop_a_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("caf\u00e9.mp4", "b.mp4"):
                (Path(directory) / name).write_bytes(b"")
            ascii_out = io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")
            d = detector()
            with patch.object(cli, "load", return_value=d), patch.object(d, "detect", return_value=result()), \
                    patch.object(cli.sys, "stdout", ascii_out), contextlib.redirect_stderr(io.StringIO()):
                code = cli.main([directory, "-o", os.path.join(directory, "out"), "-q"])
            self.assertEqual(code, 0)
            self.assertEqual(len(list(Path(directory, "out").glob("*.json"))), 2)

    def test_an_unavailable_gpu_is_reported_before_loading(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "--device cpu"):
                api._device("cuda")

    def test_broken_model_files_are_explained(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "config.json").write_text(json.dumps({"something": "else"}))
            (folder / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "not a persist_sbd model config"):
                api.load(str(folder), device="cpu")
            (folder / "config.json").write_text(json.dumps(dict(CONFIG, model={})))
            with self.assertRaisesRegex(ValueError, "Git LFS"):
                api.load(str(folder), device="cpu")
            with self.assertRaisesRegex(FileNotFoundError, "not a file"):
                api.load(str(folder / "model.safetensors"), device="cpu")

    def test_selftest_reports_failures_in_one_line(self):
        from persist_sbd import selftest

        def stub(transitions=None, error=None):
            d = detector()
            fake = result(n=128, runs=transitions or ())
            patcher = patch.object(d, "detect", side_effect=error) if error else \
                patch.object(d, "detect", return_value=fake)
            return d, patcher

        for runs, error, ok in ((((61, 66, 0.9),), None, True), ((), None, False),
                                (((61, 66, 0.9), (90, 95, 0.9)), None, False), (None, RuntimeError("no kernel"), False)):
            with self.subTest(runs=runs, error=error):
                d, patcher = stub(runs, error)
                lines = []
                with patcher:
                    self.assertEqual(selftest.run(d, out=lines.append), ok)
                self.assertTrue(lines[-1].startswith("selftest passed" if ok else "selftest FAILED"), lines)


class TimestampTests(unittest.TestCase):
    def test_constant_frame_rate_times_match_frame_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _videos.write(os.path.join(directory, "cut.mp4"), "cut")
            with video.open_video(path) as (info, frames):
                n = sum(1 for _ in frames)
                times = list(info.times)
        rel, source = api._relative_times(times, n)
        self.assertEqual(source, "timestamps")
        np.testing.assert_allclose(rel, np.arange(n) / 25.0, atol=1e-6)

    def test_variable_frame_rate_times_are_kept(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "vfr.mp4")
            try:
                with av.open(path, mode="w") as container:
                    stream = container.add_stream("mpeg4", rate=50)
                    stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
                    stream.codec_context.time_base = Fraction(1, 1000)
                    pts = 0
                    for i in range(60):
                        frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 4 * i, np.uint8), format="rgb24")
                        frame.pts, frame.time_base = pts, Fraction(1, 1000)
                        for packet in stream.encode(frame):
                            container.mux(packet)
                        pts += 40 if i < 30 else 20
                    for packet in stream.encode():
                        container.mux(packet)
            except Exception as exc:  # writing variable-rate files depends on the PyAV/FFmpeg build
                self.skipTest(f"cannot write a variable frame rate test file here: {exc}")
            with video.open_video(path) as (info, frames):
                n = sum(1 for _ in frames)
                times = list(info.times)
        rel, source = api._relative_times(times, n)
        steps = np.round(np.diff(rel), 3) if rel is not None else []
        if source != "timestamps" or not (set(steps) >= {0.04, 0.02}):
            self.skipTest(f"the written file did not keep variable timestamps (steps {sorted(set(steps))})")
        np.testing.assert_allclose(rel[:30], np.arange(30) * 0.04, atol=1e-3)
        np.testing.assert_allclose(rel[30:], 1.2 + np.arange(n - 30) * 0.02, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
