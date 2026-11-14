"""Regression tests for inference and public outputs.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from persist_sbd import api, cli, infer


class ConstantModel(torch.nn.Module):
    def forward(self, clips):
        values = torch.full((len(clips), clips.shape[2]), 0.25, device=clips.device)
        return {key: values for key in ["probs", *infer.GATES.values()]}


class FrameValueModel(torch.nn.Module):
    """Return each frame's mean intensity at its window position."""

    def forward(self, clips):
        values = clips.float().mean(dim=(1, 3, 4)) / 255.0
        return {key: values for key in ["probs", *infer.GATES.values()]}


def detector():
    return api.Detector(ConstantModel(), {
        "threshold": 0.65,
        "preprocess": dict(clip_len=64, clip_stride=16, center_keep_len=16,
                           resize_short_side=128, center_crop=112),
    }, torch.device("cpu"))


def result(fps=25.0, n=128):
    covered = np.zeros(n, dtype=bool)
    covered[24:104] = True
    return api.Result("example.mp4", fps, n, 0.5, "paper", np.zeros(n, np.float32),
                      {name: np.zeros(n, np.float32) for name in infer.GATES}, covered)


class InferenceTests(unittest.TestCase):
    def test_invalid_batch_does_not_consume_input(self):
        def frames():
            self.fail("invalid batch must fail before decoding")
            yield
        for value in (-1, 0, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                infer.run(ConstantModel(), frames(), torch.device("cpu"), batch_size=value)

    def test_paper_and_pad_coverage(self):
        for n, interval in ((63, (0, 0)), (64, (24, 40)), (128, (24, 104)), (130, (24, 104))):
            for edges in ("paper", "pad"):
                with self.subTest(n=n, edges=edges):
                    frames = (np.zeros((2, 2, 3), np.uint8) for _ in range(n))
                    scores, gates, covered = infer.run(ConstantModel(), frames, torch.device("cpu"), edges=edges, batch_size=2)
                    expected = np.zeros(n, bool)
                    expected[slice(*interval) if edges == "paper" else slice(None)] = True
                    np.testing.assert_array_equal(covered, expected)
                    np.testing.assert_array_equal(scores, expected.astype(np.float32) * 0.25)
                    self.assertEqual(set(gates), set(infer.GATES))

    def test_full_coverage_uses_real_frames_and_keeps_paper_scores(self):
        for n, edges in [(n, "full") for n in (1, 30, 63, 64, 65, 79, 80, 128, 130, 143, 1000)]:
            with self.subTest(n=n, edges=edges):
                frames = [np.full((2, 2, 3), (7 * i) % 251, np.uint8) for i in range(n)]
                truth = np.array([((7 * i) % 251) / 255.0 for i in range(n)], np.float32)
                full, _, covered = infer.run(FrameValueModel(), iter(frames), torch.device("cpu"), edges=edges,
                                             batch_size=3)
                self.assertTrue(covered.all())
                np.testing.assert_allclose(full, truth, rtol=0, atol=1e-6)
                paper, _, paper_covered = infer.run(FrameValueModel(), iter(frames), torch.device("cpu"),
                                                    edges="paper", batch_size=3)
                np.testing.assert_array_equal(full[paper_covered], paper[paper_covered])

    def test_sparse_windows_only_allowed_in_paper_mode(self):
        def unused_frames():
            self.fail("invalid pad coverage must fail before decoding")
            yield

        for edges in ("pad", "full"):
            with self.assertRaisesRegex(ValueError, f"stride.*center_keep_len.*{edges}"):
                infer.run(ConstantModel(), unused_frames(), torch.device("cpu"), stride=32, edges=edges)
        frames = (np.zeros((2, 2, 3), np.uint8) for _ in range(128))
        _, _, covered = infer.run(ConstantModel(), frames, torch.device("cpu"), stride=32, edges="paper")
        expected = np.zeros(128, bool)
        for start in (24, 56, 88):
            expected[start:start + 16] = True
        np.testing.assert_array_equal(covered, expected)

    def test_api_validation_precedes_video_open(self):
        d = detector()
        invalid = [dict(batch_size=v) for v in (-1, 0, True, 1.2)]
        invalid += [dict(threshold=v) for v in (0, -0.1, 1.01, float("nan"), float("inf"), True)]
        invalid += [dict(edges="unknown")]
        with patch.object(api, "open_video") as opened:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    d.detect("unused.mp4", **kwargs)
            opened.assert_not_called()

    def test_api_closes_reader_on_inference_error(self):
        closed = []
        @contextlib.contextmanager
        def opened(*args):
            try:
                yield object(), iter(())
            finally:
                closed.append(True)
        with patch.object(api, "open_video", opened), patch.object(infer, "run", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                detector().detect("unused.mp4")
        self.assertEqual(closed, [True])


class TransitionTests(unittest.TestCase):
    @staticmethod
    def probs(n, *runs):
        p = np.zeros(n, np.float32)
        for first, last in runs:
            p[first:last + 1] = 0.9
        return p

    def test_six_frame_run_is_a_cut_at_its_middle(self):
        # A hard cut between frames 63 and 64 is predicted on frames 61..66.
        t = infer.transitions(self.probs(128, (61, 66)), 0.5)
        self.assertEqual([(x["start"], x["end"], x["boundary"], x["kind"], x["run_start"], x["run_end"]) for x in t],
                         [(63, 64, 63, "cut", 61, 66)])
        self.assertEqual(infer.shots(self.probs(128, (61, 66)), 0.5), [(0, 63), (64, 127)])

    def test_gradual_transition_spans_its_run_and_a_boundary_on_the_last_frame_is_dropped(self):
        t = infer.transitions(self.probs(200, (100, 129)), 0.5)
        self.assertEqual([(x["start"], x["end"], x["boundary"], x["kind"]) for x in t], [(100, 129, 114, "gradual")])
        self.assertEqual(infer.transitions(self.probs(10, (9, 9)), 0.5), [])
        self.assertEqual(infer.shots(self.probs(10, (9, 9)), 0.5), [(0, 9)])

    def test_first_and_last_shots_are_at_least_six_frames(self):
        n = infer.MIN_EDGE_SHOT_FRAMES
        for runs, kept in (([(2, 4)], []), ([(7, 9)], [8]), ([(95, 97)], []), ([(90, 92)], [91]),
                           ([(0, 0), (3, 3), (60, 62)], [61]), ([(1, 2), (3 + n, 4 + n), (60, 62)], [3 + n, 61])):
            with self.subTest(runs=runs):
                p = self.probs(100, *runs)
                t = infer.transitions(p, 0.5)
                s = infer.shots(p, 0.5)
                self.assertEqual([x["boundary"] for x in t], kept)
                self.assertGreaterEqual(s[0][1] - s[0][0] + 1, n)
                self.assertGreaterEqual(s[-1][1] - s[-1][0] + 1, n)

    def test_a_one_frame_dip_does_not_split_a_transition(self):
        t = infer.transitions(self.probs(100, (40, 42), (44, 45)), 0.5)  # one frame (43) below
        self.assertEqual([(x["run_start"], x["run_end"], x["kind"]) for x in t], [(40, 45, "cut")])
        t = infer.transitions(self.probs(100, (40, 42), (45, 46)), 0.5)  # two frames below: two runs
        self.assertEqual([(x["run_start"], x["run_end"]) for x in t], [(40, 42), (45, 46)])
        t = infer.transitions(self.probs(100, (10, 12), (14, 16), (18, 30)), 0.5)
        self.assertEqual([(x["run_start"], x["run_end"], x["kind"]) for x in t], [(10, 30, "gradual")])

    def test_kind_follows_run_length(self):
        for length, kind in ((1, "cut"), (2, "cut"), (6, "cut"), (infer.CUT_MAX_FRAMES + 1, "gradual"), (30, "gradual")):
            with self.subTest(length=length):
                t = infer.transitions(self.probs(200, (100, 100 + length - 1)), 0.5)
                self.assertEqual([x["kind"] for x in t], [kind])

    def test_kind_uses_the_response_span_including_a_merged_gap(self):
        probs = self.probs(100, (40, 42), (44, 46))
        self.assertEqual(np.count_nonzero(probs >= 0.5), 6)
        transitions = infer.transitions(probs, 0.5)
        self.assertEqual([(x["run_start"], x["run_end"], x["kind"]) for x in transitions],
                         [(40, 46, "gradual")])

    def test_shots_are_consecutive_and_cover_every_frame(self):
        p = self.probs(300, (0, 2), (40, 45), (100, 129), (299, 299))
        s = infer.shots(p, 0.5)
        t = infer.transitions(p, 0.5)
        self.assertEqual(s[0][0], 0)
        self.assertEqual(s[-1][1], 299)
        for (a, b), (c, d) in zip(s, s[1:]):
            self.assertEqual(c, b + 1)
            self.assertLessEqual(a, b)
        # End detections would leave an empty or sub-six-frame edge shot.
        self.assertEqual(s, [(0, 42), (43, 114), (115, 299)])
        self.assertEqual(len(s), len(t) + 1)
        self.assertEqual([x["boundary"] for x in t], [a[1] for a in s[:-1]])
        self.assertEqual(infer.shots(np.zeros(0, np.float32), 0.5), [])
        self.assertEqual(infer.shots(np.zeros(5, np.float32), 0.5), [(0, 4)])


class OutputTests(unittest.TestCase):
    def test_csv_reports_scored_frames_per_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.csv"
            result().save_csv(target)
            with target.open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["shot", "first_frame", "last_frame", "start_time", "end_time", "frames",
                                       "duration", "scored_frames"])
            self.assertEqual(rows[1][-1], "80")
            r = result()
            r.edges = "full"
            r.save_csv(target)
            with target.open() as handle:
                self.assertEqual(next(csv.reader(handle))[-1], "duration")  # the coverage column is for paper mode

    def test_coverage_export_is_explicit(self):
        r = result()
        data = r.to_dict(per_frame=True)
        self.assertEqual(data["scored_frames"], 80)
        self.assertEqual(data["scored_ranges"], [dict(start=24, end=103)])
        self.assertEqual(sum(data["covered"]), 80)
        self.assertNotIn("covered", r.to_dict())
        self.assertEqual(result(n=12).to_dict()["scored_frames"], 0)

    def test_unknown_fps_writes_valid_json_and_blank_csv_times(self):
        for fps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as directory:
                r = result(fps)
                target = Path(directory) / "result.json"
                r.save_json(target)
                data = json.loads(target.read_text(), parse_constant=lambda value: self.fail(value))
                self.assertIsNone(data["fps"])
                self.assertIsNone(data["shots"][0]["start_time"])
                csv_path = Path(directory) / "result.csv"
                r.save_csv(csv_path)
                with csv_path.open() as handle:
                    row = list(csv.reader(handle))[1]
                self.assertEqual([row[3], row[4], row[6]], ["", "", ""])

    def test_bad_json_does_not_destroy_existing_file(self):
        r = result()
        r.scores[32] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            target.write_text("keep")
            with self.assertRaises(ValueError):
                r.save_json(target, per_frame=True)
            self.assertEqual(target.read_text(), "keep")

    def test_optional_output_protection_keeps_default_overwrite_behavior(self):
        for fmt in ("json", "csv"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / f"result.{fmt}"
                target.write_text("keep")
                save = getattr(result(), f"save_{fmt}")
                with self.assertRaises(FileExistsError):
                    save(target, overwrite=False)
                self.assertEqual(target.read_text(), "keep")
                save(target)
                self.assertNotEqual(target.read_text(), "keep")


class CLITests(unittest.TestCase):
    def rejected_before_model_load(self, argv, message=None):
        err = io.StringIO()
        with patch.object(cli, "load") as load, contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as exc:
                cli.main(argv)
            self.assertEqual(exc.exception.code, 2)
            load.assert_not_called()
        if message:
            self.assertIn(message, err.getvalue())

    def test_inputs_with_the_same_name_get_distinct_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / "a" / "test.mp4", root / "b" / "test.mkv", root / "a" / "x.mp4", root / "b" / "x.mp4"]
            for path in inputs:
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"")
            out = root / "out"
            d = detector()
            with patch.object(cli, "load", return_value=d), patch.object(d, "detect", return_value=result()), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main([*map(str, inputs), "-o", str(out)]), 0)
            self.assertEqual(sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json")),
                             ["a/x.mp4.persist.json", "b/x.mp4.persist.json", "test.mkv.persist.json",
                              "test.mp4.persist.json"])

    def test_missing_input_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            self.rejected_before_model_load([str(Path(directory) / "missing.mp4"), "-o", directory],
                                            "input not found")

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test.persist.json"
            target.write_text("existing")
            video = Path(directory) / "test.mp4"
            video.write_bytes(b"")
            self.rejected_before_model_load([str(video), "-o", directory])
            self.assertEqual(target.read_text(), "existing")

    def test_bad_cli_parameters_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            video.write_bytes(b"")
            for args, message in ((["--batch-size", "-1"], "batch_size"), (["--batch-size", "0"], "batch_size"),
                                  (["--threshold", "nan"], "threshold"), (["--threshold", "0"], "threshold"),
                                  (["--device", "gpu"], "unknown device"), (["-f", "json,xml"], "formats")):
                with self.subTest(args=args):
                    self.rejected_before_model_load([str(video), "-o", directory, *args], message)

    def test_explicit_overwrite_and_uncovered_video_message(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test.persist.json"
            target.write_text("existing")
            video = Path(directory) / "test.mp4"
            video.write_bytes(b"")
            d = detector()
            output = io.StringIO()
            with patch.object(cli, "load", return_value=d), patch.object(d, "detect", return_value=result(n=12)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(output):
                self.assertEqual(cli.main([str(video), "-o", directory, "--overwrite", "--edges", "paper"]), 0)
            self.assertEqual(json.loads(target.read_text())["scored_frames"], 0)
            self.assertIn("No frames received a prediction", output.getvalue())

    def test_output_created_during_inference_is_not_overwritten(self):
        for fmt in ("json", "csv"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / f"test.persist.{fmt}"
                video = Path(directory) / "test.mp4"
                video.write_bytes(b"")
                d = detector()

                def concurrent_output(*args, **kwargs):
                    target.write_text("another process finished first")
                    return result()

                error = io.StringIO()
                with patch.object(cli, "load", return_value=d), \
                        patch.object(d, "detect", side_effect=concurrent_output), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                    self.assertEqual(cli.main([str(video), "-o", directory, "-f", fmt]), 1)
                self.assertEqual(target.read_text(), "another process finished first")
                self.assertIn(str(target), error.getvalue())
                self.assertIn("--overwrite", error.getvalue())


class HubTests(unittest.TestCase):
    def test_hub_fetches_only_selected_checkpoint_and_forwards_options(self):
        for variant in (None, "seed11"):
            prefix = f"{variant}/" if variant else ""
            with self.subTest(variant=variant), \
                    patch("huggingface_hub.constants.HF_HUB_OFFLINE", False), \
                    patch("huggingface_hub.hf_hub_download", side_effect=lambda repo, filename, **kw: filename) as download:
                cfg, weights = api._files("example/persist", variant, "v1", "cache", False)
            expected = [f"{prefix}config.json", f"{prefix}model.safetensors"]
            self.assertEqual([call.args[1] for call in download.call_args_list], expected)
            for call in download.call_args_list:
                self.assertEqual(call.args[0], "example/persist")
                self.assertEqual(call.kwargs, dict(revision="v1", cache_dir="cache", local_files_only=False))
            self.assertEqual((cfg, weights), tuple(expected))

    def test_default_load_uses_one_root_checkpoint(self):
        config = {"model": {"example": "architecture"}, "preprocess": {}, "threshold": 0.68}
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "config.json"
            cfg.write_text(json.dumps(config))
            with patch.object(api, "_files", return_value=(str(cfg), "model.safetensors")) as files, \
                    patch("safetensors.torch.load_file", return_value={}) as weights, \
                    patch.object(api, "PERSIST") as network:
                loaded = api.load(device="cpu")
        files.assert_called_once_with("linty5/PERSIST", None, api.DEFAULT_REVISION, None, False)
        weights.assert_called_once_with("model.safetensors")
        network.assert_called_once_with(config["model"])
        network.return_value.load_state_dict.assert_called_once_with({}, strict=True)
        self.assertEqual(loaded.threshold, 0.68)

    def test_only_official_hub_defaults_to_the_release_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "config.json"
            cfg.write_text(json.dumps({"model": {}, "preprocess": {}, "threshold": 0.5}))
            cases = ((api.DEFAULT_REPO, None, api.DEFAULT_REVISION),
                     (api.DEFAULT_REPO, "main", "main"),
                     (api.DEFAULT_REPO, "my-tag", "my-tag"),
                     ("example/persist", None, None),
                     ("example/persist", "my-tag", "my-tag"),
                     (directory, None, None))
            for model, requested, expected in cases:
                with self.subTest(model=model, revision=requested), \
                        patch.object(api, "_files", return_value=(str(cfg), "model.safetensors")) as files, \
                        patch("safetensors.torch.load_file", return_value={}), patch.object(api, "PERSIST"):
                    loaded = api.load(model, device="cpu", revision=requested)
                files.assert_called_once_with(model, None, expected, None, False)
                self.assertEqual(loaded.source["revision"], expected)

    def test_results_record_the_resolved_hub_commit_instead_of_the_tag(self):
        commit = "c" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshots" / commit
            snapshot.mkdir(parents=True)
            cfg, weights = snapshot / "config.json", snapshot / "model.safetensors"
            cfg.write_text(json.dumps({"model": {}, "preprocess": {}, "threshold": 0.5}))
            weights.write_bytes(b"cached weights placeholder")
            with patch.object(api, "_files", return_value=(str(cfg), str(weights))), \
                    patch("safetensors.torch.load_file", return_value={}), patch.object(api, "PERSIST"):
                loaded = api.load(device="cpu")
            scores, covered = np.zeros(1, np.float32), np.ones(1, bool)
            run = scores, {name: scores.copy() for name in infer.GATES}, covered
            detected = loaded._result("clip.mp4", 25.0, run, 0.5, "full", "pyav", [], 0)
        self.assertEqual(detected.to_dict()["model"]["revision"], commit)

    def test_root_checkpoint_works_in_real_offline_cache(self):
        from huggingface_hub import hf_hub_download

        revision = "b" * 40
        for global_offline in (False, True):
            with self.subTest(global_offline=global_offline), tempfile.TemporaryDirectory() as directory:
                repository = Path(directory) / "models--linty5--PERSIST"
                snapshot = repository / "snapshots" / revision
                snapshot.mkdir(parents=True)
                (repository / "refs").mkdir()
                (repository / "refs" / api.DEFAULT_REVISION).write_text(revision)
                cfg, weights = snapshot / "config.json", snapshot / "model.safetensors"
                cfg.write_text('{"threshold": 0.68}')
                weights.write_bytes(b"cached weights placeholder")
                with patch("huggingface_hub.constants.HF_HUB_OFFLINE", global_offline), \
                        patch("huggingface_hub.hf_hub_download", wraps=hf_hub_download) as download, \
                        patch("huggingface_hub.file_download.get_hf_file_metadata",
                              side_effect=AssertionError("offline cache load must not request metadata")):
                    actual = api._files(api.DEFAULT_REPO, None, api.DEFAULT_REVISION, directory,
                                        local_files_only=not global_offline)
                self.assertEqual(actual, (str(cfg), str(weights)))
                self.assertEqual([call.args[1] for call in download.call_args_list],
                                 ["config.json", "model.safetensors"])
                self.assertTrue(all(call.kwargs["local_files_only"] for call in download.call_args_list))

    def test_offline_variant_loads_without_root_config_in_real_cache(self):
        from huggingface_hub import hf_hub_download

        revision = "a" * 40
        for global_offline in (False, True):
            with self.subTest(global_offline=global_offline), tempfile.TemporaryDirectory() as directory:
                snapshot = Path(directory) / "models--example--persist" / "snapshots" / revision
                variant = snapshot / "seed11"
                variant.mkdir(parents=True)
                cfg, weights = variant / "config.json", variant / "model.safetensors"
                cfg.write_text("{}")
                weights.write_bytes(b"cached weights placeholder")
                self.assertFalse((snapshot / "config.json").exists())
                with patch("huggingface_hub.constants.HF_HUB_OFFLINE", global_offline), \
                        patch("huggingface_hub.hf_hub_download", wraps=hf_hub_download) as download, \
                        patch("huggingface_hub.file_download.get_hf_file_metadata",
                              side_effect=AssertionError("offline cache load must not request metadata")):
                    actual = api._files("example/persist", "seed11", revision, directory,
                                        local_files_only=not global_offline)
                self.assertEqual(actual, (str(cfg), str(weights)))
                self.assertEqual([call.args[1] for call in download.call_args_list],
                                 ["seed11/config.json", "seed11/model.safetensors"])

    def test_local_model_never_contacts_hub(self):
        with tempfile.TemporaryDirectory() as directory, patch("huggingface_hub.hf_hub_download") as download:
            for name in ("config.json", "model.safetensors"):
                (Path(directory) / name).write_text("{}")
            api._files(directory, None, None)
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
