"""End-to-end detection tests with synthetic videos.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import contextlib
import inspect
import io
import os
import tempfile
import unittest

import numpy as np

import persist_sbd
from persist_sbd import api, cli

import _videos

MODEL = os.environ.get("PERSIST_SBD_TEST_MODEL")
DEFAULT_EDGES = inspect.signature(api.Detector.detect).parameters["edges"].default


@unittest.skipUnless(MODEL, "set PERSIST_SBD_TEST_MODEL to a local model folder to run")
class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.det = persist_sbd.load(MODEL, device="cpu")
        cls.video = {k: _videos.write(os.path.join(cls.tmp.name, f"{k}.mp4"), k) for k in ("cut", "flash", "dissolve")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_hard_cut_is_one_cut_at_the_right_frame(self):
        r = self.det.detect(self.video["cut"])
        self.assertEqual(r.n_frames, _videos.N)
        self.assertEqual(len(r.transitions), 1, r.transitions)
        t = r.transitions[0]
        self.assertEqual(t["kind"], "cut")
        self.assertIn(t["boundary"], (62, 63, 64))  # the cut is between frames 63 and 64
        self.assertEqual(r.shots, [(0, t["boundary"]), (t["boundary"] + 1, _videos.N - 1)])
        if DEFAULT_EDGES != "paper":
            self.assertEqual(r.scored_frames, _videos.N)

    def test_two_frame_flash_scores_well_below_a_cut(self):
        cut = self.det.detect(self.video["cut"]).scores.max()
        flash = self.det.detect(self.video["flash"])
        self.assertLess(flash.scores.max(), cut - 0.2)
        self.assertEqual(self.det.detect(self.video["flash"], threshold=0.68).transitions, [])

    def test_no_short_shots_at_the_end_of_a_video(self):
        # Texture b raises end-window scores to about 0.55.
        r = self.det.detect(self.video["dissolve"])
        self.assertTrue(all(t["boundary"] < _videos.N - 24 for t in r.transitions), r.transitions)
        self.assertGreaterEqual(r.shots[-1][1] - r.shots[-1][0] + 1, 6)
        self.assertEqual(len(r.shots), len(r.transitions) + 1)

    def test_result_records_its_model_and_uses_the_video_timestamps(self):
        import hashlib

        r = self.det.detect(self.video["cut"])
        with open(os.path.join(MODEL, "model.safetensors"), "rb") as fh:
            self.assertEqual(r.info["model"]["sha256"], hashlib.sha256(fh.read()).hexdigest())
        self.assertEqual(r.info["time_source"], "timestamps")
        cut = r.to_dict()["transitions"][0]
        self.assertAlmostEqual(cut["time"], (cut["boundary"] + 1) / 25, places=6)

    def test_own_frames_give_the_same_scores_as_the_video(self):
        import av

        with av.open(self.video["cut"]) as container:
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        np.testing.assert_array_equal(self.det.detect_frames(frames, fps=25).scores,
                                      self.det.detect(self.video["cut"]).scores)

    def test_selftest_passes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["--selftest", "--model", MODEL, "--device", "cpu", "-q"]), 0, out.getvalue())
        self.assertIn("selftest passed", out.getvalue())


if __name__ == "__main__":
    unittest.main()
