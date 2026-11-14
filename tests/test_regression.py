"""Numerical regression tests for the released model.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import hashlib
import os
import unittest

import numpy as np
import torch

import persist_sbd
from _synthetic import regression_clips

REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "regression_seed11.npz")
NAMES = ("probs", "trcs_change", "trcs_innov", "trcs_return", "trcs_gate", "start_probs", "end_probs")
MODEL = os.environ.get("PERSIST_SBD_TEST_MODEL")


@unittest.skipUnless(MODEL, "set PERSIST_SBD_TEST_MODEL to a local model folder to run")
class NetworkRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = np.load(REFERENCE)
        with open(os.path.join(MODEL, "model.safetensors"), "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        if sha != str(cls.ref["weights_sha256"]):
            raise unittest.SkipTest("PERSIST_SBD_TEST_MODEL holds other weights than the reference")
        cls.clips = torch.from_numpy(regression_clips()).permute(0, 4, 1, 2, 3).contiguous()

    def check(self, device, atol):
        det = persist_sbd.load(MODEL, device=device)
        with torch.no_grad():
            out = det.model(self.clips.to(device))
        for name in NAMES:
            with self.subTest(output=name, device=device):
                got = out[name].float().reshape(self.clips.shape[0], -1).cpu().numpy()
                np.testing.assert_allclose(got, self.ref[name], rtol=0, atol=atol)

    def test_cpu_matches_reference(self):
        self.check("cpu", 1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_cuda_matches_reference(self):
        # cuDNN/TF32 differences across GPUs can reach a few 1e-3.
        self.check("cuda", 1e-2)


if __name__ == "__main__":
    unittest.main()
