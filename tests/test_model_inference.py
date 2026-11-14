"""Tests for inference outputs and training gradients.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import unittest

import torch

from persist_sbd.nets.sbd_ours_head import SBDOursHead
from persist_sbd.nets.sbd_slowfast_siren_trcs_head import SlowFastSBDSirenTRCSHead


class InferenceHeadTests(unittest.TestCase):
    def make_head(self, kind, **overrides):
        options = dict(in_channels_slow=4, in_channels_fast=4, dropout_ratio=0.0,
                       siren_hidden_dim=8, siren_num_hidden_layers=1, siren_cond_dim=4,
                       siren_phi_dim=2, siren_start_epoch=0, dyn_start_epoch=0,
                       dyn_use_second_derivative=False, compute_dyn_in_eval=False,
                       enable_trcs=True, trcs_offsets=(1, 2), cond_t_smooth_kernel=3,
                       boundary_suppress_kernel=3)
        options.update(overrides)
        torch.manual_seed(7)
        head = kind(**options)
        inputs = (torch.randn(1, 4, 4, 2, 2), torch.randn(1, 4, 16, 2, 2))
        return head, inputs

    def test_eval_has_no_siren_graph_and_keeps_frame_signals(self):
        for kind in (SBDOursHead, SlowFastSBDSirenTRCSHead):
            for context in (torch.no_grad, torch.inference_mode):
                with self.subTest(head=kind.__name__, context=context.__name__):
                    head, inputs = self.make_head(kind)
                    head.eval()
                    with context():
                        out = head(inputs)
                    self.assertFalse(out["siren_phi"].requires_grad)
                    self.assertNotIn("dyn_energy_logits", out)
                    self.assertNotIn("trcs_gate_mean", out)
                    self.assertNotIn("trcs_evraw_mean", out)
                    for name in ("probs", "trcs_change", "trcs_innov", "trcs_return", "trcs_gate"):
                        self.assertEqual(tuple(out[name].shape), (1, 16))
                        self.assertTrue(torch.isfinite(out[name]).all())

    def test_debug_statistics_are_opt_in_without_parameter_changes(self):
        for kind in (SBDOursHead, SlowFastSBDSirenTRCSHead):
            with self.subTest(head=kind.__name__):
                head, inputs = self.make_head(kind)
                head.eval()
                before_keys = set(head.state_dict())
                with torch.no_grad():
                    ordinary = head(inputs)
                    head.inference_debug_stats = True
                    debug = head(inputs)
                self.assertEqual(before_keys, set(head.state_dict()))
                self.assertIn("trcs_gate_mean", debug)
                self.assertIn("trcs_evraw_mean", debug)
                for name, value in ordinary.items():
                    torch.testing.assert_close(value, debug[name], rtol=0, atol=0)

    def test_training_retains_gradients_dynamics_and_statistics(self):
        for kind in (SBDOursHead, SlowFastSBDSirenTRCSHead):
            with self.subTest(head=kind.__name__):
                head, inputs = self.make_head(kind, dyn_use_second_derivative=True)
                head.train()
                out = head(inputs)
                self.assertTrue(out["siren_phi"].requires_grad)
                self.assertIn("dyn_energy_logits", out)
                self.assertIn("trcs_gate_mean", out)
                self.assertIn("trcs_evraw_mean", out)
                (out["probs"].mean() + out["siren_phi"].square().mean()
                 + out["dyn_energy_logits"].mean() * 0.001).backward()
                grads = [p.grad for p in head.siren.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_explicit_first_derivative_eval_still_builds_required_graph(self):
        for kind in (SBDOursHead, SlowFastSBDSirenTRCSHead):
            with self.subTest(head=kind.__name__):
                head, inputs = self.make_head(kind, compute_dyn_in_eval=True)
                head.eval()
                with torch.no_grad():
                    out = head(inputs)
                self.assertIn("dyn_energy_logits", out)
                self.assertTrue(torch.isfinite(out["dyn_energy_logits"]).all())


if __name__ == "__main__":
    unittest.main()
