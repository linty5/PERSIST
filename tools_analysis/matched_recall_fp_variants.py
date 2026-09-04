"""Compare pseudo-event false positives at matched transition recall.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Support NumPy 2.x pickle module paths when loading with NumPy 1.x.
try:
    import numpy._core  # noqa: F401
except ModuleNotFoundError:
    import numpy.core as _npc
    sys.modules['numpy._core'] = _npc
    for _sub in ('multiarray', 'umath', 'numeric', '_multiarray_umath'):
        try:
            sys.modules['numpy._core.' + _sub] = getattr(_npc, _sub)
        except AttributeError:
            pass

from synth_per_type_f1 import SYNTH_GT
from matching_lib_and_tolerance_sweep import (probs_to_transitions,
                                           transitions_to_scenes,
                                           evaluate_scenes_and_diffs)

WORKSPACE_ROOT = Path(
    os.environ.get("PERSIST_ANALYSIS_ROOT", Path(__file__).resolve().parents[1])
).resolve()
MODEL_NAMES = (
    'a0_s11', 'a0_s1', 'a0_s21', 'a1_s11', 'a1_s1', 'a1_s21',
    'a2_s11_valbest', 'a2_s11', 'a2_s1', 'a2_s21', 'a4_s11', 'a4_s1',
    'a4_s21', 'b1_s11', 'b1_s1', 'b1_s21', 'cuefocal',
    'b3_single_cue', 'loo1_dropgt', 'loo2_dropgr', 'loo3_gconly',
    'loo1_dropgt_s1', 'loo1_dropgt_s21', 'loo2_dropgr_s1',
    'loo2_dropgr_s21', 'loo3_gconly_s1', 'loo3_gconly_s21', 'a3_cr_s1',
    'a3_cr_s21', 'a5_cr_s1', 'a5_cr_s21', 'a3_s11', 'a5_s11',
    'b2_nomech', 'b4_hnft', 'thetarel_s11',
)
MODELS = {
    model: str(
        WORKSPACE_ROOT
        / 'analysis_inputs'
        / 'synth_diag'
        / model
        / 'preds_prob.pkl'
    )
    for model in MODEL_NAMES
}


def load_preds(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    return {k.split('/')[-1].rsplit('.', 1)[0]: np.asarray(v, np.float32).flatten()
            for k, v in raw.items()}


def recall_and_fp(preds, gt, thr):
    """True-transition recall over ast+gst clips; pseudo FP total at thr."""
    gtn = {k.split('/')[-1].rsplit('.', 1)[0]: v for k, v in gt.items()}
    TP = FN = 0
    fp_total = 0
    for pk, probs in preds.items():
        rec = gtn.get(pk)
        if rec is None:
            continue
        arr = np.asarray(probs, np.float32).flatten()
        n = int(rec.get('frame_num', arr.shape[0]))
        transitions = [(int(s), int(e)) for s, e in (rec.get('ast') or [])]
        transitions += [(int(s), int(e)) for s, e in (rec.get('gst') or [])]
        gt_scenes = transitions_to_scenes(transitions, n)
        pred_scenes = transitions_to_scenes(probs_to_transitions(arr, thr), n)
        tp, fp, fn, _ = evaluate_scenes_and_diffs(gt_scenes, pred_scenes, 2)
        if rec.get('label_group') in ('ast', 'gst'):
            TP += tp
            FN += fn
        else:
            fp_total += fp
    rc = TP / (TP + FN) if (TP + FN) else 0.0
    return rc, fp_total


def main():
    with open(SYNTH_GT, 'r', encoding='utf-8') as file:
        gt = json.load(file)
    thrs = np.round(np.arange(0.01, 0.995, 0.01), 2)

    # anchor: PERSIST seed-11 recall at 0.50 (paper Fig. 4 protocol)
    preds0 = load_preds(MODELS['a0_s11'])
    target, fp0 = recall_and_fp(preds0, gt, 0.50)
    print(f'anchor: a0_s11 @0.50 recall={target:.4f} FP={fp0}')

    print(f"{'model':>10} | {'thr':>5} {'recall':>7} | {'FP@matched':>10} | {'FP@0.5':>7}")
    for name, pkl in MODELS.items():
        try:
            preds = load_preds(pkl)
        except FileNotFoundError:
            print(f'{name:>10} | MISSING')
            continue
        # tie-break rule: closest recall; ties broken toward the protocol's
        # reference threshold 0.50 (PERSIST's deployable operating point).
        best = None
        for t in thrs:
            rc, fp = recall_and_fp(preds, gt, float(t))
            key = (abs(rc - target), abs(float(t) - 0.50))
            if best is None or key < best[0]:
                best = (key, float(t), rc, fp)
        best = best[1:]
        _, fp05 = recall_and_fp(preds, gt, 0.50)
        print(f'{name:>10} | {best[0]:>5.2f} {best[1]:>7.4f} | {best[2]:>10} | {fp05:>7}')


if __name__ == '__main__':
    main()
