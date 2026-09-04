"""Compute per-subtype false positives at multiple recall targets.

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

from synth_per_type_f1 import per_group_evaluate, SYNTH_GT
from matching_lib_and_tolerance_sweep import (probs_to_transitions,
                                           transitions_to_scenes,
                                           evaluate_scenes_and_diffs)

WORKSPACE_ROOT = Path(
    os.environ.get("PERSIST_ANALYSIS_ROOT", Path(__file__).resolve().parents[1])
).resolve()
PSEUDO = ['flash_illumination', 'fast_pan', 'text_overlay',
          'archival_degradation', 'scratch']
MODEL_NAMES = (
    'a0_s11', 'a1_s11', 'a2_s11_valbest', 'a2_s11', 'a4_s11', 'b1_s11',
    'cuefocal', 'b3_single_cue', 'a0_s1', 'a0_s21', 'a1_s1', 'a1_s21',
    'a2_s1', 'a2_s21', 'a4_s1', 'a4_s21', 'b1_s1', 'b1_s21',
    'loo1_dropgt', 'loo2_dropgr', 'loo3_gconly', 'loo1_dropgt_s1',
    'loo1_dropgt_s21', 'loo2_dropgr_s1', 'loo2_dropgr_s21',
    'loo3_gconly_s1', 'loo3_gconly_s21', 'a3_cr_s1', 'a3_cr_s21',
    'a5_cr_s1', 'a5_cr_s21', 'a3_s11', 'a5_s11', 'b2_nomech', 'b4_hnft',
    'thetarel_s11',
)
M = {
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
    gtn = {k.split('/')[-1].rsplit('.', 1)[0]: v for k, v in gt.items()}
    TP = FN = 0
    fp_total = 0
    for pk, probs in preds.items():
        rec = gtn.get(pk)
        if rec is None:
            continue
        arr = np.asarray(probs, np.float32).flatten()
        n = int(rec.get('frame_num', arr.shape[0]))
        tr = [(int(s), int(e)) for s, e in (rec.get('ast') or [])]
        tr += [(int(s), int(e)) for s, e in (rec.get('gst') or [])]
        gs = transitions_to_scenes(tr, n)
        ps = transitions_to_scenes(probs_to_transitions(arr, thr), n)
        tp, fp, fn, _ = evaluate_scenes_and_diffs(gs, ps, 2)
        if rec.get('label_group') in ('ast', 'gst'):
            TP += tp
            FN += fn
        else:
            fp_total += fp
    return (TP / (TP + FN) if (TP + FN) else 0.0), fp_total


def matched_thr(preds, gt, target, thrs):
    best = None
    for t in thrs:
        rc, fp = recall_and_fp(preds, gt, float(t))
        key = (abs(rc - target), abs(float(t) - 0.50))
        if best is None or key < best[0]:
            best = (key, float(t), rc, fp)
    return best[1], best[2], best[3]


def main():
    with open(SYNTH_GT, 'r', encoding='utf-8') as file:
        gt = json.load(file)
    thrs = np.round(np.arange(0.01, 0.995, 0.01), 2)
    preds0 = load_preds(M['a0_s11'])
    target, _ = recall_and_fp(preds0, gt, 0.50)
    print(f'anchor recall = {target:.4f}')

    # (a) per-subtype FP at matched threshold
    print('\n== per-subtype FP at matched threshold ==')
    print(f"{'model':>14} | {'thr':>5} | " + ' '.join(f'{t[:8]:>8}' for t in PSEUDO) + ' |  TOTAL')
    for name, pkl in M.items():
        try:
            preds = load_preds(pkl)
        except FileNotFoundError:
            print(f'{name:>14} | MISSING')
            continue
        thr, rc, _ = matched_thr(preds, gt, target, thrs)
        per = per_group_evaluate(preds, gt, 'synthetic_type', threshold=thr)
        by = {r['group']: r['fp'] for r in per}
        fps = [by.get(t, -1) for t in PSEUDO]
        print(f'{name:>14} | {thr:>5.2f} | ' + ' '.join(f'{v:>8}' for v in fps) + f' | {sum(fps):>6}')

    # (b) totals at three recall anchors
    print('\n== total FP at recall anchors 0.65 / matched / 0.80 ==')
    for anchor in (0.65, target, 0.80):
        print(f'-- anchor recall {anchor:.3f} --')
        for name, pkl in M.items():
            try:
                preds = load_preds(pkl)
            except FileNotFoundError:
                continue
            thr, rc, fp = matched_thr(preds, gt, anchor, thrs)
            print(f'{name:>14} | thr {thr:.2f} recall {rc:.4f} | FP {fp}')


if __name__ == '__main__':
    main()
