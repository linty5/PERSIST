"""Compare natural-footage false positives at matched transition recall.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np

from matching_lib_and_tolerance_sweep import (
    evaluate_scenes_and_diffs,
    probs_to_transitions,
    transitions_to_scenes,
)

WORKSPACE_ROOT = Path(
    os.environ.get("PERSIST_ANALYSIS_ROOT", Path(__file__).resolve().parents[1])
).resolve()
GT_PATH = str(WORKSPACE_ROOT / 'annotations/ClipShots/test.json')


def bn(k):
    return k.split('/')[-1].rsplit('.', 1)[0]


def eval_at(preds, ground_truth, thr, tol=2):
    gtb = {bn(k): k for k in ground_truth}
    TP = FP = FN = 0
    for pk, probs in preds.items():
        b = bn(pk)
        if b not in gtb:
            continue
        rec = ground_truth[gtb[b]]
        tr = ([(int(s), int(e)) for s, e in rec.get('ast', [])]
              + [(int(s), int(e)) for s, e in rec.get('gst', [])])
        arr = np.asarray(probs, dtype=np.float32).flatten()
        n = int(rec.get('frame_num', arr.shape[0]))
        gs = transitions_to_scenes(tr, n)
        ps = transitions_to_scenes(probs_to_transitions(arr, thr), n)
        tp, fp, fn, _ = evaluate_scenes_and_diffs(gs, ps, tol)
        TP += tp
        FP += fp
        FN += fn
    rec_ = TP / max(TP + FN, 1)
    prec = TP / max(TP + FP, 1)
    return TP, FP, FN, rec_, prec


def find_thr_for_recall(preds, ground_truth, target, lo=0.01, hi=0.99):
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        _, _, _, r, _ = eval_at(preds, ground_truth, mid)
        if r > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

MODEL_NAMES = (
    'a0_s11', 'a0_s1', 'a0_s21', 'b4_hnft', 'b1_s11', 'a1_s11',
    'a2_s11', 'a4_s11', 'a1_s1', 'a1_s21', 'a2_s1', 'a2_s21', 'a4_s1',
    'a4_s21', 'b1_s1', 'b1_s21', 'cuefocal', 'loo1_dropgt_s11',
    'loo1_dropgt_s1', 'loo1_dropgt_s21', 'loo2_dropgr_s11',
    'loo2_dropgr_s1', 'loo2_dropgr_s21', 'loo3_gconly_s11',
    'loo3_gconly_s1', 'loo3_gconly_s21', 'thetarel_s11', 'a3_s11',
    'a5_s11',
)
MODEL_PATHS = {
    model: str(
        WORKSPACE_ROOT
        / 'analysis_inputs'
        / 'clipshots'
        / model
        / 'preds_prob.pkl'
    )
    for model in MODEL_NAMES
}


def main():
    with open(GT_PATH, 'r', encoding='utf-8') as file:
        ground_truth = json.load(file)
    predictions = {}
    for model, path in MODEL_PATHS.items():
        try:
            with open(path, 'rb') as file:
                predictions[model] = pickle.load(file)
        except FileNotFoundError:
            print(f'[path] {model:9s} -> MISSING (skip)')
    if 'a0_s11' not in predictions:
        raise SystemExit('Missing required PERSIST anchor predictions: a0_s11')

    print('\n=== ClipShots TEST: natural-footage variants ===')
    print('--- at fix-0.5 ---')
    base = {}
    for model, model_predictions in predictions.items():
        _, false_positives, _, recall, precision = eval_at(
            model_predictions, ground_truth, 0.5)
        base[model] = (recall, false_positives)
        print(
            f'{model:9s} recall={recall:.4f} FP={false_positives} '
            f'prec={precision:.4f}')

    target = base['a0_s11'][0]
    print(f'\n--- matched to PERSIST_s11 fix-0.5 recall={target:.4f} ---')
    for model, model_predictions in predictions.items():
        threshold = find_thr_for_recall(
            model_predictions, ground_truth, target)
        _, false_positives, _, recall, precision = eval_at(
            model_predictions, ground_truth, threshold)
        print(
            f'{model:9s} threshold={threshold:.3f} recall={recall:.4f} '
            f'FP={false_positives} precision={precision:.4f}')


if __name__ == '__main__':
    main()
