#!/usr/bin/env python
"""Compute synthetic-diagnostic metrics by event category.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matching_lib_and_tolerance_sweep import (  # type: ignore
    probs_to_transitions, transitions_to_scenes,
    evaluate_scenes_and_diffs, aggregate_f1,
)


WORKSPACE_ROOT = Path(
    os.environ.get("PERSIST_ANALYSIS_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SYNTH_GT = str(WORKSPACE_ROOT / "diagnostic/annotations/test_diag_all.json")
SYNTH_OUTPUT_BASE = WORKSPACE_ROOT / "analysis_inputs/synth_diag"
MODEL_NAMES = (
    "a0_base", "a1_notrcs", "a2_nophi", "a2_nophi_valbest",
    "a3_nodyn", "a4_singlerate", "a5_trcsonly", "b1_tnv2repro",
    "b2_tnv2nomech", "official_tnv2", "b3_noslow", "b4_hnft",
    "p1_psrv2", "p2_psrv2plus", "pmft_finetune",
)


def per_group_evaluate(
    preds: Dict[str, np.ndarray],
    gt: Dict[str, dict],
    group_field: str,
    threshold: float,
    tol_frames: int = 2,
) -> List[Dict]:
    """Compute F1 for videos grouped by a synthetic-event field."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for k, rec in gt.items():
        g = rec.get(group_field) or "_unknown"
        groups[g].append(k)
    rows = []
    gt_index_basename = {k.split("/")[-1].rsplit(".", 1)[0]: k for k in gt.keys()}
    for grp_name, gt_keys in groups.items():
        TP = FP = FN = 0
        n_videos = 0
        # build a reverse-lookup for matching preds basename to gt full key
        keys_basenames = {k.split("/")[-1].rsplit(".", 1)[0] for k in gt_keys}
        for pk, probs in preds.items():
            if pk not in keys_basenames:
                continue
            gk = gt_index_basename.get(pk)
            if gk is None:
                continue
            rec = gt[gk]
            transitions = [(int(s), int(e)) for s, e in rec.get("ast", [])]
            transitions += [(int(s), int(e)) for s, e in rec.get("gst", [])]
            arr = np.asarray(probs, dtype=np.float32).flatten()
            n_frames = int(rec.get("frame_num", arr.shape[0]))
            gt_scenes = transitions_to_scenes(transitions, n_frames)
            pred_trans = probs_to_transitions(arr, threshold)
            pred_scenes = transitions_to_scenes(pred_trans, n_frames)
            tp, fp, fn, _ = evaluate_scenes_and_diffs(gt_scenes, pred_scenes, tol_frames)
            TP += tp
            FP += fp
            FN += fn
            n_videos += 1
        p, r, f1 = aggregate_f1(TP, FP, FN)
        rows.append(dict(
            group_field=group_field, group=grp_name,
            n_videos=n_videos, tp=TP, fp=FP, fn=FN,
            precision=round(p, 5), recall=round(r, 5), f1=round(f1, 5),
        ))
    rows.sort(key=lambda r: (-r["n_videos"], r["group"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(WORKSPACE_ROOT / "analysis_outputs"))
    ap.add_argument("--threshold", type=float, default=0.50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [info] loading synth GT: {SYNTH_GT}")
    with open(SYNTH_GT) as f:
        gt = json.load(f)

    all_rows = []
    model_pkls = [
        (model, str(SYNTH_OUTPUT_BASE / model / "preds_prob.pkl"))
        for model in MODEL_NAMES
    ]
    for model, pkl in model_pkls:
        if not os.path.exists(pkl):
            print(f"  [skip] {model}: {pkl} not found")
            continue
        print(f"  [run]  {model}: {pkl}")
        with open(pkl, "rb") as f:
            preds = pickle.load(f)
        for group_field in ["synthetic_type", "category_family", "transition_kind", "label_group"]:
            rows = per_group_evaluate(preds, gt, group_field, args.threshold)
            for r in rows:
                all_rows.append(dict(model=model, threshold=args.threshold, **r))

    out = out_dir / "synth_per_type.csv"
    if all_rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"[write] {out} ({len(all_rows)} rows)")
    else:
        print("[info] no rows produced; check the configured input paths.")


if __name__ == "__main__":
    main()
