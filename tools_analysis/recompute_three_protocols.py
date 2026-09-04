#!/usr/bin/env python3
"""Recompute the three SBD threshold protocols from saved predictions.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'mmaction2'))

# Pull the matching helpers from SBDMetric so semantics stay identical.
from mmaction.evaluation.metrics.sbd_metric import SBDMetric  # noqa: E402


def parse_val_thr_from_ckpt(path: str) -> float:
    m = re.search(
        r"valthr(?:eshold)?[_-]?(0?\.\d+|\d{1,3})",
        os.path.basename(path),
    )
    if not m:
        return -1.0
    threshold = float(m.group(1))
    return threshold / 100.0 if threshold > 1.0 else threshold


def load_gt_transitions(gt_json: str) -> Dict[str, List[Tuple[int, int]]]:
    """Return per-video list of GT transitions ``[(start_frame, end_frame), ...]``
    in 0-based global frame indices, combining AST and GST.

    Matches what ``_build_per_video_arrays`` does inside SBDMetric: AST and
    GST intervals are both treated as ground-truth transitions for the
    standard SBD matching rule.
    """
    with open(gt_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, List[Tuple[int, int]]] = {}
    for key, ann in raw.items():
        base = os.path.splitext(os.path.basename(str(key)))[0]
        ts: List[Tuple[int, int]] = []
        for s, e in (ann.get("ast") or []):
            ts.append((int(s), int(e)))
        for s, e in (ann.get("gst") or []):
            ts.append((int(s), int(e)))
        out[base] = ts
    return out


def eval_at_threshold(
    pred_probs: Dict[str, np.ndarray],
    gt_transitions: Dict[str, List[Tuple[int, int]]],
    threshold: float,
    n_frames_miss_tolerance: int = 2,
) -> Tuple[float, float, float, int, int, int]:
    """Run the matching rule at one threshold across all videos, return
    (f1, precision, recall, tp, fp, fn) totals.
    """
    total_tp = total_fp = total_fn = 0
    for vid, probs in pred_probs.items():
        probs = np.asarray(probs).reshape(-1)
        F = int(len(probs))
        if F <= 0:
            continue
        one_hot = (probs >= float(threshold)).astype(np.uint8)
        pred_scenes = SBDMetric._predictions_to_scenes(one_hot)
        # Normalize pickle key the same way load_gt_transitions normalized
        # GT keys (splitext+basename). Pickle keys vary by dataset:
        # ClipShots / BBC / AutoShot strip the extension already, but
        # HIST33 keeps it (e.g., "8332.m4v"); without this strip, the
        # HIST33 lookup misses every video and F1 silently returns 0.
        vid_key = os.path.splitext(os.path.basename(vid))[0]
        gt_for_vid = gt_transitions.get(vid_key) or gt_transitions.get(vid) or []
        gt_scenes = SBDMetric._transitions_to_scenes(gt_for_vid, n_frames=F)
        tp, fp, fn = SBDMetric._evaluate_scenes(
            gt_scenes, pred_scenes,
            n_frames_miss_tolerance=n_frames_miss_tolerance,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return f1, p, r, int(total_tp), int(total_fp), int(total_fn)


def scan_oracle(
    pred_probs: Dict[str, np.ndarray],
    gt_transitions: Dict[str, List[Tuple[int, int]]],
    thresholds: List[float],
    n_frames_miss_tolerance: int = 2,
) -> Tuple[float, float, float, float, int, int, int]:
    """Return (best_thr, f1, p, r, tp, fp, fn) by scanning all thresholds."""
    best = None
    for th in thresholds:
        f1, p, r, tp, fp, fn = eval_at_threshold(
            pred_probs, gt_transitions, th, n_frames_miss_tolerance,
        )
        if best is None or f1 > best[1]:
            best = (th, f1, p, r, tp, fp, fn)
    return best  # type: ignore[return-value]


_DEFAULT_THRESHOLDS = [round(0.02 + 0.02 * i, 2) for i in range(0, 49)] + [0.95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True, help="eval_outputs/preds_prob.pkl")
    ap.add_argument("--gt_json", required=True, help="dataset GT json (test split)")
    ap.add_argument("--val_threshold", type=float, default=None,
                    help="val-best threshold to apply. If omitted and --ckpt is given, "
                         "parse it from a 'valthr' checkpoint filename tag.")
    ap.add_argument("--ckpt", default=None,
                    help="optional ckpt path; used to auto-extract val_threshold.")
    ap.add_argument("--name", default="SBD",
                    help="log prefix in the printed summary block.")
    ap.add_argument("--miss_tol", type=int, default=2,
                    help="n_frames_miss_tolerance for the matching rule (default 2).")
    ap.add_argument("--out_json", default=None,
                    help="optional JSON dump of all three protocols.")
    ap.add_argument("--thresholds", nargs="*", type=float, default=None,
                    help="custom threshold grid for oracle scan; default dense [0.02, 0.95].")
    args = ap.parse_args()

    if args.val_threshold is None and args.ckpt is not None:
        v = parse_val_thr_from_ckpt(args.ckpt)
        if v < 0:
            raise SystemExit(f"Could not parse val_threshold from ckpt name: {args.ckpt}")
        args.val_threshold = v
    if args.val_threshold is None:
        raise SystemExit("Either --val_threshold or --ckpt (with valthreshold_NN in name) must be given.")

    with open(args.pickle, "rb") as f:
        pred_prob_dict = pickle.load(f)

    gt = load_gt_transitions(args.gt_json)
    thresholds = args.thresholds if args.thresholds else _DEFAULT_THRESHOLDS

    # val_best (at supplied threshold)
    val_f1, val_p, val_r, val_tp, val_fp, val_fn = eval_at_threshold(
        pred_prob_dict, gt, args.val_threshold, args.miss_tol,
    )
    # fix_05
    fix_f1, fix_p, fix_r, fix_tp, fix_fp, fix_fn = eval_at_threshold(
        pred_prob_dict, gt, 0.50, args.miss_tol,
    )
    # oracle scan
    test_thr, test_f1, test_p, test_r, test_tp, test_fp, test_fn = scan_oracle(
        pred_prob_dict, gt, thresholds, args.miss_tol,
    )

    nm = args.name
    print(
        f"[{nm}] thr   val_best={args.val_threshold:.4f}  fix_05=0.5000  test_best={test_thr:.4f}  |  "
        f"F1   val_best={val_f1:.4f}  fix_05={fix_f1:.4f}  test_best={test_f1:.4f}"
    )
    print(
        f"[{nm}] val_best  thr={args.val_threshold:.4f}  F1={val_f1:.4f}  "
        f"P={val_p:.4f}  R={val_r:.4f}  TP={val_tp}  FP={val_fp}  FN={val_fn}"
    )
    print(
        f"[{nm}] fix_05    thr=0.5000  F1={fix_f1:.4f}  "
        f"P={fix_p:.4f}  R={fix_r:.4f}  TP={fix_tp}  FP={fix_fp}  FN={fix_fn}"
    )
    print(
        f"[{nm}] test_best thr={test_thr:.4f}  F1={test_f1:.4f}  "
        f"P={test_p:.4f}  R={test_r:.4f}  TP={test_tp}  FP={test_fp}  FN={test_fn}"
    )

    if args.out_json:
        out = {
            "pickle": args.pickle, "gt_json": args.gt_json,
            "val_threshold_used": args.val_threshold,
            "val_best":  {"threshold": args.val_threshold, "f1": val_f1, "p": val_p, "r": val_r,
                          "tp": val_tp, "fp": val_fp, "fn": val_fn},
            "fix_05":    {"threshold": 0.50, "f1": fix_f1, "p": fix_p, "r": fix_r,
                          "tp": fix_tp, "fp": fix_fp, "fn": fix_fn},
            "test_best": {"threshold": test_thr, "f1": test_f1, "p": test_p, "r": test_r,
                          "tp": test_tp, "fp": test_fp, "fn": test_fn},
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[{nm}] wrote {args.out_json}")


if __name__ == "__main__":
    main()
