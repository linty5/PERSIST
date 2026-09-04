#!/usr/bin/env python
"""Compute temporal-tolerance sweeps and absolute boundary error.

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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Set PERSIST_ANALYSIS_ROOT to the workspace containing prediction dumps.
WORKSPACE_ROOT = Path(
    os.environ.get("PERSIST_ANALYSIS_ROOT", Path(__file__).resolve().parents[1])
).resolve()
GT_PATHS: Dict[str, str] = {
    "clipshots": str(WORKSPACE_ROOT / "annotations/ClipShots/test.json"),
    "bbc": str(WORKSPACE_ROOT / "annotations/BBC_PlanetEarth/test.json"),
    "bbc_fixed": str(WORKSPACE_ROOT / "annotations/BBC_fixed/test.json"),
    "autoshot": str(WORKSPACE_ROOT / "annotations/AutoShot/test.json"),
    "hist_gst12": str(WORKSPACE_ROOT / "annotations/HISTORIAN/test_gst12.json"),
}

MODEL_NAMES = ("a0_base", "a1_notrcs", "a2_nophi", "b1_tnv2repro",
               "b2_tnv2nomech")
DATASET_NAMES = ("clipshots", "bbc", "bbc_fixed", "autoshot",
                 "hist_gst12")
MODEL_EVAL_DIRS: Dict[Tuple[str, str], str] = {
    (model, dataset): str(
        WORKSPACE_ROOT
        / "analysis_inputs"
        / dataset
        / model
        / "preds_prob.pkl"
    )
    for model in MODEL_NAMES
    for dataset in DATASET_NAMES
}


TAU_GRID_FRAMES: List[int] = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
THR_SCAN_GRID: List[float] = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05..0.95


# ---------------------------------------------------------------------------
# Core matching: ported from mmaction2/.../sbd_metric.py `_evaluate_scenes`
# so that this offline tolerance sweep is numerically consistent with the
# numbers in three_protocols.json / main paper table.
# ---------------------------------------------------------------------------

def probs_to_transitions(probs: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """Return list of [start, end] transitions (end inclusive) where probs >= threshold.

    Matches the form used by mmaction2 SBD metric: transitions are 0-indexed
    half-open before being shifted in `_evaluate_scenes`.
    """
    probs = probs.flatten()
    above = probs >= threshold
    spans: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(above):
        if v and not in_run:
            start = i
            in_run = True
        elif (not v) and in_run:
            spans.append((start, i - 1))
            in_run = False
    if in_run:
        spans.append((start, len(above) - 1))
    return spans


def transitions_to_scenes(transitions: List[Tuple[int, int]], n_frames: int) -> np.ndarray:
    """Identical to mmaction2 sbd_metric `_transitions_to_scenes`."""
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")
    if not transitions:
        return np.array([[0, n_frames - 1]], dtype=np.int32)
    transitions = sorted(transitions, key=lambda x: x[0])
    scenes: List[List[int]] = []
    cur_start = 0
    for s, e in transitions:
        s = max(0, min(int(s), n_frames - 1))
        e = max(0, min(int(e), n_frames))
        if cur_start <= s:
            scenes.append([cur_start, s])
        cur_start = e
    if cur_start <= n_frames - 1:
        scenes.append([cur_start, n_frames - 1])
    return np.asarray(scenes, dtype=np.int32)


def evaluate_scenes_and_diffs(
    gt_scenes: np.ndarray,
    pred_scenes: np.ndarray,
    n_frames_miss_tolerance: int,
) -> Tuple[int, int, int, List[float]]:
    """Port of mmaction2 `_evaluate_scenes`, extended to record matched-pair
    transition-midpoint offsets (used for ABE).

    Returns (tp, fp, fn, list of abs midpoint diffs in frames).
    """
    shift = n_frames_miss_tolerance / 2.0
    gt_s = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pr_s = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    if len(gt_s) <= 1:
        gt_trans = np.zeros((0, 2), dtype=np.float32)
    else:
        gt_trans = np.stack([gt_s[:-1, 1], gt_s[1:, 0]], 1)
    if len(pr_s) <= 1:
        pr_trans = np.zeros((0, 2), dtype=np.float32)
    else:
        pr_trans = np.stack([pr_s[:-1, 1], pr_s[1:, 0]], 1)

    i = 0
    j = 0
    tp = fp = fn = 0
    diffs: List[float] = []
    while i < len(gt_trans) or j < len(pr_trans):
        if j == len(pr_trans):
            fn += 1
            i += 1
        elif i == len(gt_trans):
            fp += 1
            j += 1
        elif pr_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            j += 1
        elif pr_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            i += 1
        else:
            gm = 0.5 * (gt_trans[i, 0] + gt_trans[i, 1])
            pm = 0.5 * (pr_trans[j, 0] + pr_trans[j, 1])
            diffs.append(abs(float(pm - gm)))
            tp += 1
            i += 1
            j += 1
    return int(tp), int(fp), int(fn), diffs


def evaluate_one_video(
    gt_transitions: List[Tuple[int, int]],
    pred_transitions: List[Tuple[int, int]],
    n_frames: int,
    tol_frames: int,
) -> Tuple[int, int, int, List[float]]:
    gt_scenes = transitions_to_scenes(gt_transitions, n_frames)
    pr_scenes = transitions_to_scenes(pred_transitions, n_frames)
    return evaluate_scenes_and_diffs(gt_scenes, pr_scenes, tol_frames)


def collect_dataset_pairs(
    preds: Dict[str, np.ndarray],
    gt: Dict[str, dict],
) -> List[Tuple[str, np.ndarray, List[Tuple[int, int]], int]]:
    """Map preds keys to GT keys (basename match). Returns list of
    (pred_key, probs, gt_transitions = ast+gst concatenated, n_frames)."""
    gt_index = {k.split("/")[-1].rsplit(".", 1)[0]: k for k in gt.keys()}
    out: List[Tuple[str, np.ndarray, List[Tuple[int, int]], int]] = []
    missing = 0
    for pk, probs in preds.items():
        gk = gt_index.get(pk)
        if gk is None:
            missing += 1
            continue
        rec = gt[gk]
        transitions: List[Tuple[int, int]] = []
        for s, e in rec.get("ast", []):
            transitions.append((int(s), int(e)))
        for s, e in rec.get("gst", []):
            transitions.append((int(s), int(e)))
        if isinstance(probs, np.ndarray):
            arr = probs.astype(np.float32).flatten()
        else:
            arr = np.asarray(probs, dtype=np.float32).flatten()
        n_frames = int(rec.get("frame_num", arr.shape[0]))
        out.append((pk, arr, transitions, n_frames))
    if missing:
        print(f"  [warn] {missing} pred-key videos had no GT match", file=sys.stderr)
    return out


def aggregate_f1(tps: int, fps: int, fns: int) -> Tuple[float, float, float]:
    p = tps / (tps + fps) if (tps + fps) else 0.0
    r = tps / (tps + fns) if (tps + fns) else 0.0
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def sweep_one_cell(
    pairs: List[Tuple[str, np.ndarray, List[Tuple[int, int]], int]],
    threshold: float,
    tau_grid: List[int],
) -> Dict[int, Dict[str, float]]:
    """Returns {tau_frames: {f1, precision, recall, abe_frames, tp, fp, fn, n_matched}}."""
    # Pre-compute pred transitions once per threshold.
    cached = [(probs_to_transitions(probs, threshold), gt_trans, n_frames)
              for _, probs, gt_trans, n_frames in pairs]
    result: Dict[int, Dict[str, float]] = {}
    for tau in tau_grid:
        TP = FP = FN = 0
        all_diffs: List[float] = []
        for pred_trans, gt_trans, n_frames in cached:
            tp, fp, fn, diffs = evaluate_one_video(gt_trans, pred_trans, n_frames, tau)
            TP += tp
            FP += fp
            FN += fn
            all_diffs.extend(diffs)
        p, r, f1 = aggregate_f1(TP, FP, FN)
        abe = float(np.mean(all_diffs)) if all_diffs else float("nan")
        result[tau] = dict(f1=f1, precision=p, recall=r, abe_frames=abe,
                           tp=TP, fp=FP, fn=FN, n_matched=len(all_diffs))
    return result


def find_test_best_threshold(
    pairs: List[Tuple[str, np.ndarray, List[Tuple[int, int]], int]],
    tau_ref: int = 2,
) -> float:
    best_thr = 0.50
    best_f1 = -1.0
    for thr in THR_SCAN_GRID:
        TP = FP = FN = 0
        for _, probs, gt_trans, n_frames in pairs:
            pred_trans = probs_to_transitions(probs, thr)
            tp, fp, fn, _ = evaluate_one_video(gt_trans, pred_trans, n_frames, tau_ref)
            TP += tp
            FP += fp
            FN += fn
        _, _, f1 = aggregate_f1(TP, FP, FN)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(WORKSPACE_ROOT / "analysis_outputs"),
                    help="output directory for the two CSVs")
    ap.add_argument("--limit_pairs", type=int, default=0,
                    help="if > 0, only process the first N (model, dataset) pairs (debug)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load all GT JSONs.
    gt_cache: Dict[str, Dict[str, dict]] = {}
    for ds, path in GT_PATHS.items():
        if not os.path.exists(path):
            print(f"  [warn] missing GT JSON for {ds}: {path}", file=sys.stderr)
            continue
        with open(path, "r") as f:
            gt_cache[ds] = json.load(f)

    sweep_rows = []
    abe_rows = []

    cells = list(MODEL_EVAL_DIRS.items())
    if args.limit_pairs > 0:
        cells = cells[: args.limit_pairs]

    for (model, ds), pkl in cells:
        if ds not in gt_cache:
            print(f"  [skip] {model}/{ds}: GT cache missing")
            continue
        if not os.path.exists(pkl):
            print(f"  [skip] {model}/{ds}: {pkl} not found")
            continue
        print(f"  [run]  {model}/{ds}: loading {pkl}")
        with open(pkl, "rb") as f:
            preds = pickle.load(f)
        pairs = collect_dataset_pairs(preds, gt_cache[ds])
        if not pairs:
            print(f"  [skip] {model}/{ds}: no overlap with GT")
            continue
        print(f"         {len(pairs)} videos overlap")

        for protocol, thr in [("fix05", 0.50), ("testbest", None)]:
            if thr is None:
                thr = find_test_best_threshold(pairs)
                print(f"         testbest threshold = {thr:.2f}")
            sweep = sweep_one_cell(pairs, thr, TAU_GRID_FRAMES)
            for tau, m in sweep.items():
                sweep_rows.append(dict(
                    model=model, dataset=ds, protocol=protocol,
                    threshold=round(float(thr), 4),
                    tau_frames=tau,
                    f1=round(m["f1"], 5),
                    precision=round(m["precision"], 5),
                    recall=round(m["recall"], 5),
                    abe_frames=round(m["abe_frames"], 4) if not np.isnan(m["abe_frames"]) else "",
                    tp=m["tp"], fp=m["fp"], fn=m["fn"],
                    n_matched=m["n_matched"],
                ))
            # Use a generous match window so ABE is not limited to easy cases.
            ref_metrics = sweep[4] if 4 in sweep else sweep[min(sweep.keys())]
            abe_rows.append(dict(
                model=model, dataset=ds, protocol=protocol,
                threshold=round(float(thr), 4),
                ref_tau_frames=4,
                abe_frames=round(ref_metrics["abe_frames"], 4) if not np.isnan(ref_metrics["abe_frames"]) else "",
                n_matched=ref_metrics["n_matched"],
                f1_at_ref=round(ref_metrics["f1"], 5),
            ))

    sweep_csv = out_dir / "tolerance_sweep.csv"
    abe_csv = out_dir / "abe.csv"
    if sweep_rows:
        keys = list(sweep_rows[0].keys())
        with open(sweep_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"[write] {sweep_csv} ({len(sweep_rows)} rows)")
    if abe_rows:
        keys = list(abe_rows[0].keys())
        with open(abe_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(abe_rows)
        print(f"[write] {abe_csv} ({len(abe_rows)} rows)")


if __name__ == "__main__":
    main()
