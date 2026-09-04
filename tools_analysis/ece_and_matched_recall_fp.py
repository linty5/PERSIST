#!/usr/bin/env python
"""Compute calibration error and matched-recall false-positive counts.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from matching_lib_and_tolerance_sweep import (  # type: ignore
    probs_to_transitions, transitions_to_scenes,
    evaluate_scenes_and_diffs,
)

ROOT = str(Path(
    os.environ.get("PERSIST_ANALYSIS_ROOT", Path(__file__).resolve().parents[1])
).resolve())
OUT = Path(ROOT) / "analysis_outputs"
N_BINS = 10

# ---------------------------------------------------------------------------
# Registry for expected calibration error.
# ---------------------------------------------------------------------------
GT = {
    "clipshots": f"{ROOT}/annotations/ClipShots/test.json",
    "bbc":       f"{ROOT}/annotations/BBC_PlanetEarth/test.json",
    "autoshot":  f"{ROOT}/annotations/AutoShot/test.json",
    "hist_gst12":f"{ROOT}/annotations/HISTORIAN/test_gst12.json",
}
ECE_MODELS = ("persist", "repro", "official")
ECE_PKLS = {
    (model, dataset): str(
        Path(ROOT)
        / "analysis_inputs"
        / dataset
        / model
        / "preds_prob.pkl"
    )
    for model in ECE_MODELS
    for dataset in GT
}

# Registry for synthetic matched-recall analysis.
SYNTH_GT = f"{ROOT}/diagnostic/annotations/test_diag_all.json"
SYNTH_PKLS = {
    model: str(
        Path(ROOT)
        / "analysis_inputs"
        / "synth_diag"
        / model
        / "preds_prob.pkl"
    )
    for model in ("persist", "official")
}
HN_TYPES = ["flash_illumination", "fast_pan", "text_overlay", "archival_degradation", "scratch"]


def norm(k: str) -> str:
    return k.split("/")[-1].rsplit(".", 1)[0]


def load_norm(pkl: str) -> dict:
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    return {norm(k): np.asarray(v, dtype=np.float64).flatten() for k, v in d.items()}


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def labels_inside(rec: dict, n: int) -> np.ndarray:
    y = np.zeros(n, dtype=np.float64)
    for s, e in rec.get("ast", []) + rec.get("gst", []):
        y[max(0, int(s)):min(n, int(e) + 1)] = 1.0
    return y


# Expected calibration error.
def ece_cell(preds: dict, gt: dict) -> dict:
    gidx = {norm(k): k for k in gt}
    P, Y = [], []
    for pk, arr in preds.items():
        gk = gidx.get(pk)
        if gk is None:
            continue
        n = min(int(gt[gk].get("frame_num", arr.shape[0])), arr.shape[0])
        if n <= 0:
            continue
        P.append(arr[:n])
        Y.append(labels_inside(gt[gk], n))
    if not P:
        return {}
    p = np.concatenate(P)
    y = np.concatenate(Y)
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, N_BINS - 1)
    ece = 0.0
    for b in range(N_BINS):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        ece += (nb / p.size) * abs(p[m].mean() - y[m].mean())
    return dict(ece=ece, brier=float(np.mean((p - y) ** 2)),
                pos_frac=float(y.mean()), n=int(p.size), matched=len(P))


def run_ece():
    gtc = {dataset: load_json(path) for dataset, path in GT.items()}
    rows = []
    for (model, ds), pkl in ECE_PKLS.items():
        if not os.path.exists(pkl):
            print(f"  [skip] {model}/{ds}: missing {pkl}")
            continue
        out = ece_cell(load_norm(pkl), gtc[ds])
        if not out:
            print(f"  [warn] {model}/{ds}: no key overlap")
            continue
        rows.append(dict(model=model, dataset=ds, ece=round(out["ece"], 6),
                         brier=round(out["brier"], 6), n_frames=out["n"],
                         n_videos=out["matched"]))
        print(f"  {model:9s} {ds:11s} ECE={out['ece']:.4f} (videos={out['matched']})")
    if not rows:
        return rows
    with open(OUT / "official_ece_4ds.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # pretty matrix
    print("\n  ECE matrix (rows=model, cols=dataset):")
    order = ["persist", "repro", "official"]
    print("    {:9s} {:>10s} {:>10s} {:>10s} {:>10s}".format("model", *GT.keys()))
    for m in order:
        cells = []
        for ds in GT:
            r = next((x for x in rows if x["model"] == m and x["dataset"] == ds), None)
            cells.append(f"{r['ece']:.4f}" if r else "   -  ")
        print("    {:9s} {:>10s} {:>10s} {:>10s} {:>10s}".format(m, *cells))
    print(f"\n  [write] {OUT/'official_ece_4ds.csv'}")
    return rows


# Matched-recall synthetic false positives.
def synth_recall_and_fp(preds: dict, gt: dict, thr: float):
    """Return (real_recall, {hn_type: fp_count}) at threshold thr."""
    gidx = {norm(k): k for k in gt}
    TP = FN = 0
    fp_by_type = defaultdict(int)
    for pk, arr in preds.items():
        gk = gidx.get(pk)
        if gk is None:
            continue
        rec = gt[gk]
        n = int(rec.get("frame_num", arr.shape[0]))
        trans = [(int(s), int(e)) for s, e in rec.get("ast", []) + rec.get("gst", [])]
        gt_sc = transitions_to_scenes(trans, n)
        pr_sc = transitions_to_scenes(probs_to_transitions(arr, thr), n)
        tp, fp, fn, _ = evaluate_scenes_and_diffs(gt_sc, pr_sc, 2)
        if rec.get("label_group") in ("ast", "gst"):   # real-transition video
            TP += tp
            FN += fn
        else:                                          # hard-negative video
            fp_by_type[rec.get("synthetic_type")] += fp
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    return recall, fp_by_type


def run_matched_recall():
    gt = load_json(SYNTH_GT)
    pr = {m: load_norm(p) for m, p in SYNTH_PKLS.items()}
    grid = [round(0.01 * i, 2) for i in range(1, 100)]

    # PERSIST reference operating point = fix 0.50
    r_persist, fp_persist = synth_recall_and_fp(pr["persist"], gt, 0.50)
    print(f"\n  PERSIST @0.50: real-transition recall = {r_persist:.4f}")

    def fp_at(model, thr):
        return synth_recall_and_fp(pr[model], gt, thr)

    # official @ fix 0.50 (reference) and @ matched recall
    r_off_05, fp_off_05 = fp_at("official", 0.50)
    # find official threshold whose recall is closest to PERSIST's
    best = min(grid, key=lambda t: abs(fp_at("official", t)[0] - r_persist))
    r_off_m, fp_off_m = fp_at("official", best)
    print(f"  official @0.50: recall = {r_off_05:.4f}")
    print(f"  official @matched thr={best:.2f}: recall = {r_off_m:.4f} (target {r_persist:.4f})")

    print(f"\n  Pseudo-event FP counts (lower better):")
    hdr = f"    {'hn subtype':22s} {'#vids':>6s} {'PERSIST@.50':>12s} {'official@.50':>13s} {'official@matched':>17s}"
    print(hdr)
    nv = Counter(v.get("synthetic_type") for v in gt.values()
                 if v.get("label_group") not in ("ast", "gst"))
    rows = []
    for t in HN_TYPES:
        a, b, c = fp_persist.get(t, 0), fp_off_05.get(t, 0), fp_off_m.get(t, 0)
        print(f"    {t:22s} {nv[t]:>6d} {a:>12d} {b:>13d} {c:>17d}")
        rows.append(dict(hn_subtype=t, n_videos=nv[t], persist_fp_thr050=a,
                         official_fp_thr050=b, official_fp_matched=c,
                         persist_recall=round(r_persist, 4),
                         official_matched_thr=best,
                         official_matched_recall=round(r_off_m, 4)))
    tot = (sum(fp_persist.get(t, 0) for t in HN_TYPES),
           sum(fp_off_05.get(t, 0) for t in HN_TYPES),
           sum(fp_off_m.get(t, 0) for t in HN_TYPES))
    print(f"    {'TOTAL (these 5)':22s} {'':>6s} {tot[0]:>12d} {tot[1]:>13d} {tot[2]:>17d}")
    with open(OUT / "official_synth_fp_matched_recall.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  [write] {OUT/'official_synth_fp_matched_recall.csv'}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 70, "\nExpected calibration error\n" + "=" * 70)
    run_ece()
    print(
        "\n" + "=" * 70,
        "\nSynthetic false positives at matched recall\n" + "=" * 70,
    )
    run_matched_recall()
