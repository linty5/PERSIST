"""Compare natural-footage false positives at matched recall.

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
GT_PATH = WORKSPACE_ROOT / "annotations/ClipShots/test.json"
MODEL_PATHS = {
    model: (
        WORKSPACE_ROOT
        / "analysis_inputs"
        / "clipshots"
        / model
        / "preds_prob.pkl"
    )
    for model in ("PERSIST_s11", "cue_B1", "official")
}


def basename(key: str) -> str:
    return key.split("/")[-1].rsplit(".", 1)[0]


def evaluate_at_threshold(predictions, ground_truth, threshold, tolerance=2):
    gt_by_basename = {basename(key): key for key in ground_truth}
    true_positives = false_positives = false_negatives = 0
    for prediction_key, probabilities in predictions.items():
        key = basename(prediction_key)
        if key not in gt_by_basename:
            continue
        record = ground_truth[gt_by_basename[key]]
        transitions = [
            (int(start), int(end))
            for start, end in record.get("ast", []) + record.get("gst", [])
        ]
        probabilities = np.asarray(probabilities, dtype=np.float32).flatten()
        n_frames = int(record.get("frame_num", probabilities.shape[0]))
        gt_scenes = transitions_to_scenes(transitions, n_frames)
        predicted_scenes = transitions_to_scenes(
            probs_to_transitions(probabilities, threshold), n_frames)
        tp, fp, fn, _ = evaluate_scenes_and_diffs(
            gt_scenes, predicted_scenes, tolerance)
        true_positives += tp
        false_positives += fp
        false_negatives += fn
    recall = true_positives / max(true_positives + false_negatives, 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    return true_positives, false_positives, false_negatives, recall, precision


def find_threshold_for_recall(predictions, ground_truth, target, lo=0.01,
                              hi=0.99):
    for _ in range(22):
        midpoint = 0.5 * (lo + hi)
        _, _, _, recall, _ = evaluate_at_threshold(
            predictions, ground_truth, midpoint)
        if recall > target:
            lo = midpoint
        else:
            hi = midpoint
    return 0.5 * (lo + hi)


def main():
    with open(GT_PATH, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)
    predictions = {}
    for model, path in MODEL_PATHS.items():
        with open(path, "rb") as file:
            predictions[model] = pickle.load(file)

    print("=== ClipShots TEST: natural footage ===")
    fixed_results = {}
    for model in MODEL_PATHS:
        _, fp, _, recall, precision = evaluate_at_threshold(
            predictions[model], ground_truth, 0.5)
        fixed_results[model] = (recall, fp)
        print(
            f"{model:12s} threshold=0.500 recall={recall:.3f} "
            f"FP={fp} precision={precision:.3f}")

    target_recall = fixed_results["PERSIST_s11"][0]
    print(f"\nMatched to PERSIST recall={target_recall:.3f}:")
    for model in MODEL_PATHS:
        threshold = find_threshold_for_recall(
            predictions[model], ground_truth, target_recall)
        _, fp, _, recall, precision = evaluate_at_threshold(
            predictions[model], ground_truth, threshold)
        print(
            f"{model:12s} threshold={threshold:.3f} recall={recall:.3f} "
            f"FP={fp} precision={precision:.3f}")


if __name__ == "__main__":
    main()
