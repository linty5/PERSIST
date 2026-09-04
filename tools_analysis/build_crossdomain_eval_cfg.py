#!/usr/bin/env python3
"""Build a resolved cross-domain evaluation config.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mmaction2"))

from mmengine.config import Config  # noqa: E402


# Per-dataset path / cache table. All cross-domain runs share the same
# SBD matching rule (n_frames_miss_tolerance=2, center crop), so only the
# input split changes.
DATASETS = {
    "clipshots": {
        # In-domain ClipShots test split, exposed here so a single
        # cross-domain runner can produce a uniform preds_prob.pkl for the
        # in-domain set alongside the transfer sets (used by the multi-seed
        # unified-inference + ensemble pipeline).
        "ann_file_test": str(REPO_ROOT / "annotations/ClipShots/test.json"),
        "data_root_test_video": str(REPO_ROOT / "data/ClipShots/videos"),
        "cache_subdir": "clipshots",
    },
    "bbc": {
        "ann_file_test": str(REPO_ROOT / "annotations/BBC_PlanetEarth/test.json"),
        "data_root_test_video": str(REPO_ROOT / "data/BBC_PlanetEarth/videos"),
        "cache_subdir": "bbc",
    },
    "bbc_fixed": {
        # Same videos as ``bbc``, with the corrected annotation variant.
        "ann_file_test": str(REPO_ROOT / "annotations/BBC_fixed/test.json"),
        "data_root_test_video": str(REPO_ROOT / "data/BBC_PlanetEarth/videos"),
        "cache_subdir": "bbc_fixed",
    },
    "autoshot": {
        "ann_file_test": str(REPO_ROOT / "annotations/AutoShot/test.json"),
        "data_root_test_video": str(REPO_ROOT / "data/AutoShot/videos/test"),
        "cache_subdir": "autoshot",
    },
    "hist33": {
        # Full HISTORIAN test annotation.
        "ann_file_test": str(REPO_ROOT / "annotations/HISTORIAN/test.json"),
        "data_root_test_video": str(REPO_ROOT / "data/HISTORIAN/videos"),
        "cache_subdir": "hist33",
    },
    "hist_gst12": {
        # HISTORIAN subset containing at least one gradual shot transition.
        "ann_file_test": str(REPO_ROOT / "annotations/HISTORIAN/test_gst12.json"),
        "data_root_test_video": str(REPO_ROOT / "data/HISTORIAN/videos"),
        "cache_subdir": "hist_gst12",
    },
    "synth_diag": {
        # Synthetic diagnostic split with event-category labels.
        "ann_file_test": str(REPO_ROOT / "diagnostic/annotations/test_diag_all.json"),
        "data_root_test_video": str(REPO_ROOT / "diagnostic/videos"),
        "cache_subdir": "synth_diag",
    },
}


DATASET_CACHE_ROOT_TEMPLATE = str(REPO_ROOT / "data/_cache/crossdomain/{subdir}")


def parse_val_thr_from_ckpt(ckpt: str) -> float:
    m = re.search(
        r"valthr(?:eshold)?[_-]?(0?\.\d+|\d{1,3})",
        os.path.basename(ckpt),
    )
    if not m:
        return 0.50
    threshold = float(m.group(1))
    return threshold / 100.0 if threshold > 1.0 else threshold


def build_one_cfg(
    train_cfg_path: str,
    ckpt_path: str,
    dataset: str,
    output_cfg_path: str,
    *,
    save_siren_debug: bool = False,
    save_raw_probs: bool = False,
    save_clipshots_json: bool = False,
    save_pred_pickle: bool = True,
) -> dict:
    """Load training cfg, apply cross-domain overrides, dump resolved cfg.

    Returns a small status dict with the resolved threshold + work_dir.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset}; expected one of {list(DATASETS)}")
    ds = DATASETS[dataset]

    cfg = Config.fromfile(train_cfg_path)

    val_thr = parse_val_thr_from_ckpt(ckpt_path)
    eval_name = Path(output_cfg_path).stem
    # exp_root is the *parent directory* of the cfg file, not the cfg
    # path with .py stripped. The cfg lives inside exp_root with the
    # same name (e.g. ``exp_root/bbc_nb_oursfinal_a0_eval.py``), so
    # eval_outputs/ is a sibling of the cfg file.
    exp_root = str(Path(output_cfg_path).resolve().parent)
    eval_root = os.path.join(exp_root, "eval_outputs")

    # ---- experiment-identity fields ----
    cfg.experiment_name = eval_name
    cfg.exp_root = exp_root
    cfg.eval_root = eval_root
    cfg.work_dir = exp_root
    cfg.load_from = ckpt_path
    cfg.resume = False

    # ---- dataset overrides ----
    cfg.ann_file_test = ds["ann_file_test"]
    cfg.data_root_test_video = ds["data_root_test_video"]
    cache_dir = DATASET_CACHE_ROOT_TEMPLATE.format(subdir=ds["cache_subdir"])
    cfg.dataset_cache_root = cache_dir

    # Patch the resolved test_dataloader (the dict was built at base-load
    # time with the old paths, so we have to overwrite the keys directly).
    cfg.test_dataloader["dataset"]["ann_file"] = ds["ann_file_test"]
    cfg.test_dataloader["dataset"]["data_prefix"] = dict(video=ds["data_root_test_video"])
    cfg.test_dataloader["dataset"]["cache_dir"] = os.path.join(cache_dir, "test")

    # Long HISTORIAN videos use fewer, non-persistent workers to limit memory.
    if dataset in {"hist33", "hist_gst12"}:
        cfg.test_dataloader["num_workers"] = 4
        cfg.test_dataloader["persistent_workers"] = False

    # ---- evaluator overrides ----
    # Threshold for the val_best protocol (= ckpt source-val-best).
    cfg.best_threshold = val_thr
    cfg.test_evaluator["threshold"] = val_thr
    if "val_evaluator" in cfg:
        cfg.val_evaluator["threshold"] = val_thr

    # Output paths.
    cfg.test_evaluator["save_raw"] = bool(save_raw_probs)
    cfg.test_evaluator["save_clipshots_json"] = bool(save_clipshots_json)
    cfg.test_evaluator["save_pred_pickle"] = bool(save_pred_pickle)
    cfg.test_evaluator["save_siren_debug"] = bool(save_siren_debug)
    if save_raw_probs:
        cfg.test_evaluator["raw_out_dir"] = os.path.join(eval_root, "raw_probs")
    else:
        cfg.test_evaluator["raw_out_dir"] = ""
    if save_clipshots_json:
        cfg.test_evaluator["clipshots_json_path"] = os.path.join(eval_root, "preds_clipshots.json")
    else:
        cfg.test_evaluator["clipshots_json_path"] = ""
    if save_pred_pickle:
        cfg.test_evaluator["pred_pickle_path"] = os.path.join(eval_root, "preds_prob.pkl")
    else:
        cfg.test_evaluator["pred_pickle_path"] = ""
    if save_siren_debug:
        cfg.test_evaluator["siren_debug_path"] = os.path.join(eval_root, "siren_debug.pkl")
    else:
        cfg.test_evaluator["siren_debug_path"] = ""

    # ---- launcher + wandb off (eval-only) ----
    cfg.launcher = "none"
    cfg.use_wandb = False
    cfg.vis_backends = [dict(type="LocalVisBackend")]
    cfg.visualizer = dict(type="Visualizer", vis_backends=cfg.vis_backends)

    # ---- write ----
    Path(output_cfg_path).parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(output_cfg_path)
    Path(eval_root).mkdir(parents=True, exist_ok=True)

    return {
        "eval_cfg": output_cfg_path,
        "work_dir": exp_root,
        "eval_root": eval_root,
        "val_threshold": val_thr,
        "ann_file_test": ds["ann_file_test"],
        "data_root_test_video": ds["data_root_test_video"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cfg", required=True, help="ClipShots training config of the source ckpt.")
    ap.add_argument("--ckpt", required=True, help="path to .pth with valthreshold_NN tag in filename.")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--output_cfg", required=True, help="path to write the resolved eval cfg.")
    ap.add_argument("--save_siren_debug", action="store_true")
    ap.add_argument("--no_pred_pickle", action="store_true",
                    help="skip preds_prob.pkl (saves disk + IO, but blocks offline three-protocol recompute).")
    args = ap.parse_args()

    info = build_one_cfg(
        train_cfg_path=args.train_cfg,
        ckpt_path=args.ckpt,
        dataset=args.dataset,
        output_cfg_path=args.output_cfg,
        save_siren_debug=args.save_siren_debug,
        save_pred_pickle=not args.no_pred_pickle,
    )
    print(f"[build_crossdomain_eval_cfg] wrote {info['eval_cfg']}  val_thr={info['val_threshold']:.2f}")


if __name__ == "__main__":
    main()
