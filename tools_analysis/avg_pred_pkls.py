#!/usr/bin/env python3
"""Average matching per-frame prediction arrays from multiple pickle files.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""
import argparse
import pickle
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--in", dest="inputs", required=True, nargs="+",
                    help="2+ preds_prob.pkl paths to average")
    args = ap.parse_args()
    if len(args.inputs) < 2:
        ap.error("need at least 2 input pkls to average")

    dicts = []
    for p in args.inputs:
        with open(p, "rb") as f:
            dicts.append(pickle.load(f))
        print(f"[avg] loaded {p}  n_keys={len(dicts[-1])}")

    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    keys = sorted(keys)
    print(f"[avg] common video ids across {len(dicts)} inputs: {len(keys)}")
    if not keys:
        sys.exit("[avg] FATAL: no common video ids across inputs")

    out = {}
    skipped = 0
    for k in keys:
        arrs = [np.asarray(d[k], dtype=np.float64).reshape(-1) for d in dicts]
        lens = {a.shape[0] for a in arrs}
        if len(lens) != 1:
            # length mismatch for this video across seeds -> skip, report.
            skipped += 1
            continue
        out[k] = (np.mean(np.stack(arrs, axis=0), axis=0)).astype(np.float32).reshape(-1, 1)
    if skipped:
        print(f"[avg] WARNING: skipped {skipped} ids with per-seed length mismatch")

    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"[avg] wrote {args.out}  n_keys={len(out)}")


if __name__ == "__main__":
    main()
