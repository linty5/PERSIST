#!/usr/bin/env python3
"""Generate an evaluation config from a training config and checkpoint.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""

import argparse
import os
import re
from pathlib import Path
from typing import Tuple


def parse_thr_from_ckpt(ckpt_path: str) -> float:
    m = re.search(
        r'valthr(?:eshold)?[_-]?(0?\.\d+|\d{1,3})',
        os.path.basename(ckpt_path),
    )
    if not m:
        return 0.50
    threshold = float(m.group(1))
    return threshold / 100.0 if threshold > 1.0 else threshold


def replace_or_append(text: str, key: str, new_line: str) -> str:
    pattern = rf'^(?P<prefix>\s*{re.escape(key)}\s*=\s*).*$'
    if re.search(pattern, text, flags=re.M):
        return re.sub(pattern, new_line, text, flags=re.M)
    return new_line + "\n" + text


def disable_eval_outputs(text: str) -> str:
    text = replace_or_append(text, "eval_root", "eval_root = exp_root")
    text = replace_or_append(text, "save_raw", "    save_raw=False,")
    text = replace_or_append(text, "save_clipshots_json", "    save_clipshots_json=False,")
    text = replace_or_append(text, "save_pred_pickle", "    save_pred_pickle=False,")
    text = replace_or_append(text, "save_siren_debug", "    save_siren_debug=False,")
    return text


def inject_threshold_into_evaluators(text: str, thr: float) -> str:
    """Set evaluator thresholds directly and idempotently."""
    def _set_in_block(t: str, key: str) -> Tuple[str, bool]:
        marker = f"  # injected by gen_eval_cfg.py: {key}"
        injected_line = f"    threshold={thr:.4f},{marker}"
        lines = t.splitlines(keepends=True)
        opener = re.compile(rf"^{re.escape(key)}\s*=\s*dict\(")
        for start, line in enumerate(lines):
            if opener.search(line) is None:
                continue
            if "threshold=" in line:
                lines[start] = re.sub(
                    r"threshold=[-\d.eE]+",
                    f"threshold={thr:.4f}",
                    line,
                    count=1,
                )
                return "".join(lines), True

            depth = 0
            end = start
            for end in range(start, len(lines)):
                code = lines[end].split("#", 1)[0]
                depth += code.count("(") - code.count(")")
                if depth == 0:
                    break
            for index in range(start + 1, end + 1):
                if re.match(r"^\s+threshold\s*=", lines[index]):
                    newline = "\n" if lines[index].endswith("\n") else ""
                    lines[index] = injected_line + newline
                    return "".join(lines), True
            lines.insert(start + 1, injected_line + "\n")
            return "".join(lines), True
        return t, False

    text, t_ok = _set_in_block(text, "test_evaluator")
    text, v_ok = _set_in_block(text, "val_evaluator")

    appended_blocks = []
    if not t_ok:
        appended_blocks.append(
            f"\n# auto-appended by gen_eval_cfg.py: test_evaluator threshold override\n"
            f"test_evaluator = dict(threshold={thr:.4f})\n"
        )
    if not v_ok:
        appended_blocks.append(
            f"\n# auto-appended by gen_eval_cfg.py: val_evaluator threshold override\n"
            f"val_evaluator = dict(threshold={thr:.4f})\n"
        )
    if appended_blocks:
        text = text + "\n" + "".join(appended_blocks)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cfg", required=True)
    ap.add_argument("--eval_cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--disable_eval_outputs",
        action="store_true",
        help="Disable raw/json/pickle/debug saves in the generated eval cfg.",
    )
    args = ap.parse_args()

    train_cfg = Path(args.train_cfg)
    eval_cfg = Path(args.eval_cfg)
    ckpt = args.ckpt

    thr = parse_thr_from_ckpt(ckpt)

    txt = train_cfg.read_text(encoding="utf-8")

    eval_exp_name = eval_cfg.stem
    txt = replace_or_append(txt, "experiment_name", f"experiment_name = '{eval_exp_name}'")

    # disable wandb
    txt = replace_or_append(txt, "use_wandb", "use_wandb = False")

    # set best_threshold (still kept so any downstream variable reference
    # uses the right value; not sufficient on its own, see next call)
    txt = replace_or_append(txt, "best_threshold", f"best_threshold = {thr:.2f}")

    # Inject threshold INTO the evaluator dict overrides themselves. This
    # is the actual fix for the "sbd_val_f1 stuck at 0.50" issue.
    txt = inject_threshold_into_evaluators(txt, thr)

    txt = replace_or_append(txt, "resume", "resume = False")

    if args.disable_eval_outputs:
        txt = disable_eval_outputs(txt)

    eval_cfg.parent.mkdir(parents=True, exist_ok=True)
    eval_cfg.write_text(txt, encoding="utf-8")
    print(
        f"[gen_eval_cfg] wrote {eval_cfg}  "
        f"(best_threshold={thr:.2f} from {os.path.basename(ckpt)}, "
        f"disable_eval_outputs={args.disable_eval_outputs})"
    )


if __name__ == "__main__":
    main()
