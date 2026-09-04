"""Append the validation-selected threshold to best-checkpoint names.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""

import os
import re
import glob
import shutil
from typing import Optional, Dict, Any, Set

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.dist import is_main_process


@HOOKS.register_module()
class SBDAppendValThresholdToBestHook(Hook):

    priority = 'LOWEST'

    def __init__(
        self,
        threshold_key: str = 'sbd_best_threshold',
        scale: int = 100,
        best_glob: str = 'best_*.pth',
        retry: int = 1,
    ):
        self.threshold_key = threshold_key
        self.scale = scale
        self.best_glob = best_glob
        self.retry = retry
        self._pending_thr_int: Optional[int] = None
        self._best_before_val: Set[str] = set()

    @staticmethod
    def _metric_prefix(path: str) -> str:
        """Return the metric part of a MMEngine best checkpoint name."""
        base = os.path.basename(path)
        match = re.match(r'best_(.+?)_epoch_\d+\.pth$', base)
        if match:
            return match.group(1)
        match = re.match(r'best_(.+?)_epoch_\d+_valthreshold_\d+\.pth$', base)
        if match:
            return match.group(1)
        return 'unknown'

    def before_val_epoch(self, runner) -> None:
        if not is_main_process():
            return

        work_dir = runner.work_dir
        cands = glob.glob(os.path.join(work_dir, self.best_glob))
        cands = [
            p for p in cands
            if os.path.isfile(p) and 'valthreshold_' not in os.path.basename(p)
        ]
        self._best_before_val = set(os.path.abspath(p) for p in cands)

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, Any]] = None) -> None:
        if not is_main_process():
            return

        if not metrics or self.threshold_key not in metrics:
            runner.logger.info(
                f'[SBDAppendValThresholdToBestHook] no threshold key "{self.threshold_key}" in metrics, skip.'
            )
            self._pending_thr_int = None
            return

        thr = float(metrics[self.threshold_key])
        self._pending_thr_int = int(round(thr * self.scale))
        thr_int = self._pending_thr_int

        runner.logger.info(
            f'[SBDAppendValThresholdToBestHook] cached {self.threshold_key}={thr:.4f} '
            f'-> valthreshold_{thr_int:02d}'
        )

        if thr_int is None:
            return

        work_dir = runner.work_dir

        def _do_once() -> bool:
            cands = glob.glob(os.path.join(work_dir, self.best_glob))
            cands = [
                p for p in cands
                if os.path.isfile(p) and 'valthreshold_' not in os.path.basename(p)
            ]

            if not cands:
                runner.logger.info(
                    '[SBDAppendValThresholdToBestHook] no base best ckpt '
                    f'matching "{self.best_glob}" in {work_dir}, skip.'
                )
                return True

            current_best = set(os.path.abspath(p) for p in cands)
            new_best = sorted(
                current_best - self._best_before_val,
                key=lambda p: (self._metric_prefix(p), os.path.getmtime(p)),
            )

            if not new_best:
                runner.logger.info(
                    '[SBDAppendValThresholdToBestHook] no new best checkpoint generated in this val epoch, skip.'
                )
                return True

            all_tagged_best = [
                os.path.abspath(p)
                for p in glob.glob(os.path.join(work_dir, 'best_*_valthreshold_*.pth'))
                if os.path.isfile(p)
            ]

            for latest in new_best:
                base = os.path.basename(latest)
                latest_abs = os.path.abspath(latest)
                metric_prefix = self._metric_prefix(latest_abs)
                new_path = re.sub(r'\.pth$', f'_valthreshold_{thr_int:02d}.pth', latest_abs)
                new_base = os.path.basename(new_path)

                for p in all_tagged_best:
                    if p == new_path:
                        continue
                    if self._metric_prefix(p) != metric_prefix:
                        continue
                    try:
                        os.remove(p)
                        runner.logger.info(
                            f'[SBDAppendValThresholdToBestHook] removed stale tagged best '
                            f'for {metric_prefix}: {os.path.basename(p)}'
                        )
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        runner.logger.warning(
                            f'[SBDAppendValThresholdToBestHook] failed to remove stale tagged best "{p}": {e}'
                        )

                if os.path.exists(new_path):
                    try:
                        os.remove(new_path)
                        runner.logger.info(
                            f'[SBDAppendValThresholdToBestHook] removed existing tagged target before refresh: '
                            f'{new_base}'
                        )
                    except Exception as e:
                        runner.logger.warning(
                            '[SBDAppendValThresholdToBestHook] failed to remove '
                            f'existing tagged target "{new_path}": {e}'
                        )

                try:
                    shutil.copy2(latest_abs, new_path)
                    runner.logger.info(
                        f'[SBDAppendValThresholdToBestHook] copied tagged best: {base} -> {new_base}'
                    )
                except FileNotFoundError:
                    return False
                except Exception as e:
                    runner.logger.warning(
                        f'[SBDAppendValThresholdToBestHook] copy failed: {e}'
                    )
            return True

        ok = _do_once()
        if not ok:
            for _ in range(self.retry):
                ok = _do_once()
                if ok:
                    break
            if not ok:
                runner.logger.warning(
                    '[SBDAppendValThresholdToBestHook] still hit FileNotFound race after retry; skip this epoch.'
                )
