"""Synchronize the training epoch with the PERSIST head.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""

from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.model.wrappers import is_model_wrapper


@HOOKS.register_module()
class SBDSirenEpochHook(Hook):
    """Sync runner.epoch into model.cls_head._runtime_epoch."""

    priority = 'LOWEST'

    @staticmethod
    def _unwrap_model(runner: Runner):
        model = runner.model
        # unwrap common MMEngine wrappers
        if is_model_wrapper(model):
            model = model.module
        # fallback for other wrappers
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "model"):  # some wrappers may store the real model here
            try:
                if hasattr(model.model, "cls_head"):
                    model = model.model
            except Exception:
                pass
        return model

    def _get_epoch(self, runner, stage: str) -> int:
        ep = int(getattr(runner, "epoch", 0))
        if stage in ("val", "test"):
            return ep
        else:
            return ep + 1

    def _set_head_epoch(self, runner: Runner, stage: str) -> None:
        model = self._unwrap_model(runner)
        head = getattr(model, "cls_head", None)

        ep = self._get_epoch(runner, stage)

        if head is not None and hasattr(head, "set_epoch"):
            head.set_epoch(ep)

            # Limit routine hook logging to the first few epochs.
            if ep <= 3:
                runner.logger.info(
                    "[SBDSirenEpochHook] set head epoch=%d (head=%s)",
                    ep, head.__class__.__name__
                )
        else:
            if ep <= 3:
                runner.logger.warning(
                    "[SBDSirenEpochHook] head not found or no set_epoch (model=%s)",
                    model.__class__.__name__
                )

    def before_train_epoch(self, runner: Runner) -> None:
        self._set_head_epoch(runner, "train")

    def before_val_epoch(self, runner: Runner) -> None:
        self._set_head_epoch(runner, "val")

    def before_test_epoch(self, runner: Runner) -> None:
        self._set_head_epoch(runner, "test")
