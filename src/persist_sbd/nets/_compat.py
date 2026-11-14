"""Minimal head attributes and weight initialization without MMAction2.

Adapted from MMAction2 and MMEngine interfaces.
Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

from typing import Dict, Optional, Tuple, Union

import torch.nn as nn


def normal_init(module: nn.Module, mean: float = 0.0, std: float = 1.0, bias: float = 0.0) -> None:
    """Match mmengine.model.weight_init.normal_init."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class BaseHead(nn.Module):
    """Provide MMAction2 head attributes without constructing training losses."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_cls: Optional[Dict] = None,
        multi_class: bool = False,
        label_smooth_eps: float = 0.0,
        topk: Union[int, Tuple[int]] = (1, 5),
        average_clips: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = None
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        self.average_clips = average_clips
        self.topk = (topk,) if isinstance(topk, int) else topk
