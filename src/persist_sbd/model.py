"""Build the PERSIST inference network from its configuration.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .nets import SBDOursHead, SlowFastTransNetHybridBackbone

# Restore tuple-valued parameters from JSON lists.
_TUPLE_KEYS = {"input_size", "trcs_offsets"}


def _tupled(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in cfg.items():
        if k == "type":
            continue
        if isinstance(v, dict):
            v = _tupled(v)
        elif k in _TUPLE_KEYS and isinstance(v, list):
            v = tuple(v)
        out[k] = v
    return out


class PERSIST(nn.Module):
    """Map RGB clips [B, 3, T, H, W] in 0..255 to probabilities and persistence gates."""

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = SlowFastTransNetHybridBackbone(**_tupled(model_cfg["backbone"]))
        self.cls_head = SBDOursHead(**_tupled(model_cfg["cls_head"]))

    def forward(self, clips: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = clips.float() / 255.0
        return self.cls_head(self.backbone(x))
