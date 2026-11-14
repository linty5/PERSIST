"""Dual-rate TransNet backbone with configurable pathway adapters.

Adapted from github.com/linty5/PERSIST (MMAction2 fork).
Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .transnetv2_dualrate import TransNetV2DualRateBackbone
from .transnetv2_style import StackedDDCNNV2


class TransNetResidualAdapter(nn.Module):
    """Apply shape-preserving DDCNN blocks with residual or replacement fusion."""

    def __init__(
        self,
        in_channels: int,
        n_blocks: int = 1,
        batch_norm: bool = True,
        separable: bool = True,
        shortcut: bool = True,
        fusion_mode: str = "residual",
        fusion_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if int(in_channels) % 4 != 0:
            raise ValueError(f'in_channels must be divisible by 4 for TransNetResidualAdapter, got {in_channels}')
        if fusion_mode not in ("residual", "replace"):
            raise ValueError(f'Unsupported fusion_mode={fusion_mode}')

        self.fusion_mode = str(fusion_mode)
        self.fusion_scale = float(fusion_scale)
        self.block = StackedDDCNNV2(
            in_filters=int(in_channels),
            n_blocks=int(n_blocks),
            filters=int(in_channels) // 4,
            shortcut=shortcut,
            pool_type="none",
            batch_norm=batch_norm,
            separable=separable,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.fusion_mode == "replace":
            return y
        return x + self.fusion_scale * y


class SlowFastTransNetHybridBackbone(nn.Module):
    """Combine dual-rate TransNet features and optional residual adapters.

    Only backbone_mode="transnet" is supported; SlowFast arguments remain for
    configuration compatibility. Outputs are (slow, fast) or fast, with optional
    stage features and resized RGB.
    """

    def __init__(
        self,
        backbone_mode: str = "slowfast",
        output_mode: str = "tuple",
        enable_slow_path: bool = True,
        return_transnet_block_features: bool = False,
        return_transnet_resized_inputs: bool = False,
        slowfast_cfg: Optional[Dict] = None,
        slow_out_channels: int = 2048,
        fast_out_channels: int = 512,
        transnet_cfg: Optional[Dict] = None,
        enable_transnet_fast_adapter: bool = False,
        enable_transnet_slow_adapter: bool = False,
        transnet_fast_adapter_blocks: int = 1,
        transnet_slow_adapter_blocks: int = 1,
        transnet_adapter_batch_norm: bool = True,
        transnet_adapter_separable: bool = True,
        transnet_adapter_shortcut: bool = True,
        transnet_adapter_fusion_mode: str = "residual",
        transnet_adapter_fusion_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if backbone_mode not in ("slowfast", "transnet"):
            raise ValueError(f'Unsupported backbone_mode={backbone_mode}')
        if output_mode not in ("tuple", "fast"):
            raise ValueError(f'Unsupported output_mode={output_mode}')

        self.backbone_mode = str(backbone_mode)
        self.output_mode = str(output_mode)
        self.enable_slow_path = bool(enable_slow_path)
        self.return_transnet_block_features = bool(return_transnet_block_features)
        self.return_transnet_resized_inputs = bool(return_transnet_resized_inputs)

        self.enable_transnet_fast_adapter = bool(enable_transnet_fast_adapter)
        self.enable_transnet_slow_adapter = bool(enable_transnet_slow_adapter) and bool(enable_slow_path)

        self.slowfast_backbone = None
        self.transnet_backbone = None

        if self.backbone_mode == "slowfast":
            raise NotImplementedError("persist_sbd includes only backbone_mode='transnet'")
        else:
            cfg = dict(
                in_channels=3,
                fast_base_filters=16,
                fast_num_stages=3,
                fast_blocks_per_stage=2,
                enable_slow=bool(enable_slow_path),
                slow_base_filters=32,
                slow_num_stages=3,
                slow_blocks_per_stage=2,
                slow_temporal_stride=8,
                use_lateral=True,
                lateral_ratio=0.25,
                pool_type='avg',
                batch_norm=True,
                separable=True,
                shortcut=True,
                resize_input=True,
                input_size=(27, 48),
            )
            if transnet_cfg is not None:
                cfg.update(transnet_cfg)
            cfg["enable_slow"] = bool(enable_slow_path)
            cfg["return_fast_stage_features"] = bool(return_transnet_block_features)
            cfg["return_resized_inputs"] = bool(return_transnet_resized_inputs)
            self.transnet_backbone = TransNetV2DualRateBackbone(**cfg)
            slow_channels = (
                self.transnet_backbone.slow_out_channels
                if enable_slow_path
                else self.transnet_backbone.fast_out_channels
            )
            self.slow_out_channels = int(slow_channels)
            self.fast_out_channels = int(self.transnet_backbone.fast_out_channels)

        self.fast_adapter = None
        if self.enable_transnet_fast_adapter:
            self.fast_adapter = TransNetResidualAdapter(
                in_channels=self.fast_out_channels,
                n_blocks=transnet_fast_adapter_blocks,
                batch_norm=transnet_adapter_batch_norm,
                separable=transnet_adapter_separable,
                shortcut=transnet_adapter_shortcut,
                fusion_mode=transnet_adapter_fusion_mode,
                fusion_scale=transnet_adapter_fusion_scale,
            )

        self.slow_adapter = None
        if self.enable_transnet_slow_adapter:
            self.slow_adapter = TransNetResidualAdapter(
                in_channels=self.slow_out_channels,
                n_blocks=transnet_slow_adapter_blocks,
                batch_norm=transnet_adapter_batch_norm,
                separable=transnet_adapter_separable,
                shortcut=transnet_adapter_shortcut,
                fusion_mode=transnet_adapter_fusion_mode,
                fusion_scale=transnet_adapter_fusion_scale,
            )

    def forward(self, inputs: torch.Tensor):
        if self.backbone_mode == "slowfast":
            x_slow, x_fast = self.slowfast_backbone(inputs)
            if not self.enable_slow_path:
                x_slow = x_fast
        else:
            out = self.transnet_backbone(inputs)
            block_features = None
            resized_inputs = None
            if isinstance(out, tuple):
                if len(out) == 4:
                    x_slow, x_fast, block_features, resized_inputs = out
                elif len(out) == 3:
                    x_slow, x_fast, third = out
                    if isinstance(third, (list, tuple)):
                        block_features = third
                    else:
                        resized_inputs = third
                else:
                    x_slow, x_fast = out
            else:
                x_fast = out
                x_slow = out

        if self.fast_adapter is not None:
            x_fast = self.fast_adapter(x_fast)
        if self.slow_adapter is not None:
            x_slow = self.slow_adapter(x_slow)

        if self.output_mode == "fast":
            if self.backbone_mode == "transnet":
                if block_features is not None and resized_inputs is not None:
                    return (x_fast, block_features, resized_inputs)
                if block_features is not None:
                    return (x_fast, block_features)
                if resized_inputs is not None:
                    return (x_fast, resized_inputs)
            return x_fast
        if self.backbone_mode == "transnet":
            if block_features is not None and resized_inputs is not None:
                return ((x_slow, x_fast), block_features, resized_inputs)
            if block_features is not None:
                return ((x_slow, x_fast), block_features)
            if resized_inputs is not None:
                return ((x_slow, x_fast), resized_inputs)
        return (x_slow, x_fast)
