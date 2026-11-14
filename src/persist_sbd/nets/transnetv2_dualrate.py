"""Dual-rate TransNetV2 features with fast-to-slow lateral connections.

Adapted from github.com/linty5/PERSIST (MMAction2 fork).
Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transnetv2_style import StackedDDCNNV2


class TransNetV2DualRateBackbone(nn.Module):
    """Extract fast features at T and slow features at T // slow_temporal_stride.

    Lateral connections project fast stage i into slow stage i+1. Without
    a slow pathway, return (fast, fast) and disable use_slow_logits in the head.
    The default three-stage layout returns 256 fast and 512 slow channels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        fast_base_filters: int = 16,
        fast_num_stages: int = 3,
        fast_blocks_per_stage: int = 2,
        enable_slow: bool = True,
        slow_base_filters: int = 32,
        slow_num_stages: int = 3,
        slow_blocks_per_stage: int = 2,
        slow_temporal_stride: int = 8,
        use_lateral: bool = True,
        lateral_ratio: float = 0.25,
        pool_type: str = "avg",
        batch_norm: bool = True,
        separable: bool = True,
        shortcut: bool = True,
        resize_input: bool = True,
        input_size: Union[Tuple[int, int], Sequence[int]] = (27, 48),
        return_fast_stage_features: bool = False,
        return_resized_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.fast_base_filters = int(fast_base_filters)
        self.fast_num_stages = int(fast_num_stages)
        self.fast_blocks_per_stage = int(fast_blocks_per_stage)
        self.enable_slow = bool(enable_slow)
        self.slow_base_filters = int(slow_base_filters)
        self.slow_num_stages = int(slow_num_stages)
        self.slow_blocks_per_stage = int(slow_blocks_per_stage)
        self.slow_temporal_stride = int(slow_temporal_stride)
        self.use_lateral = bool(use_lateral) and bool(enable_slow)
        self.lateral_ratio = float(lateral_ratio)
        self.resize_input = bool(resize_input)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.return_fast_stage_features = bool(return_fast_stage_features)
        self.return_resized_inputs = bool(return_resized_inputs)

        fast_stages: List[nn.Module] = []
        fast_in = self.in_channels
        for i in range(self.fast_num_stages):
            f = self.fast_base_filters * (2 ** i)
            fast_stages.append(
                StackedDDCNNV2(
                    in_filters=fast_in,
                    n_blocks=self.fast_blocks_per_stage,
                    filters=f,
                    shortcut=shortcut,
                    pool_type=pool_type,
                    batch_norm=batch_norm,
                    separable=separable,
                )
            )
            fast_in = f * 4
        self.fast_stages = nn.ModuleList(fast_stages)
        self.fast_out_channels = fast_in

        self._fast_stage_ch = [
            self.fast_base_filters * (2 ** i) * 4
            for i in range(self.fast_num_stages)
        ]

        if self.enable_slow:
            slow_stages: List[nn.Module] = []
            lat_convs: List[nn.Module] = []
            lat_stage_indices: List[int] = []
            slow_in = self.in_channels

            for i in range(self.slow_num_stages):
                lat_ch = 0
                # Fast stage i-1 contributes lateral features to slow stage i.
                if self.use_lateral and i > 0 and (i - 1) < self.fast_num_stages:
                    fast_ch = self._fast_stage_ch[i - 1]
                    lat_ch = max(1, int(fast_ch * self.lateral_ratio))
                    lat_convs.append(
                        nn.Conv3d(fast_ch, lat_ch, kernel_size=1, bias=False)
                    )
                    lat_stage_indices.append(i)

                f = self.slow_base_filters * (2 ** i)
                slow_stages.append(
                    StackedDDCNNV2(
                        in_filters=slow_in + lat_ch,
                        n_blocks=self.slow_blocks_per_stage,
                        filters=f,
                        shortcut=shortcut,
                        pool_type=pool_type,
                        batch_norm=batch_norm,
                        separable=separable,
                    )
                )
                slow_in = f * 4

            self.slow_stages = nn.ModuleList(slow_stages)
            self.lat_convs = nn.ModuleList(lat_convs)
            self._lat_stage_indices = lat_stage_indices
            self.slow_out_channels = slow_in
        else:
            self.slow_stages = None
            self.lat_convs = None
            self._lat_stage_indices = []
            self.slow_out_channels = 0

        h, w = self.input_size
        for _ in range(self.fast_num_stages):
            h //= 2
            w //= 2
        self.out_spatial_h = h
        self.out_spatial_w = w

    def _resize_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if not self.resize_input:
            return x
        b, c, t, h, w = x.shape
        th, tw = self.input_size
        if h == th and w == tw:
            return x
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x2d = F.interpolate(x2d, size=(th, tw), mode="bilinear", align_corners=False)
        return x2d.view(b, t, c, th, tw).permute(0, 2, 1, 3, 4).contiguous()

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._resize_spatial(inputs)

        x_fast = x
        fast_stage_outs: List[torch.Tensor] = []
        for stage in self.fast_stages:
            x_fast = stage(x_fast)
            fast_stage_outs.append(x_fast)

        if not self.enable_slow:
            # Aliased outputs preserve the dual-pathway interface in fast-only mode.
            if self.return_fast_stage_features and self.return_resized_inputs:
                return (x_fast, x_fast, tuple(fast_stage_outs), x)
            if self.return_fast_stage_features:
                return (x_fast, x_fast, tuple(fast_stage_outs))
            if self.return_resized_inputs:
                return (x_fast, x_fast, x)
            return (x_fast, x_fast)

        # Downsample time once before the slow stages, preserving spatial size.
        t_slow = max(1, inputs.shape[2] // self.slow_temporal_stride)
        x_slow = F.adaptive_avg_pool3d(x, (t_slow, x.shape[3], x.shape[4]))

        lat_iter = iter(self.lat_convs)
        for i, stage in enumerate(self.slow_stages):
            if i in self._lat_stage_indices:
                lat_conv = next(lat_iter)
                # Both branches have passed i spatial pools, so lateral features align.
                fast_prev = fast_stage_outs[i - 1]
                lat = F.adaptive_avg_pool3d(
                    fast_prev,
                    (x_slow.shape[2], fast_prev.shape[3], fast_prev.shape[4]),
                )
                lat = lat_conv(lat)
                x_slow = torch.cat([x_slow, lat], dim=1)
            x_slow = stage(x_slow)

        if self.return_fast_stage_features and self.return_resized_inputs:
            return (x_slow, x_fast, tuple(fast_stage_outs), x)
        if self.return_fast_stage_features:
            return (x_slow, x_fast, tuple(fast_stage_outs))
        if self.return_resized_inputs:
            return (x_slow, x_fast, x)
        return (x_slow, x_fast)
