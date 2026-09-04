"""TransNetV2-style convolutional building blocks used by PERSIST.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at

Portions adapted from TransNetV2.
Copyright (c) 2020 Tomáš Souček.
Licensed under the MIT License; see THIRD_PARTY_LICENSES/TransNetV2-LICENSE.
"""

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS


class Conv3DConfigurable(nn.Module):
    """TransNet-style configurable 3D conv.

    This is adapted from the official TransNetV2 design, but kept trainable
    inside the current MMAction pipeline.
    """

    def __init__(
        self,
        in_filters: int,
        filters: int,
        dilation_rate: int,
        separable: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        if separable:
            conv1 = nn.Conv3d(
                in_filters,
                2 * filters,
                kernel_size=(1, 3, 3),
                dilation=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            )
            conv2 = nn.Conv3d(
                2 * filters,
                filters,
                kernel_size=(3, 1, 1),
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 0, 0),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(
                in_filters,
                filters,
                kernel_size=3,
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 1, 1),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class DilatedDCNNV2(nn.Module):
    """Multi-dilation local temporal block from TransNetV2."""

    def __init__(
        self,
        in_filters: int,
        filters: int,
        batch_norm: bool = True,
        activation=None,
        separable: bool = True,
    ) -> None:
        super().__init__()
        self.conv_1 = Conv3DConfigurable(
            in_filters, filters, dilation_rate=1, separable=separable, use_bias=not batch_norm
        )
        self.conv_2 = Conv3DConfigurable(
            in_filters, filters, dilation_rate=2, separable=separable, use_bias=not batch_norm
        )
        self.conv_4 = Conv3DConfigurable(
            in_filters, filters, dilation_rate=4, separable=separable, use_bias=not batch_norm
        )
        self.conv_8 = Conv3DConfigurable(
            in_filters, filters, dilation_rate=8, separable=separable, use_bias=not batch_norm
        )
        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                self.conv_1(inputs),
                self.conv_2(inputs),
                self.conv_4(inputs),
                self.conv_8(inputs),
            ],
            dim=1,
        )
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class StackedDDCNNV2(nn.Module):
    """Stacked TransNetV2 dilated blocks plus spatial pooling."""

    def __init__(
        self,
        in_filters: int,
        n_blocks: int,
        filters: int,
        shortcut: bool = True,
        pool_type: str = "avg",
        batch_norm: bool = True,
        separable: bool = True,
    ) -> None:
        super().__init__()
        assert pool_type in ("avg", "max", "none")
        self.shortcut = bool(shortcut)
        self.blocks = nn.ModuleList(
            [
                DilatedDCNNV2(
                    in_filters if i == 0 else filters * 4,
                    filters,
                    batch_norm=batch_norm,
                    activation=F.relu if i != (n_blocks - 1) else None,
                    separable=separable,
                )
                for i in range(n_blocks)
            ]
        )
        if pool_type == "avg":
            self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
        elif pool_type == "max":
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        else:
            self.pool = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        shortcut = None
        for block in self.blocks:
            x = block(x)
            if shortcut is None:
                shortcut = x
        x = F.relu(x)
        if self.shortcut and shortcut is not None:
            x = x + shortcut
        x = self.pool(x)
        return x


@MODELS.register_module()
class TransNetV2StyleBackbone(nn.Module):
    """Trainable TransNet-style feature extractor.

    The goal is not to reproduce the full official TransNetV2 training stack,
    but to reuse its local multi-scale temporal block design as a controllable
    feature extractor inside the current SBD framework.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 16,
        num_stages: int = 3,
        blocks_per_stage: int = 2,
        pool_type: str = "avg",
        batch_norm: bool = True,
        separable: bool = True,
        shortcut: bool = True,
        resize_input: bool = True,
        input_size: Union[Tuple[int, int], Sequence[int]] = (27, 48),
        return_block_features: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_filters = int(base_filters)
        self.num_stages = int(num_stages)
        self.blocks_per_stage = int(blocks_per_stage)
        self.resize_input = bool(resize_input)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.return_block_features = bool(return_block_features)

        blocks = []
        in_filters = self.in_channels
        for stage_idx in range(self.num_stages):
            filters = self.base_filters * (2 ** stage_idx)
            blocks.append(
                StackedDDCNNV2(
                    in_filters=in_filters,
                    n_blocks=self.blocks_per_stage,
                    filters=filters,
                    shortcut=shortcut,
                    pool_type=pool_type,
                    batch_norm=batch_norm,
                    separable=separable,
                )
            )
            in_filters = filters * 4
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_filters

        # Compute output spatial dimensions for downstream use (e.g. head dim calculation).
        # Each stage applies AvgPool3d/MaxPool3d with kernel (1,2,2), halving H and W.
        h, w = self.input_size
        for _ in range(self.num_stages):
            h = h // 2
            w = w // 2
        self.out_spatial_h = h
        self.out_spatial_w = w
        # Per-stage output channels: base_filters * 2^i * 4 (four dilations concatenated).
        self.stage_out_channels = [self.base_filters * (2 ** i) * 4 for i in range(self.num_stages)]
        # Total input channels for FrameSimilarity (cat of all stage outputs after spatial mean).
        self.frame_similarity_total_in = sum(self.stage_out_channels)
        # Flattened spatial feature dim from the last stage (used when use_flatten_map=True).
        self.flatten_feat_dim = self.out_channels * h * w

    def _resize_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if not self.resize_input:
            return x
        b, c, t, h, w = x.shape
        target_h, target_w = self.input_size
        if h == target_h and w == target_w:
            return x
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x2d = F.interpolate(x2d, size=(target_h, target_w), mode="bilinear", align_corners=False)
        x = x2d.view(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()
        return x

    def forward(self, inputs: torch.Tensor):
        x = self._resize_spatial(inputs)
        block_features = []
        for block in self.blocks:
            x = block(x)
            block_features.append(x)
        if self.return_block_features:
            return x, tuple(block_features)
        return x
