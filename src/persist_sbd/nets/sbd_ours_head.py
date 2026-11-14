"""Boundary prediction with latent-state and optional TransNetV2 descriptor cues.

Adapted from github.com/linty5/PERSIST (MMAction2 fork).
Author: Tingyu Lin, Computer Vision Lab, TU Wien.

Portions adapted from TransNetV2.
Copyright (c) 2020 Tomáš Souček.
Licensed under the MIT License; see THIRD_PARTY_LICENSES/TransNetV2-LICENSE.
"""

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ._compat import normal_init
from torch import Tensor

from .sbd_slowfast_siren_trcs_head import SlowFastSBDSirenTRCSHead


class _OfficialFrameSimilarity(nn.Module):
    """TransNetV2 frame similarity with a ReLU-activated output projection."""

    def __init__(
        self,
        in_filters: int,
        similarity_dim: int = 128,
        lookup_window: int = 101,
        output_dim: int = 128,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        assert int(in_filters) > 0, f"FrameSimilarity requires in_filters > 0, got {in_filters}"
        assert int(lookup_window) % 2 == 1, "lookup_window must be odd"
        self.projection = nn.Linear(int(in_filters), int(similarity_dim), bias=use_bias)
        self.fc = nn.Linear(int(lookup_window), int(output_dim))
        self.lookup_window = int(lookup_window)

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        x = torch.cat([torch.mean(item, dim=[3, 4]) for item in inputs], dim=1)
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=2)

        b, t = x.shape[0], x.shape[1]
        sims = torch.bmm(x, x.transpose(1, 2))
        pad = (self.lookup_window - 1) // 2
        sims_padded = F.pad(sims, [pad, pad])

        bi = torch.arange(b, device=x.device).view(b, 1, 1).expand(b, t, self.lookup_window)
        ti = torch.arange(t, device=x.device).view(1, t, 1).expand(b, t, self.lookup_window)
        li = (
            torch.arange(self.lookup_window, device=x.device)
            .view(1, 1, self.lookup_window)
            .expand(b, t, self.lookup_window)
            + ti
        )
        gathered = sims_padded[bi, ti, li]
        return F.relu(self.fc(gathered))


class _OfficialColorHistograms(nn.Module):
    """TransNetV2 RGB histogram similarity with an optional ReLU projection."""

    def __init__(self, lookup_window: int = 101, output_dim: int = 128) -> None:
        super().__init__()
        assert int(lookup_window) % 2 == 1, "lookup_window must be odd"
        self.lookup_window = int(lookup_window)
        self.fc = nn.Linear(self.lookup_window, int(output_dim)) if output_dim is not None else None

    @staticmethod
    def compute_color_histograms(frames: Tensor) -> Tensor:
        frames = frames.int()

        def _bin(x: Tensor) -> Tensor:
            r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
            r, g, b = r >> 5, g >> 5, b >> 5
            return (r << 6) + (g << 3) + b

        bsz, tw, h, w, c = frames.shape
        assert c == 3
        flat = frames.view(bsz * tw, h * w, 3)
        binned = _bin(flat)
        prefix = (torch.arange(0, bsz * tw, device=frames.device) << 9).view(-1, 1)
        binned = (binned + prefix).view(-1)

        hist = torch.zeros(bsz * tw * 512, dtype=torch.int32, device=frames.device)
        hist.scatter_add_(0, binned, torch.ones(len(binned), dtype=torch.int32, device=frames.device))
        hist = hist.view(bsz, tw, 512).float()
        return F.normalize(hist, p=2, dim=2)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.compute_color_histograms(inputs)
        b, t = x.shape[:2]
        sims = torch.bmm(x, x.transpose(1, 2))
        pad = (self.lookup_window - 1) // 2
        sims_padded = F.pad(sims, [pad, pad])

        bi = torch.arange(b, device=x.device).view(b, 1, 1).expand(b, t, self.lookup_window)
        ti = torch.arange(t, device=x.device).view(1, t, 1).expand(b, t, self.lookup_window)
        li = (
            torch.arange(self.lookup_window, device=x.device)
            .view(1, 1, self.lookup_window)
            .expand(b, t, self.lookup_window)
            + ti
        )
        gathered = sims_padded[bi, ti, li]
        if self.fc is not None:
            return F.relu(self.fc(gathered))
        return gathered


class SBDOursHead(SlowFastSBDSirenTRCSHead):
    """Combine boundary evidence, FiLM-SIREN trajectories and persistence gates.

    Optional TransNetV2 descriptors preserve the reference cue ordering.
    """

    def __init__(
        self,
        in_channels: Optional[int] = None,
        in_channels_slow: Optional[int] = None,
        in_channels_fast: Optional[int] = None,
        loss_cls: Dict[str, Any] = dict(type="BCELossWithLogits"),
        dropout_ratio: float = 0.5,
        init_std: float = 0.01,
        input_mode: str = "auto",
        feature_mode: str = "pooled",
        use_base_branch: bool = True,
        use_boundary_factor: bool = True,
        use_logit_calibration: bool = True,
        predict_with_calibrated_probs: bool = False,
        main_loss_mode: str = "pos_weight",
        transition_weight: float = 1.0,
        descriptor_in_dim: int = -1,
        descriptor_dim: int = 1024,
        use_flatten_map: bool = True,
        use_mean_pooling: Optional[bool] = None,
        concat_order: str = "official",
        frame_similarity_on_last_layer: bool = False,
        use_frame_similarity: bool = False,
        frame_similarity_in_dim: int = 0,
        frame_similarity_lookup_window: int = 101,
        frame_similarity_dim: int = 128,
        frame_similarity_output_dim: int = 128,
        use_color_histograms: bool = False,
        color_hist_lookup_window: int = 101,
        color_hist_output_dim: int = 128,
        use_many_hot_targets: bool = False,
        many_hot_loss_weight: float = 0.10,
        loss_many_hot: Dict[str, Any] = dict(type="BCELossWithLogits"),
        enable_local_evidence_gate: bool = False,
        local_evidence_strength: float = 0.35,
        local_evidence_pool_kernel: int = 3,
        init_local_evidence_prob: float = 0.02,
        lambda_local_evidence_loss: float = 0.0,
        local_evidence_pos_weight: float = 5.0,
        enable_video_logit_norm: bool = False,
        init_video_norm_strength: float = 0.35,
        init_video_norm_scale: float = 1.0,
        init_video_norm_bias: float = -2.0,
        video_logit_norm_eps: float = 1e-4,
        video_logit_norm_mad_scale: float = 1.4826,
        video_logit_norm_detach_stats: bool = True,
        siren_condition_source: str = "auto",
        enable_siren: bool = True,
        siren_hidden_dim: int = 128,
        siren_num_hidden_layers: int = 3,
        siren_omega_0: float = 30.0,
        siren_final_w_init: float = 1e-4,
        siren_start_epoch: int = 0,
        siren_conditioned: bool = True,
        siren_cond_dim: int = 32,
        siren_use_timestep_cond: bool = True,
        cond_t_smooth_kernel: int = 5,
        cond_gamma_init: float = 0.50,
        pos_weight: float = 2.0,
        lambda_base_aux: float = 1.0,
        lambda_siren_cls: float = 0.0,
        lambda_boundary: float = 1.0,
        boundary_pos_weight: float = 5.0,
        lambda_bg_suppress: float = 0.20,
        bg_hard_gamma: float = 2.0,
        bg_hard_power: float = 1.0,
        dyn_start_epoch: int = 6,
        use_dyn_in_boundary_evidence: bool = False,
        lambda_dyn_energy: float = 0.0,
        dyn_pos_weight: float = 5.0,
        dyn_use_second_derivative: bool = False,
        dyn_dd_weight: float = 0.5,
        dyn_energy_scale_init: float = 5.0,
        dyn_energy_bias_init: float = 0.0,
        dyn_energy_norm: str = "mean",
        dyn_energy_clip: float = 10.0,
        boundary_suppress_kernel: int = 9,
        init_boundary_beta: float = 0.20,
        init_logit_bias: float = 0.0,
        init_temperature: float = 1.0,
        min_temperature: float = 0.7,
        max_temperature: float = 4.0,
        siren_phi_dim: int = 16,
        lambda_phi_distill: float = 0.2,
        lambda_phi_tv: float = 0.0,
        enable_trcs: bool = False,
        trcs_start_epoch: int = 0,
        trcs_offsets: Optional[Sequence[int]] = None,
        trcs_offset: int = 8,
        trcs_return_thr: float = 0.6,
        trcs_return_temp: float = 0.1,
        trcs_use_innov: bool = True,
        trcs_innov_thr: float = 0.05,
        trcs_innov_temp: float = 0.03,
        trcs_change_source: str = "ev",
        trcs_change_thr: float = 0.6,
        trcs_change_temp: float = 0.05,
        trcs_strength: float = 0.6,
        lambda_trcs_loss: float = 0.0,
        enable_artifact_suppression: Optional[bool] = None,
        artifact_start_epoch: Optional[int] = None,
        artifact_offsets: Optional[Sequence[int]] = None,
        return_similarity_thr: Optional[float] = None,
        return_similarity_temp: Optional[float] = None,
        use_innovation_gate: Optional[bool] = None,
        innovation_thr: Optional[float] = None,
        innovation_temp: Optional[float] = None,
        change_source: Optional[str] = None,
        change_thr: Optional[float] = None,
        change_temp: Optional[float] = None,
        artifact_strength: Optional[float] = None,
        lambda_artifact_loss: Optional[float] = None,
        compute_dyn_in_eval: bool = False,
        save_debug_fields: bool = True,
        inference_debug_stats: bool = False,
        freeze_inactive_params: bool = False,
        **kwargs,
    ) -> None:
        feature_mode = str(feature_mode)
        if feature_mode not in ("pooled", "descriptor", "transnet_descriptor"):
            raise ValueError(f"Unsupported feature_mode={feature_mode}")
        input_mode = str(input_mode)
        if input_mode not in ("auto", "slowfast", "transnet", "fast"):
            raise ValueError(f"Unsupported input_mode={input_mode}")
        siren_condition_source = str(siren_condition_source)
        if siren_condition_source not in ("auto", "slow", "seq"):
            raise ValueError(f"Unsupported siren_condition_source={siren_condition_source}")
        main_loss_mode = str(main_loss_mode)
        if main_loss_mode not in ("pos_weight", "transition_weight"):
            raise ValueError(f"Unsupported main_loss_mode={main_loss_mode}")
        concat_order = str(concat_order)
        if concat_order not in ("official", "append"):
            raise ValueError(f"Unsupported concat_order={concat_order}")
        if in_channels_fast is None:
            if in_channels is None:
                raise ValueError("Either in_channels_fast or in_channels must be set.")
            in_channels_fast = int(in_channels)
        if in_channels_slow is None:
            in_channels_slow = int(in_channels_fast if in_channels is None else in_channels)
        if use_mean_pooling is not None:
            use_flatten_map = not bool(use_mean_pooling)

        if enable_artifact_suppression is not None:
            enable_trcs = bool(enable_artifact_suppression)
        if artifact_start_epoch is not None:
            trcs_start_epoch = int(artifact_start_epoch)
        if artifact_offsets is not None:
            trcs_offsets = artifact_offsets
        if return_similarity_thr is not None:
            trcs_return_thr = float(return_similarity_thr)
        if return_similarity_temp is not None:
            trcs_return_temp = float(return_similarity_temp)
        if use_innovation_gate is not None:
            trcs_use_innov = bool(use_innovation_gate)
        if innovation_thr is not None:
            trcs_innov_thr = float(innovation_thr)
        if innovation_temp is not None:
            trcs_innov_temp = float(innovation_temp)
        if change_source is not None:
            trcs_change_source = str(change_source)
        if change_thr is not None:
            trcs_change_thr = float(change_thr)
        if change_temp is not None:
            trcs_change_temp = float(change_temp)
        if artifact_strength is not None:
            trcs_strength = float(artifact_strength)
        if lambda_artifact_loss is not None:
            lambda_trcs_loss = float(lambda_artifact_loss)

        self.input_mode = input_mode
        self.feature_mode = feature_mode
        self.raw_in_channels_slow = int(in_channels_slow)
        self.raw_in_channels_fast = int(in_channels_fast)
        self.use_descriptor_branch = feature_mode in ("descriptor", "transnet_descriptor")
        self.use_flatten_map = bool(use_flatten_map)
        self.concat_order = concat_order
        self.frame_similarity_on_last_layer = bool(frame_similarity_on_last_layer)
        self.use_frame_similarity = bool(use_frame_similarity)
        self.use_color_histograms = bool(use_color_histograms)
        self.use_many_hot_targets = bool(use_many_hot_targets)
        self.many_hot_loss_weight = float(many_hot_loss_weight)
        self.enable_local_evidence_gate = bool(enable_local_evidence_gate)
        self.local_evidence_strength = float(local_evidence_strength)
        self.local_evidence_pool_kernel = int(local_evidence_pool_kernel)
        if self.local_evidence_pool_kernel < 1:
            self.local_evidence_pool_kernel = 1
        if self.local_evidence_pool_kernel % 2 == 0:
            self.local_evidence_pool_kernel += 1
        init_local_evidence_prob = min(max(float(init_local_evidence_prob), 1e-4), 1.0 - 1e-4)
        self.init_local_evidence_logit = float(
            math.log(init_local_evidence_prob / (1.0 - init_local_evidence_prob))
        )
        self.lambda_local_evidence_loss = float(lambda_local_evidence_loss)
        self.local_evidence_pos_weight = float(local_evidence_pos_weight)
        self.enable_video_logit_norm = bool(enable_video_logit_norm)
        init_video_norm_strength = min(max(float(init_video_norm_strength), 1e-4), 1.0 - 1e-4)
        self.init_video_norm_strength = float(init_video_norm_strength)
        self.init_video_norm_scale = float(max(init_video_norm_scale, 1e-6))
        self.init_video_norm_bias = float(init_video_norm_bias)
        self.video_logit_norm_eps = float(max(video_logit_norm_eps, 1e-8))
        self.video_logit_norm_mad_scale = float(max(video_logit_norm_mad_scale, 1e-6))
        self.video_logit_norm_detach_stats = bool(video_logit_norm_detach_stats)
        self.main_loss_mode = main_loss_mode
        self.transition_weight = float(transition_weight)
        self.use_boundary_factor = bool(use_boundary_factor)
        self.use_logit_calibration = bool(use_logit_calibration)
        self.predict_with_calibrated_probs = bool(predict_with_calibrated_probs)
        self.siren_condition_source = siren_condition_source
        self.freeze_inactive_params = bool(freeze_inactive_params)

        # Descriptor mode uses post-fc1 features rather than backbone channels.
        sequence_dim = int(descriptor_dim) if self.use_descriptor_branch else int(in_channels_fast)

        # A late similarity cue widens the classifier without changing sequence features.
        cls_in_dim = sequence_dim
        if self.use_descriptor_branch and self.use_frame_similarity and self.frame_similarity_on_last_layer:
            cls_in_dim = sequence_dim + int(frame_similarity_output_dim)
        self.cls_in_dim = cls_in_dim

        super().__init__(
            in_channels_slow=int(in_channels_slow),
            in_channels_fast=cls_in_dim,
            loss_cls=loss_cls,
            dropout_ratio=dropout_ratio,
            init_std=init_std,
            use_base_branch=use_base_branch,
            enable_siren=enable_siren,
            siren_hidden_dim=siren_hidden_dim,
            siren_num_hidden_layers=siren_num_hidden_layers,
            siren_omega_0=siren_omega_0,
            siren_final_w_init=siren_final_w_init,
            siren_start_epoch=siren_start_epoch,
            siren_conditioned=siren_conditioned,
            siren_cond_dim=siren_cond_dim,
            siren_use_timestep_cond=siren_use_timestep_cond,
            cond_t_smooth_kernel=cond_t_smooth_kernel,
            cond_gamma_init=cond_gamma_init,
            pos_weight=pos_weight,
            lambda_base_aux=lambda_base_aux,
            lambda_siren_cls=lambda_siren_cls,
            lambda_boundary=lambda_boundary,
            boundary_pos_weight=boundary_pos_weight,
            lambda_bg_suppress=lambda_bg_suppress,
            bg_hard_gamma=bg_hard_gamma,
            bg_hard_power=bg_hard_power,
            dyn_start_epoch=dyn_start_epoch,
            use_dyn_in_boundary_evidence=use_dyn_in_boundary_evidence,
            lambda_dyn_energy=lambda_dyn_energy,
            dyn_pos_weight=dyn_pos_weight,
            dyn_use_second_derivative=dyn_use_second_derivative,
            dyn_dd_weight=dyn_dd_weight,
            dyn_energy_scale_init=dyn_energy_scale_init,
            dyn_energy_bias_init=dyn_energy_bias_init,
            dyn_energy_norm=dyn_energy_norm,
            dyn_energy_clip=dyn_energy_clip,
            boundary_suppress_kernel=boundary_suppress_kernel,
            init_boundary_beta=init_boundary_beta,
            init_logit_bias=init_logit_bias,
            init_temperature=init_temperature,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            siren_phi_dim=siren_phi_dim,
            lambda_phi_distill=lambda_phi_distill,
            lambda_phi_tv=lambda_phi_tv,
            enable_trcs=enable_trcs,
            trcs_start_epoch=trcs_start_epoch,
            trcs_offsets=trcs_offsets,
            trcs_offset=trcs_offset,
            trcs_return_thr=trcs_return_thr,
            trcs_return_temp=trcs_return_temp,
            trcs_use_innov=trcs_use_innov,
            trcs_innov_thr=trcs_innov_thr,
            trcs_innov_temp=trcs_innov_temp,
            trcs_change_source=trcs_change_source,
            trcs_change_thr=trcs_change_thr,
            trcs_change_temp=trcs_change_temp,
            trcs_strength=trcs_strength,
            lambda_trcs_loss=lambda_trcs_loss,
            compute_dyn_in_eval=compute_dyn_in_eval,
            save_debug_fields=save_debug_fields,
            inference_debug_stats=inference_debug_stats,
            **kwargs,
        )

        self.descriptor_in_dim = int(descriptor_in_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.frame_similarity_output_dim = int(frame_similarity_output_dim)
        self.descriptor_fc: Optional[nn.Linear] = None
        self.frame_similarity: Optional[nn.Module] = None
        self.color_histograms: Optional[nn.Module] = None

        if self.use_descriptor_branch:
            assert self.descriptor_in_dim > 0, (
                "descriptor_in_dim must be set for feature_mode='descriptor'. "
                "For full TransNetV2 cues use 256*3*6 + 128 + 128 = 4864 (when "
                "frame_similarity_on_last_layer=False), or 256*3*6 + 128 = 4736 "
                "when frame_similarity_on_last_layer=True."
            )
            self.descriptor_fc = nn.Linear(self.descriptor_in_dim, self.descriptor_dim)

            if self.use_frame_similarity:
                assert int(frame_similarity_in_dim) > 0, (
                    "frame_similarity_in_dim must be > 0 when use_frame_similarity=True."
                )
                self.frame_similarity = _OfficialFrameSimilarity(
                    in_filters=int(frame_similarity_in_dim),
                    similarity_dim=int(frame_similarity_dim),
                    lookup_window=int(frame_similarity_lookup_window),
                    output_dim=int(frame_similarity_output_dim),
                    use_bias=True,
                )

            if self.use_color_histograms:
                self.color_histograms = _OfficialColorHistograms(
                    lookup_window=int(color_hist_lookup_window),
                    output_dim=int(color_hist_output_dim),
                )

        self.loss_many_hot = None
        self.many_hot_cls = (
            nn.Conv1d(cls_in_dim, 1, kernel_size=1) if self.use_many_hot_targets else None
        )
        self.local_evidence_head = (
            nn.Conv1d(cls_in_dim, 1, kernel_size=1) if self.enable_local_evidence_gate else None
        )
        if self.enable_video_logit_norm:
            self.logit_video_norm_strength = nn.Parameter(
                torch.tensor(
                    math.log(self.init_video_norm_strength / (1.0 - self.init_video_norm_strength)),
                    dtype=torch.float32,
                )
            )
            initial_scale = torch.tensor(
                math.log(self.init_video_norm_scale), dtype=torch.float32)
            self.log_video_norm_scale = nn.Parameter(initial_scale)
            self.video_norm_bias = nn.Parameter(torch.tensor(self.init_video_norm_bias, dtype=torch.float32))
        else:
            self.logit_video_norm_strength = None
            self.log_video_norm_scale = None
            self.video_norm_bias = None

        self.seq_siren_cond_proj: Optional[nn.Sequential] = None
        if self.enable_siren and self.siren_conditioned:
            self.seq_siren_cond_proj = nn.Sequential(
                nn.Linear(cls_in_dim, self.siren_cond_dim),
                nn.LayerNorm(self.siren_cond_dim),
            )

        self._init_extra_weights()
        if self.freeze_inactive_params:
            self._freeze_inactive_parameters()

    def init_weights(self) -> None:
        # The parent initializes weights before descriptor modules exist.
        super().init_weights()
        if hasattr(self, "descriptor_fc"):
            self._init_extra_weights()

    def _init_extra_weights(self) -> None:
        if self.descriptor_fc is not None:
            normal_init(self.descriptor_fc, std=self.init_std)
        if self.many_hot_cls is not None:
            normal_init(self.many_hot_cls, std=self.init_std)
        if getattr(self, "local_evidence_head", None) is not None:
            normal_init(self.local_evidence_head, std=self.init_std)
            if self.local_evidence_head.bias is not None:
                nn.init.constant_(self.local_evidence_head.bias, self.init_local_evidence_logit)

    @staticmethod
    def _freeze_module_params(module: Optional[nn.Module]) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _freeze_param(param: Optional[nn.Parameter]) -> None:
        if param is not None:
            param.requires_grad_(False)

    def _freeze_inactive_parameters(self) -> None:
        """Freeze disabled branches so DDP does not expect gradients from them."""
        if (not self.use_boundary_factor) and self.lambda_boundary <= 0:
            self._freeze_module_params(self.start_head)
            self._freeze_module_params(self.end_head)
            self._freeze_param(self.logit_boundary_beta)

        if not self.use_logit_calibration:
            self._freeze_param(self.logit_bias)
            self._freeze_param(self.log_temperature)

        if not self.enable_video_logit_norm:
            self._freeze_param(getattr(self, "logit_video_norm_strength", None))
            self._freeze_param(getattr(self, "log_video_norm_scale", None))
            self._freeze_param(getattr(self, "video_norm_bias", None))

        if not self.enable_local_evidence_gate:
            self._freeze_module_params(getattr(self, "local_evidence_head", None))

        if not self.enable_siren:
            self._freeze_module_params(self.siren)
            self._freeze_module_params(getattr(self, "siren_cond_proj", None))
            self._freeze_module_params(getattr(self, "seq_siren_cond_proj", None))
            self._freeze_module_params(getattr(self, "cond_feat_norm", None))
            self._freeze_param(self.logit_cond_gamma)

        if (not self.enable_siren) or ((self.lambda_dyn_energy <= 0) and (not self.use_dyn_in_boundary_evidence)):
            self._freeze_param(self.log_dyn_energy_scale)
            self._freeze_param(self.dyn_energy_bias)

        if (not self.enable_siren) or self.lambda_phi_distill <= 0:
            self._freeze_module_params(self.phi_distill_proj)

    @staticmethod
    def _prepare_rgb_uint8(rgb_inputs: Tensor) -> Tensor:
        rgb = rgb_inputs.detach()
        rgb = rgb.permute(0, 2, 3, 4, 1).contiguous()
        rgb = torch.clamp(rgb * 255.0, 0.0, 255.0).round().to(torch.uint8)
        return rgb

    def _unpack_inputs(
        self,
        inputs: Union[Tensor, Tuple],
    ) -> Tuple[Optional[Tensor], Tensor, Optional[Tuple[Tensor, ...]], Optional[Tensor]]:
        """Return slow/fast features, optional stage features and resized RGB.

        Accept a tensor, (slow, fast), or nested backbone outputs with optional extras.
        """
        x_slow: Optional[Tensor] = None
        block_features: Optional[Tuple[Tensor, ...]] = None
        resized_rgb: Optional[Tensor] = None

        if torch.is_tensor(inputs):
            x_fast = inputs
            return x_slow, x_fast, block_features, resized_rgb

        items = list(inputs)
        if len(items) == 0:
            raise RuntimeError("SBDOursHead got empty inputs")

        first = items[0]
        rest = items[1:]

        if isinstance(first, (tuple, list)):
            if len(first) != 2:
                raise RuntimeError(f"Expected (slow, fast) sub-tuple, got len={len(first)}")
            x_slow, x_fast = first
        elif (
            len(items) >= 2
            and torch.is_tensor(items[0])
            and torch.is_tensor(items[1])
            and self.input_mode not in ("transnet", "fast")
            and not (items[1].dim() == 5 and items[1].shape[1] == 3)
        ):
            x_slow, x_fast = items[0], items[1]
            rest = items[2:]
        else:
            x_fast = first

        for item in rest:
            if isinstance(item, (tuple, list)):
                block_features = tuple(item)
            elif torch.is_tensor(item):
                if item.dim() == 5 and item.shape[1] == 3:
                    resized_rgb = item
                elif block_features is None:
                    block_features = (item,)
                else:
                    resized_rgb = item

        if not torch.is_tensor(x_fast):
            raise RuntimeError(f"SBDOursHead expected a tensor x_fast, got {type(x_fast)}")
        return x_slow, x_fast, block_features, resized_rgb

    def _build_descriptor(
        self,
        x_fast: Tensor,
        block_features: Optional[Tuple[Tensor, ...]],
        resized_rgb: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Return per-frame descriptors and an optional late similarity cue.

        The reference layout prepends [color_hist, frame_sim] before fc1.
        A late frame_sim cue is concatenated after fc1, ReLU and dropout.
        """
        if self.use_flatten_map:
            desc = x_fast.permute(0, 2, 3, 4, 1).reshape(x_fast.shape[0], x_fast.shape[2], -1)
        else:
            desc = x_fast.mean(dim=[3, 4]).transpose(1, 2).contiguous()

        sim_feat: Optional[Tensor] = None
        if self.use_frame_similarity and self.frame_similarity is not None:
            assert block_features is not None, "FrameSimilarity requires block features from backbone"
            sim_feat = self.frame_similarity(block_features)

        hist_feat: Optional[Tensor] = None
        if self.use_color_histograms and self.color_histograms is not None:
            assert resized_rgb is not None, "ColorHistograms requires resized RGB inputs from backbone"
            hist_feat = self.color_histograms(self._prepare_rgb_uint8(resized_rgb))

        if self.concat_order == "official":
            if sim_feat is not None and not self.frame_similarity_on_last_layer:
                desc = torch.cat([sim_feat, desc], dim=-1)
            if hist_feat is not None:
                desc = torch.cat([hist_feat, desc], dim=-1)
        else:
            # Preserve the configured append layout.
            if sim_feat is not None and not self.frame_similarity_on_last_layer:
                desc = torch.cat([desc, sim_feat], dim=-1)
            if hist_feat is not None:
                desc = torch.cat([desc, hist_feat], dim=-1)

        assert self.descriptor_fc is not None
        seq_feat = F.relu(self.descriptor_fc(desc))
        if self.dropout is not None:
            seq_feat = self.dropout(seq_feat)

        frame_sim_late = sim_feat if (sim_feat is not None and self.frame_similarity_on_last_layer) else None
        if frame_sim_late is not None:
            seq_feat = torch.cat([frame_sim_late, seq_feat], dim=-1) if self.concat_order == "official" \
                else torch.cat([seq_feat, frame_sim_late], dim=-1)
        return seq_feat, frame_sim_late

    def _build_sequence_features(
        self,
        x_fast: Tensor,
        block_features: Optional[Tuple[Tensor, ...]],
        resized_rgb: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if self.use_descriptor_branch:
            seq_feat, _ = self._build_descriptor(x_fast, block_features, resized_rgb)  # [N,T,C]
            return seq_feat, seq_feat.transpose(1, 2).contiguous()

        x_pooled = self.avg_pool_fast(x_fast)
        if self.dropout is not None:
            x_pooled = self.dropout(x_pooled)
        seq_ch = x_pooled.squeeze(-1).squeeze(-1)  # [N,C,T]
        seq_feat = seq_ch.transpose(1, 2).contiguous()
        return seq_feat, seq_ch

    def _compute_cond_feat_unified(
        self,
        seq_feat: Tensor,
        x_slow: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor]:
        n, t_fast, _ = seq_feat.shape
        device = seq_feat.device
        dtype = seq_feat.dtype

        if (not self.siren_conditioned) or self.siren is None:
            cond_feat = torch.zeros((n, t_fast, max(1, self.siren_cond_dim)), device=device, dtype=dtype)
            return None, None, cond_feat

        use_slow = False
        if self.siren_condition_source == "slow":
            use_slow = x_slow is not None and self.siren_cond_proj is not None
        elif self.siren_condition_source == "auto":
            use_slow = (
                x_slow is not None
                and self.siren_cond_proj is not None
                and x_slow.dim() == 5
                and x_slow.shape[1] == self.raw_in_channels_slow
            )

        if use_slow:
            clip_code_g = x_slow.mean(dim=(2, 3, 4)).detach()
            cond_global = self.siren_cond_proj(clip_code_g)

            if self.siren_use_timestep_cond:
                code_t = x_slow.mean(dim=(3, 4)).detach()  # [N,C,Tslow]
                code_t = F.interpolate(code_t, size=t_fast, mode="linear", align_corners=False)
                code_t = code_t.permute(0, 2, 1).contiguous()  # [N,Tfast,C]
                cond_t = self.siren_cond_proj(code_t)
                cond_t = self._smooth_cond_t(cond_t, self.cond_t_smooth_kernel)
            else:
                cond_t = None
        else:
            assert self.seq_siren_cond_proj is not None, "seq_siren_cond_proj is required for seq conditioning"
            cond_global = self.seq_siren_cond_proj(seq_feat.mean(dim=1).detach())
            cond_t = self.seq_siren_cond_proj(seq_feat.detach()) if self.siren_use_timestep_cond else None
            if cond_t is not None:
                cond_t = self._smooth_cond_t(cond_t, self.cond_t_smooth_kernel)

        if cond_t is not None:
            gamma = torch.sigmoid(self.logit_cond_gamma)
            cond_feat = (1.0 - gamma) * cond_global.unsqueeze(1) + gamma * cond_t
        else:
            cond_feat = cond_global.unsqueeze(1).expand(-1, t_fast, -1)
        if self.cond_feat_norm is not None:
            cond_feat = self.cond_feat_norm(cond_feat)
        return cond_global, cond_t, cond_feat

    def _compute_siren_outputs_unified(
        self,
        seq_feat: Tensor,
        x_slow: Optional[Tensor],
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        n, t_fast, _ = seq_feat.shape
        device = seq_feat.device

        cond_global, cond_t, cond_feat = self._compute_cond_feat_unified(seq_feat, x_slow)
        tcoords = self._build_tcoords(t_fast, device=device, dtype=dtype).expand(n, -1, -1).contiguous()
        cond_flat = cond_feat.reshape(n * t_fast, -1)

        need_dyn = (self._runtime_epoch >= self.dyn_start_epoch) and (self.training or self.compute_dyn_in_eval)
        # Time derivatives are needed only for training or explicit dynamics.
        track_siren_grad = self.training or need_dyn
        with torch.set_grad_enabled(track_siren_grad):
            tcoords_req = tcoords.detach().requires_grad_(track_siren_grad)
            tflat_req = tcoords_req.reshape(n * t_fast, 1)
            out = self.siren(tflat_req, cond_flat).view(n, t_fast, self.siren_phi_dim + 1)
            phi = out[:, :, : self.siren_phi_dim]
            siren_logits = out[:, :, self.siren_phi_dim]

            energy_norm = None
            dyn_energy_logits = None

            if need_dyn:
                grads_1 = []
                grads_2 = [] if self.dyn_use_second_derivative else None
                for k in range(self.siren_phi_dim):
                    phi_k = phi[:, :, k]
                    dphi_k_dt = torch.autograd.grad(
                        outputs=phi_k,
                        inputs=tcoords_req,
                        grad_outputs=torch.ones_like(phi_k),
                        create_graph=self.training and self.dyn_use_second_derivative,
                        retain_graph=True,
                        only_inputs=True,
                    )[0].squeeze(-1)
                    grads_1.append(dphi_k_dt)

                    if self.dyn_use_second_derivative:
                        d2phi_k_dt2 = torch.autograd.grad(
                            outputs=dphi_k_dt,
                            inputs=tcoords_req,
                            grad_outputs=torch.ones_like(dphi_k_dt),
                            create_graph=self.training,
                            retain_graph=True,
                            only_inputs=True,
                        )[0].squeeze(-1)
                        grads_2.append(d2phi_k_dt2)

                dphi_dt = torch.stack(grads_1, dim=-1)
                if self.dyn_use_second_derivative:
                    d2phi_dt2 = torch.stack(grads_2, dim=-1)
                else:
                    d2phi_dt2 = torch.zeros_like(dphi_dt)
                e1 = torch.sqrt((dphi_dt ** 2).sum(dim=-1) + 1e-12)
                e2 = torch.sqrt((d2phi_dt2 ** 2).sum(dim=-1) + 1e-12)
                energy = e1 + self.dyn_dd_weight * e2

                if self.dyn_energy_norm == "median":
                    denom = energy.detach().median(dim=1, keepdim=True).values.clamp(min=1e-6)
                else:
                    denom = energy.detach().mean(dim=1, keepdim=True).clamp(min=1e-6)
                energy_norm = (energy / denom).clamp(0.0, self.dyn_energy_clip)
                energy_scale = torch.exp(self.log_dyn_energy_scale).clamp(min=1e-6, max=1e6)
                dyn_energy_logits = energy_scale * (energy_norm - 1.0) + self.dyn_energy_bias

        outs: Dict[str, Tensor] = {
            "siren_phi": phi,
            "siren_logits": siren_logits,
            "cond_gamma": torch.sigmoid(self.logit_cond_gamma).detach(),
            "siren_cond_feat": cond_feat,
        }
        if cond_global is not None:
            outs["siren_cond_global"] = cond_global
        if cond_t is not None:
            outs["siren_cond_t"] = cond_t
        if energy_norm is not None and dyn_energy_logits is not None:
            outs["dyn_energy_norm"] = energy_norm.detach()
            outs["dyn_energy_logits"] = dyn_energy_logits
            outs["dyn_probs"] = torch.sigmoid(dyn_energy_logits).detach()
        return outs

    def _compute_local_evidence(self, seq_ch: Tensor) -> Tuple[Tensor, Tensor]:
        """Estimate local state jumps from adjacent sequence features."""
        if self.local_evidence_head is None:
            empty = seq_ch.new_zeros(seq_ch.shape[0], seq_ch.shape[2])
            return empty, empty

        left = torch.zeros_like(seq_ch)
        right = torch.zeros_like(seq_ch)
        left[:, :, 1:] = (seq_ch[:, :, 1:] - seq_ch[:, :, :-1]).abs()
        right[:, :, :-1] = (seq_ch[:, :, 1:] - seq_ch[:, :, :-1]).abs()
        diff = torch.maximum(left, right)

        local_logits = self.local_evidence_head(diff).squeeze(1)
        if self.local_evidence_pool_kernel > 1:
            pad = self.local_evidence_pool_kernel // 2
            local_logits = F.max_pool1d(
                local_logits.unsqueeze(1),
                kernel_size=self.local_evidence_pool_kernel,
                stride=1,
                padding=pad,
            ).squeeze(1)
        return local_logits, torch.sigmoid(local_logits)

    def _apply_video_logit_norm(self, logits: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Blend logits with per-clip median/MAD-normalized logits."""
        if (
            (not self.enable_video_logit_norm)
            or self.logit_video_norm_strength is None
            or self.log_video_norm_scale is None
            or self.video_norm_bias is None
        ):
            return logits, {}

        stats_logits = logits.detach() if self.video_logit_norm_detach_stats else logits
        center = stats_logits.median(dim=1, keepdim=True).values
        mad = (stats_logits - center).abs().median(dim=1, keepdim=True).values
        spread = (mad * self.video_logit_norm_mad_scale).clamp(min=self.video_logit_norm_eps)

        normed = (logits - center) / spread
        scale = torch.exp(self.log_video_norm_scale).clamp(min=1e-3, max=1e3)
        norm_logits = scale * normed + self.video_norm_bias
        strength = torch.sigmoid(self.logit_video_norm_strength)
        out = (1.0 - strength) * logits + strength * norm_logits
        return out, {
            "video_norm_logits": norm_logits,
            "video_norm_center": center.detach(),
            "video_norm_spread": spread.detach(),
            "video_norm_strength": strength.detach(),
            "video_norm_scale": scale.detach(),
            "video_norm_bias": self.video_norm_bias.detach(),
        }

    def forward(self, inputs: Union[Tensor, Tuple], **kwargs) -> Dict[str, Tensor]:
        x_slow, x_fast, block_features, resized_rgb = self._unpack_inputs(inputs)
        seq_feat, seq_ch = self._build_sequence_features(x_fast, block_features, resized_rgb)
        _, t_fast, _ = seq_feat.shape

        outputs: Dict[str, Tensor] = {"seq_feat": seq_feat}
        if self.temporal_conv is None:
            base_logits = seq_feat.new_zeros(seq_feat.shape[0], t_fast)
        else:
            base_logits = self.temporal_conv(seq_ch).squeeze(1)
        outputs["base_logits"] = base_logits

        if self.many_hot_cls is not None:
            outputs["many_hot_logits"] = self.many_hot_cls(seq_ch).squeeze(1)

        start_logits = self.start_head(seq_ch).squeeze(1)
        end_logits = self.end_head(seq_ch).squeeze(1)
        outputs["start_logits"] = start_logits
        outputs["end_logits"] = end_logits

        start_probs = torch.sigmoid(start_logits)
        end_probs = torch.sigmoid(end_logits)

        siren_active = (
            self.enable_siren
            and self._runtime_epoch >= self.siren_start_epoch
            and self.siren is not None
        )
        dyn_probs = None
        if siren_active:
            outputs.update(self._compute_siren_outputs_unified(seq_feat, x_slow, dtype=x_fast.dtype))
            if ("dyn_probs" in outputs) and (self._runtime_epoch >= self.dyn_start_epoch):
                dyn_probs = outputs["dyn_probs"]

        p_mix = torch.sigmoid(base_logits)
        outputs["p_mix"] = p_mix
        outputs["p_base"] = p_mix

        dyn_for_ev = dyn_probs if (self.use_dyn_in_boundary_evidence and (dyn_probs is not None)) else None
        ev = self._boundary_evidence(start_probs, end_probs, dyn_for_ev)
        ev_raw = ev
        outputs["ev_raw"] = ev_raw
        outputs["start_probs"] = start_probs
        outputs["end_probs"] = end_probs

        if self.enable_local_evidence_gate:
            local_logits, local_probs = self._compute_local_evidence(seq_ch)
            ev = (ev + self.local_evidence_strength * local_probs * (1.0 - ev)).clamp(0.0, 1.0)
            outputs["local_evidence_logits"] = local_logits
            outputs["local_evidence_probs"] = local_probs
            outputs["ev_local"] = ev

        gate_artifact = ev_raw.new_zeros(ev_raw.shape)
        if (
            self.enable_trcs
            and siren_active
            and (self._runtime_epoch >= self.trcs_start_epoch)
            and ("siren_phi" in outputs)
            and ("siren_cond_t" in outputs)
        ):
            phi = outputs["siren_phi"]
            phi_n = F.normalize(phi, dim=-1)
            gate_return, sim_max = self._return_gate_multiscale(phi_n)
            if not getattr(self, "trcs_use_return", True):
                # Neutralize the return factor for leave-one-out configurations.
                gate_return = gate_return.new_ones(gate_return.shape)

            cond_t = outputs["siren_cond_t"].detach()
            u = self.phi_distill_proj(cond_t)
            du = torch.sqrt(((u[:, 1:, :] - u[:, :-1, :]) ** 2).sum(dim=-1) + 1e-12)
            dphi = torch.sqrt(((phi[:, 1:, :] - phi[:, :-1, :]) ** 2).sum(dim=-1) + 1e-12)
            innov = F.relu(du - dphi)
            innov = F.pad(innov, (1, 0), value=0.0)

            if self.trcs_use_innov:
                gate_innov = torch.sigmoid((innov - self.trcs_innov_thr) / self.trcs_innov_temp)
            else:
                gate_innov = gate_return.new_ones(gate_return.shape)

            if self.trcs_change_source == "dyn" and (dyn_probs is not None):
                gate_change = torch.sigmoid((dyn_probs.detach() - self.trcs_change_thr) / self.trcs_change_temp)
            elif self.trcs_change_source == "mix" and (dyn_probs is not None):
                gate_dyn = torch.sigmoid((dyn_probs.detach() - self.trcs_change_thr) / self.trcs_change_temp)
                gate_ev = torch.sigmoid((ev_raw.detach() - self.trcs_change_thr) / self.trcs_change_temp)
                gate_change = torch.maximum(gate_dyn, gate_ev)
            else:
                gate_change = torch.sigmoid((ev_raw.detach() - self.trcs_change_thr) / self.trcs_change_temp)

            gate_artifact = gate_return * gate_innov * gate_change
            ev = (ev - self.trcs_strength * gate_artifact).clamp(0.0, 1.0)

            outputs["trcs_gate"] = gate_artifact
            if self.training or self.inference_debug_stats:
                outputs["trcs_gate_mean"] = gate_artifact.mean().detach()
                outputs["trcs_return_mean"] = gate_return.mean().detach()
                outputs["trcs_innov_mean"] = gate_innov.mean().detach()
                outputs["trcs_change_mean"] = gate_change.mean().detach()
                outputs["trcs_sim_max_mean"] = sim_max.mean().detach()
            outputs["trcs_return"] = gate_return
            outputs["trcs_innov"] = gate_innov
            outputs["trcs_change"] = gate_change
            outputs["trcs_sim_max"] = sim_max
            if self.training or self.inference_debug_stats:
                with torch.no_grad():
                    m = gate_artifact > 0.5
                    outputs["trcs_evraw_mean"] = ev_raw[m].mean().detach() if m.any() else ev_raw.mean().detach()

        outputs["ev_used"] = ev

        beta = torch.sigmoid(self.logit_boundary_beta)
        if self.use_boundary_factor:
            factor = (1.0 - beta) + beta * ev
        else:
            factor = torch.ones_like(p_mix)
        outputs["factor"] = factor

        # A probability-to-logit round trip would clamp extreme base predictions.
        if (not self.use_boundary_factor) and (not self.use_logit_calibration) and (not self.enable_video_logit_norm):
            logits_out = base_logits
            p_raw = p_mix
            p_out = p_mix
        else:
            p_raw = (p_mix * factor).clamp(0.0, 1.0)
            logits_out = self._logit_from_prob(p_raw)
            if self.use_logit_calibration:
                temperature = self._get_temperature()
                logits_out = (logits_out + self.logit_bias) / temperature

            if self.enable_video_logit_norm:
                outputs["logits_before_video_norm"] = logits_out
                logits_out, norm_dbg = self._apply_video_logit_norm(logits_out)
                outputs.update(norm_dbg)

            if self.enable_video_logit_norm:
                p_out = torch.sigmoid(logits_out)
            elif self.predict_with_calibrated_probs and self.use_logit_calibration:
                p_out = torch.sigmoid(logits_out)
            else:
                p_out = p_raw

        outputs["logits"] = logits_out
        outputs["probs"] = p_out
        outputs["probs_raw"] = p_raw
        outputs["boundary_beta"] = beta.detach()
        outputs["temperature"] = self._get_temperature().detach()
        outputs["logit_bias"] = self.logit_bias.detach()
        return outputs

    def _classification_loss(self, logits: Tensor, targets: Tensor, gt_hard: Tensor) -> Tensor:
        if self.main_loss_mode == "transition_weight":
            elem_weight = 1.0 + gt_hard * (self.transition_weight - 1.0)
            return F.binary_cross_entropy_with_logits(logits, targets, weight=elem_weight, reduction="mean")
        return self._bce_with_pos_weight(logits, targets, self.pos_weight)

    def _get_targets(self, data_samples, device: torch.device, t_pred: int) -> Dict[str, Tensor]:
        targets = super()._get_targets(data_samples, device=device, t_pred=t_pred)
        if not isinstance(data_samples, (list, tuple)):
            data_samples = [data_samples]
        gt_many_hot_list = []
        for d in data_samples:
            v = getattr(d, "gt_many_hot", None)
            if v is None:
                continue
            v = v.to(device=device, dtype=torch.float32) if isinstance(v, torch.Tensor) else torch.as_tensor(
                v, device=device, dtype=torch.float32
            )
            gt_many_hot_list.append(v)
        if len(gt_many_hot_list) == len(data_samples):
            stacked = torch.stack(gt_many_hot_list, dim=0)
            if stacked.shape[1] != t_pred:
                stacked = F.interpolate(
                    stacked.unsqueeze(1), size=t_pred, mode="linear",
                    align_corners=False).squeeze(1)
            targets["gt_many_hot"] = stacked
        return targets

    def loss(self, inputs: Union[Tensor, Tuple], data_samples, **kwargs) -> Dict[str, Tensor]:
        outs = self.forward(inputs, **kwargs)

        logits = outs["logits"]
        probs = outs["probs"]
        base_logits = outs["base_logits"]
        start_logits = outs["start_logits"]
        end_logits = outs["end_logits"]

        _, t_pred = logits.shape
        targets = self._get_targets(data_samples, device=logits.device, t_pred=t_pred)
        gt_hard = targets["gt"]
        gt_main = targets.get("gt_soft", gt_hard)
        gt_many_hot = targets.get("gt_many_hot", gt_main)

        losses: Dict[str, Tensor] = {}
        losses["loss_cls"] = self._classification_loss(logits, gt_main, gt_hard)

        if self.lambda_base_aux > 0:
            losses["loss_base_aux"] = self.lambda_base_aux * self._classification_loss(base_logits, gt_main, gt_hard)

        if self.use_many_hot_targets and ("many_hot_logits" in outs) and self.loss_many_hot is not None:
            losses["loss_many_hot"] = self.many_hot_loss_weight * self.loss_many_hot(
                outs["many_hot_logits"], gt_many_hot
            )

        if (
            self.enable_local_evidence_gate
            and ("local_evidence_logits" in outs)
            and (self.lambda_local_evidence_loss > 0)
        ):
            losses["loss_local_evidence"] = self.lambda_local_evidence_loss * self._bce_with_pos_weight(
                outs["local_evidence_logits"], gt_main, self.local_evidence_pos_weight
            )

        if ("gt_start" in targets) and ("gt_end" in targets) and (self.lambda_boundary > 0):
            losses["loss_boundary_start"] = self.lambda_boundary * self._bce_with_pos_weight(
                start_logits, targets["gt_start"], self.boundary_pos_weight
            )
            losses["loss_boundary_end"] = self.lambda_boundary * self._bce_with_pos_weight(
                end_logits, targets["gt_end"], self.boundary_pos_weight
            )

        if self.lambda_bg_suppress > 0:
            bg_mask = (gt_hard < 0.5).float()
            denom = bg_mask.sum().clamp(min=1.0)
            p = probs.clamp(0.0, 1.0)
            hard_w = torch.pow(p, self.bg_hard_gamma)
            base_pen = p if self.bg_hard_power <= 1.0 else torch.pow(p, self.bg_hard_power)
            losses["loss_bg_suppress"] = self.lambda_bg_suppress * (hard_w * base_pen * bg_mask).sum() / denom

        if self.enable_trcs and ("trcs_gate" in outs) and ("ev_raw" in outs) and (self.lambda_trcs_loss > 0):
            bg_mask = (gt_hard < 0.5).float()
            denom = bg_mask.sum().clamp(min=1.0)
            gate = outs["trcs_gate"].detach()
            ev_raw = outs["ev_raw"]
            ev_pos = torch.sigmoid((ev_raw.detach() - self.trcs_change_thr) / self.trcs_change_temp)
            losses["loss_trcs_bg"] = self.lambda_trcs_loss * ((ev_raw * gate * ev_pos * bg_mask).sum() / denom)

        if self.enable_siren and ("siren_logits" in outs) and (self.lambda_siren_cls > 0):
            losses["loss_siren_cls"] = self.lambda_siren_cls * self._classification_loss(
                outs["siren_logits"], gt_main, gt_hard
            )

        if (
            self.enable_siren
            and ("dyn_energy_logits" in outs)
            and (self._runtime_epoch >= self.dyn_start_epoch)
            and (self.lambda_dyn_energy > 0)
        ):
            losses["loss_dyn_energy"] = self.lambda_dyn_energy * self._bce_with_pos_weight(
                outs["dyn_energy_logits"], gt_main, self.dyn_pos_weight
            )

        if self.enable_siren and self.lambda_phi_distill > 0 and ("siren_phi" in outs) and ("siren_cond_t" in outs):
            phi = outs["siren_phi"]
            cond_t = outs["siren_cond_t"].detach()
            target = self.phi_distill_proj(cond_t)
            losses["loss_phi_distill"] = self.lambda_phi_distill * F.mse_loss(phi, target)

        if self.enable_siren and ("siren_phi" in outs) and (self.lambda_phi_tv > 0):
            phi = outs["siren_phi"]
            tv = (phi[:, 1:, :] - phi[:, :-1, :]).pow(2).mean()
            losses["loss_phi_tv"] = self.lambda_phi_tv * tv

        with torch.no_grad():
            if self.logit_cond_gamma is not None:
                losses["cond_gamma"] = torch.sigmoid(self.logit_cond_gamma).detach()
            losses["boundary_beta"] = torch.sigmoid(self.logit_boundary_beta).detach()
            losses["temperature"] = self._get_temperature().detach()
            losses["logit_bias"] = self.logit_bias.detach()

            for k in [
                "trcs_gate_mean",
                "trcs_return_mean",
                "trcs_innov_mean",
                "trcs_change_mean",
                "trcs_evraw_mean",
                "trcs_sim_max_mean",
            ]:
                if k in outs:
                    losses[k] = outs[k]

            if (self.lambda_dyn_energy > 0) or self.use_dyn_in_boundary_evidence or ("dyn_energy_norm" in outs):
                losses["dyn_energy_scale"] = torch.exp(self.log_dyn_energy_scale).detach()
                losses["dyn_energy_bias"] = self.dyn_energy_bias.detach()

            if "dyn_energy_norm" in outs:
                e = outs["dyn_energy_norm"]
                pos = gt_hard > 0.5
                neg = ~pos
                if pos.any() and neg.any():
                    losses["dyn_e_pos_mean"] = e[pos].mean().detach()
                    losses["dyn_e_neg_mean"] = e[neg].mean().detach()
                    losses["dyn_e_gap_mean"] = (e[pos].mean() - e[neg].mean()).detach()

            if "local_evidence_probs" in outs:
                le = outs["local_evidence_probs"]
                pos = gt_hard > 0.5
                neg = ~pos
                losses["local_evidence_mean"] = le.mean().detach()
                if pos.any() and neg.any():
                    losses["local_evidence_pos_mean"] = le[pos].mean().detach()
                    losses["local_evidence_neg_mean"] = le[neg].mean().detach()

            if "video_norm_strength" in outs:
                losses["video_norm_strength"] = outs["video_norm_strength"]
                losses["video_norm_scale"] = outs["video_norm_scale"]
                losses["video_norm_bias"] = outs["video_norm_bias"]

        return losses

    def predict(self, inputs: Union[Tensor, Tuple], data_samples, **kwargs):
        if not isinstance(data_samples, (list, tuple)):
            data_samples = [data_samples]

        outs = self.forward(inputs, **kwargs)
        probs = outs["probs"]
        start_probs = torch.sigmoid(outs["start_logits"])
        end_probs = torch.sigmoid(outs["end_logits"])

        for i, sample in enumerate(data_samples):
            sample.pred_scores = probs[i].detach().cpu()
            if not self.save_debug_fields:
                continue

            sample.set_field(start_probs[i].detach().cpu(), "pred_start_scores")
            sample.set_field(end_probs[i].detach().cpu(), "pred_end_scores")

            for k in [
                "p_mix",
                "p_base",
                "probs_raw",
                "ev_raw",
                "ev_local",
                "ev_used",
                "factor",
                "start_probs",
                "end_probs",
                "local_evidence_probs",
                "video_norm_logits",
            ]:
                if k in outs:
                    sample.set_field(outs[k][i].detach().cpu(), k)
            if "many_hot_logits" in outs:
                sample.set_field(torch.sigmoid(outs["many_hot_logits"][i]).detach().cpu(), "many_hot_probs")

            for k in ["trcs_gate", "trcs_return", "trcs_innov", "trcs_change", "trcs_sim_max"]:
                if k in outs:
                    sample.set_field(outs[k][i].detach().cpu(), k)

            for k in ["siren_phi", "siren_logits", "dyn_probs", "dyn_energy_norm"]:
                if k in outs:
                    v = outs[k][i].detach().cpu() if outs[k].dim() >= 2 else outs[k].detach().cpu()
                    sample.set_field(v, k)

            if "siren_cond_t" in outs:
                sample.set_field(outs["siren_cond_t"][i].detach().cpu(), "siren_cond_t")
            if "cond_gamma" in outs:
                sample.set_field(outs["cond_gamma"].detach().cpu(), "cond_gamma")

            sample.set_field(outs["boundary_beta"].detach().cpu(), "boundary_beta")
            sample.set_field(outs["temperature"].detach().cpu(), "temperature")
            sample.set_field(outs["logit_bias"].detach().cpu(), "logit_bias")
            if "video_norm_strength" in outs:
                sample.set_field(outs["video_norm_strength"].detach().cpu(), "video_norm_strength")
                sample.set_field(outs["video_norm_scale"].detach().cpu(), "video_norm_scale")
                sample.set_field(outs["video_norm_bias"].detach().cpu(), "video_norm_bias")

        return data_samples
