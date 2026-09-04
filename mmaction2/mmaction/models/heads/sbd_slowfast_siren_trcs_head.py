"""FiLM-SIREN and persistence-gate components for the PERSIST head.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""

import math
from typing import Any, Dict, List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.model.weight_init import normal_init
from mmaction.registry import MODELS
from mmaction.models.heads.base import BaseHead


# ----------------------------
#   FiLM-SIREN building blocks
# ----------------------------
class FiLMSineLayer(nn.Module):
    """
    SIREN layer with FiLM modulation on pre-activation:
      y = sin( omega0 * ( (Wx+b) * (1+gamma) + beta ) )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.is_first = bool(is_first)
        self.omega_0 = float(omega_0)

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.in_features, 1.0 / self.in_features)
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x: Tensor, gamma: Optional[Tensor] = None, beta: Optional[Tensor] = None) -> Tensor:
        z = self.linear(x)
        if gamma is not None:
            z = z * (1.0 + gamma)
        if beta is not None:
            z = z + beta
        return torch.sin(self.omega_0 * z)


class SirenFiLMNet(nn.Module):
    """
    Temporal FiLM-SIREN:
    input:  tcoord [B,1]
    cond:   [B,D]
    output: [B,out_dim]
    """

    def __init__(
        self,
        cond_dim: int,
        out_dim: int = 2,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        omega_0: float = 30.0,
        final_w_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.omega_0 = float(omega_0)

        layers: List[FiLMSineLayer] = []
        layers.append(FiLMSineLayer(1, self.hidden_dim, is_first=True, omega_0=self.omega_0))
        for _ in range(self.num_hidden_layers - 1):
            layers.append(FiLMSineLayer(self.hidden_dim, self.hidden_dim, is_first=False, omega_0=self.omega_0))
        self.layers = nn.ModuleList(layers)

        self.final_linear = nn.Linear(self.hidden_dim, out_dim)
        with torch.no_grad():
            self.final_linear.weight.uniform_(-final_w_init, final_w_init)
            if self.final_linear.bias is not None:
                self.final_linear.bias.zero_()

        total = 0
        for lyr in self.layers:
            total += 2 * lyr.out_features
        self.hyper = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.GELU(),
            nn.Linear(self.cond_dim, total),
        )

    def forward(self, t: Tensor, cond: Tensor) -> Tensor:
        film = self.hyper(cond)
        idx = 0
        h = t
        for lyr in self.layers:
            of = lyr.out_features
            gamma = film[:, idx: idx + of]
            beta = film[:, idx + of: idx + 2 * of]
            idx += 2 * of
            h = lyr(h, gamma=gamma, beta=beta)
        return self.final_linear(h)


# ----------------------------
#   TRCS Head v3
# ----------------------------
@MODELS.register_module()
class SlowFastSBDSirenTRCSHead(BaseHead):
    """
    TRCS v3:
      - Return gate from implicit trajectory phi(t): multi-scale consistency
      - Change gate from dyn energy (preferred) or ev_raw (fallback) or mix
      - Optional innovation gate from (du - dphi) impulse
      - Suppress boundary evidence ev (not probabilities) => "mechanism A suppresses other mechanisms"
      - Hard-negative auxiliary loss on background high-evidence peaks
    """

    def __init__(
        self,
        in_channels_slow: int,
        in_channels_fast: int,
        loss_cls: Dict[str, Any] = dict(type="BCELossWithLogits"),
        dropout_ratio: float = 0.5,
        init_std: float = 0.01,

        # baseline
        use_base_branch: bool = True,

        # siren
        enable_siren: bool = True,
        siren_hidden_dim: int = 128,
        siren_num_hidden_layers: int = 3,
        siren_omega_0: float = 30.0,
        siren_final_w_init: float = 1e-4,
        siren_start_epoch: int = 0,

        # conditioning
        siren_conditioned: bool = True,
        siren_cond_dim: int = 32,
        siren_use_timestep_cond: bool = True,
        cond_t_smooth_kernel: int = 5,
        cond_gamma_init: float = 0.50,

        # losses
        pos_weight: float = 2.0,
        lambda_base_aux: float = 1.0,
        lambda_siren_cls: float = 0.0,

        # boundary heads
        lambda_boundary: float = 1.0,
        boundary_pos_weight: float = 5.0,

        # bg suppress
        lambda_bg_suppress: float = 0.20,
        bg_hard_gamma: float = 2.0,
        bg_hard_power: float = 1.0,

        # dyn (phi'(t))
        dyn_start_epoch: int = 6,
        use_dyn_in_boundary_evidence: bool = False,
        lambda_dyn_energy: float = 0.0,
        dyn_pos_weight: float = 5.0,
        dyn_use_second_derivative: bool = False,
        dyn_dd_weight: float = 0.5,
        dyn_energy_scale_init: float = 5.0,
        dyn_energy_bias_init: float = 0.0,
        dyn_energy_norm: str = "mean",  # mean|median
        dyn_energy_clip: float = 10.0,

        # boundary evidence factor
        boundary_suppress_kernel: int = 9,
        init_boundary_beta: float = 0.20,

        # calibration
        init_logit_bias: float = 0.0,
        init_temperature: float = 1.0,
        min_temperature: float = 0.7,
        max_temperature: float = 4.0,

        # phi distill
        siren_phi_dim: int = 16,
        lambda_phi_distill: float = 0.2,
        lambda_phi_tv: float = 0.0,

        # TRCS
        enable_trcs: bool = False,
        trcs_start_epoch: int = 0,
        trcs_offsets: Optional[Sequence[int]] = None,   # if provided, multi-scale; else use trcs_offset
        trcs_offset: int = 8,
        trcs_return_thr: float = 0.6,
        trcs_return_temp: float = 0.1,

        trcs_use_innov: bool = True,
        trcs_use_return: bool = True,
        trcs_return_mode: str = 'absolute',   # 'absolute' | 'relative' (theta_r reparam)
        trcs_return_rel_k: float = 1.0,
        trcs_innov_thr: float = 0.05,
        trcs_innov_temp: float = 0.03,

        # change/peak source: "dyn" | "ev" | "mix"
        trcs_change_source: str = "ev",
        trcs_change_thr: float = 0.6,
        trcs_change_temp: float = 0.05,

        trcs_strength: float = 0.6,

        # auxiliary loss
        lambda_trcs_loss: float = 0.0,

        # compute dyn in eval
        compute_dyn_in_eval: bool = False,

        save_debug_fields: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes=1,
            in_channels=in_channels_fast,
            loss_cls=loss_cls,
            **kwargs,
        )

        self.in_channels_slow = int(in_channels_slow)
        self.in_channels_fast = int(in_channels_fast)
        self.dropout_ratio = float(dropout_ratio)
        self.init_std = float(init_std)

        self.use_base_branch = bool(use_base_branch)

        self.enable_siren = bool(enable_siren)
        self.siren_start_epoch = int(siren_start_epoch)
        self._runtime_epoch = 0

        self.siren_conditioned = bool(siren_conditioned)
        self.siren_cond_dim = int(siren_cond_dim)
        self.siren_use_timestep_cond = bool(siren_use_timestep_cond)
        self.cond_t_smooth_kernel = int(cond_t_smooth_kernel)

        cond_gamma_init = float(cond_gamma_init)
        cond_gamma_init = min(max(cond_gamma_init, 0.01), 0.99)
        self.logit_cond_gamma = nn.Parameter(
            torch.tensor(math.log(cond_gamma_init / (1.0 - cond_gamma_init)), dtype=torch.float32)
        )

        self.pos_weight = float(pos_weight)
        self.lambda_base_aux = float(lambda_base_aux)
        self.lambda_siren_cls = float(lambda_siren_cls)

        self.lambda_boundary = float(lambda_boundary)
        self.boundary_pos_weight = float(boundary_pos_weight)

        self.lambda_bg_suppress = float(lambda_bg_suppress)
        self.bg_hard_gamma = float(bg_hard_gamma)
        self.bg_hard_power = float(bg_hard_power)

        self.dyn_start_epoch = int(dyn_start_epoch)
        self.use_dyn_in_boundary_evidence = bool(use_dyn_in_boundary_evidence)
        self.lambda_dyn_energy = float(lambda_dyn_energy)
        self.dyn_pos_weight = float(dyn_pos_weight)
        self.dyn_use_second_derivative = bool(dyn_use_second_derivative)
        self.dyn_dd_weight = float(dyn_dd_weight)
        self.dyn_energy_clip = float(max(0.0, dyn_energy_clip))
        self.dyn_energy_norm = str(dyn_energy_norm)

        dyn_energy_scale_init = float(max(1e-6, dyn_energy_scale_init))
        self.log_dyn_energy_scale = nn.Parameter(torch.tensor(math.log(dyn_energy_scale_init), dtype=torch.float32))
        self.dyn_energy_bias = nn.Parameter(torch.tensor(float(dyn_energy_bias_init), dtype=torch.float32))

        self.compute_dyn_in_eval = bool(compute_dyn_in_eval)

        # heads
        self.avg_pool_fast = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.dropout = nn.Dropout(self.dropout_ratio) if self.dropout_ratio > 0 else None

        self.temporal_conv = nn.Conv1d(self.in_channels_fast, 1, kernel_size=1) if self.use_base_branch else None
        self.start_head = nn.Conv1d(self.in_channels_fast, 1, kernel_size=1)
        self.end_head = nn.Conv1d(self.in_channels_fast, 1, kernel_size=1)

        # conditioning projection from slow
        if self.enable_siren and self.siren_conditioned:
            self.siren_cond_proj = nn.Sequential(
                nn.Linear(self.in_channels_slow, self.siren_cond_dim),
                nn.LayerNorm(self.siren_cond_dim),
            )
            self.cond_feat_norm = nn.LayerNorm(self.siren_cond_dim)
        else:
            self.siren_cond_proj = None
            self.cond_feat_norm = None

        # boundary evidence factor
        init_boundary_beta = float(init_boundary_beta)
        init_boundary_beta = min(max(init_boundary_beta, 0.01), 0.99)
        self.logit_boundary_beta = nn.Parameter(
            torch.tensor(math.log(init_boundary_beta / (1.0 - init_boundary_beta)), dtype=torch.float32)
        )
        self.boundary_suppress_kernel = int(boundary_suppress_kernel)
        if self.boundary_suppress_kernel < 1:
            self.boundary_suppress_kernel = 1
        if self.boundary_suppress_kernel % 2 == 0:
            self.boundary_suppress_kernel += 1

        # calibration
        self.logit_bias = nn.Parameter(torch.tensor(float(init_logit_bias), dtype=torch.float32))
        init_temperature = float(max(init_temperature, 1e-6))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature), dtype=torch.float32))
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)

        # phi
        self.siren_phi_dim = int(siren_phi_dim)
        self.lambda_phi_distill = float(lambda_phi_distill)
        self.lambda_phi_tv = float(lambda_phi_tv)

        # TRCS
        self.enable_trcs = bool(enable_trcs)
        self.trcs_start_epoch = int(trcs_start_epoch)

        if trcs_offsets is not None:
            self.trcs_offsets = tuple(int(x) for x in trcs_offsets)
        else:
            self.trcs_offsets = (int(trcs_offset),)

        self.trcs_return_thr = float(trcs_return_thr)
        self.trcs_return_temp = float(trcs_return_temp)

        self.trcs_use_innov = bool(trcs_use_innov)
        self.trcs_use_return = bool(trcs_use_return)
        self.trcs_return_mode = str(trcs_return_mode)
        self.trcs_return_rel_k = float(trcs_return_rel_k)
        self.trcs_innov_thr = float(trcs_innov_thr)
        self.trcs_innov_temp = float(trcs_innov_temp)

        self.trcs_change_source = str(trcs_change_source)
        self.trcs_change_thr = float(trcs_change_thr)
        self.trcs_change_temp = float(trcs_change_temp)

        self.trcs_strength = float(trcs_strength)
        self.lambda_trcs_loss = float(lambda_trcs_loss)

        self.save_debug_fields = bool(save_debug_fields)

        self._tcoord_cache: Dict[int, Tensor] = {}

        # FiLM-SIREN
        self.siren = SirenFiLMNet(
            cond_dim=self.siren_cond_dim if (self.enable_siren and self.siren_conditioned) else 1,
            out_dim=self.siren_phi_dim + 1,
            hidden_dim=int(siren_hidden_dim),
            num_hidden_layers=int(siren_num_hidden_layers),
            omega_0=float(siren_omega_0),
            final_w_init=float(siren_final_w_init),
        ) if self.enable_siren else None

        self.phi_distill_proj = nn.Linear(self.siren_cond_dim, self.siren_phi_dim)

        self.init_weights()

    def init_weights(self) -> None:
        if self.temporal_conv is not None:
            normal_init(self.temporal_conv, std=self.init_std)
        normal_init(self.start_head, std=self.init_std)
        normal_init(self.end_head, std=self.init_std)

    def set_epoch(self, epoch: int) -> None:
        self._runtime_epoch = int(epoch)

    def _get_temperature(self) -> Tensor:
        t = torch.exp(self.log_temperature)
        return t.clamp(min=self.min_temperature, max=self.max_temperature)

    @staticmethod
    def _logit_from_prob(p: Tensor, eps: float = 1e-5) -> Tensor:
        p = p.clamp(min=eps, max=1.0 - eps)
        return torch.log(p / (1.0 - p))

    def _forward_base_branch(self, x_fast: Tensor) -> Tuple[Tensor, Tensor]:
        x_pooled = self.avg_pool_fast(x_fast)
        if self.dropout is not None:
            x_pooled = self.dropout(x_pooled)
        x_pooled = x_pooled.squeeze(-1).squeeze(-1)  # [N,C,T]
        if self.temporal_conv is None:
            base_logits = x_pooled.new_zeros(x_pooled.shape[0], x_pooled.shape[2])
        else:
            base_logits = self.temporal_conv(x_pooled).squeeze(1)
        return base_logits, x_pooled

    def _build_tcoords(self, t: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if t in self._tcoord_cache:
            coord = self._tcoord_cache[t].to(device=device, dtype=dtype)
        else:
            coord = torch.linspace(-1.0, 1.0, t, dtype=torch.float32).view(1, t, 1)
            self._tcoord_cache[t] = coord.cpu()
            coord = coord.to(device=device, dtype=dtype)
        return coord

    @staticmethod
    def _smooth_cond_t(cond_t: Tensor, kernel: int) -> Tensor:
        k = int(kernel)
        if k <= 1:
            return cond_t
        if k % 2 == 0:
            k += 1
        x = cond_t.permute(0, 2, 1).contiguous()  # [N,D,T]
        pad = k // 2
        x = F.pad(x, (pad, pad), mode="replicate")
        x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=0)
        return x.permute(0, 2, 1).contiguous()

    def _compute_cond_feat(self, x_slow: Tensor, t_fast: int) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor]:
        n = x_slow.shape[0]
        device = x_slow.device
        dtype = x_slow.dtype

        if (not self.siren_conditioned) or (self.siren_cond_proj is None):
            cond_feat = torch.zeros((n, t_fast, max(1, self.siren_cond_dim)), device=device, dtype=dtype)
            return None, None, cond_feat

        clip_code_g = x_slow.mean(dim=(2, 3, 4)).detach()
        cond_global = self.siren_cond_proj(clip_code_g)

        if self.siren_use_timestep_cond:
            code_t = x_slow.mean(dim=(3, 4)).permute(0, 2, 1).detach()  # [N,Tslow,C]
            code_t = code_t.permute(0, 2, 1).contiguous()               # [N,C,Tslow]
            code_t = F.interpolate(code_t, size=t_fast, mode="linear", align_corners=False)
            code_t = code_t.permute(0, 2, 1).contiguous()               # [N,Tfast,C]
            cond_t = self.siren_cond_proj(code_t)
            cond_t = self._smooth_cond_t(cond_t, self.cond_t_smooth_kernel)

            gamma = torch.sigmoid(self.logit_cond_gamma)
            cond_feat = (1.0 - gamma) * cond_global.unsqueeze(1) + gamma * cond_t
            if self.cond_feat_norm is not None:
                cond_feat = self.cond_feat_norm(cond_feat)
            return cond_global, cond_t, cond_feat

        cond_t = None
        cond_feat = cond_global.unsqueeze(1).expand(-1, t_fast, -1)
        if self.cond_feat_norm is not None:
            cond_feat = self.cond_feat_norm(cond_feat)
        return cond_global, cond_t, cond_feat

    def _compute_siren_outputs(self, x_slow: Tensor, t_fast: int, dtype: torch.dtype) -> Dict[str, Tensor]:
        n = x_slow.shape[0]
        device = x_slow.device

        cond_global, cond_t, cond_feat = self._compute_cond_feat(x_slow, t_fast)
        tcoords = self._build_tcoords(t_fast, device=device, dtype=dtype).expand(n, -1, -1).contiguous()

        tcoords_flat = tcoords.reshape(n * t_fast, 1)
        cond_flat = cond_feat.reshape(n * t_fast, -1)

        with torch.enable_grad():
            tcoords_req = tcoords.detach().requires_grad_(True)
            tflat_req = tcoords_req.reshape(n * t_fast, 1)

            out = self.siren(tflat_req, cond_flat)
            out = out.view(n, t_fast, self.siren_phi_dim + 1)
            phi = out[:, :, : self.siren_phi_dim]
            siren_logits = out[:, :, self.siren_phi_dim]

            need_dyn = (self._runtime_epoch >= self.dyn_start_epoch) and (self.training or self.compute_dyn_in_eval)

            energy_norm = None
            dyn_energy_logits = None

            if need_dyn:
                N, T, K = phi.shape
                grads_1 = []
                grads_2 = [] if self.dyn_use_second_derivative else None

                for k in range(K):
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

                dphi_dt = torch.stack(grads_1, dim=-1)  # [N,T,K]
                if self.dyn_use_second_derivative:
                    d2phi_dt2 = torch.stack(grads_2, dim=-1)
                else:
                    d2phi_dt2 = torch.zeros_like(dphi_dt)

                e1 = torch.sqrt((dphi_dt ** 2).sum(dim=-1) + 1e-12)
                e2 = torch.sqrt((d2phi_dt2 ** 2).sum(dim=-1) + 1e-12)
                energy = e1 + self.dyn_dd_weight * e2  # [N,T]

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
        }
        if cond_global is not None:
            outs["siren_cond_global"] = cond_global
        if cond_t is not None:
            outs["siren_cond_t"] = cond_t
        outs["siren_cond_feat"] = cond_feat

        if energy_norm is not None and dyn_energy_logits is not None:
            outs["dyn_energy_norm"] = energy_norm.detach()
            outs["dyn_energy_logits"] = dyn_energy_logits
            outs["dyn_probs"] = torch.sigmoid(dyn_energy_logits).detach()

        return outs

    def _boundary_evidence(self, start_probs: Tensor, end_probs: Tensor, dyn_probs: Optional[Tensor]) -> Tensor:
        k = self.boundary_suppress_kernel
        if k <= 1:
            ev = torch.maximum(start_probs, end_probs)
            if dyn_probs is not None:
                ev = torch.maximum(ev, dyn_probs)
            return ev.clamp(0.0, 1.0)

        pad = k // 2
        s_pool = F.max_pool1d(start_probs.unsqueeze(1), kernel_size=k, stride=1, padding=pad).squeeze(1)
        e_pool = F.max_pool1d(end_probs.unsqueeze(1), kernel_size=k, stride=1, padding=pad).squeeze(1)
        ev = torch.maximum(s_pool, e_pool)

        if dyn_probs is not None:
            d_pool = F.max_pool1d(dyn_probs.unsqueeze(1), kernel_size=k, stride=1, padding=pad).squeeze(1)
            ev = torch.maximum(ev, d_pool)

        return ev.clamp(0.0, 1.0)

    def _return_gate_multiscale(self, phi_n: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute multi-scale return gate.
        Returns:
          gate_return: [B,T]
          sim_max:     [B,T] (max similarity across scales)
        """
        B, T, _ = phi_n.shape
        sim_max = phi_n.new_full((B, T), fill_value=-1.0)

        for d in self.trcs_offsets:
            d = int(d)
            if T <= 2 * d:
                continue
            sim = phi_n.new_full((B, T), fill_value=-1.0)
            zl = phi_n[:, : T - 2 * d, :]
            zr = phi_n[:, 2 * d : T, :]
            sim_mid = (zl * zr).sum(dim=-1).clamp(-1.0, 1.0)
            sim[:, d : T - d] = sim_mid
            sim_max = torch.maximum(sim_max, sim)

        sim_max = sim_max.clamp(-1.0, 1.0)
        if getattr(self, "trcs_return_mode", "absolute") == "relative":
            # theta_r reparameterisation (camera-ready candidate): threshold and
            # temperature relative to the clip's own similarity distribution, so
            # the gate cannot saturate when sims concentrate near 1 (E12a).
            med = sim_max.median(dim=1, keepdim=True).values
            q75 = sim_max.quantile(0.75, dim=1, keepdim=True)
            q25 = sim_max.quantile(0.25, dim=1, keepdim=True)
            scale = (q75 - q25).clamp_min(1e-4) * float(
                getattr(self, "trcs_return_rel_k", 1.0))
            gate_return = torch.sigmoid((sim_max - med) / scale)
        else:
            gate_return = torch.sigmoid((sim_max - self.trcs_return_thr) / self.trcs_return_temp)
        return gate_return, sim_max

    @staticmethod
    def _bce_with_pos_weight(logits: Tensor, targets: Tensor, pos_weight: float) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if pos_weight == 1.0:
            return loss.mean()
        w = torch.ones_like(loss)
        w = w + (pos_weight - 1.0) * (targets > 0.5).float()
        return (loss * w).mean()

    def forward(self, x: Tuple[Tensor, Tensor], **kwargs) -> Dict[str, Tensor]:
        x_slow, x_fast = x
        n, _, t_fast, _, _ = x_fast.shape
        outputs: Dict[str, Tensor] = {}

        base_logits, fast_feat = self._forward_base_branch(x_fast)
        outputs["base_logits"] = base_logits

        start_logits = self.start_head(fast_feat).squeeze(1)
        end_logits = self.end_head(fast_feat).squeeze(1)
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
            siren_outs = self._compute_siren_outputs(x_slow=x_slow, t_fast=t_fast, dtype=x_fast.dtype)
            outputs.update(siren_outs)
            if ("dyn_probs" in outputs) and (self._runtime_epoch >= self.dyn_start_epoch):
                dyn_probs = outputs["dyn_probs"]

        p_mix = torch.sigmoid(base_logits)
        outputs["p_mix"] = p_mix  # [B,T]

        dyn_for_ev = dyn_probs if (self.use_dyn_in_boundary_evidence and (dyn_probs is not None)) else None
        ev = self._boundary_evidence(start_probs, end_probs, dyn_for_ev)
        ev_raw = ev
        outputs["ev_raw"] = ev_raw  # [B,T]

        outputs["start_probs"] = start_probs  # [B,T]
        outputs["end_probs"] = end_probs      # [B,T]

        # ----------------------------
        # TRCS suppression on evidence
        # ----------------------------
        gate_artifact = ev_raw.new_zeros(ev_raw.shape)
        trcs_active = (
            self.enable_trcs
            and siren_active
            and self._runtime_epoch >= self.trcs_start_epoch
            and "siren_phi" in outputs
            and "siren_cond_t" in outputs
        )
        if trcs_active:
            phi = outputs["siren_phi"]  # [B,T,K]
            phi_n = F.normalize(phi, dim=-1)

            # return gate (multi-scale)
            gate_return, sim_max = self._return_gate_multiscale(phi_n)
            if not self.trcs_use_return:
                # LOO ablation: drop g_r from the gate product (loss + adjustment)
                gate_return = gate_return.new_ones(gate_return.shape)

            # observed state u(t) (teacher)
            cond_t = outputs["siren_cond_t"].detach()  # [B,T,D]
            u = self.phi_distill_proj(cond_t)          # [B,T,K]

            # innovation (impulse): du - dphi
            du = torch.sqrt(((u[:, 1:, :] - u[:, :-1, :]) ** 2).sum(dim=-1) + 1e-12)
            dphi = torch.sqrt(((phi[:, 1:, :] - phi[:, :-1, :]) ** 2).sum(dim=-1) + 1e-12)
            innov = F.relu(du - dphi)
            innov = F.pad(innov, (1, 0), value=0.0)

            if self.trcs_use_innov:
                gate_innov = torch.sigmoid((innov - self.trcs_innov_thr) / self.trcs_innov_temp)
            else:
                gate_innov = gate_return.new_ones(gate_return.shape)

            # change/peak gate
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

            # Cache signals used by losses and diagnostics.
            outputs["trcs_gate"] = gate_artifact
            outputs["ev_raw"] = ev_raw
            outputs["trcs_gate_mean"] = gate_artifact.mean().detach()
            outputs["trcs_return_mean"] = gate_return.mean().detach()
            outputs["trcs_innov_mean"] = gate_innov.mean().detach()
            outputs["trcs_change_mean"] = gate_change.mean().detach()
            outputs["trcs_sim_max_mean"] = sim_max.mean().detach()
            # Per-frame TRCS signals for visualization.
            outputs["trcs_return"] = gate_return       # [B,T]
            outputs["trcs_innov"] = gate_innov         # [B,T]
            outputs["trcs_change"] = gate_change       # [B,T]
            outputs["trcs_sim_max"] = sim_max          # [B,T]

            with torch.no_grad():
                m = (gate_artifact > 0.5)
                outputs["trcs_evraw_mean"] = (ev_raw[m].mean().detach() if m.any() else ev_raw.mean().detach())

        outputs["ev_used"] = ev  # [B,T]

        beta = torch.sigmoid(self.logit_boundary_beta)
        factor = (1.0 - beta) + beta * ev
        outputs["factor"] = factor  # [B,T]

        p_out = (p_mix * factor).clamp(0.0, 1.0)

        logits_out = self._logit_from_prob(p_out)
        temperature = self._get_temperature()
        logits_out = (logits_out + self.logit_bias) / temperature

        outputs["logits"] = logits_out
        outputs["probs"] = p_out
        outputs["boundary_beta"] = beta.detach()
        outputs["temperature"] = temperature.detach()
        outputs["logit_bias"] = self.logit_bias.detach()

        return outputs

    def _get_targets(self, data_samples, device: torch.device, t_pred: int) -> Dict[str, Tensor]:
        if not isinstance(data_samples, (list, tuple)):
            data_samples = [data_samples]

        gt_list = []
        gt_soft_list = []
        gt_start_list = []
        gt_end_list = []

        for d in data_samples:
            gt = getattr(d, "gt_labels", None)
            if gt is None:
                raise RuntimeError("gt_labels not found in data_sample")
            if isinstance(gt, torch.Tensor):
                gt = gt.to(device=device, dtype=torch.float32)
            else:
                gt = torch.as_tensor(
                    gt, device=device, dtype=torch.float32)
            gt_list.append(gt)

            gt_soft = getattr(d, "gt_soft", None)
            if gt_soft is not None:
                if isinstance(gt_soft, torch.Tensor):
                    gt_soft = gt_soft.to(device=device, dtype=torch.float32)
                else:
                    gt_soft = torch.as_tensor(
                        gt_soft, device=device, dtype=torch.float32)
                gt_soft_list.append(gt_soft)

            gt_start = getattr(d, "gt_start", None)
            if gt_start is not None:
                if isinstance(gt_start, torch.Tensor):
                    gt_start = gt_start.to(device=device, dtype=torch.float32)
                else:
                    gt_start = torch.as_tensor(
                        gt_start, device=device, dtype=torch.float32)
                gt_start_list.append(gt_start)

            gt_end = getattr(d, "gt_end", None)
            if gt_end is not None:
                if isinstance(gt_end, torch.Tensor):
                    gt_end = gt_end.to(device=device, dtype=torch.float32)
                else:
                    gt_end = torch.as_tensor(
                        gt_end, device=device, dtype=torch.float32)
                gt_end_list.append(gt_end)

        gt = torch.stack(gt_list, dim=0)

        def _align_1d(y: Tensor) -> Tensor:
            if y.shape[1] == t_pred:
                return y
            return F.interpolate(y.unsqueeze(1), size=t_pred, mode="linear", align_corners=False).squeeze(1)

        out = {"gt": _align_1d(gt)}
        if len(gt_soft_list) == len(gt_list):
            out["gt_soft"] = _align_1d(torch.stack(gt_soft_list, dim=0))
        if len(gt_start_list) == len(gt_list):
            out["gt_start"] = _align_1d(torch.stack(gt_start_list, dim=0))
        if len(gt_end_list) == len(gt_list):
            out["gt_end"] = _align_1d(torch.stack(gt_end_list, dim=0))
        return out

    def loss(self, inputs: Tuple[Tensor, Tensor], data_samples, **kwargs) -> Dict[str, Tensor]:
        x_slow, x_fast = inputs
        outs = self.forward((x_slow, x_fast), **kwargs)

        logits = outs["logits"]
        probs = outs["probs"]
        base_logits = outs["base_logits"]
        start_logits = outs["start_logits"]
        end_logits = outs["end_logits"]

        _, t_pred = logits.shape
        targets = self._get_targets(data_samples, device=logits.device, t_pred=t_pred)
        gt_hard = targets["gt"]
        gt_main = targets.get("gt_soft", gt_hard)

        losses: Dict[str, Tensor] = {}
        losses["loss_cls"] = self._bce_with_pos_weight(logits, gt_main, self.pos_weight)

        if self.lambda_base_aux > 0:
            losses["loss_base_aux"] = self.lambda_base_aux * self._bce_with_pos_weight(
                base_logits, gt_main, self.pos_weight
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

        # TRCS hard-negative loss on background peaks
        if self.enable_trcs and ("trcs_gate" in outs) and ("ev_raw" in outs) and (self.lambda_trcs_loss > 0):
            bg_mask = (gt_hard < 0.5).float()
            denom = bg_mask.sum().clamp(min=1.0)

            gate = outs["trcs_gate"].detach()
            ev_raw = outs["ev_raw"]

            # focus on high-evidence background peaks
            ev_pos = torch.sigmoid((ev_raw.detach() - self.trcs_change_thr) / self.trcs_change_temp)
            losses["loss_trcs_bg"] = self.lambda_trcs_loss * ((ev_raw * gate * ev_pos * bg_mask).sum() / denom)

        if self.enable_siren and ("siren_logits" in outs) and (self.lambda_siren_cls > 0):
            losses["loss_siren_cls"] = self.lambda_siren_cls * self._bce_with_pos_weight(
                outs["siren_logits"], gt_main, self.pos_weight
            )

        if (
            self.enable_siren
            and "dyn_energy_logits" in outs
            and self._runtime_epoch >= self.dyn_start_epoch
            and self.lambda_dyn_energy > 0
        ):
            losses["loss_dyn_energy"] = self.lambda_dyn_energy * self._bce_with_pos_weight(
                outs["dyn_energy_logits"], gt_main, self.dyn_pos_weight
            )

        # phi distill
        if self.lambda_phi_distill > 0 and ("siren_phi" in outs) and ("siren_cond_t" in outs):
            phi = outs["siren_phi"]
            cond_t = outs["siren_cond_t"].detach()
            target = self.phi_distill_proj(cond_t)
            losses["loss_phi_distill"] = self.lambda_phi_distill * F.mse_loss(phi, target)

        # phi smoothness
        if self.enable_siren and ("siren_phi" in outs) and (self.lambda_phi_tv > 0):
            phi = outs["siren_phi"]
            tv = (phi[:, 1:, :] - phi[:, :-1, :]).pow(2).mean()
            losses["loss_phi_tv"] = self.lambda_phi_tv * tv

        with torch.no_grad():
            losses["cond_gamma"] = torch.sigmoid(self.logit_cond_gamma).detach()
            losses["boundary_beta"] = torch.sigmoid(self.logit_boundary_beta).detach()
            losses["temperature"] = self._get_temperature().detach()
            losses["logit_bias"] = self.logit_bias.detach()

            diagnostic_keys = [
                "trcs_gate_mean", "trcs_return_mean", "trcs_innov_mean",
                "trcs_change_mean", "trcs_evraw_mean", "trcs_sim_max_mean",
            ]
            for k in diagnostic_keys:
                if k in outs:
                    losses[k] = outs[k]

            if (self.lambda_dyn_energy > 0) or self.use_dyn_in_boundary_evidence or ("dyn_energy_norm" in outs):
                losses["dyn_energy_scale"] = torch.exp(self.log_dyn_energy_scale).detach()
                losses["dyn_energy_bias"] = self.dyn_energy_bias.detach()

            if "dyn_energy_norm" in outs:
                e = outs["dyn_energy_norm"]
                pos = (gt_hard > 0.5)
                neg = ~pos
                if pos.any() and neg.any():
                    losses["dyn_e_pos_mean"] = e[pos].mean().detach()
                    losses["dyn_e_neg_mean"] = e[neg].mean().detach()
                    losses["dyn_e_gap_mean"] = (e[pos].mean() - e[neg].mean()).detach()

        return losses

    def predict(self, inputs: Tuple[Tensor, Tensor], data_samples, **kwargs):
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
            # Base/evidence chain.
            for k in ["p_mix", "ev_raw", "ev_used", "factor", "start_probs", "end_probs"]:
                if k in outs:
                    sample.set_field(outs[k][i].detach().cpu(), k)

            # Per-frame TRCS gates.
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

        return data_samples
