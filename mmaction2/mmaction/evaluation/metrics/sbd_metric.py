"""Shot-boundary metrics and prediction export for PERSIST.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""

from typing import Optional, List, Dict, Any, Tuple

import os
import os.path as osp
import json
import pickle

import numpy as np
import torch

from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmaction.registry import METRICS


# --------------------------------------------------------------------------
# Three-protocol naming convention (used by both SBDMetric and
# SBDRandomClipMetric).  Each protocol ships exactly one set of seven
# values (threshold + F1 + precision + recall + tp + fp + fn) so the
# log dict has no duplicates.
#
#   configured: metric evaluated at ``self.threshold``. During training this
#               is usually 0.50; during cross-domain eval this can be the
#               source-validation threshold.
#   fix_05    : threshold fixed at 0.50, matching TransNetV2-style
#               deployment defaults.
#   split_best: threshold scanned directly on the evaluation split
#               (oracle); a ranking-quality diagnostic, not a deployable
#               number.
#
# Legacy keys ``sbd_f1`` / ``sbd_best_threshold`` / ``sbd_precision`` /
# ``sbd_recall`` / ``sbd_tp`` / ``sbd_fp`` / ``sbd_fn`` are kept as
# aliases of the split_best protocol so checkpoint hooks and downstream
# log-parsing scripts that assumed the legacy schema still work.
# --------------------------------------------------------------------------

def _print_three_protocol_summary(
    name_prefix: str,
    val_th: float, val_f1: float, val_p: float, val_r: float,
    val_tp: int, val_fp: int, val_fn: int,
    fix_th: float, fix_f1: float, fix_p: float, fix_r: float,
    fix_tp: int, fix_fp: int, fix_fn: int,
    test_th: float, test_f1: float, test_p: float, test_r: float,
    test_tp: int, test_fp: int, test_fn: int,
    logger: str = 'current',
) -> None:
    """Emit a four-line human-readable three-protocol summary block.

    The first line lists the three thresholds and three F1 values
    side-by-side so the reader can compare them at a glance. The next
    three lines give per-protocol precision / recall / tp / fp / fn.

    The block is printed via mmengine's ``print_log`` so it lands in the
    same log file as the keyed metric dict; the keyed dict is still
    returned by the metric for downstream parsing.
    """
    head = (
        f"[{name_prefix}] "
        f"thr   configured={val_th:.4f}  fix_05={fix_th:.4f}  split_best={test_th:.4f}  |  "
        f"F1   configured={val_f1:.4f}  fix_05={fix_f1:.4f}  split_best={test_f1:.4f}"
    )
    print_log(head, logger=logger)
    print_log(
        f"[{name_prefix}] configured thr={val_th:.4f}  F1={val_f1:.4f}  "
        f"P={val_p:.4f}  R={val_r:.4f}  TP={val_tp}  FP={val_fp}  FN={val_fn}",
        logger=logger,
    )
    print_log(
        f"[{name_prefix}] fix_05    thr={fix_th:.4f}  F1={fix_f1:.4f}  "
        f"P={fix_p:.4f}  R={fix_r:.4f}  TP={fix_tp}  FP={fix_fp}  FN={fix_fn}",
        logger=logger,
    )
    print_log(
        f"[{name_prefix}] split_best thr={test_th:.4f}  F1={test_f1:.4f}  "
        f"P={test_p:.4f}  R={test_r:.4f}  TP={test_tp}  FP={test_fp}  FN={test_fn}",
        logger=logger,
    )


def _safe_threshold_list(thresholds: Optional[List[float]]) -> List[float]:
    if thresholds is None or len(thresholds) == 0:
        return [round(x, 3) for x in np.concatenate([
            np.linspace(0.02, 0.20, 10),
            np.linspace(0.22, 0.60, 20),
            np.linspace(0.62, 0.95, 14),
        ]).tolist()]
    out = []
    for t in thresholds:
        try:
            tt = float(t)
            if 0.0 <= tt <= 1.0:
                out.append(tt)
        except Exception:
            continue
    if len(out) == 0:
        return _safe_threshold_list(None)
    return out


@METRICS.register_module()
class SBDMetric(BaseMetric):
    """
    Sliding-window SBD metric with:
      - One-pass per-video aggregation (fast threshold scan)
      - Optional center-only accumulation per clip
      - Automatic compatibility with both:
          * rawframe datasets: frame_inds are 1-based file/global ids
          * video datasets:    frame_inds are 0-based time/global ids
      - Optional saving of raw probs / clipshots json / prob pickle / siren debug

    Expected per-sample fields:
      - vid or frame_dir or video_id
      - frame_inds: [L]
          rawframe: 1-based global frame ids
          video:    0-based global time/frame indices
      - pred_scores: [L] probabilities for clip frames
      - gt_labels (optional): [L] 0/1 for clip frames
      - ast_gst_json (optional): json string with {"ast":[...], "gst":[...]} in 0-based global time indices

    Important naming note for returned metrics:
      - legacy `sbd_best_threshold / sbd_f1 / sbd_precision / sbd_recall`
        mean "threshold scanned directly on the evaluation split"
      - legacy `sbd_val_threshold / sbd_val_f1 / sbd_val_precision / sbd_val_recall`
        mean "threshold selected from validation, then applied on the evaluation split"

    The implementation keeps the legacy keys for backward compatibility, but also
    returns clearer alias keys so new logs are easier to interpret.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        thresholds: Optional[List[float]] = None,
        n_frames_miss_tolerance: int = 2,

        save_raw: bool = False,
        raw_out_dir: Optional[str] = None,

        save_clipshots_json: bool = False,
        clipshots_json_path: Optional[str] = None,

        save_pred_pickle: bool = False,
        pred_pickle_path: Optional[str] = None,

        restrict_to_pred_span: bool = False,

        use_center_crop: bool = True,
        center_keep_len: Optional[int] = None,
        center_keep_ratio: float = 0.5,

        save_siren_debug: bool = False,
        siren_debug_path: Optional[str] = None,

        collect_device: str = 'cpu',
    ) -> None:
        super().__init__(collect_device=collect_device)

        self.threshold = float(threshold)
        self.thresholds = thresholds
        self.n_frames_miss_tolerance = int(n_frames_miss_tolerance)
        self.restrict_to_pred_span = bool(restrict_to_pred_span)

        self.save_raw = bool(save_raw)
        self.raw_out_dir = raw_out_dir

        self.save_clipshots_json = bool(save_clipshots_json)
        self.clipshots_json_path = clipshots_json_path

        self.save_pred_pickle = bool(save_pred_pickle)
        self.pred_pickle_path = pred_pickle_path

        self.use_center_crop = bool(use_center_crop)
        self.center_keep_len = None if center_keep_len is None else int(center_keep_len)
        self.center_keep_ratio = float(center_keep_ratio)

        self.save_siren_debug = bool(save_siren_debug)
        self.siren_debug_path = siren_debug_path
        self._siren_debug_records: List[Dict[str, Any]] = []

        if self.save_raw and self.raw_out_dir is None:
            raise ValueError('save_raw=True but raw_out_dir is None.')
        if self.save_clipshots_json and self.clipshots_json_path is None:
            raise ValueError('save_clipshots_json=True but clipshots_json_path is None.')
        if self.save_pred_pickle and self.pred_pickle_path is None:
            raise ValueError('save_pred_pickle=True but pred_pickle_path is None.')
        if self.save_siren_debug and self.siren_debug_path is None:
            raise ValueError('save_siren_debug=True but siren_debug_path is None.')

    # ------------------------
    # helpers
    # ------------------------
    @staticmethod
    def _get_field(sample, name: str):
        if hasattr(sample, 'get'):
            val = sample.get(name, None)
            if val is not None:
                return val
        if hasattr(sample, name):
            return getattr(sample, name)
        if isinstance(sample, dict):
            return sample.get(name, None)
        return None

    @staticmethod
    def _predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def _evaluate_scenes(
        gt_scenes: np.ndarray,
        pred_scenes: np.ndarray,
        n_frames_miss_tolerance: int = 2,
    ):
        shift = n_frames_miss_tolerance / 2.0
        gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
        pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

        gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
        pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

        i, j = 0, 0
        tp, fp, fn = 0, 0, 0

        while i < len(gt_trans) or j < len(pred_trans):
            if j == len(pred_trans):
                fn += 1
                i += 1
            elif i == len(gt_trans):
                fp += 1
                j += 1
            elif pred_trans[j, 1] < gt_trans[i, 0]:
                fp += 1
                j += 1
            elif pred_trans[j, 0] > gt_trans[i, 1]:
                fn += 1
                i += 1
            else:
                i += 1
                j += 1
                tp += 1

        return int(tp), int(fp), int(fn)

    @staticmethod
    def _transitions_to_scenes(transitions, n_frames: int) -> np.ndarray:
        if n_frames <= 0:
            raise ValueError("n_frames must be > 0")

        if not transitions:
            return np.array([[0, n_frames - 1]], dtype=np.int32)

        transitions = sorted(transitions, key=lambda x: x[0])
        scenes = []
        cur_start = 0

        for s, e in transitions:
            s = int(s)
            e = int(e)

            s = max(0, min(s, n_frames - 1))
            e = max(0, min(e, n_frames))

            if cur_start <= s:
                scenes.append([cur_start, s])
            cur_start = e

        if cur_start <= n_frames - 1:
            scenes.append([cur_start, n_frames - 1])

        return np.asarray(scenes, dtype=np.int32)

    @staticmethod
    def _bin_to_clipshots_segments(one_hot: np.ndarray, n_frames: int) -> Dict[str, List[List[int]]]:
        """
        Convert binary per-frame predictions (0/1) into ClipShots-like segments.

        Indices are 0-based time indices:
        - "cut":     [s, s+1]
        - "gradual": [s, e] where s and e are boundaries

        Heuristic:
        - length 1 or 2 -> cut
        - length >= 3   -> gradual, expand boundaries by 1 on both sides
        """
        F = int(len(one_hot))
        if n_frames is None:
            n_frames = F
        n_frames = int(n_frames)

        cuts: List[List[int]] = []
        gradual: List[List[int]] = []

        i = 0
        while i < F:
            if one_hot[i] == 0:
                i += 1
                continue

            start = i
            while i + 1 < F and one_hot[i + 1] == 1:
                i += 1
            end = i
            length = end - start + 1

            if length <= 2:
                s = int(start)
                e = int(start + 1)
                if e > n_frames:
                    e = n_frames
                if s < 0:
                    s = 0
                cuts.append([s, e])
            else:
                s = int(start - 1)
                e = int(end + 1)
                if s < 0:
                    s = 0
                if e > n_frames - 1:
                    e = n_frames - 1
                if e > s:
                    gradual.append([s, e])

            i += 1

        return dict(cut=cuts, gradual=gradual)

    def _center_slice(self, L: int) -> slice:
        if (not self.use_center_crop) or L <= 0:
            return slice(0, L)

        keep_len = self.center_keep_len
        if keep_len is None:
            keep_len = max(1, int(round(L * max(0.1, min(self.center_keep_ratio, 1.0)))))
        keep_len = max(1, min(int(keep_len), L))

        start = (L - keep_len) // 2
        end = start + keep_len
        return slice(start, end)

    @staticmethod
    def _infer_frame_base_from_segment(seg: Dict[str, Any]) -> int:
        """
        Infer whether seg['frame_inds'] are 0-based or 1-based.

        Priority:
        1) If clip_start exists and frame_inds length > 0:
           compare frame_inds[0] with clip_start
           - equal      -> 0-based
           - equal + 1  -> 1-based
        2) Otherwise fallback:
           - contains 0 -> 0-based
           - min >= 1   -> 1-based
           - default    -> 0-based
        """
        frame_inds = np.asarray(seg['frame_inds']).reshape(-1)
        if frame_inds.size == 0:
            return 0

        clip_start = seg.get('clip_start', None)
        if clip_start is not None:
            try:
                cs = int(clip_start)
                f0 = int(frame_inds[0])
                if f0 == cs:
                    return 0
                if f0 == cs + 1:
                    return 1
            except Exception:
                pass

        try:
            if int(frame_inds.min()) == 0:
                return 0
        except Exception:
            pass

        try:
            if int(frame_inds.min()) >= 1:
                return 1
        except Exception:
            pass

        return 0

    @staticmethod
    def _global_time_from_frame_inds(frame_inds: np.ndarray, frame_base: int) -> np.ndarray:
        """
        Convert frame_inds to unified 0-based global time indices.
        """
        frame_inds = np.asarray(frame_inds, dtype=np.int64).reshape(-1)
        if frame_base == 1:
            return frame_inds - 1
        return frame_inds

    # ------------------------
    # process: per-clip
    # ------------------------
    def process(self, data_batch, data_samples: List[Any]) -> None:
        for sample in data_samples:
            vid = self._get_field(sample, 'frame_dir')
            if vid is None:
                vid = self._get_field(sample, 'video_id')
            if vid is None:
                vid = self._get_field(sample, 'filename')
            if vid is None:
                vid = 'unknown'
            vid = str(vid)

            frame_inds = self._get_field(sample, 'frame_inds')
            if frame_inds is None:
                raise RuntimeError('frame_inds not found in data_sample.')
            frame_inds_np = (
                frame_inds.detach().cpu().numpy()
                if isinstance(frame_inds, torch.Tensor)
                else np.asarray(frame_inds)
            ).astype(np.int64)

            gt = self._get_field(sample, 'gt_labels')
            if gt is not None:
                gt_np = (
                    gt.detach().cpu().numpy().astype(np.int32)
                    if isinstance(gt, torch.Tensor)
                    else np.asarray(gt).astype(np.int32)
                )
            else:
                gt_np = None

            pred_scores = self._get_field(sample, 'pred_scores')
            if pred_scores is None:
                raise RuntimeError('pred_scores not found in data_sample.')
            pred_scores_np = (
                pred_scores.detach().cpu().numpy()
                if isinstance(pred_scores, torch.Tensor)
                else np.asarray(pred_scores, dtype=np.float32)
            ).astype(np.float32)

            clip_start = self._get_field(sample, 'clip_start')
            clip_end = self._get_field(sample, 'clip_end')
            clip_start = None if clip_start is None else int(clip_start)
            clip_end = None if clip_end is None else int(clip_end)

            ast_list, gst_list = [], []
            meta_json = self._get_field(sample, 'ast_gst_json')
            if meta_json is not None:
                try:
                    meta = json.loads(meta_json)
                    ast_list = meta.get('ast', []) or []
                    gst_list = meta.get('gst', []) or []
                except Exception:
                    ast_list, gst_list = [], []

            frame_base = self._infer_frame_base_from_segment(
                dict(frame_inds=frame_inds_np, clip_start=clip_start)
            )

            self.results.append(
                dict(
                    vid=vid,
                    frame_inds=frame_inds_np,
                    frame_base=frame_base,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    gt=gt_np,
                    pred_scores=pred_scores_np,
                    ast_list=ast_list,
                    gst_list=gst_list,
                )
            )

            if not self.save_siren_debug:
                continue

            L = int(len(frame_inds_np))
            if L <= 0:
                continue

            sl = self._center_slice(L)
            frame_inds_c = frame_inds_np[sl]
            if frame_inds_c.size == 0:
                continue

            def _to_np_field(name: str):
                v = self._get_field(sample, name)
                if v is None:
                    return None
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                else:
                    v = np.asarray(v)
                return v

            signal_names_1d = [
                "pred_scores",
                "pred_start_scores",
                "pred_end_scores",
                "p_mix",
                "ev_raw",
                "ev_used",
                "factor",
                "start_probs",
                "end_probs",
                "trcs_gate",
                "trcs_return",
                "trcs_innov",
                "trcs_change",
                "trcs_sim_max",
                "dyn_probs",
                "dyn_energy_norm",
            ]

            signals: Dict[str, Any] = {}
            for name in signal_names_1d:
                arr = _to_np_field(name)
                if arr is None:
                    continue
                arr = arr.reshape(-1)
                if len(arr) != L:
                    continue
                signals[name] = arr[sl].astype(np.float32)

            phi = _to_np_field("siren_phi")
            if phi is not None and phi.ndim == 2 and phi.shape[0] == L:
                signals["siren_phi"] = phi[sl].astype(np.float32)

            cond_t = _to_np_field("siren_cond_t")
            if cond_t is not None and cond_t.ndim == 2 and cond_t.shape[0] == L:
                signals["siren_cond_t"] = cond_t[sl].astype(np.float32)

            scalars: Dict[str, float] = {}
            for name in [
                "cond_gamma",
                "boundary_beta",
                "temperature",
                "logit_bias",
                "dyn_energy_scale",
                "dyn_energy_bias",
            ]:
                v = _to_np_field(name)
                if v is None:
                    continue
                try:
                    scalars[name] = float(np.asarray(v).reshape(-1)[0])
                except Exception:
                    pass

            vid_stem = osp.basename(str(vid).rstrip('/'))

            # Save debug frame indices in unified 0-based global time indices
            frame_inds_c_time0 = self._global_time_from_frame_inds(frame_inds_c, frame_base)

            self._siren_debug_records.append(
                dict(
                    vid=vid_stem,
                    frame_inds=frame_inds_c_time0.astype(np.int64),
                    signals=signals,
                    scalars=scalars,
                )
            )

    # ------------------------
    # pre-aggregate once per video
    # ------------------------
    def _build_per_video_arrays(
        self,
        by_vid: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        For each vid:
        - pred_probs: [F] aggregated per-frame prob, index == 0-based time index
        - gt_labels:  [F] aggregated per-frame gt label (max over clips)
        - gt_transitions: list of [s,e] in 0-based time indices
        - span_time:  (min_time, max_time) in 0-based time indices
        - frame_base: inferred per-video frame indexing convention (0 or 1)
        """
        per_vid: Dict[str, Dict[str, Any]] = {}

        for vid, segs in by_vid.items():
            first_seg = segs[0]
            ast = first_seg.get('ast_list', None)
            gst = first_seg.get('gst_list', None)
            if ast is None and gst is None:
                gt_transitions_full: List[List[int]] = []
            else:
                ast = ast or []
                gst = gst or []
                gt_transitions_full = (ast + gst)

            frame_base_votes = [int(seg.get('frame_base', 0)) for seg in segs]
            frame_base = 1 if sum(frame_base_votes) > (len(frame_base_votes) / 2.0) else 0

            min_time = 10**18
            max_time = -1
            for seg in segs:
                fi = np.asarray(seg['frame_inds']).reshape(-1)
                if fi.size == 0:
                    continue
                ti = self._global_time_from_frame_inds(fi, int(seg.get('frame_base', frame_base)))
                min_time = min(min_time, int(ti.min()))
                max_time = max(max_time, int(ti.max()))

            if max_time < 0 or min_time == 10**18:
                continue

            span_time = (int(min_time), int(max_time))
            F = int(max_time) + 1
            if F <= 0:
                continue

            if self.restrict_to_pred_span:
                s_t, e_t = span_time
                gt_transitions: List[List[int]] = [
                    [int(s), int(e)] for (s, e) in gt_transitions_full
                    if not (int(e) < s_t or int(s) > e_t)
                ]
            else:
                gt_transitions = [[int(s), int(e)] for (s, e) in gt_transitions_full]

            pred_sum = np.zeros(F, dtype=np.float32)
            pred_cnt = np.zeros(F, dtype=np.int32)
            gt_labels = np.zeros(F, dtype=np.int32)

            for seg in segs:
                frame_inds = np.asarray(seg['frame_inds']).reshape(-1)
                pred_scores = np.asarray(seg['pred_scores'], dtype=np.float32).reshape(-1)
                gt = seg.get('gt', None)

                if len(frame_inds) == 0:
                    continue
                if len(pred_scores) != len(frame_inds):
                    raise RuntimeError(
                        f'pred_scores length {len(pred_scores)} != frame_inds length {len(frame_inds)}'
                    )

                L = len(frame_inds)
                sl = self._center_slice(L)

                frame_inds_c = frame_inds[sl]
                pred_scores_c = pred_scores[sl]
                gt_c = gt[sl] if gt is not None else None

                time_inds_c = self._global_time_from_frame_inds(
                    frame_inds_c,
                    int(seg.get('frame_base', frame_base))
                )

                for i, t in enumerate(time_inds_c):
                    idx = int(t)
                    if idx < 0 or idx >= F:
                        continue
                    pred_sum[idx] += float(pred_scores_c[i])
                    pred_cnt[idx] += 1
                    if gt_c is not None and int(gt_c[i]) > 0:
                        gt_labels[idx] = 1

            mask = pred_cnt > 0
            pred_probs = np.zeros(F, dtype=np.float32)
            pred_probs[mask] = pred_sum[mask] / pred_cnt[mask]

            per_vid[vid] = dict(
                pred_probs=pred_probs,
                gt_labels=gt_labels,
                gt_transitions=gt_transitions,
                span_time=span_time,
                frame_base=frame_base,
            )

        return per_vid

    def _eval_per_video_at_threshold(
        self,
        per_vid: Dict[str, Dict[str, Any]],
        th: float
    ) -> Tuple[int, int, int, Dict[str, Dict[str, List[List[int]]]], Dict[str, np.ndarray]]:
        tol = self.n_frames_miss_tolerance
        total_tp = total_fp = total_fn = 0

        clipshots_dict: Dict[str, Dict[str, List[List[int]]]] = {}
        pred_prob_dict: Dict[str, np.ndarray] = {}

        for vid, pack in per_vid.items():
            pred_probs = pack['pred_probs']
            gt_transitions = pack['gt_transitions']
            F = len(pred_probs)
            if F <= 0:
                continue

            vid_stem = osp.basename(str(vid).rstrip('/'))
            pred_prob_dict[vid_stem] = pred_probs.reshape(-1, 1).astype(np.float32)

            one_hot_pred = (pred_probs >= th).astype(np.uint8)
            pred_scenes = self._predictions_to_scenes(one_hot_pred)

            gt_scenes = self._transitions_to_scenes(gt_transitions, n_frames=F)
            tp, fp, fn = self._evaluate_scenes(
                gt_scenes,
                pred_scenes,
                n_frames_miss_tolerance=tol
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            if self.save_clipshots_json:
                pred_seg_dict = self._bin_to_clipshots_segments(one_hot_pred, n_frames=F)
                clipshots_dict[vid_stem] = {
                    'cut': pred_seg_dict['cut'],
                    'gradual': pred_seg_dict['gradual'],
                }

        return total_tp, total_fp, total_fn, clipshots_dict, pred_prob_dict

    # ------------------------
    # compute metrics
    # ------------------------
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        by_vid: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            by_vid.setdefault(r['vid'], []).append(r)

        per_vid = self._build_per_video_arrays(by_vid)

        th_list = _safe_threshold_list(self.thresholds)

        best = None  # (f1, p, r, tp, fp, fn, th, clipshots_dict, pred_prob_dict)
        for th in th_list:
            tp, fp, fn, clipshots_dict, pred_prob_dict = self._eval_per_video_at_threshold(per_vid, th)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            cand = (f1, p, r, tp, fp, fn, th, clipshots_dict, pred_prob_dict)
            if best is None or cand[0] > best[0]:
                best = cand

        assert best is not None
        f1, p, r, tp, fp, fn, best_th, clipshots_dict, pred_prob_dict = best

        if self.save_raw:
            os.makedirs(self.raw_out_dir, exist_ok=True)
            for vid_stem, arr in pred_prob_dict.items():
                np.save(osp.join(self.raw_out_dir, f'{vid_stem}_probs.npy'), arr.squeeze(1))

        if self.save_clipshots_json:
            out_dir = os.path.dirname(self.clipshots_json_path)
            if out_dir != '':
                os.makedirs(out_dir, exist_ok=True)
            with open(self.clipshots_json_path, 'w') as f_out:
                json.dump(clipshots_dict, f_out)

        if self.save_pred_pickle:
            out_dir = os.path.dirname(self.pred_pickle_path)
            if out_dir != '':
                os.makedirs(out_dir, exist_ok=True)
            with open(self.pred_pickle_path, 'wb') as f_out:
                pickle.dump(pred_prob_dict, f_out, protocol=pickle.HIGHEST_PROTOCOL)

        val_th = float(self.threshold)
        val_tp, val_fp, val_fn, _, _ = self._eval_per_video_at_threshold(per_vid, val_th)
        val_p = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        val_r = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
        val_f1 = (2 * val_p * val_r / (val_p + val_r)) if (val_p + val_r) > 0 else 0.0

        fixed_th = 0.5
        fixed_tp, fixed_fp, fixed_fn, _, _ = self._eval_per_video_at_threshold(per_vid, fixed_th)
        fixed_p = fixed_tp / (fixed_tp + fixed_fp) if (fixed_tp + fixed_fp) > 0 else 0.0
        fixed_r = fixed_tp / (fixed_tp + fixed_fn) if (fixed_tp + fixed_fn) > 0 else 0.0
        fixed_f1 = (2 * fixed_p * fixed_r / (fixed_p + fixed_r)) if (fixed_p + fixed_r) > 0 else 0.0
        split_best_thr_delta = abs(float(best_th) - fixed_th)
        split_best_fix05_gap = float(f1) - float(fixed_f1)
        calib_score = float(f1) - 0.10 * split_best_thr_delta

        if self.save_siren_debug:
            out_dir = os.path.dirname(self.siren_debug_path)
            if out_dir != '':
                os.makedirs(out_dir, exist_ok=True)
            with open(self.siren_debug_path, 'wb') as f_out:
                pickle.dump(self._siren_debug_records, f_out, protocol=pickle.HIGHEST_PROTOCOL)

        # Human-readable three-protocol summary block (4 log lines).
        _print_three_protocol_summary(
            name_prefix='SBD',
            val_th=val_th, val_f1=val_f1, val_p=val_p, val_r=val_r,
            val_tp=int(val_tp), val_fp=int(val_fp), val_fn=int(val_fn),
            fix_th=fixed_th, fix_f1=fixed_f1, fix_p=fixed_p, fix_r=fixed_r,
            fix_tp=int(fixed_tp), fix_fp=int(fixed_fp), fix_fn=int(fixed_fn),
            test_th=best_th, test_f1=f1, test_p=p, test_r=r,
            test_tp=int(tp), test_fp=int(fp), test_fn=int(fn),
        )

        # Keyed metric dict, no duplicates.
        # Three protocols x seven fields each = 21 keys, plus seven legacy
        # aliases (== test_best) so existing checkpoint hooks that key on
        # ``sbd_f1`` / ``sbd_best_threshold`` continue to work.
        return dict(
            # ---- val_best protocol (val-selected threshold on this split) ----
            sbd_val_threshold=float(val_th),
            sbd_val_f1=float(val_f1),
            sbd_val_precision=float(val_p),
            sbd_val_recall=float(val_r),
            sbd_val_tp=int(val_tp),
            sbd_val_fp=int(val_fp),
            sbd_val_fn=int(val_fn),

            # ---- fix_05 protocol (deployment-default 0.50) ----
            sbd_fix05_threshold=float(fixed_th),
            sbd_fix05_f1=float(fixed_f1),
            sbd_fix05_precision=float(fixed_p),
            sbd_fix05_recall=float(fixed_r),
            sbd_fix05_tp=int(fixed_tp),
            sbd_fix05_fp=int(fixed_fp),
            sbd_fix05_fn=int(fixed_fn),

            # ---- test_best protocol (oracle on this split) ----
            sbd_test_threshold=float(best_th),
            sbd_test_f1=float(f1),
            sbd_test_precision=float(p),
            sbd_test_recall=float(r),
            sbd_test_tp=int(tp),
            sbd_test_fp=int(fp),
            sbd_test_fn=int(fn),

            # ---- legacy aliases for backward compatibility ----
            # These mirror test_best; checkpoint hooks select best by
            # ``sbd_f1`` and stamp threshold via ``sbd_best_threshold``.
            sbd_best_threshold=float(best_th),
            sbd_f1=float(f1),
            sbd_precision=float(p),
            sbd_recall=float(r),
            sbd_tp=int(tp),
            sbd_fp=int(fp),
            sbd_fn=int(fn),

            # ---- calibration diagnostics ----
            # Positive gap / large delta means the split-oracle peak is sharp
            # and offset from the deployable 0.50 operating point.
            sbd_split_best_thr_delta=float(split_best_thr_delta),
            sbd_split_best_fix05_gap=float(split_best_fix05_gap),
            sbd_calib_score=float(calib_score),

            sbd_use_center_crop=float(1.0 if self.use_center_crop else 0.0),
            sbd_center_keep_len=float(-1.0 if self.center_keep_len is None else float(self.center_keep_len)),
            sbd_center_keep_ratio=float(self.center_keep_ratio),
        )


@METRICS.register_module()
class SBDRandomClipMetric(BaseMetric):
    """
    Random-clip metric for fast validation.
    Works on clip-local evaluation and does not need rawframe/video frame base conversion,
    because it relies on clip_start/clip_end (0-based dataset timeline).

    Here `sbd_best_threshold / sbd_f1 / ...` still mean "threshold scanned on the
    current evaluation split". Clear alias keys are returned as well.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        thresholds: Optional[List[float]] = None,
        n_frames_miss_tolerance: int = 2,

        use_center_crop: bool = False,
        center_keep_len: Optional[int] = None,
        center_keep_ratio: float = 0.5,

        collect_device: str = 'cpu',
    ) -> None:
        super().__init__(collect_device=collect_device)
        self.threshold = float(threshold)
        self.thresholds = thresholds
        self.n_frames_miss_tolerance = int(n_frames_miss_tolerance)

        self.use_center_crop = bool(use_center_crop)
        self.center_keep_len = None if center_keep_len is None else int(center_keep_len)
        self.center_keep_ratio = float(center_keep_ratio)

    @staticmethod
    def _get_field(sample, name: str):
        if hasattr(sample, 'get'):
            val = sample.get(name, None)
            if val is not None:
                return val
        if hasattr(sample, name):
            return getattr(sample, name)
        if isinstance(sample, dict):
            return sample.get(name, None)
        return None

    @staticmethod
    def _predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def _evaluate_scenes(gt_scenes: np.ndarray,
                         pred_scenes: np.ndarray,
                         n_frames_miss_tolerance: int = 2):
        shift = n_frames_miss_tolerance / 2.0
        gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
        pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

        gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
        pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

        i, j = 0, 0
        tp, fp, fn = 0, 0, 0

        while i < len(gt_trans) or j < len(pred_trans):
            if j == len(pred_trans):
                fn += 1
                i += 1
            elif i == len(gt_trans):
                fp += 1
                j += 1
            elif pred_trans[j, 1] < gt_trans[i, 0]:
                fp += 1
                j += 1
            elif pred_trans[j, 0] > gt_trans[i, 1]:
                fn += 1
                i += 1
            else:
                i += 1
                j += 1
                tp += 1

        return int(tp), int(fp), int(fn)

    @staticmethod
    def _transitions_to_scenes(transitions, n_frames: int) -> np.ndarray:
        if n_frames <= 0:
            raise ValueError("n_frames must be > 0")

        if not transitions:
            return np.array([[0, n_frames - 1]], dtype=np.int32)

        transitions = sorted(transitions, key=lambda x: x[0])
        scenes = []
        cur_start = 0

        for s, e in transitions:
            s = int(s)
            e = int(e)

            s = max(0, min(s, n_frames - 1))
            e = max(0, min(e, n_frames))

            if cur_start <= s:
                scenes.append([cur_start, s])
            cur_start = e

        if cur_start <= n_frames - 1:
            scenes.append([cur_start, n_frames - 1])

        return np.asarray(scenes, dtype=np.int32)

    def _center_slice(self, L: int) -> slice:
        if (not self.use_center_crop) or L <= 0:
            return slice(0, L)

        keep_len = self.center_keep_len
        if keep_len is None:
            keep_len = max(1, int(round(L * max(0.1, min(self.center_keep_ratio, 1.0)))))
        keep_len = max(1, min(int(keep_len), L))

        start = (L - keep_len) // 2
        end = start + keep_len
        return slice(start, end)

    def process(self, data_batch, data_samples: List[Any]) -> None:
        for sample in data_samples:
            frame_inds = self._get_field(sample, 'frame_inds')
            if frame_inds is None:
                raise RuntimeError('frame_inds not found in data_sample.')
            if isinstance(frame_inds, torch.Tensor):
                frame_inds = frame_inds.detach().cpu().numpy()
            else:
                frame_inds = np.asarray(frame_inds)

            gt = self._get_field(sample, 'gt_labels')
            if gt is not None:
                if isinstance(gt, torch.Tensor):
                    gt = gt.detach().cpu().numpy().astype(np.int32)
                else:
                    gt = np.asarray(gt).astype(np.int32)

            pred_scores = self._get_field(sample, 'pred_scores')
            if pred_scores is None:
                raise RuntimeError('pred_scores not found in data_sample.')
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.detach().cpu().numpy()
            else:
                pred_scores = np.asarray(pred_scores, dtype=np.float32)

            clip_start = self._get_field(sample, 'clip_start')
            clip_end = self._get_field(sample, 'clip_end')
            if clip_start is None or clip_end is None:
                raise RuntimeError('clip_start/clip_end not found in data_sample.')

            ast_list, gst_list = [], []
            meta_json = self._get_field(sample, 'ast_gst_json')
            if meta_json is not None:
                try:
                    meta = json.loads(meta_json)
                    ast_list = meta.get('ast', []) or []
                    gst_list = meta.get('gst', []) or []
                except Exception:
                    ast_list, gst_list = [], []

            self.results.append(
                dict(
                    frame_inds=frame_inds,
                    gt=gt,
                    pred_scores=pred_scores,
                    clip_start=int(clip_start),
                    clip_end=int(clip_end),
                    ast_list=ast_list,
                    gst_list=gst_list,
                )
            )

    def _eval_one(self, r: Dict[str, Any], th: float) -> Tuple[int, int, int]:
        tol = self.n_frames_miss_tolerance

        pred_scores = np.asarray(r['pred_scores'], dtype=np.float32)
        T = len(pred_scores)
        if T == 0:
            return 0, 0, 0

        sl = self._center_slice(T)
        pred_scores_c = pred_scores[sl]

        clip_start_t = int(r['clip_start'])
        clip_end_t = int(r['clip_end'])
        offset = int(sl.start)

        ast_list = r.get('ast_list', []) or []
        gst_list = r.get('gst_list', []) or []

        one_hot_pred = (pred_scores_c >= th).astype(np.uint8)
        pred_scenes = self._predictions_to_scenes(one_hot_pred)

        gt_transitions_clip: List[List[int]] = []
        for s, e in (ast_list + gst_list):
            s_global = int(s)
            e_global = int(e)

            if e_global < clip_start_t or s_global > clip_end_t:
                continue

            s_clip = max(s_global, clip_start_t)
            e_clip = min(e_global, clip_end_t)

            s_local = (s_clip - clip_start_t) - offset
            e_local = (e_clip - clip_start_t) - offset

            if e_local < 0 or s_local > (len(pred_scores_c) - 1):
                continue
            s_local = max(0, s_local)
            e_local = min(len(pred_scores_c) - 1, e_local)

            if e_local > s_local:
                gt_transitions_clip.append([int(s_local), int(e_local)])

        gt_scenes = self._transitions_to_scenes(gt_transitions_clip, n_frames=len(pred_scores_c))
        tp, fp, fn = self._evaluate_scenes(gt_scenes, pred_scenes, n_frames_miss_tolerance=tol)
        return tp, fp, fn

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        th_list = _safe_threshold_list(self.thresholds)

        def _eval_total_at_threshold(th: float):
            total_tp = total_fp = total_fn = 0
            for r in results:
                tp, fp, fn = self._eval_one(r, th)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            r_ = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2 * p * r_ / (p + r_)) if (p + r_) > 0 else 0.0
            return f1, p, r_, total_tp, total_fp, total_fn

        best = None
        for th in th_list:
            f1, p, r_, total_tp, total_fp, total_fn = _eval_total_at_threshold(th)

            cand = (f1, p, r_, total_tp, total_fp, total_fn, th)
            if best is None or cand[0] > best[0]:
                best = cand

        assert best is not None
        f1, p, r_, tp, fp, fn, best_th = best

        val_th = float(self.threshold)
        val_f1, val_p, val_r, val_tp, val_fp, val_fn = _eval_total_at_threshold(val_th)

        fixed_th = 0.5
        fixed_f1, fixed_p, fixed_r, fixed_tp, fixed_fp, fixed_fn = _eval_total_at_threshold(fixed_th)
        split_best_thr_delta = abs(float(best_th) - fixed_th)
        split_best_fix05_gap = float(f1) - float(fixed_f1)
        calib_score = float(f1) - 0.10 * split_best_thr_delta

        # Human-readable three-protocol summary block (random-clip path).
        _print_three_protocol_summary(
            name_prefix='SBD-rc',
            val_th=val_th, val_f1=val_f1, val_p=val_p, val_r=val_r,
            val_tp=int(val_tp), val_fp=int(val_fp), val_fn=int(val_fn),
            fix_th=fixed_th, fix_f1=fixed_f1, fix_p=fixed_p, fix_r=fixed_r,
            fix_tp=int(fixed_tp), fix_fp=int(fixed_fp), fix_fn=int(fixed_fn),
            test_th=best_th, test_f1=f1, test_p=p, test_r=r_,
            test_tp=int(tp), test_fp=int(fp), test_fn=int(fn),
        )

        # Same three-protocol schema as SBDMetric (no duplicate keys).
        return dict(
            # ---- val_best protocol ----
            sbd_val_threshold=float(val_th),
            sbd_val_f1=float(val_f1),
            sbd_val_precision=float(val_p),
            sbd_val_recall=float(val_r),
            sbd_val_tp=int(val_tp),
            sbd_val_fp=int(val_fp),
            sbd_val_fn=int(val_fn),

            # ---- fix_05 protocol ----
            sbd_fix05_threshold=float(fixed_th),
            sbd_fix05_f1=float(fixed_f1),
            sbd_fix05_precision=float(fixed_p),
            sbd_fix05_recall=float(fixed_r),
            sbd_fix05_tp=int(fixed_tp),
            sbd_fix05_fp=int(fixed_fp),
            sbd_fix05_fn=int(fixed_fn),

            # ---- test_best protocol ----
            sbd_test_threshold=float(best_th),
            sbd_test_f1=float(f1),
            sbd_test_precision=float(p),
            sbd_test_recall=float(r_),
            sbd_test_tp=int(tp),
            sbd_test_fp=int(fp),
            sbd_test_fn=int(fn),

            # ---- legacy aliases ----
            sbd_best_threshold=float(best_th),
            sbd_f1=float(f1),
            sbd_precision=float(p),
            sbd_recall=float(r_),
            sbd_tp=int(tp),
            sbd_fp=int(fp),
            sbd_fn=int(fn),

            # ---- calibration diagnostics ----
            sbd_split_best_thr_delta=float(split_best_thr_delta),
            sbd_split_best_fix05_gap=float(split_best_fix05_gap),
            sbd_calib_score=float(calib_score),
        )
