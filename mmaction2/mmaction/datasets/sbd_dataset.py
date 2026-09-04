"""Datasets and sampling utilities for shot boundary detection.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""

import os.path as osp
import json
from copy import deepcopy
from typing import Callable, List, Optional, Union, Dict, Any, Tuple

import numpy as np
from mmengine.fileio import exists

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset

from torch.utils.data import Dataset


def _sigmoid_ramp(x: np.ndarray, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / max(float(k), 1e-6)))


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _time0_to_file1(frame_inds_0based: np.ndarray) -> np.ndarray:
    return frame_inds_0based.astype(np.int64) + 1


def _normalize_transitions_0based(
    frame_num: int,
    ast_list: List[List[int]],
    gst_list: List[List[int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    max_idx = frame_num - 1
    ast_pairs: List[Tuple[int, int]] = []
    gst_pairs: List[Tuple[int, int]] = []

    for s, e in ast_list:
        s = _clamp_int(int(s), 0, max_idx)
        e = _clamp_int(int(e), 0, max_idx)
        if e <= s:
            continue
        ast_pairs.append((s, e))

    for s, e in gst_list:
        s = _clamp_int(int(s), 0, max_idx)
        e = _clamp_int(int(e), 0, max_idx)
        if e <= s:
            continue
        gst_pairs.append((s, e))

    return ast_pairs, gst_pairs


def _build_clip_labels_s_e_0based(
    frame_num: int,
    ast_list: List[List[int]],
    gst_list: List[List[int]],
    clip_start: int,
    clip_end: int,
    boundary_half_width: int = 2,
    soft_edge_k: float = 2.0,
) -> Dict[str, np.ndarray]:
    assert 0 <= clip_start <= clip_end < frame_num
    L = clip_end - clip_start + 1
    gt = np.zeros(L, dtype=np.float32)
    gt_start = np.zeros(L, dtype=np.float32)
    gt_end = np.zeros(L, dtype=np.float32)
    gt_soft = np.zeros(L, dtype=np.float32)

    ast_pairs, gst_pairs = _normalize_transitions_0based(frame_num, ast_list, gst_list)

    def _mark_window(arr: np.ndarray, center_0based: int) -> None:
        s0 = _clamp_int(center_0based - boundary_half_width, 0, frame_num - 1)
        e0 = _clamp_int(center_0based + boundary_half_width, 0, frame_num - 1)
        s_int = max(s0, clip_start)
        e_int = min(e0, clip_end)
        if e_int >= s_int:
            arr[s_int - clip_start : e_int - clip_start + 1] = 1.0

    t_all = np.arange(clip_start, clip_end + 1, dtype=np.float32)

    for s, e in gst_pairs:
        inner_s = s + 1
        inner_e = e - 1
        if inner_e >= inner_s:
            s_int = max(inner_s, clip_start)
            e_int = min(inner_e, clip_end)
            if e_int >= s_int:
                gt[s_int - clip_start : e_int - clip_start + 1] = 1.0
            left = _sigmoid_ramp(t_all - float(inner_s), soft_edge_k)
            right = _sigmoid_ramp(float(inner_e) - t_all, soft_edge_k)
            seg_soft = left * right
            gt_soft = np.maximum(gt_soft, seg_soft.astype(np.float32))
        _mark_window(gt_start, s)
        _mark_window(gt_end, e)

    for s, e in ast_pairs:
        _mark_window(gt, s)
        _mark_window(gt, e)
        _mark_window(gt_start, s)
        _mark_window(gt_end, e)
        _mark_window(gt_soft, s)
        _mark_window(gt_soft, e)

    gt_soft = np.maximum(gt_soft, gt)
    return dict(
        gt_labels=gt.astype(np.float32),
        gt_start=gt_start.astype(np.float32),
        gt_end=gt_end.astype(np.float32),
        gt_soft=gt_soft.astype(np.float32),
    )


def _build_sampling_intervals_0based(
    frame_num: int,
    clip_len: int,
    transitions: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    max_start = frame_num - clip_len
    if max_start < 0:
        return []
    intervals: List[Tuple[int, int]] = []
    for s, e in transitions:
        start_min = max(0, s - clip_len + 1)
        start_max = min(max_start, e)
        if start_max >= start_min:
            intervals.append((start_min, start_max))
    return intervals


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


@DATASETS.register_module()
class SBDMixedRatioDataset(Dataset):
    """Mix a real dataset and a synthetic dataset with a fixed ratio.

    Notes:
    - `real_dataset` and `syn_dataset` should themselves be valid dataset
      configs already registered in `DATASETS`.
    - `epoch_length` controls the number of mixed samples seen per epoch.
    - `mix_real_ratio=0.75` means approximately 3:1 real:synthetic.
    - `set_epoch()` is forwarded to sub-datasets so their random clip sampling
      continues to refresh every epoch.
    """

    def __init__(
        self,
        real_dataset: Dict[str, Any],
        syn_dataset: Dict[str, Any],
        mix_real_ratio: float = 0.75,
        epoch_length: Optional[int] = None,
        fixed_seed: int = 12345,
    ) -> None:
        assert 0.0 < float(mix_real_ratio) < 1.0
        self.real_dataset = DATASETS.build(real_dataset)
        self.syn_dataset = DATASETS.build(syn_dataset)
        self.mix_real_ratio = float(mix_real_ratio)
        self.fixed_seed = int(fixed_seed)
        self._epoch = 0

        if epoch_length is None:
            real_len = max(1, len(self.real_dataset))
            syn_need = int(round(real_len * (1.0 - self.mix_real_ratio) / self.mix_real_ratio))
            self.epoch_length = real_len + max(1, syn_need)
        else:
            self.epoch_length = int(epoch_length)
        assert self.epoch_length > 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if hasattr(self.real_dataset, 'set_epoch'):
            self.real_dataset.set_epoch(epoch)
        if hasattr(self.syn_dataset, 'set_epoch'):
            self.syn_dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return self.epoch_length

    def _normalize_sample_keys(self, sample):
        """Make real/rawframe and synthetic/video samples collate-compatible.

        Important:
        - If the child dataset has already gone through its pipeline and
          returned the final training dict, only keep `inputs` and
          `data_samples`. This prevents backend-specific temporary fields
          such as `video_reader` from entering pseudo_collate.
        """
        if not isinstance(sample, dict):
            return sample

        # Final pipeline output: keep only the universally collate-safe fields.
        if 'inputs' in sample and 'data_samples' in sample:
            return {
                'inputs': sample['inputs'],
                'data_samples': sample['data_samples'],
            }

        sample = deepcopy(sample)

        has_frame_dir = 'frame_dir' in sample
        has_filename = 'filename' in sample

        if has_frame_dir and not has_filename:
            sample['filename'] = sample['frame_dir']
        elif has_filename and not has_frame_dir:
            sample['frame_dir'] = sample['filename']

        # Keep optional meta fields aligned across branches.
        sample.setdefault('video_id', '')
        sample.setdefault('ast_gst_json', '')
        sample.setdefault('filename_tmpl', 'img_{:07}.jpg')

        # Remove common backend-specific temporary keys if present.
        sample.pop('video_reader', None)
        sample.pop('decord_vr', None)
        sample.pop('file_client', None)
        sample.pop('io_backend', None)
        sample.pop('reader', None)

        return sample

    def _rng_for_idx(self, idx: int) -> np.random.RandomState:
        # Stable per-(epoch, idx) RNG so the sample source is reproducible.
        seed = (self.fixed_seed + self._epoch * 1000003 + int(idx) * 9176) & 0xFFFFFFFF
        return np.random.RandomState(seed)

    def __getitem__(self, idx: int):
        rng = self._rng_for_idx(idx)
        if rng.rand() < self.mix_real_ratio:
            sub_idx = int(rng.randint(0, len(self.real_dataset)))
            return self._normalize_sample_keys(self.real_dataset[sub_idx])
        sub_idx = int(rng.randint(0, len(self.syn_dataset)))
        return self._normalize_sample_keys(self.syn_dataset[sub_idx])


@DATASETS.register_module()
class SBDVideoRandomClipDataset(BaseActionDataset):
    """Random one-clip-per-video dataset for video files (Decord).

    Best suited for synthetic 128-frame videos. Each epoch samples one 64-frame
    clip per video. Positive videos preferentially sample a clip intersecting
    AST/GST; hard negatives sample randomly.
    """

    def __init__(
        self,
        ann_file: str,
        pipeline: List[Union[ConfigType, Callable]],
        data_prefix: ConfigType = dict(video=""),
        clip_len: int = 64,
        ast_half_width: int = 1,
        random_shift: bool = True,
        modality: str = "RGB",
        test_mode: bool = False,
        clips_per_video: int = 1,
        fixed_seed: Optional[int] = None,
        pos_sample_prob: float = 1.0,
        boundary_half_width: int = 2,
        soft_edge_k: float = 2.0,
        **kwargs,
    ) -> None:
        self.clip_len = int(clip_len)
        self.ast_half_width = int(ast_half_width)
        self.random_shift = bool(random_shift)
        self.clips_per_video = int(clips_per_video)
        self._epoch = 0
        self.fixed_seed = fixed_seed
        self.pos_sample_prob = float(pos_sample_prob)
        self.boundary_half_width = int(boundary_half_width)
        self.soft_edge_k = float(soft_edge_k)

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=False,
            num_classes=None,
            start_index=0,
            modality=modality,
            **kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if self.random_shift and not self.test_mode:
            self.data_list = self.load_data_list()

    def _load_json_ann(self) -> Dict[str, Any]:
        exists(self.ann_file)
        with open(self.ann_file, "r") as f:
            return json.load(f)

    def _get_rng(self) -> np.random.RandomState:
        if self.fixed_seed is not None:
            seed = int(self.fixed_seed)
        elif not self.random_shift or self.test_mode:
            seed = 12345
        else:
            seed = 12345 + self._epoch
        return np.random.RandomState(seed)

    def _sample_positive_start(
        self,
        frame_num: int,
        transitions: List[Tuple[int, int]],
        rng: np.random.RandomState,
    ) -> int:
        max_start = frame_num - self.clip_len
        if max_start < 0:
            raise RuntimeError("Video too short for the given clip_len.")
        intervals = _build_sampling_intervals_0based(frame_num, self.clip_len, transitions)
        if not intervals:
            return int(rng.randint(0, max_start + 1))
        merged = _merge_intervals(intervals)
        lengths = [e - s + 1 for s, e in merged]
        total_len = int(sum(lengths))
        r = int(rng.randint(0, total_len))
        for (s, e), L in zip(merged, lengths):
            if r < L:
                return int(s + r)
            r -= L
        return int(merged[-1][0])

    def load_data_list(self) -> List[dict]:
        ann = self._load_json_ann()
        data_list: List[dict] = []
        rng = self._get_rng()

        for vid, meta in ann.items():
            frame_num = int(meta["frame_num"])
            ast_list = meta.get("ast", [])
            gst_list = meta.get("gst", [])
            video_prefix = self.data_prefix.get("video", None)
            filename = osp.join(video_prefix, vid) if video_prefix else vid
            vid_stem = vid[:-4] if vid.endswith(".mp4") else vid
            if frame_num < self.clip_len:
                continue

            max_start = frame_num - self.clip_len
            ast_pairs, gst_pairs = _normalize_transitions_0based(frame_num, ast_list, gst_list)
            all_transitions = ast_pairs + gst_pairs

            for _ in range(self.clips_per_video):
                if all_transitions and rng.rand() < self.pos_sample_prob:
                    clip_start = self._sample_positive_start(frame_num, all_transitions, rng)
                else:
                    clip_start = int(rng.randint(0, max_start + 1))
                clip_end = clip_start + self.clip_len - 1

                labels_dict = _build_clip_labels_s_e_0based(
                    frame_num=frame_num,
                    ast_list=ast_list,
                    gst_list=gst_list,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    boundary_half_width=self.boundary_half_width,
                    soft_edge_k=self.soft_edge_k,
                )

                clip_info = dict(
                    filename=filename,
                    frame_dir=filename,  # keep compatibility with rawframe branch
                    filename_tmpl='img_{:07}.jpg',
                    total_frames=frame_num,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    frame_inds=np.arange(clip_start, clip_end + 1, dtype=np.int64),
                    gt_labels=labels_dict["gt_labels"],
                    gt_start=labels_dict["gt_start"],
                    gt_end=labels_dict["gt_end"],
                    gt_soft=labels_dict["gt_soft"],
                    label=int(labels_dict["gt_labels"].max()),
                )
                if self.test_mode:
                    clip_info["video_id"] = vid_stem
                    clip_info["ast_gst_json"] = json.dumps(dict(ast=ast_list, gst=gst_list))
                data_list.append(clip_info)

        print(f"[SBDVideoRandomClipDataset] Epoch {self._epoch} TOTAL clips: {len(data_list)}")
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info.setdefault("filename_tmpl", 'img_{:07}.jpg')
        if "filename" in data_info and "frame_dir" not in data_info:
            data_info["frame_dir"] = data_info["filename"]
        return data_info


@DATASETS.register_module()
class SBDRandomClipDataset(BaseActionDataset):
    """Random one-clip-per-video dataset for shot boundary detection."""

    def __init__(
        self,
        ann_file: str,
        pipeline: List[Union[ConfigType, Callable]],
        data_prefix: ConfigType = dict(img=""),
        filename_tmpl: str = "img_{:07}.jpg",
        clip_len: int = 64,
        clip_stride: int = 32,
        ast_half_width: int = 1,
        random_shift: bool = True,
        modality: str = "RGB",
        test_mode: bool = False,
        clips_per_video: int = 1,
        fixed_seed: Optional[int] = None,
        pos_sample_prob: float = 0.3,
        boundary_half_width: int = 2,
        soft_edge_k: float = 2.0,
        **kwargs,
    ) -> None:
        self.filename_tmpl = filename_tmpl
        self.clip_len = int(clip_len)
        self.clip_stride = int(clip_stride)
        self.ast_half_width = int(ast_half_width)
        self.random_shift = bool(random_shift)
        self.clips_per_video = int(clips_per_video)
        self._epoch = 0
        self.fixed_seed = fixed_seed
        self.pos_sample_prob = float(pos_sample_prob)
        self.boundary_half_width = int(boundary_half_width)
        self.soft_edge_k = float(soft_edge_k)

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=False,
            num_classes=None,
            start_index=0,
            modality=modality,
            **kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if self.random_shift and not self.test_mode:
            self.data_list = self.load_data_list()

    def _load_json_ann(self) -> Dict[str, Any]:
        exists(self.ann_file)
        with open(self.ann_file, "r") as f:
            return json.load(f)

    def _get_rng(self) -> np.random.RandomState:
        if self.fixed_seed is not None:
            seed = int(self.fixed_seed)
        elif not self.random_shift or self.test_mode:
            seed = 12345
        else:
            seed = 12345 + self._epoch
        return np.random.RandomState(seed)

    def _sample_positive_start(
        self,
        frame_num: int,
        transitions: List[Tuple[int, int]],
        rng: np.random.RandomState,
    ) -> int:
        max_start = frame_num - self.clip_len
        if max_start < 0:
            raise RuntimeError("Video too short for the given clip_len.")

        intervals = _build_sampling_intervals_0based(frame_num, self.clip_len, transitions)
        if not intervals:
            return int(rng.randint(0, max_start + 1))

        merged = _merge_intervals(intervals)
        lengths = [e - s + 1 for s, e in merged]
        total_len = int(sum(lengths))

        r = int(rng.randint(0, total_len))
        for (s, e), L in zip(merged, lengths):
            if r < L:
                return int(s + r)
            r -= L
        return int(merged[-1][0])

    def load_data_list(self) -> List[dict]:
        ann = self._load_json_ann()
        data_list: List[dict] = []
        rng = self._get_rng()

        num_videos = 0
        num_pos = 0
        num_neg = 0

        for vid, meta in ann.items():
            frame_num = int(meta["frame_num"])
            ast_list = meta.get("ast", [])
            gst_list = meta.get("gst", [])

            num_videos += 1

            vid_stem = vid[:-4] if vid.endswith(".mp4") else vid
            frame_dir = (
                osp.join(self.data_prefix["img"], vid_stem)
                if self.data_prefix.get("img", None) is not None
                else vid_stem
            )

            if frame_num < self.clip_len:
                continue

            max_start = frame_num - self.clip_len

            ast_pairs, gst_pairs = _normalize_transitions_0based(frame_num, ast_list, gst_list)
            all_transitions = ast_pairs + gst_pairs

            for _ in range(self.clips_per_video):
                if all_transitions and rng.rand() < self.pos_sample_prob:
                    clip_start = self._sample_positive_start(frame_num, all_transitions, rng)
                else:
                    clip_start = int(rng.randint(0, max_start + 1))

                clip_end = clip_start + self.clip_len - 1

                labels_dict = _build_clip_labels_s_e_0based(
                    frame_num=frame_num,
                    ast_list=ast_list,
                    gst_list=gst_list,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    boundary_half_width=self.boundary_half_width,
                    soft_edge_k=self.soft_edge_k,
                )

                gt_labels = labels_dict["gt_labels"]
                if int(gt_labels.max()) > 0:
                    num_pos += 1
                else:
                    num_neg += 1

                frame_inds_0 = np.arange(clip_start, clip_end + 1, dtype=np.int64)
                frame_inds = _time0_to_file1(frame_inds_0)

                clip_info = dict(
                    frame_dir=frame_dir,
                    total_frames=frame_num,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    frame_inds=frame_inds,
                    gt_labels=labels_dict["gt_labels"],
                    gt_start=labels_dict["gt_start"],
                    gt_end=labels_dict["gt_end"],
                    gt_soft=labels_dict["gt_soft"],
                    label=int(gt_labels.max()),
                )

                if self.test_mode:
                    clip_info["video_id"] = vid_stem
                    clip_info["ast_gst_json"] = json.dumps(dict(ast=ast_list, gst=gst_list))

                data_list.append(clip_info)

        print(
            f"[SBDRandomClipDataset] Epoch {self._epoch}  Videos processed: {num_videos}  "
            f"Positive clips: {num_pos}  Negative clips: {num_neg}  TOTAL clips: {len(data_list)}"
        )
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info["filename_tmpl"] = self.filename_tmpl
        return data_info


@DATASETS.register_module()
class SBDSegmentSlidingDataset(BaseActionDataset):
    def __init__(
        self,
        ann_file: str,
        pipeline: List[Union[ConfigType, Callable]],
        data_prefix: ConfigType = dict(img=""),
        filename_tmpl: str = "img_{:07}.jpg",
        clip_len: int = 64,
        clip_stride: int = 32,
        segment_factor: int = 1,
        ast_half_width: int = 1,
        modality: str = "RGB",
        test_mode: bool = False,
        fixed_seed: Optional[int] = None,
        boundary_half_width: int = 2,
        soft_edge_k: float = 2.0,
        **kwargs,
    ) -> None:
        self.filename_tmpl = filename_tmpl
        self.clip_len = int(clip_len)
        self.clip_stride = int(clip_stride)
        self.segment_factor = int(segment_factor)
        self.segment_len = self.segment_factor * self.clip_len
        self.ast_half_width = int(ast_half_width)
        self.fixed_seed = fixed_seed
        self.boundary_half_width = int(boundary_half_width)
        self.soft_edge_k = float(soft_edge_k)
        self._epoch = 0
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=False,
            num_classes=None,
            start_index=0,
            modality=modality,
            **kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if not self.test_mode:
            self.data_list = self.load_data_list()

    def _get_rng(self) -> np.random.RandomState:
        seed = int(self.fixed_seed) if self.fixed_seed is not None else (12345 + int(self._epoch))
        return np.random.RandomState(seed)

    def _load_json_ann(self) -> Dict[str, Any]:
        exists(self.ann_file)
        with open(self.ann_file, "r") as f:
            return json.load(f)

    def load_data_list(self) -> List[dict]:
        ann = self._load_json_ann()
        data_list: List[dict] = []
        rng = self._get_rng()
        for vid, meta in ann.items():
            frame_num = int(meta["frame_num"])
            ast_list = meta.get("ast", [])
            gst_list = meta.get("gst", [])
            vid_stem = vid[:-4] if vid.endswith(".mp4") else vid
            frame_dir = (
                osp.join(self.data_prefix["img"], vid_stem)
                if self.data_prefix.get("img", None) is not None
                else vid_stem
            )
            if frame_num < self.segment_len:
                continue
            max_seg_start = frame_num - self.segment_len
            seg_start = int(rng.randint(0, max_seg_start + 1))
            seg_end = seg_start + self.segment_len - 1
            start = seg_start
            while start + self.clip_len - 1 <= seg_end:
                end = start + self.clip_len - 1
                labels_dict = _build_clip_labels_s_e_0based(
                    frame_num=frame_num,
                    ast_list=ast_list,
                    gst_list=gst_list,
                    clip_start=start,
                    clip_end=end,
                    boundary_half_width=self.boundary_half_width,
                    soft_edge_k=self.soft_edge_k,
                )
                clip_info = dict(
                    frame_dir=frame_dir,
                    total_frames=frame_num,
                    clip_start=start,
                    clip_end=end,
                    frame_inds=_time0_to_file1(np.arange(start, end + 1, dtype=np.int64)),
                    gt_labels=labels_dict["gt_labels"],
                    gt_start=labels_dict["gt_start"],
                    gt_end=labels_dict["gt_end"],
                    gt_soft=labels_dict["gt_soft"],
                    label=int(labels_dict["gt_labels"].max()),
                )
                if self.test_mode:
                    clip_info["video_id"] = vid_stem
                    clip_info["ast_gst_json"] = json.dumps(dict(ast=ast_list, gst=gst_list))
                data_list.append(clip_info)
                start += self.clip_stride
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info["filename_tmpl"] = self.filename_tmpl
        return data_info


@DATASETS.register_module()
class SBDVideoSlidingWindowDataset(BaseActionDataset):
    def __init__(
        self,
        ann_file: str,
        pipeline: List[Union[ConfigType, Callable]],
        data_prefix: ConfigType = dict(video=""),
        clip_len: int = 64,
        clip_stride: int = 32,
        ast_half_width: int = 1,
        random_shift: bool = False,
        modality: str = "RGB",
        test_mode: bool = True,
        neg_keep_ratio: float = 1.0,
        ast_keep_ratio: float = 1.0,
        gst_keep_ratio: float = 1.0,
        boundary_half_width: int = 2,
        soft_edge_k: float = 2.0,
        cache_data_list: bool = False,
        cache_dir: Optional[str] = None,
        progress_interval: int = 200,
        **kwargs,
    ) -> None:
        self.clip_len = int(clip_len)
        self.clip_stride = int(clip_stride)
        self.ast_half_width = int(ast_half_width)
        self.random_shift = bool(random_shift)
        self.neg_keep_ratio = float(neg_keep_ratio)
        self.ast_keep_ratio = float(ast_keep_ratio)
        self.gst_keep_ratio = float(gst_keep_ratio)
        self.boundary_half_width = int(boundary_half_width)
        self.soft_edge_k = float(soft_edge_k)
        self.cache_data_list = bool(cache_data_list)
        self.cache_dir = cache_dir
        self.progress_interval = int(progress_interval)
        self._epoch = 0
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=False,
            num_classes=None,
            start_index=0,
            modality=modality,
            **kwargs,
        )

    def _load_json_ann(self) -> Dict[str, Any]:
        exists(self.ann_file)
        with open(self.ann_file, "r") as f:
            return json.load(f)

    def load_data_list(self) -> List[dict]:
        ann = self._load_json_ann()
        data_list: List[dict] = []
        for vid, meta in ann.items():
            frame_num = int(meta["frame_num"])
            ast_list = meta.get("ast", [])
            gst_list = meta.get("gst", [])
            video_prefix = self.data_prefix.get("video", None)
            filename = osp.join(video_prefix, vid) if video_prefix else vid
            vid_stem = vid[:-4] if vid.endswith(".mp4") else vid
            if frame_num < self.clip_len:
                continue
            start = 0
            while start + self.clip_len - 1 <= frame_num - 1:
                end = start + self.clip_len - 1
                labels_dict = _build_clip_labels_s_e_0based(
                    frame_num=frame_num,
                    ast_list=ast_list,
                    gst_list=gst_list,
                    clip_start=start,
                    clip_end=end,
                    boundary_half_width=self.boundary_half_width,
                    soft_edge_k=self.soft_edge_k,
                )
                clip_info = dict(
                    filename=filename,
                    frame_dir=filename,
                    filename_tmpl='img_{:07}.jpg',
                    total_frames=frame_num,
                    clip_start=start,
                    clip_end=end,
                    frame_inds=np.arange(start, end + 1, dtype=np.int64),
                    gt_labels=labels_dict["gt_labels"],
                    gt_start=labels_dict["gt_start"],
                    gt_end=labels_dict["gt_end"],
                    gt_soft=labels_dict["gt_soft"],
                    label=int(labels_dict["gt_labels"].max()),
                    video_id=vid_stem,
                    ast_gst_json=json.dumps(dict(ast=ast_list, gst=gst_list)),
                )
                data_list.append(clip_info)
                start += self.clip_stride
        return data_list


@DATASETS.register_module()
class SBDMixedScheduledDataset(Dataset):
    """Mix real and synthetic data while traversing real samples systematically.

    This wrapper is intended for the next synthetic-data phase where the real
    branch during mixed finetune should stay closer to the strong
    sliding-window baseline. The key difference from the random-ratio wrapper
    above is:

    - real samples are consumed in a shuffled-but-systematic order each epoch;
    - synthetic samples are interleaved at positions implied by
      `mix_real_ratio`;
    - synthetic indices can still be drawn with replacement.
    """

    def __init__(
        self,
        real_dataset: Dict[str, Any],
        syn_dataset: Dict[str, Any],
        mix_real_ratio: float = 0.75,
        epoch_length: Optional[int] = None,
        fixed_seed: int = 12345,
        real_shuffle: bool = True,
    ) -> None:
        assert 0.0 < float(mix_real_ratio) < 1.0
        self.real_dataset = DATASETS.build(real_dataset)
        self.syn_dataset = DATASETS.build(syn_dataset)
        self.mix_real_ratio = float(mix_real_ratio)
        self.fixed_seed = int(fixed_seed)
        self.real_shuffle = bool(real_shuffle)
        self._epoch = 0

        if epoch_length is None:
            real_len = max(1, len(self.real_dataset))
            syn_need = int(round(real_len * (1.0 - self.mix_real_ratio) / self.mix_real_ratio))
            self.epoch_length = real_len + max(1, syn_need)
        else:
            self.epoch_length = int(epoch_length)
        assert self.epoch_length > 0

        self._source_schedule: List[str] = []
        self._real_indices: List[int] = []
        self._syn_indices: List[int] = []
        self._real_pos_map: List[int] = []
        self._syn_pos_map: List[int] = []
        self._build_epoch_schedule()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if hasattr(self.real_dataset, 'set_epoch'):
            self.real_dataset.set_epoch(epoch)
        if hasattr(self.syn_dataset, 'set_epoch'):
            self.syn_dataset.set_epoch(epoch)
        self._build_epoch_schedule()

    def __len__(self) -> int:
        return self.epoch_length

    def _normalize_sample_keys(self, sample):
        """Make real/rawframe and synthetic/video samples collate-compatible."""
        if not isinstance(sample, dict):
            return sample

        # Final pipeline output: keep only the universally collate-safe fields.
        if 'inputs' in sample and 'data_samples' in sample:
            return {
                'inputs': sample['inputs'],
                'data_samples': sample['data_samples'],
            }

        sample = deepcopy(sample)

        has_frame_dir = 'frame_dir' in sample
        has_filename = 'filename' in sample

        if has_frame_dir and not has_filename:
            sample['filename'] = sample['frame_dir']
        elif has_filename and not has_frame_dir:
            sample['frame_dir'] = sample['filename']

        sample.setdefault('video_id', '')
        sample.setdefault('ast_gst_json', '')
        sample.setdefault('filename_tmpl', 'img_{:07}.jpg')

        sample.pop('video_reader', None)
        sample.pop('decord_vr', None)
        sample.pop('file_client', None)
        sample.pop('io_backend', None)
        sample.pop('reader', None)

        return sample

    def _rng(self, salt: int = 0) -> np.random.RandomState:
        seed = (self.fixed_seed + self._epoch * 1000003 + int(salt) * 9176) & 0xFFFFFFFF
        return np.random.RandomState(seed)

    def _build_real_indices(self, count: int) -> List[int]:
        if count <= 0:
            return []
        real_len = len(self.real_dataset)
        assert real_len > 0
        rng = self._rng(1)
        out: List[int] = []
        while len(out) < count:
            if self.real_shuffle:
                order = rng.permutation(real_len).tolist()
            else:
                order = list(range(real_len))
            need = count - len(out)
            out.extend(order[:need])
        return out

    def _build_syn_indices(self, count: int) -> List[int]:
        if count <= 0:
            return []
        syn_len = len(self.syn_dataset)
        assert syn_len > 0
        rng = self._rng(2)
        return rng.randint(0, syn_len, size=count).astype(np.int64).tolist()

    def _build_epoch_schedule(self) -> None:
        syn_count = int(round(self.epoch_length * (1.0 - self.mix_real_ratio)))
        syn_count = min(max(syn_count, 0), self.epoch_length)
        real_count = self.epoch_length - syn_count

        source_schedule = np.array(['real'] * self.epoch_length, dtype=object)
        if syn_count > 0:
            rng = self._rng(3)
            syn_positions = rng.choice(self.epoch_length, size=syn_count, replace=False)
            source_schedule[syn_positions] = 'syn'

        self._source_schedule = source_schedule.tolist()
        self._real_indices = self._build_real_indices(real_count)
        self._syn_indices = self._build_syn_indices(syn_count)
        self._real_pos_map = [-1] * self.epoch_length
        self._syn_pos_map = [-1] * self.epoch_length

        real_ptr = 0
        syn_ptr = 0
        for i, source in enumerate(self._source_schedule):
            if source == 'real':
                self._real_pos_map[i] = real_ptr
                real_ptr += 1
            else:
                self._syn_pos_map[i] = syn_ptr
                syn_ptr += 1

    def __getitem__(self, idx: int):
        source = self._source_schedule[int(idx)]
        if source == 'real':
            sub_idx = self._real_indices[self._real_pos_map[int(idx)]]
            return self._normalize_sample_keys(self.real_dataset[sub_idx])

        sub_idx = self._syn_indices[self._syn_pos_map[int(idx)]]
        return self._normalize_sample_keys(self.syn_dataset[sub_idx])
