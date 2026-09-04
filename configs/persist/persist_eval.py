"""Evaluate PERSIST with the validation-selected operating threshold.

Author: Tingyu Lin, Computer Vision Lab, TU Wien
Email: tylin@cvl.tuwien.ac.at
"""

# ============================== User settings ==============================
# 1) Benchmark - set ANN_FILE / VIDEO_ROOT / CACHE_DIR together. Presets:
#    ClipShots:  annotations/ClipShots/test.json; data/ClipShots/videos;
#                data/_cache/crossdomain/clipshots
#    BBC:        annotations/BBC_PlanetEarth/test.json;
#                data/BBC_PlanetEarth/videos; data/_cache/crossdomain/bbc
#    BBC_fixed:  annotations/BBC_fixed/test.json; data/BBC_PlanetEarth/videos;
#                data/_cache/crossdomain/bbc_fixed
#    AutoShot:   annotations/AutoShot/test.json; data/AutoShot/videos/test;
#                data/_cache/crossdomain/autoshot
#    HIST-GST12: annotations/HISTORIAN/test_gst12.json; data/HISTORIAN/videos;
#                data/_cache/crossdomain/hist_gst12
#    Synth diag: diagnostic/annotations/test_diag_all.json; diagnostic/videos;
#                data/_cache/crossdomain/synth_diag
# 2) Seed - set CHECKPOINT / THRESHOLD / SEED together:
#    seed 11  checkpoints/persist_seed11_epoch10_valthr0.68.pth  THRESHOLD = 0.68
#    seed 1   checkpoints/persist_seed1_epoch7_valthr0.65.pth    THRESHOLD = 0.65
#    seed 21  checkpoints/persist_seed21_epoch8_valthr0.65.pth   THRESHOLD = 0.65
# 3) OUT_DIR - where metrics and per-frame probabilities are written.
# (The train/val dataloader entries keep their ClipShots defaults; they are
#  not used by tools/test.py.)
# ===========================================================================
ANN_FILE   = 'annotations/ClipShots/test.json'
VIDEO_ROOT = 'data/ClipShots/videos'
CACHE_DIR  = 'data/_cache/crossdomain/clipshots'
CHECKPOINT = 'checkpoints/persist_seed11_epoch10_valthr0.68.pth'
THRESHOLD  = 0.68
SEED       = 11
OUT_DIR    = 'work_dirs/persist_eval'
# ===========================================================================

ann_file_test = ANN_FILE
ann_file_train = 'annotations/ClipShots/train.json'
ann_file_val = 'annotations/ClipShots/val.json'
ast_half_width = 1
auto_scale_lr = dict(base_batch_size=224, enable=False)
best_threshold = THRESHOLD
clip_len = 64
custom_hooks = [
    dict(type='SBDSirenEpochHook'),
    dict(
        best_glob='best_*.pth',
        priority='LOWEST',
        scale=100,
        threshold_key='sbd_best_threshold',
        type='SBDAppendValThresholdToBestHook'),
]
data_root_test_video = VIDEO_ROOT
data_root_train_video = 'data/ClipShots/videos'
data_root_val_video = 'data/ClipShots/videos'
dataset_cache_root = CACHE_DIR
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best=[
            'sbd_fix05_f1',
            'sbd_f1',
        ],
        type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl', timeout=2400),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
epochs = 10
eval_root = OUT_DIR + '/eval_outputs'
exp_root = OUT_DIR
experiment_base = 'work_dirs'
experiment_group = 'persist'
experiment_name = 'persist_eval'
gpu_num = 4
is_main_rank = True
launcher = 'none'
load_from = CHECKPOINT
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
lr = 0.03
model = dict(
    backbone=dict(
        backbone_mode='transnet',
        enable_slow_path=True,
        enable_transnet_fast_adapter=False,
        enable_transnet_slow_adapter=False,
        output_mode='tuple',
        return_transnet_block_features=True,
        return_transnet_resized_inputs=True,
        transnet_cfg=dict(
            batch_norm=True,
            enable_slow=True,
            fast_base_filters=16,
            fast_blocks_per_stage=2,
            fast_num_stages=3,
            in_channels=3,
            input_size=(
                27,
                48,
            ),
            lateral_ratio=0.25,
            pool_type='avg',
            resize_input=True,
            separable=True,
            shortcut=True,
            slow_base_filters=32,
            slow_blocks_per_stage=2,
            slow_num_stages=3,
            slow_temporal_stride=8,
            use_lateral=True),
        type='SlowFastTransNetHybridBackbone'),
    cls_head=dict(
        bg_hard_gamma=2.0,
        bg_hard_power=1.0,
        boundary_pos_weight=5.0,
        boundary_suppress_kernel=9,
        compute_dyn_in_eval=False,
        cond_gamma_init=0.5,
        cond_t_smooth_kernel=5,
        dropout_ratio=0.5,
        dyn_dd_weight=0.25,
        dyn_energy_bias_init=0.0,
        dyn_energy_clip=10.0,
        dyn_energy_norm='mean',
        dyn_energy_scale_init=5.0,
        dyn_pos_weight=5.0,
        dyn_start_epoch=0,
        dyn_use_second_derivative=True,
        enable_siren=True,
        enable_trcs=True,
        feature_mode='pooled',
        freeze_inactive_params=False,
        in_channels_fast=256,
        in_channels_slow=512,
        init_boundary_beta=0.15,
        init_logit_bias=0.0,
        init_std=0.01,
        init_temperature=1.0,
        input_mode='transnet',
        lambda_base_aux=1.0,
        lambda_bg_suppress=0.1,
        lambda_boundary=1.0,
        lambda_dyn_energy=0.01,
        lambda_phi_distill=0.2,
        lambda_phi_tv=0.01,
        lambda_siren_cls=0.05,
        lambda_trcs_loss=0.05,
        loss_cls=dict(type='BCELossWithLogits'),
        main_loss_mode='pos_weight',
        max_temperature=4.0,
        min_temperature=0.7,
        pos_weight=2.0,
        save_debug_fields=True,
        siren_cond_dim=32,
        siren_condition_source='auto',
        siren_conditioned=True,
        siren_final_w_init=0.0001,
        siren_hidden_dim=128,
        siren_num_hidden_layers=3,
        siren_omega_0=30.0,
        siren_phi_dim=16,
        siren_start_epoch=0,
        siren_use_timestep_cond=True,
        trcs_change_source='ev',
        trcs_change_temp=0.05,
        trcs_change_thr=0.6,
        trcs_innov_temp=0.03,
        trcs_innov_thr=0.05,
        trcs_offsets=(
            4,
            8,
            12,
        ),
        trcs_return_temp=0.1,
        trcs_return_thr=0.6,
        trcs_start_epoch=0,
        trcs_strength=0.4,
        trcs_use_innov=True,
        type='SBDOursHead',
        use_boundary_factor=True,
        use_color_histograms=False,
        use_dyn_in_boundary_evidence=False,
        use_frame_similarity=False,
        use_logit_calibration=True,
        use_many_hot_targets=False),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.03, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            dyn_energy_bias=dict(decay_mult=0.0, lr_mult=0.1),
            log_dyn_energy_scale=dict(decay_mult=0.0, lr_mult=0.1),
            logit_=dict(decay_mult=0.0, lr_mult=0.1),
            seq_siren_cond_proj=dict(lr_mult=0.1),
            siren=dict(lr_mult=0.1),
            siren_cond_proj=dict(lr_mult=0.1))))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        start_factor=0.1,
        type='LinearLR'),
    dict(by_epoch=True, gamma=0.5, milestones=[
        2,
        5,
        8,
    ], type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=SEED)
rank = 0
resume = False
test_ast_half_width = 1
test_batch_size = 64
test_cfg = dict(type='TestLoop')
test_clip_stride = 16
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file=
        ANN_FILE,
        ast_half_width=1,
        ast_keep_ratio=1.0,
        cache_data_list=True,
        cache_dir=
        CACHE_DIR + '/test',
        clip_len=64,
        clip_stride=16,
        data_prefix=dict(
            video=VIDEO_ROOT),
        gst_keep_ratio=1.0,
        neg_keep_ratio=1.0,
        pipeline=[
            dict(type='DecordInit'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(crop_size=112, type='CenterCrop'),
            dict(keep_ratio=False, scale=(
                112,
                112,
            ), type='Resize'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                extra_gt_keys=(
                    'gt_start',
                    'gt_end',
                    'gt_soft',
                ),
                gt_key='gt_labels',
                input_key='imgs',
                meta_keys=(
                    'frame_inds',
                    'clip_start',
                    'clip_end',
                    'video_id',
                    'ast_gst_json',
                ),
                type='SBDPackActionInputs'),
        ],
        progress_interval=200,
        random_shift=False,
        test_mode=True,
        type='SBDVideoSlidingWindowDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataset_type = 'SBDVideoSlidingWindowDataset'
test_evaluator = dict(
    center_keep_len=16,
    center_keep_ratio=0.5,
    clipshots_json_path='',
    n_frames_miss_tolerance=2,
    pred_pickle_path=
    OUT_DIR + '/eval_outputs/preds_prob.pkl',
    raw_out_dir='',
    restrict_to_pred_span=False,
    save_clipshots_json=False,
    save_pred_pickle=True,
    save_raw=False,
    save_siren_debug=False,
    siren_debug_path='',
    threshold=THRESHOLD,
    thresholds=[
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.12,
        0.14,
        0.16,
        0.18,
        0.2,
        0.22,
        0.24,
        0.26,
        0.28,
        0.3,
        0.32,
        0.34,
        0.36,
        0.38,
        0.4,
        0.42,
        0.44,
        0.46,
        0.48,
        0.5,
        0.52,
        0.54,
        0.56,
        0.58,
        0.6,
        0.62,
        0.65,
        0.68,
        0.71,
        0.74,
        0.77,
        0.8,
        0.83,
        0.86,
        0.89,
        0.92,
        0.95,
    ],
    type='SBDMetric',
    use_center_crop=True)
test_num_workers = 8
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(crop_size=112, type='CenterCrop'),
    dict(keep_ratio=False, scale=(
        112,
        112,
    ), type='Resize'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(
        extra_gt_keys=(
            'gt_start',
            'gt_end',
            'gt_soft',
        ),
        gt_key='gt_labels',
        input_key='imgs',
        meta_keys=(
            'frame_inds',
            'clip_start',
            'clip_end',
            'video_id',
            'ast_gst_json',
        ),
        type='SBDPackActionInputs'),
]
thresholds = [
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.12,
    0.14,
    0.16,
    0.18,
    0.2,
    0.22,
    0.24,
    0.26,
    0.28,
    0.3,
    0.32,
    0.34,
    0.36,
    0.38,
    0.4,
    0.42,
    0.44,
    0.46,
    0.48,
    0.5,
    0.52,
    0.54,
    0.56,
    0.58,
    0.6,
    0.62,
    0.65,
    0.68,
    0.71,
    0.74,
    0.77,
    0.8,
    0.83,
    0.86,
    0.89,
    0.92,
    0.95,
]
train_batch_size = 56
train_cfg = dict(
    max_epochs=10, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_clip_stride = 64
train_dataloader = dict(
    batch_size=56,
    dataset=dict(
        ann_file=
        'annotations/ClipShots/train.json',
        ast_half_width=1,
        ast_keep_ratio=1.0,
        cache_data_list=True,
        cache_dir=
        'data/_cache/ClipShots/train',
        clip_len=64,
        clip_stride=64,
        data_prefix=dict(
            video='data/ClipShots/videos'),
        gst_keep_ratio=1.0,
        neg_keep_ratio=1.0,
        pipeline=[
            dict(type='DecordInit'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(
                input_size=112,
                max_wh_scale_gap=0,
                random_crop=False,
                scales=(
                    1,
                    0.8,
                ),
                type='MultiScaleCrop'),
            dict(crop_size=112, type='CenterCrop'),
            dict(keep_ratio=False, scale=(
                112,
                112,
            ), type='Resize'),
            dict(
                brightness=0.4,
                contrast=0.4,
                hue=0.1,
                saturation=0.4,
                type='ColorJitter'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                extra_gt_keys=(
                    'gt_start',
                    'gt_end',
                    'gt_soft',
                ),
                gt_key='gt_labels',
                input_key='imgs',
                meta_keys=(
                    'frame_inds',
                    'clip_start',
                    'clip_end',
                    'video_id',
                    'ast_gst_json',
                ),
                type='SBDPackActionInputs'),
        ],
        progress_interval=200,
        random_shift=False,
        test_mode=False,
        type='SBDVideoSlidingWindowDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset_type = 'SBDVideoSlidingWindowDataset'
train_num_workers = 8
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(
        input_size=112,
        max_wh_scale_gap=0,
        random_crop=False,
        scales=(
            1,
            0.8,
        ),
        type='MultiScaleCrop'),
    dict(crop_size=112, type='CenterCrop'),
    dict(keep_ratio=False, scale=(
        112,
        112,
    ), type='Resize'),
    dict(
        brightness=0.4,
        contrast=0.4,
        hue=0.1,
        saturation=0.4,
        type='ColorJitter'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(
        extra_gt_keys=(
            'gt_start',
            'gt_end',
            'gt_soft',
        ),
        gt_key='gt_labels',
        input_key='imgs',
        meta_keys=(
            'frame_inds',
            'clip_start',
            'clip_end',
            'video_id',
            'ast_gst_json',
        ),
        type='SBDPackActionInputs'),
]
use_wandb = False
val_batch_size = 48
val_cfg = dict(type='ValLoop')
val_clip_stride = 16
val_dataloader = dict(
    batch_size=48,
    dataset=dict(
        ann_file=
        'annotations/ClipShots/val.json',
        ast_half_width=1,
        ast_keep_ratio=1.0,
        cache_data_list=True,
        cache_dir='data/_cache/ClipShots/val',
        clip_len=64,
        clip_stride=16,
        data_prefix=dict(
            video='data/ClipShots/videos'),
        gst_keep_ratio=1.0,
        neg_keep_ratio=1.0,
        pipeline=[
            dict(type='DecordInit'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(crop_size=112, type='CenterCrop'),
            dict(keep_ratio=False, scale=(
                112,
                112,
            ), type='Resize'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(
                extra_gt_keys=(
                    'gt_start',
                    'gt_end',
                    'gt_soft',
                ),
                gt_key='gt_labels',
                input_key='imgs',
                meta_keys=(
                    'frame_inds',
                    'clip_start',
                    'clip_end',
                    'video_id',
                    'ast_gst_json',
                ),
                type='SBDPackActionInputs'),
        ],
        progress_interval=200,
        random_shift=False,
        test_mode=True,
        type='SBDVideoSlidingWindowDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataset_type = 'SBDVideoSlidingWindowDataset'
val_evaluator = dict(
    center_keep_len=16,
    center_keep_ratio=0.5,
    n_frames_miss_tolerance=2,
    restrict_to_pred_span=False,
    save_clipshots_json=False,
    save_pred_pickle=False,
    save_raw=False,
    save_siren_debug=False,
    threshold=THRESHOLD,
    thresholds=[
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.12,
        0.14,
        0.16,
        0.18,
        0.2,
        0.22,
        0.24,
        0.26,
        0.28,
        0.3,
        0.32,
        0.34,
        0.36,
        0.38,
        0.4,
        0.42,
        0.44,
        0.46,
        0.48,
        0.5,
        0.52,
        0.54,
        0.56,
        0.58,
        0.6,
        0.62,
        0.65,
        0.68,
        0.71,
        0.74,
        0.77,
        0.8,
        0.83,
        0.86,
        0.89,
        0.92,
        0.95,
    ],
    type='SBDMetric',
    use_center_crop=True)
val_num_workers = 4
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(crop_size=112, type='CenterCrop'),
    dict(keep_ratio=False, scale=(
        112,
        112,
    ), type='Resize'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(
        extra_gt_keys=(
            'gt_start',
            'gt_end',
            'gt_soft',
        ),
        gt_key='gt_labels',
        input_key='imgs',
        meta_keys=(
            'frame_inds',
            'clip_start',
            'clip_end',
            'video_id',
            'ast_gst_json',
        ),
        type='SBDPackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Visualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = OUT_DIR
