# Dataset layout and PERSIST source map

Author: Tingyu Lin, Computer Vision Lab, TU Wien

Email: [tylin@cvl.tuwien.ac.at](mailto:tylin@cvl.tuwien.ac.at)

This directory holds the (not redistributed) benchmark videos. Download them
from their original sources and lay them out as:

```
data/ClipShots/videos/        # https://github.com/Tangshitao/ClipShots
data/BBC_PlanetEarth/videos/  # BBC Planet Earth episodes (TransNetV2 evaluation set)
data/AutoShot/videos/test/    # https://github.com/wentaozhu/AutoShot
data/HISTORIAN/videos/        # HISTORIAN films
data/_cache/                  # created automatically (decoding caches)
```

## PERSIST-specific source files

The ten PERSIST files added to the `mmaction2/` fork are:

| File | Content |
|---|---|
| `mmaction/models/heads/sbd_ours_head.py` | `SBDOursHead`, the paper's model head: FiLM-conditioned SIREN latent state, the three persistence gates (change / transient-impulse / return), evidence suppression, and all training losses. |
| `mmaction/models/heads/sbd_slowfast_siren_trcs_head.py` | `SlowFastSBDSirenTRCSHead`, the parent class of `SBDOursHead` (shared SIREN/gate machinery). |
| `mmaction/models/backbones/slowfast_transnet_hybrid.py` | `SlowFastTransNetHybridBackbone`, the dual-rate backbone entry point instantiated by the configs. |
| `mmaction/models/backbones/transnetv2_dualrate.py` | `TransNetV2DualRateBackbone`, the dual-rate TransNet trunk (fast + slow pathways with lateral fusion). |
| `mmaction/models/backbones/transnetv2_style.py` | `StackedDDCNNV2`, the TransNetV2-style dilated 3D-convolution cells used by both pathways. |
| `mmaction/datasets/sbd_dataset.py` | `SBDVideoSlidingWindowDataset` and companions: sliding-window video datasets over the per-video `{frame_num, ast, gst}` annotation format. |
| `mmaction/evaluation/metrics/sbd_metric.py` | `SBDMetric`, the three-protocol evaluator (configured / fixed-0.50 / oracle threshold sweep) with the standard +/-2-frame matching rule; also dumps per-frame probabilities. |
| `mmaction/engine/hooks/sbd_siren_epoch_hook.py` | `SBDSirenEpochHook`, pushes the current epoch into the SIREN head during training. |
| `mmaction/engine/hooks/sbd_best_ckpt_name_hook.py` | `SBDAppendValThresholdToBestHook`, appends the validation-swept threshold to the best checkpoint's filename (the val-best selection rule). |
| `tools/gen_eval_cfg.py` | Generates an evaluation config with the checkpoint's validation-selected threshold. |

In addition, `mmaction/datasets/transforms/formatting.py` carries the added
`SBDPackActionInputs` transform inside an otherwise stock file, and the package
`__init__.py` files carry the matching registrations.
