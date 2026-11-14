"""Backbone and head modules for PERSIST inference.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

from .sbd_ours_head import SBDOursHead
from .slowfast_transnet_hybrid import SlowFastTransNetHybridBackbone

__all__ = ["SBDOursHead", "SlowFastTransNetHybridBackbone"]
