"""Public exports for PERSIST shot boundary detection.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

from .api import DEFAULT_REPO, DEFAULT_REVISION, Detector, Result, load

__version__ = "1.1.0"
__all__ = ["DEFAULT_REPO", "DEFAULT_REVISION", "Detector", "Result", "load", "__version__"]
