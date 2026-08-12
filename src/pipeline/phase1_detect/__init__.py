"""Phase 1 — face detection + tracking.

Decision (research/pipeline.md Part 4): SCRFD-10GF pretrained on WIDER FACE,
inference-only, for the best cost/AP trade-off among verified detectors.
ByteTrack for tracking.
"""

from .detector import FaceDetector, Detection
from .tracker import FaceTracker, Track

__all__ = ["FaceDetector", "Detection", "FaceTracker", "Track"]
