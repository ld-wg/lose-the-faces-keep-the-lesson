"""Phase 1 — face detection + tracking.

Decision (research/pipeline.md Part 4): RetinaFace pretrained on WIDER FACE,
inference-only, for maximum classroom recall. ByteTrack for tracking.
"""

from .detector import FaceDetector, Detection
from .tracker import FaceTracker, Track

__all__ = ["FaceDetector", "Detection", "FaceTracker", "Track"]
