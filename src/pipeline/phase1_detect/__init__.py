"""Phase 1 — face detection + tracking.

Decision (research/stages/identification.md): SCRFD-10GF pretrained on WIDER
FACE, inference-only, for the best cost/AP trade-off among verified
detectors — the deployed default. SCRFD-34GF and YOLO-FaceV2-l are the two
Tier 0.5 candidates also selectable via `FaceDetector(model=...)` for
head-to-head comparison (see `models/`). ByteTrack for tracking.
"""

from .detector import FaceDetector, Detection
from .models import MODEL_NAMES
from .tracker import FaceTracker, Track

__all__ = ["FaceDetector", "Detection", "FaceTracker", "Track", "MODEL_NAMES"]
