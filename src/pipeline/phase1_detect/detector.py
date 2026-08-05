"""RetinaFace face detector (Phase 1).

Decision (research/pipeline.md Part 4): RetinaFace pretrained on WIDER FACE,
inference-only. Chosen for perfect classroom recall (Ananda et al., ICVEE 2024)
over YOLOv8n (our mAP50 0.616 baseline) — recall is the only currency that
matters in a privacy pipeline (a missed face = a leaked identity).

Backend: InsightFace (buffalo_l pack ships RetinaFace R50 pretrained on WIDER FACE).
Falls back to a clear error if insightface is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single face detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: Optional[np.ndarray] = None  # (5, 2) facial landmarks if available

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class FaceDetector:
    """RetinaFace detector via InsightFace.

    Usage:
        detector = FaceDetector(conf_threshold=0.3)  # low threshold → high recall
        detections = detector.detect(frame_bgr)
    """

    def __init__(
        self,
        conf_threshold: float = 0.3,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        model_name: str = "buffalo_l",
    ):
        """Args:
            conf_threshold: low default (0.3) favours recall over precision —
                false positives are filtered downstream by tracking.
            det_size: detector input size; larger = better small-face recall, slower.
            ctx_id: 0 for GPU/MPS, -1 for CPU.
            model_name: InsightFace model pack (buffalo_l ships RetinaFace R50).
        """
        self.conf_threshold = conf_threshold
        self.det_size = det_size
        self.ctx_id = ctx_id
        self.model_name = model_name
        self._app = None  # lazy load

    def _load(self):
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "insightface is required for Phase 1 detection. "
                "Install with: pip install insightface onnxruntime"
            ) from e
        logger.info(f"Loading InsightFace pack '{self.model_name}' (RetinaFace, det_size={self.det_size})")
        app = FaceAnalysis(name=self.model_name)
        app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        self._app = app

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR frame (OpenCV format).

        Returns a list of Detection sorted by confidence (desc).
        """
        self._load()
        faces = self._app.get(frame)
        detections: list[Detection] = []
        for f in faces:
            conf = float(f.det_score)
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(float, f.bbox)
            landmarks = getattr(f, "kps", None)
            detections.append(
                Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, landmarks=landmarks)
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_with_crops(
        self, frame: np.ndarray, pad: int = 32
    ) -> list[tuple[Detection, np.ndarray]]:
        """Detect faces and return (Detection, padded crop) pairs.

        The 32px default padding follows LDFA's verified trick — gives the
        downstream generator boundary context and hides the seam.
        """
        h, w = frame.shape[:2]
        out = []
        for det in self.detect(frame):
            x1 = max(0, int(det.x1) - pad)
            y1 = max(0, int(det.y1) - pad)
            x2 = min(w, int(det.x2) + pad)
            y2 = min(h, int(det.y2) + pad)
            crop = frame[y1:y2, x1:x2].copy()
            out.append((det, crop))
        return out
