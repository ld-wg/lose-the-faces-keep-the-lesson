"""Phase 1 face detector facade — dispatches to a swappable backend.

Decision (research/stages/identification.md): SCRFD-10GF (InsightFace's
`buffalo_l` pack) is the deployed default. SCRFD-34GF and YOLO-FaceV2-l are
the two Tier 0.5 candidates queued for head-to-head comparison on our own
hardware — same method (same tracker, same Phase 1 -> Phase 2 contract in
`contracts.py`), just a different detector underneath. All three implement
`detect(frame) -> list[Detection]` and produce the same `Detection` shape,
so `run.py`, `tracker.py`, and `contracts.py` don't care which one runs.

Backend code lives one-per-model under `models/<name>/backend.py` (see
`models/__init__.py`), imported lazily: only the selected model's module —
and its optional dependency, if any — is ever imported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .models import MODEL_NAMES, load_backend_class

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
    """Facade over the three Phase 1 detector backends (see `MODEL_NAMES`).

    Usage:
        detector = FaceDetector(model="scrfd-10gf", conf_threshold=0.3)  # low threshold -> high recall
        detections = detector.detect(frame_bgr)
    """

    def __init__(
        self,
        model: str = "scrfd-10gf",
        weights: Optional[Path] = None,
        conf_threshold: float = 0.3,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
    ):
        """Args:
            model: which detector to run — one of `MODEL_NAMES`
                ("scrfd-10gf", "scrfd-34gf", "yolo-facev2-l").
            weights: path to that model's weights file. Unused for
                "scrfd-10gf" (resolved by name via InsightFace's own model
                cache); required for the other two — resolve a default path
                under `CONFIG.weights_dir` at the call site if not given
                explicitly (see `run.py`).
            conf_threshold: low default (0.3) favours recall over precision —
                false positives are filtered downstream by tracking.
            det_size: detector input size; larger = better small-face recall, slower.
            ctx_id: 0 for GPU/MPS, -1 for CPU.
        """
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(MODEL_NAMES)}")
        self.model = model
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.det_size = det_size
        self.ctx_id = ctx_id
        self._backend = None  # lazy load

    def _load(self):
        if self._backend is not None:
            return
        backend_cls = load_backend_class(self.model)
        logger.info(f"Loading detector backend '{self.model}' (weights={self.weights})")
        self._backend = backend_cls(
            conf_threshold=self.conf_threshold,
            det_size=self.det_size,
            ctx_id=self.ctx_id,
            weights=self.weights,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR frame (OpenCV format).

        Returns a list of Detection sorted by confidence (desc).
        """
        self._load()
        return self._backend.detect(frame)

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
