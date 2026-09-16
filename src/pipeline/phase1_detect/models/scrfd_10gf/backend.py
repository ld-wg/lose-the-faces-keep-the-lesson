"""SCRFD-10GF backend — InsightFace's `buffalo_l` pack.

Decision (research/stages/identification.md): SCRFD-10GF pretrained on WIDER
FACE, inference-only. Chosen for the best cost/AP trade-off among verified
detectors — recall is the primary currency in a privacy pipeline (a missed
face = a leaked identity), and SCRFD-10GF gets it at an order of magnitude
fewer params/FLOPs than the alternatives.

`buffalo_l`'s detector is SCRFD-10GF, not RetinaFace, despite RetinaFace being
the commonly assumed default for InsightFace pipelines — verified against
insightface's own model router (see identification.md's "Correction"
section). This backend is otherwise unchanged from Phase 1's original
implementation; the other two candidates in `..scrfd_34gf` / `..yolo_facev2_l`
are new, queued for head-to-head comparison (Tier 0.5).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ...detector import Detection

logger = logging.getLogger(__name__)


class Backend:
    """SCRFD-10GF via InsightFace's `buffalo_l` pack.

    `weights` is accepted for interface parity with the other backends but
    unused: `buffalo_l` is resolved by name through InsightFace's own model
    cache (`~/.insightface`), not a local file in this repo's weights dir.
    """

    def __init__(
        self,
        conf_threshold: float = 0.3,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        weights: Optional[Path] = None,
    ):
        self.conf_threshold = conf_threshold
        self.det_size = det_size
        self.ctx_id = ctx_id
        self._app = None  # lazy load

    def _load(self):
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "insightface is required for the scrfd-10gf backend. "
                "Install with: pip install insightface onnxruntime"
            ) from e
        logger.info(f"Loading InsightFace pack 'buffalo_l' (SCRFD-10GF, det_size={self.det_size})")
        app = FaceAnalysis(name="buffalo_l")
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
