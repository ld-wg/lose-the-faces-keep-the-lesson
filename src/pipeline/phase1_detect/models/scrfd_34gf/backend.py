"""SCRFD-34GF backend — same SCRFD architecture family as SCRFD-10GF, at a
higher WIDER FACE Hard AP (85.29 vs 83.05), loaded from a standalone
converted `.onnx` file rather than an InsightFace `buffalo_*` pack (no pack
ships it — see `convert.py` in this directory and
research/stages/identification.md, Tier 0.5 items 7/9).

Routed through InsightFace's own `SCRFD` class directly (the same class
`buffalo_l`'s SCRFD-10GF resolves to via its model router — see
identification.md's "Correction" section) rather than the full
`FaceAnalysis` pipeline: we only need detection, not the recognition /
attribute models a full pack would also load.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ...detector import Detection

logger = logging.getLogger(__name__)


class Backend:
    """SCRFD-34GF via a standalone ONNX file (see `convert.py`)."""

    def __init__(
        self,
        conf_threshold: float = 0.3,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        weights: Optional[Path] = None,
    ):
        if weights is None:
            raise ValueError(
                "scrfd-34gf requires --weights pointing at a converted .onnx file "
                "(no pre-built ONNX is publicly distributed for SCRFD-34GF — "
                "convert one with: python -m src.pipeline.phase1_detect.models.scrfd_34gf.convert)"
            )
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"SCRFD-34GF weights not found: {self.weights}\n"
                "Convert them first: python -m src.pipeline.phase1_detect.models.scrfd_34gf.convert "
                f"--checkpoint <downloaded .pth> --output {self.weights}"
            )
        self.conf_threshold = conf_threshold
        self.det_size = det_size
        self.ctx_id = ctx_id
        self._model = None  # lazy load

    def _load(self):
        if self._model is not None:
            return
        try:
            from insightface.model_zoo.model_zoo import ModelRouter
        except ImportError as e:
            raise ImportError(
                "insightface is required for the scrfd-34gf backend. "
                "Install with: pip install insightface onnxruntime"
            ) from e
        logger.info(f"Loading SCRFD-34GF from {self.weights} (det_size={self.det_size})")
        model = ModelRouter(str(self.weights)).get_model()
        model.prepare(ctx_id=self.ctx_id, input_size=self.det_size, det_thresh=self.conf_threshold)
        self._model = model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR frame (OpenCV format).

        Returns a list of Detection sorted by confidence (desc).
        """
        self._load()
        bboxes, kpss = self._model.detect(frame, input_size=self.det_size)
        detections: list[Detection] = []
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2, conf = box
            landmarks = kpss[i] if kpss is not None else None
            detections.append(
                Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    confidence=float(conf), landmarks=landmarks,
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
