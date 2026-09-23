"""Phase 2 face generator facade — dispatches to a swappable backend.

Mirrors `phase1_detect.detector.FaceDetector`'s shape (lazy backend load,
one public method), with one deliberate difference: Phase 1's three
detector backends all share one exact constructor signature, so `FaceDetector`
could hardcode it. Phase 2's backends don't (CIAGAN needs `num_classes`/
`img_size`/`dlib_predictor`; a future diffusion backend would need entirely
different knobs) — so `FaceGenerator` forwards arbitrary backend-specific
kwargs instead of enumerating them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .models import MODEL_NAMES, load_backend_class

logger = logging.getLogger(__name__)


class FaceGenerator:
    """Facade over the Phase 2 generation backends (see `MODEL_NAMES`).

    Usage:
        gen = FaceGenerator(model="ciagan", weights=..., dlib_predictor=...)
        out = gen.generate(face_crop_bgr, identity.seed)
    """

    def __init__(
        self,
        model: str = "ciagan",
        weights: Optional[Path] = None,
        ctx_id: int = 0,
        **backend_kwargs,
    ):
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(MODEL_NAMES)}")
        self.model = model
        self.weights = weights
        self.ctx_id = ctx_id
        self._backend_kwargs = backend_kwargs
        self._backend = None  # lazy load

    def _load(self):
        if self._backend is not None:
            return
        backend_cls = load_backend_class(self.model)
        logger.info(f"Loading generation backend '{self.model}' (weights={self.weights})")
        self._backend = backend_cls(weights=self.weights, ctx_id=self.ctx_id, **self._backend_kwargs)

    def generate(self, crop: np.ndarray, seed: int) -> Optional[np.ndarray]:
        """Anonymize the face in a BGR crop, seeded by `seed`.

        Returns a same-shape/dtype image, or None if the backend couldn't
        produce output for this crop (e.g. no usable landmarks) — caller
        decides the fallback (see `run.py`).
        """
        self._load()
        return self._backend.generate(crop, seed)

    def identity_class(self, seed: int) -> int:
        """The backend's deterministic identity index for `seed`, for logging."""
        self._load()
        return self._backend.identity_class(seed)
