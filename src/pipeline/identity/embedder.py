"""ArcFace embeddings and head pose from insightface's `buffalo_l` pack (ONNX).

Both models ship in the same pack Phase 1's SCRFD-10GF detector already
loads (`FaceAnalysis(name="buffalo_l")` loads all five models and runs
them per face, then `scrfd_10gf/backend.py` keeps only box, score and
keypoints). This module reuses two of the discarded ones, fed with the
5-point landmarks Phase 1 already wrote to `detections.jsonl`, so no
second detection pass is needed:

- `w600k_r50.onnx`: ArcFace ResNet-50 trained on WebFace600K. The same
  architecture and training set as the `arcface_w600k_r50.onnx`
  FaceFusion uses inside the BLANKET bridge, with the same alignment
  template and `(x-127.5)/127.5` RGB normalization (verified in both
  sources 2026-09-25) — one embedding space across stages.
- `1k3d68.onnx`: 3D 68-point landmarks; insightface derives
  `pose = (pitch, yaw, roll)` from them (`model_zoo/landmark.py`).

No new dependency: insightface and onnxruntime(-gpu) are base
dependencies of this project.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import align

logger = logging.getLogger(__name__)

BUFFALO_L = "buffalo_l"
ARCFACE_FILENAME = "w600k_r50.onnx"
POSE_FILENAME = "1k3d68.onnx"
LANDMARK_106_FILENAME = "2d106det.onnx"


def buffalo_model_path(filename: str, root: str = "~/.insightface") -> Path:
    """Path to one model file of the buffalo_l pack, downloading the pack if absent
    (the same cache Phase 1's detector uses)."""
    from insightface.utils import ensure_available

    model_dir = ensure_available("models", BUFFALO_L, root=root)
    return Path(model_dir) / filename


def _load_onnx_model(path: Path, ctx_id: int):
    from insightface.model_zoo import get_model
    from insightface.model_zoo.onnxruntime_utils import get_default_providers

    logger.info(f"Loading {path.name} from {path}")
    model = get_model(str(path), providers=get_default_providers())
    if model is None:
        raise RuntimeError(f"insightface could not route {path} to a known model type")
    model.prepare(ctx_id=ctx_id)
    return model


class ArcFaceEmbedder:
    """512-d ArcFace embeddings (`w600k_r50`), raw — not L2-normalized.

    The raw norm is kept because whether it tracks image quality for this
    standard (non-MagFace) model is an open question the identity report
    measures (plan D3); callers normalize themselves.
    """

    def __init__(self, onnx_path: Optional[Path] = None, ctx_id: int = 0):
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.ctx_id = ctx_id
        self._model = None

    def _load(self):
        if self._model is None:
            path = self.onnx_path or buffalo_model_path(ARCFACE_FILENAME)
            self._model = _load_onnx_model(path, self.ctx_id)

    def embed(self, image_bgr: np.ndarray, landmarks: Sequence[Sequence[float]]) -> np.ndarray:
        """Embedding of the face whose 5 landmarks (in `image_bgr`'s coordinates) are given."""
        return self.embed_aligned(align.warp(image_bgr, landmarks, align.ARCFACE_SIZE))

    def embed_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        self._load()
        return self._model.get_feat(aligned_bgr).ravel().astype(np.float32)


class PoseEstimator:
    """(pitch, yaw, roll) in degrees from buffalo_l's 3D-68 landmark model."""

    def __init__(self, onnx_path: Optional[Path] = None, ctx_id: int = 0):
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.ctx_id = ctx_id
        self._model = None

    def _load(self):
        if self._model is None:
            path = self.onnx_path or buffalo_model_path(POSE_FILENAME)
            self._model = _load_onnx_model(path, self.ctx_id)

    def pose(self, image_bgr: np.ndarray, box: Sequence[float]) -> tuple[float, float, float]:
        from insightface.app.common import Face

        self._load()
        face = Face(bbox=np.asarray(box, dtype=np.float32))
        self._model.get(image_bgr, face)
        pitch, yaw, roll = (float(v) for v in face.pose)
        return pitch, yaw, roll


class Landmark106:
    """106 2D landmarks from buffalo_l's `2d106det` model — the evaluation
    harness's expression proxy compares these between real and anonymized crops."""

    def __init__(self, onnx_path: Optional[Path] = None, ctx_id: int = 0):
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.ctx_id = ctx_id
        self._model = None

    def _load(self):
        if self._model is None:
            path = self.onnx_path or buffalo_model_path(LANDMARK_106_FILENAME)
            self._model = _load_onnx_model(path, self.ctx_id)

    def landmarks(self, image_bgr: np.ndarray, box: Sequence[float]) -> np.ndarray:
        from insightface.app.common import Face

        self._load()
        face = Face(bbox=np.asarray(box, dtype=np.float32))
        return np.asarray(self._model.get(image_bgr, face), dtype=np.float32)[:, :2]
