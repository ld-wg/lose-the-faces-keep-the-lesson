"""Utility probes the evaluator compares between a real face and its anonymized
version: is the behavioral signal kept?

- `EmotionClassifier`: categorical facial expression (8 AffectNet classes)
  from HSEmotion's EfficientNet-B0 (`enet_b0_8_best_vgaf.onnx`, Savchenko
  2022). Code Apache-2.0; weights trained on AffectNet, whose terms allow
  research use only — fine for this thesis, a stated limitation beyond it.
  Downloaded once into `CONFIG.weights_dir/fer/` with its sha256 logged.
- Head pose and age/gender come from buffalo_l (`..pipeline.identity.embedder`
  `PoseEstimator`, `AttributeEstimator`); the 106-landmark expression proxy
  from `Landmark106`.
- Gaze is not probed: no permissively licensed estimator was validated on
  faces this small (experiments/DECISIONS.md 10, a stated limitation).

Every probe returns scalars or labels; nothing is written (LGPD).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402

logger = logging.getLogger(__name__)

EMOTION_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
               "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx")
EMOTION_FILENAME = "enet_b0_8_best_vgaf.onnx"
EMOTIONS = ("anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
EMOTION_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        logger.info(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as r, tmp_path.open("wb") as f:
            shutil.copyfileobj(r, f)
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info(f"{dest.name} sha256 {hashlib.sha256(dest.read_bytes()).hexdigest()}")


class EmotionClassifier:
    """8-class expression probabilities for the face in `box`."""

    def __init__(self, onnx_path: Optional[Path] = None, ctx_id: int = 0):
        self.onnx_path = Path(onnx_path) if onnx_path else CONFIG.weights_dir / "fer" / EMOTION_FILENAME
        self.ctx_id = ctx_id
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        import onnxruntime as ort

        if not self.onnx_path.is_file():
            _download(EMOTION_URL, self.onnx_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.ctx_id >= 0 else ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        self._session = ort.InferenceSession(str(self.onnx_path),
                                             providers=[p for p in providers if p in available])
        self._input = self._session.get_inputs()[0].name

    def probabilities(self, image_bgr: np.ndarray, box: Sequence[float]) -> np.ndarray:
        self._load()
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        crop = image_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return np.full(len(EMOTIONS), 1.0 / len(EMOTIONS), dtype=np.float32)
        x = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (EMOTION_SIZE, EMOTION_SIZE)).astype(np.float32) / 255
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        logits = self._session.run(None, {self._input: x})[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        return (e / e.sum()).astype(np.float32)


def pose_error(real: tuple[float, float, float], anon: tuple[float, float, float]) -> float:
    """Mean absolute difference of (pitch, yaw, roll), degrees — the usual head-pose MAE."""
    return float(np.mean(np.abs(np.subtract(real, anon))))
