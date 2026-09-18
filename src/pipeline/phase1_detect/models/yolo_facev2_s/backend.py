"""YOLO-FaceV2-s backend — loads a converted ONNX graph (see `convert.py`).

"-s" (small) specifically, not "-l" (large) as originally planned: `-l`/`-m`/
`-n` are functionally broken in the published v2.1 release (near-zero
backbone variance on real images, confirmed by direct inspection) — `-s` is
the only size that actually works. See `convert.py`'s docstring and
`NOTICE.md` for the full investigation.

Same shape as `..scrfd_10gf`/`..scrfd_34gf`: an .onnx file + `onnxruntime`,
nothing else. `vendor/`'s PyTorch model code and the licensing situation it
carries (see NOTICE.md) matter only for the one-time conversion in
`convert.py` — this module has no `torch` dependency.

Also the better choice on this project's actual hardware (Apple Silicon, no
GPU cluster): `onnxruntime` gets a CoreML execution provider (Neural
Engine/GPU), while PyTorch's MPS backend has known gaps for less-common ops
like this model's CBAM/SE/EMA attention blocks, silently falling back to
CPU per-op. It also puts all three Tier 0.5 candidates on the same runtime,
so latency comparisons reflect the architectures, not the backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ...detector import Detection

logger = logging.getLogger(__name__)

#: Matches upstream's own `detect_face.py` default (`iou_thres=0.5`). Not
#: exposed as a constructor parameter since the shared `Backend` interface
#: (see `..scrfd_10gf`/`..scrfd_34gf`) only has one threshold knob.
_IOU_THRESHOLD = 0.5

#: YOLOv5-family models downsample by 32x; letterbox padding is rounded to
#: this stride so every feature-map size the network sees stays integral.
_STRIDE = 32


def _letterbox(
    img: np.ndarray, new_shape: tuple[int, int], stride: int = _STRIDE, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize+pad to `new_shape`, preserving aspect ratio, padded to a stride multiple.

    Standard YOLO preprocessing (same algorithm as e.g. ultralytics/yolov5's
    own `utils/datasets.py`) — generic, reimplemented here rather than
    imported from `vendor/` so this backend carries no vendored-code
    dependency at inference time.
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / h, new_w / w)
    unpad_w, unpad_h = int(round(w * r)), int(round(h * r))
    dw, dh = (new_w - unpad_w) % stride, (new_h - unpad_h) % stride
    dw, dh = dw / 2, dh / 2

    if (w, h) != (unpad_w, unpad_h):
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Standard greedy IoU-based NMS on xyxy boxes, sorted by score.

    Same algorithm as `tracker.py`'s `_iou()`-based association, applied
    here to suppress overlapping detections rather than associate tracks
    across frames — generic, not specific to this model.
    """
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / np.maximum(area_i + area_rest - inter, 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


class Backend:
    """YOLO-FaceV2-s via a converted ONNX graph (see `convert.py`)."""

    def __init__(
        self,
        conf_threshold: float = 0.3,
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        weights: Optional[Path] = None,
    ):
        if weights is None:
            raise ValueError(
                "yolo-facev2-s requires --weights pointing at a converted .onnx file "
                "(no pre-built ONNX is distributed — convert one with: "
                "python -m src.pipeline.phase1_detect.models.yolo_facev2_s.convert)"
            )
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"YOLO-FaceV2-s weights not found: {self.weights}\n"
                "Convert them first: download 'yolo-facev2s-preweight.pt' from "
                "https://github.com/Krasjet-Yu/YOLO-FaceV2/releases, then run "
                "python -m src.pipeline.phase1_detect.models.yolo_facev2_s.convert "
                f"--checkpoint <downloaded .pt> --output {self.weights}"
            )
        self.conf_threshold = conf_threshold
        self.det_size = det_size  # (width, height) — this backend's own convention
        self.ctx_id = ctx_id
        self._session = None  # lazy load
        self._input_name = None

    def _load(self):
        if self._session is not None:
            return
        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for the yolo-facev2-s backend. "
                "Install with: pip install onnxruntime"
            ) from e
        providers = ["CPUExecutionProvider"] if self.ctx_id < 0 else onnxruntime.get_available_providers()
        logger.info(f"Loading YOLO-FaceV2-s from {self.weights} (det_size={self.det_size})")
        session = onnxruntime.InferenceSession(str(self.weights), providers=providers)
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR frame (OpenCV format).

        Returns a list of Detection sorted by confidence (desc).
        """
        self._load()
        h0, w0 = frame.shape[:2]

        # The checkpoint was trained on RGB (standard YOLOv5 convention);
        # `frame` arrives as BGR (OpenCV format).
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target_w, target_h = self.det_size
        img, ratio, (pad_x, pad_y) = _letterbox(img_rgb, (target_h, target_w))
        img = img.astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))[None]  # HWC -> NCHW

        # Output layout (see convert.py): per-anchor [cx, cy, w, h, obj_conf,
        # cls_conf] — this checkpoint has no landmark channels (confirmed by
        # inspecting it directly; contrary to the repo's own "landmarks
        # version" release label, see convert.py's docstring). conf = obj_conf
        # * cls_conf (single class: face), same combination upstream's own
        # NMS applies, reimplemented here since this backend carries no
        # vendored-code dependency.
        pred = self._session.run(None, {self._input_name: img})[0][0]
        conf = pred[:, 4] * pred[:, 5]
        keep = conf > self.conf_threshold
        pred, conf = pred[keep], conf[keep]
        if pred.shape[0] == 0:
            return []

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        keep_idx = _nms(boxes, conf, _IOU_THRESHOLD)
        boxes, conf = boxes[keep_idx], conf[keep_idx]

        # Undo letterbox: subtract padding, divide by the resize ratio, clip
        # to the original frame — inverse of _letterbox() above.
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / ratio).clip(0, w0)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / ratio).clip(0, h0)

        detections = [
            Detection(
                x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                confidence=float(c), landmarks=None,
            )
            for b, c in zip(boxes, conf)
        ]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
