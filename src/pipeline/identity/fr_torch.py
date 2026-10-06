"""Differentiable ArcFace (`w600k_r50`) and alignment, for gradient-based identity push (P3, P1).

`TorchArcFace` converts buffalo_l's own `w600k_r50.onnx` to PyTorch with
onnx2torch (Apache-2.0), so its embeddings live in the same space as the
pre-pass's track identity (ONNX, `embedder.ArcFaceEmbedder`) and FaceFusion's
recognizer. `check_parity()` compares the two on real aligned crops; use it
before trusting gradients (contribution plan D6, Step 1(a): cos > 0.999).
Measured on serra1 (2026-10-05, 50 faces of demo1): the conversion itself,
min 0.99986. Through `affine_sample` instead of cv2: min 0.986, mean 0.998.
The two warps differ only by cv2's uint8 rounding (0.22/255 per pixel on
average), and that alone moves ArcFace by up to 0.014 on small faces: the
recognizer's sensitivity to noise, not a conversion error.

`affine_sample()` is the differentiable counterpart of `cv2.warpAffine`: it
samples an output image from an input through a pixel-space 2x3 matrix that
maps OUTPUT pixels to INPUT pixels (the inverse of what warpAffine takes).

Embeddings stay in memory (LGPD).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .embedder import ARCFACE_FILENAME, buffalo_model_path


class TorchArcFace:
    """RGB in [0, 1], (B, 3, 112, 112) -> unit embeddings (B, 512); no parameter gradients."""

    def __init__(self, device, onnx_path: Optional[Path] = None):
        import onnx2torch

        self.device = device
        path = Path(onnx_path) if onnx_path else buffalo_model_path(ARCFACE_FILENAME)
        model = onnx2torch.convert(str(path))
        model.eval()  # converted BatchNorm would otherwise run in training mode
        model.requires_grad_(False)
        self.model = model.to(device)

    def __call__(self, rgb01):
        import torch.nn.functional as F

        return F.normalize(self.model(2 * rgb01 - 1), dim=1)  # insightface: (x - 127.5) / 127.5


def affine_sample(image, out_to_in: np.ndarray, out_size: int):
    """Differentiable warp. `image` (B, C, H, W); `out_to_in` maps output pixel
    (x, y) to input pixel coordinates, OpenCV's pixel-center convention."""
    import torch
    import torch.nn.functional as F

    _, _, h, w = image.shape
    a = np.vstack([np.asarray(out_to_in, np.float64), [0, 0, 1]])
    # normalized (align_corners=False) <-> pixel: n = (2p + 1) / size - 1
    to_pixel_out = np.array([[out_size / 2, 0, out_size / 2 - 0.5], [0, out_size / 2, out_size / 2 - 0.5], [0, 0, 1]])
    to_norm_in = np.array([[2 / w, 0, 1 / w - 1], [0, 2 / h, 1 / h - 1], [0, 0, 1]])
    theta = (to_norm_in @ a @ to_pixel_out)[:2]
    theta_t = torch.as_tensor(theta, dtype=image.dtype, device=image.device).unsqueeze(0).expand(image.shape[0], 2, 3)
    grid = F.affine_grid(theta_t, (image.shape[0], image.shape[1], out_size, out_size), align_corners=False)
    # zeros outside the input, like cv2.warpAffine's black border in the pre-pass's alignment
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def check_parity(images_bgr, landmarks, device, onnx_embedder=None) -> dict[str, list[float]]:
    """Cosine between TorchArcFace and the ONNX model on the same faces: `model`
    on the identical cv2-aligned crop (the conversion itself), `pipeline` with
    TorchArcFace fed through `affine_sample` (conversion + differentiable warp)."""
    import cv2
    import torch

    from . import align
    from .embedder import ArcFaceEmbedder

    onnx_embedder = onnx_embedder or ArcFaceEmbedder(ctx_id=-1)
    fr = TorchArcFace(device)
    out: dict[str, list[float]] = {"model": [], "pipeline": []}
    to_t = lambda bgr: torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).permute(2, 0, 1)[None].to(device)  # noqa: E731
    for img, lm in zip(images_bgr, landmarks):
        aligned = align.warp(img, lm, align.ARCFACE_SIZE)
        ref = onnx_embedder.embed_aligned(aligned)
        ref = ref / np.linalg.norm(ref)
        out_to_in = cv2.invertAffineTransform(align.similarity_matrix(lm))  # aligned -> image
        with torch.no_grad():
            out["model"].append(float(fr(to_t(aligned))[0].cpu().numpy() @ ref))
            out["pipeline"].append(float(fr(affine_sample(to_t(img), out_to_in, align.ARCFACE_SIZE))[0].cpu().numpy() @ ref))
    return out


def main() -> None:
    """Parity check on real faces of a Phase 1 run (scalars printed, nothing written):
    python -m src.pipeline.identity.fr_torch --phase1-dir DET --video V [--faces 50]."""
    import argparse
    import json

    import cv2
    import torch

    from ..contracts import Frame, Manifest

    p = argparse.ArgumentParser()
    p.add_argument("--phase1-dir", required=True)
    p.add_argument("--video", default=None)
    p.add_argument("--faces", type=int, default=50)
    args = p.parse_args()
    det = Path(args.phase1_dir)
    cap = cv2.VideoCapture(args.video or Manifest.load(det / "tracks.json").video.source)
    images, lms = [], []
    with (det / "detections.jsonl").open() as f:
        for fr in map(Frame.from_json, f):
            ok, frame = cap.read()
            if not ok or len(images) >= args.faces:
                break
            for face in fr.faces:
                if face.detected and face.landmarks and face.confidence >= 0.5 and len(images) < args.faces:
                    images.append(frame)
                    lms.append(face.landmarks)
    cos = check_parity(images, lms, "cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({k: {"faces": len(v), "min": round(min(v), 5), "mean": round(float(np.mean(v)), 5)}
                      for k, v in cos.items()}))


if __name__ == "__main__":
    main()
