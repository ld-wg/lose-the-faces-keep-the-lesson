"""5-point similarity alignment to the ArcFace template.

`ARCFACE_DST` is insightface's `utils/face_align.py::arcface_dst`, the same
template FaceFusion calls `arcface_112_v2` — so a face aligned here embeds
into the same space as both insightface's `norm_crop` path and the swap
stage inside the BLANKET bridge.

The similarity transform is a closed-form least-squares fit (Umeyama 1991),
the same estimator `skimage.transform.SimilarityTransform.estimate` uses
under insightface's `estimate_norm` — reimplemented in numpy rather than
called through skimage so one code path serves every output size and
margin (insightface's `estimate_norm` only accepts multiples of 112/128,
and FaceNet wants 160).
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
ARCFACE_SIZE = 112


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity matrix (rotation, uniform scale, translation) mapping src onto dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / n
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[-1] = -1.0
    u, s, vt = np.linalg.svd(cov)
    rot = u @ np.diag(d) @ vt
    var_src = (src_c ** 2).sum() / n
    scale = float((s * d).sum() / var_src)
    t = dst_mean - scale * rot @ src_mean
    return np.hstack([scale * rot, t[:, None]]).astype(np.float32)


def template(size: int, margin: float = 0.0) -> np.ndarray:
    """The ArcFace template scaled to a `size`x`size` output with `margin`
    (fraction of the template's own extent) of extra context on each side.
    `margin=0, size=112` is exactly insightface's template."""
    scale = size / (ARCFACE_SIZE * (1.0 + 2.0 * margin))
    return (ARCFACE_DST + ARCFACE_SIZE * margin) * scale


def similarity_matrix(landmarks: Sequence[Sequence[float]], size: int = ARCFACE_SIZE,
                      margin: float = 0.0) -> np.ndarray:
    lm = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    return fit_similarity(lm, template(size, margin))


def warp(image: np.ndarray, landmarks: Sequence[Sequence[float]], size: int = ARCFACE_SIZE,
         margin: float = 0.0) -> np.ndarray:
    """Aligned `size`x`size` crop. Same border handling as insightface's `norm_crop`."""
    m = similarity_matrix(landmarks, size, margin)
    return cv2.warpAffine(image, m, (size, size), borderValue=0.0)


def shift(landmarks: Sequence[Sequence[float]], dx: float, dy: float) -> list[tuple[float, float]]:
    """Landmarks translated by (-dx, -dy) — frame coordinates into a crop's own."""
    return [(float(x) - dx, float(y) - dy) for x, y in landmarks]
