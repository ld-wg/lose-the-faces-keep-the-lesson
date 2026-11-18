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


def landmark_error(reference: np.ndarray, moving: np.ndarray, iod: float) -> float:
    """Mean distance between two landmark sets after mapping `moving` onto `reference`
    with a similarity transform, in units of `iod` (inter-ocular distance). The
    evaluator's expression error (real vs shown face) and the adaptive sampler's
    change measure (last generated vs current real face) are this one function."""
    m = fit_similarity(moving, reference)
    mapped = moving @ m[:, :2].T + m[:, 2]
    return float(np.linalg.norm(mapped - reference, axis=1).mean() / max(iod, 1e-6))


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


# The face core in the 112 template: brows to chin, cheek to cheek. The ellipse's top passes the
# brow line (~0.35 inter-ocular distances above the eyes) and its bottom the chin (~0.55 below the
# mouth); everything outside it (hair, forehead, ears, jaw outline, background) is "context".
# The evaluator's --face-mask splits an aligned crop into the two, to measure where a recognizer
# finds the identity (2026-11-18 review).
FACE_CORE_ELLIPSE = (56.0, 74.0, 41.0, 38.0)   # (cx, cy, half-width, half-height), template pixels
MASK_GRAY = 128                                 # ~zero after both recognizers' input normalization


def face_core_mask(size: int, margin: float = 0.0) -> np.ndarray:
    """Boolean `size`x`size` mask of the face core, for a crop aligned with `template(size, margin)`."""
    scale = size / (ARCFACE_SIZE * (1.0 + 2.0 * margin))
    cx, cy, ax, ay = FACE_CORE_ELLIPSE
    cx, cy = (cx + ARCFACE_SIZE * margin) * scale, (cy + ARCFACE_SIZE * margin) * scale
    yy, xx = np.mgrid[0:size, 0:size]
    return ((xx - cx) / (ax * scale)) ** 2 + ((yy - cy) / (ay * scale)) ** 2 <= 1.0


def mask_aligned(aligned: np.ndarray, margin: float, keep: str) -> np.ndarray:
    """An aligned crop with `keep="core"` (gray outside the face core) or `keep="context"`
    (gray inside it)."""
    if keep not in ("core", "context"):
        raise ValueError(f"keep must be 'core' or 'context', not {keep!r}")
    inside = face_core_mask(aligned.shape[0], margin)
    out = aligned.copy()
    out[~inside if keep == "core" else inside] = MASK_GRAY
    return out


def shift(landmarks: Sequence[Sequence[float]], dx: float, dy: float) -> list[tuple[float, float]]:
    """Landmarks translated by (-dx, -dy) — frame coordinates into a crop's own."""
    return [(float(x) - dx, float(y) - dy) for x, y in landmarks]
