"""Shared gradient-domain (Poisson) compositing for Phase 2 backends.

Extracted from ciagan/backend.py's original `_poisson_composite()` — both
ciagan and ganonymization paste a generated region back into a real crop
via the same `cv2.seamlessClone` strategy with the same hard-paste
fallback. The operation itself (crop, generated image, boolean mask ->
blended image) is generic and backend-agnostic; neither backend's own
geometry/preprocessing lives here. No behavior change from ciagan's
original private function — see ciagan/NOTICE.md for this extraction.
"""

from __future__ import annotations

import cv2
import numpy as np


def poisson_composite(crop: np.ndarray, generated: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """Gradient-domain blend of `generated` into `crop` within `mask_bool`.

    Replaces a hard-edged paste (a visible skin-tone/lighting seam at the
    mask boundary) with `cv2.seamlessClone`: it solves for pixel values
    inside the mask whose *gradients* match `generated`'s, so the interior
    content is preserved but the boundary blends into the surrounding
    lighting/color instead of cutting sharply across it.

    Falls back to the hard-mask paste if `seamlessClone` raises — it needs
    a non-degenerate mask that doesn't touch the image border, which can
    happen for a face very close to the crop's edge. A cosmetic blending
    step failing shouldn't take down the whole frame.
    """
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return crop.copy()
    center = ((int(xs.min()) + int(xs.max())) // 2, (int(ys.min()) + int(ys.max())) // 2)
    mask_u8 = mask_bool.astype(np.uint8) * 255
    try:
        return cv2.seamlessClone(generated, crop, mask_u8, center, cv2.NORMAL_CLONE)
    except cv2.error:
        output = crop.copy()
        output[mask_bool] = generated[mask_bool]
        return output
