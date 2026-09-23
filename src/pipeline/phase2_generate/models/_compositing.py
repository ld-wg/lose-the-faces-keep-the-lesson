"""Shared compositing strategies for Phase 2 backends.

`poisson_composite()` was extracted from ciagan/backend.py's original
`_poisson_composite()` — both ciagan and ganonymization paste a generated
region back into a real crop via the same `cv2.seamlessClone` strategy
with the same hard-paste fallback. No behavior change from ciagan's
original private function — see ciagan/NOTICE.md for this extraction.

`feathered_alpha_composite()` is a second, opt-in strategy added for
ganonymization only (`--blend-mode feather`) — see its own docstring.
`ciagan` never calls it; `poisson_composite()`'s behavior is unchanged by
its presence.
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


def feathered_alpha_composite(crop: np.ndarray, generated: np.ndarray, mask_bool: np.ndarray,
                               feather_px: int = 15) -> np.ndarray:
    """Hard-paste `generated` inside `mask_bool`; blend only a ~feather_px-wide
    boundary band via a Gaussian-blurred alpha.

    Added for ganonymization (opt-in, `--blend-mode feather`) after real-video
    testing found `poisson_composite()` lets the real face show through when
    `generated` has weak local contrast (a blurry/checkerboard-degraded face,
    which this backend's checkpoint sometimes produces — see
    ganonymization/NOTICE.md's anti-bleed-through entry): `cv2.seamlessClone`
    solves a Poisson equation using `generated`'s own gradient field as
    guidance, with values pinned to `crop` at the mask boundary — when that
    gradient signal is weak, the solution leans toward harmonic interpolation
    of the surrounding real pixels *into* the masked region. This function
    avoids that mechanism structurally: deep inside the mask, alpha≈1, so
    `crop` contributes ~nothing to the result there, regardless of how weak
    `generated`'s own local contrast is. Trade-off: the transition band itself
    may show a more visible seam than Poisson's smooth gradient-domain blend.

    `ciagan`'s own `poisson_composite()` call is untouched — this is a
    separate function, not a mode switch on the existing one, so `ciagan`'s
    already-calibrated behavior can't shift by adding this.
    """
    mask_u8 = mask_bool.astype(np.uint8) * 255
    k = feather_px | 1  # odd kernel size, required by GaussianBlur
    alpha = cv2.GaussianBlur(mask_u8, (k, k), 0).astype(np.float32)[..., None] / 255.0
    blended = crop.astype(np.float32) * (1 - alpha) + generated.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
