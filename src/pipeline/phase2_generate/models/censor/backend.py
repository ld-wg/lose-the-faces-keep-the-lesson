"""Censorship baseline: blur or mosaic of the detected face box (E2's utility floor).

Not an anonymizer this project recommends: both are reversible or attackable
for small faces (research/stages/identification-occlusion.md), which is why
the final video's fail-safe never uses them (experiments/DECISIONS.md 32).
They are here because the literature reports them and the paper compares
against them: they set the privacy and utility floor a generative method has
to beat on behavior (expression, pose, emotion), not on suppression.

Strength is relative to the face, so a small face is not left readable:
    blur    Gaussian, sigma = box_size / 6 (kernel ~ the box), applied twice
    mosaic  the box at `mosaic_blocks` cells across its shorter side, nearest upscale
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

MODES = ("blur", "mosaic")


class Backend:
    def __init__(self, weights: Optional[Path] = None, ctx_id: int = 0, random_init: bool = False,
                 mode: str = "blur", mosaic_blocks: int = 8):
        del weights, ctx_id, random_init  # uniform constructor contract, nothing to load
        if mode not in MODES:
            raise ValueError(f"censor mode {mode!r}: choose from {MODES}")
        self.mode = mode
        self.mosaic_blocks = mosaic_blocks
        self.last_skip_reason: Optional[str] = None

    def identity_class(self, seed: int) -> int:
        return 0  # no identity: every face gets the same treatment

    def generate(self, crop: np.ndarray, seed: int, context=None) -> Optional[np.ndarray]:
        self.last_skip_reason = None
        h, w = crop.shape[:2]
        x1, y1, x2, y2 = context.box_in_crop if context is not None else (0, 0, w, h)
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            self.last_skip_reason = "empty_box"
            return None
        out = crop.copy()
        face = out[y1:y2, x1:x2]
        side = min(face.shape[:2])
        if self.mode == "blur":
            sigma = max(1.0, side / 6)
            face[:] = cv2.GaussianBlur(cv2.GaussianBlur(face, (0, 0), sigma), (0, 0), sigma)
        else:
            cells = max(1, self.mosaic_blocks)
            small = cv2.resize(face, (max(1, round(face.shape[1] * cells / side)), cells), interpolation=cv2.INTER_AREA)
            face[:] = cv2.resize(small, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_NEAREST)
        return out
