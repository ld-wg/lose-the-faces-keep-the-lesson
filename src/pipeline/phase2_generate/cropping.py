"""Crop geometry shared by Phase 2's runner, its compositor, and the identity pre-pass.

Moved out of `run.py` unchanged so that `src/pipeline/identity/prepass.py`
can cut exactly the same crops `run.py` feeds the generators without
importing the runner module (which would be circular once `run.py` itself
calls the pre-pass). `run.py` still re-exports both names.
"""

from __future__ import annotations

import numpy as np


def crop_box(h: int, w: int, box: tuple[float, float, float, float], context_ratio: float
             ) -> tuple[int, int, int, int]:
    """The (cx1, cy1, cx2, cy2) pixel rectangle `context_crop` cuts, exposed
    separately so `compose_video.py` can recompute the exact same rectangle
    to paste a generated crop back into its source frame — same formula,
    single source of truth, not duplicated.
    """
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = context_ratio * box_w, context_ratio * box_h
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    return cx1, cy1, cx2, cy2


def context_crop(frame: np.ndarray, box: tuple[float, float, float, float], context_ratio: float) -> np.ndarray:
    """Cut a crop around `box`, padded proportionally to its own size.

    Unlike a fixed-pixel pad, this scales with how big the face already is
    in the frame — needed so the crop's framing stays proportionally
    consistent across near/far faces (see `models/ciagan/NOTICE.md` for why
    a fixed small pad broke CIAGAN's own portrait-crop assumption).
    """
    h, w = frame.shape[:2]
    cx1, cy1, cx2, cy2 = crop_box(h, w, box, context_ratio)
    return frame[cy1:cy2, cx1:cx2]
