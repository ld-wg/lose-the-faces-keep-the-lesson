"""Generation-time decisions that run.py takes per face, kept free of models and I/O
so they can be tested on synthetic arrays (tests/test_policies.py).

- `changed_fraction`: how much of the detection box a generated crop actually
  changed. A backend that falls back to its input (an empty paste mask makes
  `_compositing.poisson_composite` return the real crop) produces 0; run.py marks
  such a face failsafe instead of ok, so it is hidden, never shown as generated.
- `should_generate`: fixed (`--generate-every`) and adaptive (`--generate-adaptive`)
  frame sampling. Faces that are not generated are covered by compose_video.py's
  reuse of the track's last generated face.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def changed_fraction(original: np.ndarray, generated: np.ndarray, box: Sequence[float]) -> float:
    """Share of pixels inside `box` (x1, y1, x2, y2 in crop coordinates, clipped to the
    crop) where any channel of `generated` differs from `original`. 0 for an empty box."""
    if original.shape != generated.shape:
        return 1.0   # a resized output cannot be the input passed through
    h, w = original.shape[:2]
    x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a, b = original[y1:y2, x1:x2], generated[y1:y2, x1:x2]
    diff = a != b
    if diff.ndim == 3:
        diff = diff.any(axis=2)
    return float(diff.mean())


def should_generate(gap: Optional[int], *, every: int = 1, adaptive_tau: Optional[float] = None,
                    max_gap: int = 6, change: Optional[float] = None) -> bool:
    """Whether a track's face is generated in this frame.

    gap: frames since the track's last generated ("ok") face, None if it has none yet
    (a gated or failed generation does not count, so the next frame tries again).
    Fixed sampling: generate once `gap` reaches `every`. Adaptive sampling
    (`adaptive_tau` set): generate when the real face changed by at least `adaptive_tau`
    since that frame (`change`, inter-ocular units; unknown counts as changed) or once
    `gap` reaches `max_gap`."""
    if gap is None:
        return True
    if adaptive_tau is not None:
        return gap >= max_gap or change is None or change >= adaptive_tau
    return gap >= every
