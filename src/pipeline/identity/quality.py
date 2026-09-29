"""Per-observation quality score (plan D3).

`q = det · size · pose`, each term in [0, 1], from signals Phase 1 and the
pre-pass already have — no separate face-image-quality model:

- `det = clip((conf − σ) / (1 − σ))`: detector confidence rescaled the way
  Deep OC-SORT (Maggiolino et al., arXiv:2302.11813) rescales it for its
  adaptive EMA; σ = 0.3 is Phase 1's own `--conf` threshold, the lowest
  confidence a stored detection can have.
- `size = min(1, sqrt(box area) / 112)`: faces smaller than the ArcFace
  input side get upsampled before embedding, and the pixelated tracks in
  `models/blanket/NOTICE.md` are exactly the small ones.
- `pose = max(0, 1 − |yaw| / 60°)`: profile views embed poorly.

The constants are starting points. `src/eval/identity_report.py` measures
which terms actually predict agreement with the rest of the track (and
whether the raw embedding norm does, the MagFace/AdaFace hypothesis) — the
terms can be switched off individually once that evidence exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class QualityParams:
    det_sigma: float = 0.3
    size_ref: float = 112.0
    yaw_limit: float = 60.0
    use_det: bool = True
    use_size: bool = True
    use_pose: bool = True


@dataclass(frozen=True)
class QualityTerms:
    det: float
    size: float
    pose: float
    params: QualityParams = QualityParams()

    @property
    def score(self) -> float:
        q = 1.0
        if self.params.use_det:
            q *= self.det
        if self.params.use_size:
            q *= self.size
        if self.params.use_pose:
            q *= self.pose
        return q


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def quality_terms(confidence: float, box: Sequence[float], yaw: Optional[float],
                  params: QualityParams = QualityParams()) -> QualityTerms:
    """`yaw` in degrees; None (pose unavailable) leaves the pose term at 1."""
    x1, y1, x2, y2 = box
    det = _clip01((confidence - params.det_sigma) / (1.0 - params.det_sigma))
    size = _clip01(math.sqrt(max(0.0, x2 - x1) * max(0.0, y2 - y1)) / params.size_ref)
    pose = 1.0 if yaw is None else _clip01(1.0 - abs(yaw) / params.yaw_limit)
    return QualityTerms(det=det, size=size, pose=pose, params=params)
