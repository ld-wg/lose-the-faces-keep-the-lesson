"""Per-observation context passed to generation backends (contribution plan D11).

`Backend.generate(crop, seed, context=None)`: the crop and seed are what
every backend has always received; `context` adds what the runner knows
about this face beyond its pixels. Backends that have no use for it accept
and ignore it, so the contract stays backward compatible.

Nothing in here is serialized. `track` carries a real-identity embedding
(biometric data, LGPD), which stays in memory for the run's lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..identity.aggregate import TrackIdentity


@dataclass
class FaceContext:
    frame_id: int
    box_in_crop: tuple[float, float, float, float]            # Phase 1's detection, crop coordinates
    landmarks_in_crop: Optional[list[tuple[float, float]]]    # Phase 1's 5 points, crop coordinates
    track: Optional[TrackIdentity] = None                      # only with --identity-prepass

    @property
    def push_embedding(self) -> Optional[np.ndarray]:
        """Unit real-identity estimate to push away from at this frame, if known."""
        return self.track.embedding_at(self.frame_id) if self.track is not None else None
