"""Face tracking for Phase 1 (ByteTrack-style).

Decision (research/pipeline.md): tracking is MANDATORY — Node 3 needs a fixed
synthetic identity per track. ByteTrack keeps low-confidence detections in
association, serving our high-recall priority and mitigating hard-condition
misses (occlusion, low light).

This is a self-contained IoU-based tracker following the ByteTrack principle
(associate high-conf first, then recover with low-conf). It avoids an external
dependency so Phase 1 runs standalone; swap for the official `bytetrack` package
if preferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


def _iou(a: tuple, b: tuple) -> float:
    """IoU of two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """A tracked face across frames."""
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    age: int = 0          # frames since first seen
    hits: int = 1         # total successful associations
    missed: int = 0       # consecutive frames without a detection
    landmarks: object = None

    def predict(self) -> None:
        self.age += 1
        self.missed += 1

    def update(self, det: Detection) -> None:
        self.box = det.box
        self.confidence = det.confidence
        self.landmarks = det.landmarks
        self.hits += 1
        self.missed = 0


class FaceTracker:
    """ByteTrack-style multi-face tracker.

    Two-stage association: match high-confidence detections first, then match
    remaining tracks against low-confidence detections (recall recovery).
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        iou_thresh: float = 0.3,
        max_missed: int = 30,
        min_hits: int = 3,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed  # frames to keep a lost track (occlusion gaps)
        self.min_hits = min_hits      # hits before a track is confirmed
        self._next_id = 1
        self._tracks: list[Track] = []

    def _assign(self, tracks: list[Track], dets: list[Detection]) -> tuple:
        """Greedy IoU assignment. Returns (matches, unmatched_tracks, unmatched_dets)."""
        if not tracks or not dets:
            return [], list(tracks), list(dets)
        # cost = 1 - IoU
        cost = np.zeros((len(tracks), len(dets)), dtype=float)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                cost[i, j] = 1.0 - _iou(t.box, d.box)
        matches, used_t, used_d = [], set(), set()
        # greedy: lowest cost first
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for ti, di in order:
            if ti in used_t or di in used_d:
                continue
            if cost[ti, di] > 1.0 - self.iou_thresh:
                continue
            matches.append((tracks[ti], dets[di]))
            used_t.add(ti)
            used_d.add(di)
        unmatched_t = [t for i, t in enumerate(tracks) if i not in used_t]
        unmatched_d = [d for j, d in enumerate(dets) if j not in used_d]
        return matches, unmatched_t, unmatched_d

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance one frame. Returns confirmed active tracks."""
        # split by confidence (ByteTrack core)
        high = [d for d in detections if d.confidence >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        # predict
        for t in self._tracks:
            t.predict()

        # stage 1: match all tracks against high-conf
        matches, unmatched_t, unmatched_high = self._assign(self._tracks, high)
        for t, d in matches:
            t.update(d)

        # stage 2: match remaining tracks against low-conf (recall recovery)
        matches2, unmatched_t, _ = self._assign(unmatched_t, low)
        for t, d in matches2:
            t.update(d)

        # spawn new tracks from unmatched high-conf detections
        for d in unmatched_high:
            self._tracks.append(
                Track(track_id=self._next_id, box=d.box, confidence=d.confidence, landmarks=d.landmarks)
            )
            self._next_id += 1

        # drop dead tracks
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

        # return confirmed, currently-matched tracks
        return [t for t in self._tracks if t.hits >= self.min_hits and t.missed == 0]

    @property
    def active_tracks(self) -> list[Track]:
        return list(self._tracks)
