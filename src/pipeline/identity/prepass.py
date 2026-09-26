"""One pass over the source video: per-observation embeddings and quality,
aggregated into one `TrackIdentity` per track (plan D2, D5).

Reads `detections.jsonl` in lockstep with the video, exactly like
`phase2_generate/run.py` does (line i of the JSONL is frame i). Uses the
5-point landmarks Phase 1 already stored, so there is no second detection
pass. Keeps a top-K heap of the best real crops per track as seed
candidates, cut with the same `crop_box` geometry `run.py` uses so a
candidate can go straight to a generator.

Everything stays in memory (LGPD, see the package docstring). The
optional `observations` in the result exist for `src/eval/identity_report.py`
and are never written out.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from ..contracts import Frame, Identity
from ..phase2_generate.cropping import crop_box
from . import align
from .aggregate import Observation, SeedCandidate, TrackIdentity, aggregate
from .embedder import ArcFaceEmbedder, PoseEstimator
from .quality import QualityParams, quality_terms

logger = logging.getLogger(__name__)


@dataclass
class PrepassResult:
    tracks: dict[int, TrackIdentity]
    observations: dict[int, list[Observation]] = field(default_factory=dict)  # only if requested
    num_frames: int = 0
    num_skipped_no_landmarks: int = 0

    def summary(self) -> dict:
        """JSON-safe, no embeddings."""
        return {
            "num_frames": self.num_frames,
            "num_tracks": len(self.tracks),
            "num_skipped_no_landmarks": self.num_skipped_no_landmarks,
            "tracks": [t.summary() for t in self.tracks.values()],
        }


def run_prepass(
    detections_path: Path,
    video_path: str,
    identities: dict[int, Identity],
    context_ratio: float,
    embedder: ArcFaceEmbedder,
    pose_estimator: Optional[PoseEstimator] = None,
    mode: str = "quality_mean",
    k_candidates: int = 5,
    quality_params: QualityParams = QualityParams(),
    keep_observations: bool = False,
    limit: Optional[int] = None,
    **aggregate_kwargs,
) -> PrepassResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open source video: {video_path}")

    per_track: dict[int, list[Observation]] = {}
    heaps: dict[int, list] = {}  # track_id -> min-heap of (quality, frame_id, SeedCandidate)
    num_frames = skipped = 0

    with Path(detections_path).open() as jf:
        for line in jf:
            if limit is not None and num_frames >= limit:
                break
            frame_rec = Frame.from_json(line)
            num_frames += 1
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]

            for face in frame_rec.faces:
                if face.track_id not in identities:
                    continue
                if not face.landmarks:
                    skipped += 1
                    continue
                embedding = embedder.embed(frame, face.landmarks)
                yaw = pose_estimator.pose(frame, face.box)[1] if pose_estimator is not None else None
                terms = quality_terms(face.confidence, face.box, yaw, quality_params)
                per_track.setdefault(face.track_id, []).append(Observation(
                    frame_id=frame_rec.frame_id, embedding=embedding, confidence=face.confidence,
                    box=tuple(face.box), yaw=yaw, terms=terms,
                ))

                if k_candidates > 0:
                    heap = heaps.setdefault(face.track_id, [])
                    key = (terms.score, frame_rec.frame_id)
                    if len(heap) < k_candidates or key > heap[0][:2]:
                        cx1, cy1, cx2, cy2 = crop_box(h, w, face.box, context_ratio)
                        x1, y1, x2, y2 = face.box
                        cand = SeedCandidate(
                            frame_id=frame_rec.frame_id,
                            crop=frame[cy1:cy2, cx1:cx2].copy(),
                            box_in_crop=(x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
                            landmarks_in_crop=align.shift(face.landmarks, cx1, cy1),
                            quality=terms.score,
                        )
                        entry = (terms.score, frame_rec.frame_id, cand)
                        if len(heap) < k_candidates:
                            heapq.heappush(heap, entry)
                        else:
                            heapq.heapreplace(heap, entry)

            if num_frames % 100 == 0:
                logger.info(f"identity prepass: frame {num_frames}")
    cap.release()

    tracks: dict[int, TrackIdentity] = {}
    for track_id, obs in per_track.items():
        candidates = [c for _, _, c in sorted(heaps.get(track_id, []), key=lambda e: e[:2], reverse=True)]
        tracks[track_id] = TrackIdentity(
            track_id=track_id, seed=identities[track_id].seed, mode=mode,
            num_observations=len(obs), result=aggregate(obs, mode, **aggregate_kwargs),
            seed_candidates=candidates,
        )

    logger.info(f"identity prepass: {num_frames} frames, {len(tracks)} tracks, "
                f"{sum(len(o) for o in per_track.values())} observations "
                f"({skipped} skipped for missing landmarks)")
    return PrepassResult(
        tracks=tracks, observations=per_track if keep_observations else {},
        num_frames=num_frames, num_skipped_no_landmarks=skipped,
    )


def load_identities(phase1_dir: Path) -> tuple[str, dict[int, Identity]]:
    """(video source, track_id -> Identity) from a Phase 1 run's `tracks.json`."""
    from ..contracts import Manifest

    manifest = Manifest.load(Path(phase1_dir) / "tracks.json")
    return manifest.video.source, {i.track_id: i for i in manifest.identities}
