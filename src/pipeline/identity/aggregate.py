"""Per-track identity aggregation (plan D1, D2, D4).

What is aggregated is the *real* person's identity, not the synthetic one:
the real identity does not change along a track, so its estimate converges
as frames accumulate, while the synthetic identity stays seed-locked
(Stage 3 of the method) and never drifts.

Modes:
    first         — the track's first observation (what BLANKET does today)
    best          — the highest-quality inlier observation
    mean          — unweighted mean of inlier unit embeddings (the Diffusion
                    Video Autoencoders precedent, Kim et al., CVPR 2023)
    quality_mean  — quality-weighted mean of inlier unit embeddings
                    (offline, full track — the primary mode, D2)
    ema_adaptive  — causal running estimate, one value per frame:
                    e_t = α_t e_{t−1} + (1 − α_t) u_t,  α_t = α_f + (1 − α_f)(1 − q_t).
                    With q_t reduced to its detector term this is exactly
                    Deep OC-SORT's dynamic appearance update
                    (α_t = α_f + (1 − α_f)(1 − (s − σ)/(1 − σ)), α_f = 0.95).

Outlier gating: a robust center is fit twice (quality-weighted mean, then
re-fit on observations within `outlier_cos` of it); observations below
the threshold are excluded from `best`/`mean`/`quality_mean` and reported.
A run of `min_switch_run` consecutive outliers flags a probable tracker ID
switch — a second person's frames contaminating the track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .quality import QualityTerms

MODES = ("first", "best", "mean", "quality_mean", "ema_adaptive")
CAUSAL_MODES = ("ema_adaptive",)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass
class Observation:
    """One face of one track in one frame. Memory only — never serialized."""
    frame_id: int
    embedding: np.ndarray            # raw w600k_r50 output, not normalized
    confidence: float
    box: tuple[float, float, float, float]
    yaw: Optional[float]
    terms: QualityTerms

    @property
    def quality(self) -> float:
        return self.terms.score

    @property
    def unit(self) -> np.ndarray:
        return normalize(self.embedding)

    @property
    def embedding_norm(self) -> float:
        return float(np.linalg.norm(self.embedding))


@dataclass
class SeedCandidate:
    """A high-quality real crop of a track, ranked for building its synthetic identity."""
    frame_id: int
    crop: np.ndarray                                   # BGR, context-padded exactly like run.py's crops
    box_in_crop: tuple[float, float, float, float]
    landmarks_in_crop: list[tuple[float, float]]
    quality: float


@dataclass
class AggregateResult:
    embedding: Optional[np.ndarray]                    # unit vector; for causal modes, the last value
    per_frame: dict[int, np.ndarray] = field(default_factory=dict)   # causal modes only
    inlier_frames: list[int] = field(default_factory=list)
    outlier_frames: list[int] = field(default_factory=list)
    id_switch_suspected: bool = False


@dataclass
class TrackIdentity:
    track_id: int
    seed: int
    mode: str
    num_observations: int
    result: AggregateResult
    seed_candidates: list[SeedCandidate] = field(default_factory=list)

    @property
    def embedding(self) -> Optional[np.ndarray]:
        return self.result.embedding

    def embedding_at(self, frame_id: int) -> Optional[np.ndarray]:
        """The estimate to use at `frame_id`: the running value for causal
        modes, the whole-track value otherwise."""
        return self.result.per_frame.get(frame_id, self.result.embedding)

    def summary(self) -> dict:
        """JSON-safe description, no embeddings (LGPD — see package docstring)."""
        return {
            "track_id": self.track_id,
            "mode": self.mode,
            "num_observations": self.num_observations,
            "num_outliers": len(self.result.outlier_frames),
            "outlier_frame_ranges": frame_ranges(self.result.outlier_frames),
            "id_switch_suspected": self.result.id_switch_suspected,
            "seed_candidate_frames": [c.frame_id for c in self.seed_candidates],
            "seed_candidate_quality": [round(c.quality, 3) for c in self.seed_candidates],
        }


def frame_ranges(frames: list[int]) -> list[list[int]]:
    """[3, 4, 5, 9] -> [[3, 5], [9, 9]]"""
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def _weighted_center(units: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    return normalize((units * weights[:, None]).sum(axis=0))


def _longest_run(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def aggregate(observations: list[Observation], mode: str = "quality_mean", *,
              outlier_cos: float = 0.2, min_switch_run: int = 10,
              ema_alpha_floor: float = 0.95, ema_warmup: int = 3) -> AggregateResult:
    """Aggregate one track's observations. `observations` need not be sorted."""
    if mode not in MODES:
        raise ValueError(f"unknown aggregation mode {mode!r}, choose from {MODES}")
    obs = sorted(observations, key=lambda o: o.frame_id)
    if not obs:
        return AggregateResult(embedding=None)

    units = np.stack([o.unit for o in obs])
    quality = np.array([o.quality for o in obs], dtype=np.float64)
    frames = [o.frame_id for o in obs]

    # Robust center, fit twice, independent of `mode` so outlier flags mean
    # the same thing whichever estimate is returned.
    center = _weighted_center(units, quality)
    inlier = units @ center >= outlier_cos
    if inlier.any():
        center = _weighted_center(units[inlier], quality[inlier])
        inlier = units @ center >= outlier_cos
    if not inlier.any():  # degenerate: keep everything rather than return nothing
        inlier = np.ones(len(obs), dtype=bool)

    result = AggregateResult(
        embedding=None,
        inlier_frames=[f for f, k in zip(frames, inlier) if k],
        outlier_frames=[f for f, k in zip(frames, inlier) if not k],
        id_switch_suspected=_longest_run([not k for k in inlier]) >= min_switch_run,
    )

    if mode == "first":
        result.embedding = units[0]
    elif mode == "best":
        idx = np.flatnonzero(inlier)
        result.embedding = units[idx[np.argmax(quality[idx])]]
    elif mode == "mean":
        result.embedding = normalize(units[inlier].mean(axis=0))
    elif mode == "quality_mean":
        result.embedding = _weighted_center(units[inlier], quality[inlier])
    elif mode == "ema_adaptive":
        e = units[0].copy()
        result.per_frame[frames[0]] = normalize(e)
        updates = 1
        for u, q, f in zip(units[1:], quality[1:], frames[1:]):
            # Causal gating: after a short warm-up, an observation that
            # disagrees with the running estimate is skipped rather than
            # pulling it toward a possibly different person.
            if updates < ema_warmup or float(u @ normalize(e)) >= outlier_cos:
                alpha = ema_alpha_floor + (1.0 - ema_alpha_floor) * (1.0 - float(q))
                e = alpha * e + (1.0 - alpha) * u
                updates += 1
            result.per_frame[f] = normalize(e)
        result.embedding = result.per_frame[frames[-1]]
    return result
