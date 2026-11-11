"""Re-identification attacks of the evaluator (src/eval/evaluate.py), on synthetic embeddings.

Added with the 2026-10-30 critical review: the least-similar-first attacker (the P2 push
goes towards the antipode of the real identity), the track-level attacker (the pseudonym is
stable), the shown-faces view (hiding cannot lower it) and the identity audit (tracks are the
evaluation's identities, so a person split into two tracks must be flagged).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.evaluate import (FILLED_RANK, RECOGNIZERS, AnonObs, Gallery, RealObs, identity_audit, rank_of,
                               run_metrics, track_probe)

DIM = 512


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def identities(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [unit(rng.normal(size=DIM)) for _ in range(n)]


def reals_for(ids: list[np.ndarray], frames: dict[int, list[int]], noise: float = 0.05, seed: int = 1):
    rng = np.random.default_rng(seed)
    out = {}
    for t, fids in frames.items():
        for f in fids:
            e = unit(ids[t] + noise * rng.normal(size=DIM))
            out[(f, t)] = RealObs(track_id=t, frame_id=f, emb={rec: e for rec in RECOGNIZERS}, probes={})
    return out


FRAMES = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3], 3: [0, 1, 2, 3]}


def test_rank_of():
    assert rank_of(np.array([0.1, 0.9, 0.5]), 1) == 1
    assert rank_of(np.array([0.1, 0.9, 0.5]), 0) == 3
    assert rank_of(np.array([0.5, 0.5]), 1) == 1   # ties favor the true entry (worst case for privacy)


def test_rank_and_inverse_rank():
    ids = identities(4)
    reals = reals_for(ids, FRAMES)
    g = Gallery(reals)
    real = reals[(0, 2)]
    assert g.rank("facenet", ids[2], real) == 1
    assert g.rank("facenet", -ids[2], real) == 4
    assert g.rank("facenet", -ids[2], real, inverse=True) == 1


def test_push_to_the_antipode_is_found_by_the_inverse_attacker():
    """P2's target, normalize(e_synth - beta e_real), for a large beta."""
    ids = identities(4)
    reals = reals_for(ids, FRAMES)
    g = Gallery(reals)
    e_synth = identities(1, seed=7)[0]
    pushed = unit(e_synth - 1.6 * ids[1])
    real = reals[(2, 1)]
    assert g.rank("facenet", pushed, real) > 1
    assert g.rank("facenet", pushed, real, inverse=True) == 1


def test_single_frame_track_has_no_rank():
    ids = identities(2)
    reals = reals_for(ids, {0: [0], 1: [0, 1]})
    assert Gallery(reals).rank("facenet", ids[0], reals[(0, 0)]) is None


def test_track_probe():
    assert track_probe([]) is None
    p = track_probe([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert np.allclose(p, np.array([1.0, 1.0]) / np.sqrt(2))


def test_identity_audit_flags_one_person_in_two_tracks():
    ids = identities(3)
    ids.append(ids[0])   # track 3 is track 0's person, after it left the frame
    frames = {0: [0, 1, 2], 1: [0, 1, 2, 3, 4, 5], 2: [0, 1, 2, 3, 4, 5], 3: [3, 4, 5]}
    reals = reals_for(ids, frames)
    g = Gallery(reals)
    audit = identity_audit(list(reals.values()), g, {rec: {"threshold": 0.5} for rec in RECOGNIZERS})
    assert audit["tracks"] == 4
    assert audit["never_covisible_pairs"] == 1
    assert [c[:2] for c in audit["candidates"]["facenet"]] == [[0, 3]]


def anon(track: int, frame: int, emb: np.ndarray) -> AnonObs:
    return AnonObs(track_id=track, frame_id=frame, status="ok", reason=None, anonymized=True, redetected=True,
                   emb={rec: emb for rec in RECOGNIZERS}, utility={})


def metrics(obs, reals, compose=None):
    g = Gallery(reals)
    rows: list[dict] = []
    out = run_metrics(obs, reals, {rec: {"threshold": 0.5} for rec in RECOGNIZERS}, np.random.default_rng(0), g,
                      {rec: {"rank1_rate": 1.0} for rec in RECOGNIZERS}, rows, "arm", compose)
    return out, rows


def test_track_rank_is_one_value_per_track():
    ids = identities(4)
    reals = reals_for(ids, FRAMES)
    rng = np.random.default_rng(3)
    # each frame alone is mostly noise, but the noise averages out over the track
    obs = [anon(t, f, unit(0.35 * ids[t] + rng.normal(size=DIM) / np.sqrt(DIM))) for t, fs in FRAMES.items()
           for f in fs]
    _, rows = metrics(obs, reals)
    for t in FRAMES:
        trows = sorted((r for r in rows if r["track_id"] == t), key=lambda r: r["frame_id"])
        assert trows[0]["facenet_track_rank"] == 1
        assert all(r.get("facenet_track_rank") is None for r in trows[1:])


def test_shown_rank_ignores_filled_boxes():
    ids = identities(4)
    reals = reals_for(ids, FRAMES)
    obs = [anon(t, f, ids[t]) for t, fs in FRAMES.items() for f in fs]   # a leaking method
    compose = {(f, t): ("filled" if t in (0, 1) else "generated") for t, fs in FRAMES.items() for f in fs}
    _, rows = metrics(obs, reals, compose)
    final = [r["facenet_final_rank"] for r in rows]
    shown = [r["facenet_shown_rank"] for r in rows if r["facenet_shown_rank"] is not None]
    assert np.mean([k == 1 for k in final]) == pytest.approx(0.5)   # hiding half "halves" the leak
    assert np.mean([k == 1 for k in shown]) == 1.0                    # among shown faces it still leaks
    assert np.mean([r["shown"] for r in rows]) == pytest.approx(0.5)
    hidden_track = [r for r in rows if r["track_id"] == 0 and r.get("facenet_track_rank") is not None]
    assert hidden_track[0]["facenet_track_rank"] == FILLED_RANK
