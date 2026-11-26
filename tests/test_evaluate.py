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


# --- 2026-11-18 review: anonymized faces scored on the frame, face-core masks -------------------

import cv2  # noqa: E402

from src.eval.evaluate import pasted, to_frame  # noqa: E402
from src.pipeline.identity import align  # noqa: E402
from src.pipeline.phase2_generate.cropping import crop_box  # noqa: E402

BOX = (150.0, 150.0, 250.0, 250.0)   # a 100 px face in a 400 px frame


def probe_window(image: np.ndarray, box, size: int = 192) -> np.ndarray:
    """What insightface's 106-landmark, 3D-68 and gender-age models see: a square of 1.5x the box
    side around its center, black outside the image (model_zoo/landmark.py, attribute.py)."""
    x1, y1, x2, y2 = box
    s = size / (max(x2 - x1, y2 - y1) * 1.5)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    m = np.float32([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]])
    return cv2.warpAffine(image, m, (size, size), borderValue=0.0)


def frame_and_crop(context_ratio: float):
    frame = np.random.default_rng(0).integers(1, 255, (400, 400, 3), dtype=np.uint8)   # no zero pixel
    cx1, cy1, cx2, cy2 = crop_box(400, 400, BOX, context_ratio)
    return frame, frame[cy1:cy2, cx1:cx2].copy(), cx1, cy1


def test_small_context_crop_alone_gives_the_probes_black_borders():
    frame, crop, cx1, cy1 = frame_and_crop(0.1)   # the censor backends' margin
    in_crop = (BOX[0] - cx1, BOX[1] - cy1, BOX[2] - cx1, BOX[3] - cy1)
    assert (probe_window(crop, in_crop) == 0).all(axis=2).any()     # the bug: black the real face never has
    assert not (probe_window(frame, BOX) == 0).all(axis=2).any()


def test_unchanged_crop_pasted_back_is_scored_like_the_real_face():
    for ratio in (0.1, 0.15, 0.8):
        frame, crop, cx1, cy1 = frame_and_crop(ratio)
        before = frame.copy()
        with pasted(frame, crop, cx1, cy1) as anon_frame:
            assert np.array_equal(probe_window(anon_frame, BOX), probe_window(before, BOX))
        assert np.array_equal(frame, before)   # restored


def test_pasted_writes_the_crop_and_restores_on_error():
    frame, crop, cx1, cy1 = frame_and_crop(0.1)
    before = frame.copy()
    with pytest.raises(RuntimeError):
        with pasted(frame, np.zeros_like(crop), cx1, cy1) as anon_frame:
            assert (anon_frame[cy1:cy1 + crop.shape[0], cx1:cx1 + crop.shape[1]] == 0).all()
            raise RuntimeError
    assert np.array_equal(frame, before)


def test_wide_context_crop_is_scored_the_same_either_way():
    # BLANKET and GANonymization (margin 0.8): the probes' window stays inside the crop, so the
    # fix leaves their numbers unchanged
    frame, crop, cx1, cy1 = frame_and_crop(0.8)
    in_crop = (BOX[0] - cx1, BOX[1] - cy1, BOX[2] - cx1, BOX[3] - cy1)
    a, b = probe_window(crop, in_crop).astype(int), probe_window(frame, BOX).astype(int)
    assert np.abs(a - b).max() <= 1   # same pixels; at most interpolation rounding


def test_to_frame_shifts_box_and_landmarks():
    box, kps = to_frame(np.array([1.0, 2.0, 3.0, 4.0]), np.array([[1.0, 1.0], [2.0, 2.0]]), 10, 20)
    assert box == (11.0, 22.0, 13.0, 24.0)
    assert kps == [(11.0, 21.0), (12.0, 22.0)]


def test_face_core_holds_the_five_landmarks_and_not_the_corners():
    for size, margin in ((112, 0.0), (160, 0.1)):
        inside = align.face_core_mask(size, margin)
        for x, y in align.template(size, margin):
            assert inside[int(round(y)), int(round(x))]
        for y, x in ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1), (0, size // 2)):
            assert not inside[y, x]   # corners and the top edge: hair, background


def test_core_and_context_masks_are_complementary():
    aligned = np.random.default_rng(1).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    core, context = align.mask_aligned(aligned, 0.1, "core"), align.mask_aligned(aligned, 0.1, "context")
    inside = align.face_core_mask(160, 0.1)
    assert np.array_equal(core[inside], aligned[inside]) and (core[~inside] == align.MASK_GRAY).all()
    assert np.array_equal(context[~inside], aligned[~inside]) and (context[inside] == align.MASK_GRAY).all()
    with pytest.raises(ValueError):
        align.mask_aligned(aligned, 0.1, "face")
