"""Stage 3 fail-closed composition (compose_video.compose_frame), on synthetic frames.

The thesis's privacy claim for known faces: every detector box in the output video is
either a real replacement or hidden (reused synthetic face, or a neutral fill). These
tests check that claim, including the fail-open path the 2026-10-29 review found (a
backend returning the real crop as "ok").
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.contracts import MIN_CHANGED_FRACTION, REUSE_FRAMES, Face
from src.pipeline.phase2_generate.compose_video import compose_frame
from src.pipeline.phase2_generate.cropping import crop_box
from src.pipeline.phase2_generate.policies import changed_fraction

H, W, RATIO = 120, 160, 0.8
BOX = (60.0, 40.0, 100.0, 80.0)


def frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def face(track: int = 1, box=BOX) -> Face:
    return Face(track_id=track, box=box, confidence=0.9)


def generated_crop(src: np.ndarray, box=BOX) -> np.ndarray:
    """The context crop with the face box replaced by a flat synthetic patch."""
    cx1, cy1, cx2, cy2 = crop_box(H, W, box, RATIO)
    crop = src[cy1:cy2, cx1:cx2].copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    crop[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = (10, 200, 30)
    return crop


def box_changed(out: np.ndarray, src: np.ndarray, box=BOX) -> float:
    return changed_fraction(src, out, box)


def ok(fid: int, track: int = 1) -> dict:
    return {(fid, track): {"status": "ok", "output_path": "x.png"}}


def test_generated_face_is_pasted():
    src = frame()
    out = src.copy()
    result, changed = compose_frame(out, 0, [face()], ok(0), lambda rec: generated_crop(src), RATIO, {})
    assert result == {1: "generated"}
    assert changed[1] == pytest.approx(1.0)
    x1, y1, x2, y2 = (int(v) for v in BOX)
    assert (out[y1:y2, x1:x2] == (10, 200, 30)).all()


def test_real_crop_returned_as_ok_is_hidden():
    """Fail-open path: the backend returned the real crop; it must not be shown."""
    src = frame()
    out = src.copy()
    cx1, cy1, cx2, cy2 = crop_box(H, W, BOX, RATIO)
    real = src[cy1:cy2, cx1:cx2].copy()
    result, changed = compose_frame(out, 0, [face()], ok(0), lambda rec: real, RATIO, {})
    assert result == {1: "filled"}
    assert 1 not in changed
    assert box_changed(out, src) > 0.9


def test_below_min_changed_is_hidden():
    src = frame()
    out = src.copy()
    cx1, cy1, cx2, cy2 = crop_box(H, W, BOX, RATIO)
    almost_real = src[cy1:cy2, cx1:cx2].copy()
    almost_real[0, 0] ^= 1   # one pixel outside the face box differs
    result, _ = compose_frame(out, 0, [face()], ok(0), lambda rec: almost_real, RATIO, {},
                              min_changed=MIN_CHANGED_FRACTION)
    assert result == {1: "filled"}


@pytest.mark.parametrize("ledger", [{}, {(0, 1): {"status": "failsafe", "reason": "subsampled",
                                                  "output_path": None}}])
def test_missing_or_failed_face_is_filled(ledger):
    src = frame()
    out = src.copy()
    result, _ = compose_frame(out, 0, [face()], ledger, lambda rec: None, RATIO, {})
    assert result == {1: "filled"}
    assert box_changed(out, src) > 0.9


def test_mismatched_crop_is_filled():
    src = frame()
    out = src.copy()
    result, _ = compose_frame(out, 0, [face()], ok(0), lambda rec: np.zeros((5, 5, 3), np.uint8), RATIO, {})
    assert result == {1: "filled"}


def test_reuse_within_window_then_fill():
    src = frame()
    last: dict = {}
    out0 = src.copy()
    compose_frame(out0, 0, [face()], ok(0), lambda rec: generated_crop(src), RATIO, last)
    assert 1 in last
    out = src.copy()
    result, _ = compose_frame(out, REUSE_FRAMES, [face()], {}, lambda rec: None, RATIO, last)
    assert result == {1: "reused"}
    out = src.copy()
    result, _ = compose_frame(out, REUSE_FRAMES + 1, [face()], {}, lambda rec: None, RATIO, last)
    assert result == {1: "filled"}


def test_failsafe_off_exposes():
    src = frame()
    out = src.copy()
    result, _ = compose_frame(out, 0, [face()], {}, lambda rec: None, RATIO, {}, failsafe=False)
    assert result == {}
    assert (out == src).all()


def test_no_known_face_keeps_real_pixels():
    """Invariant over random scenarios: with the fail-safe on, every face box changes."""
    rng = np.random.default_rng(1)
    last: dict = {}
    for fid in range(40):
        src = frame(fid)
        out = src.copy()
        boxes = {1: (20.0, 20.0, 60.0, 60.0), 2: (90.0, 50.0, 130.0, 95.0)}
        faces = [face(t, b) for t, b in boxes.items()]
        ledger, crops = {}, {}
        for t, b in boxes.items():
            kind = rng.choice(["ok", "real", "failsafe", "none"])
            if kind == "ok":
                ledger[(fid, t)] = {"status": "ok", "output_path": f"{t}.png"}
                crops[f"{t}.png"] = generated_crop(src, b)
            elif kind == "real":
                cx1, cy1, cx2, cy2 = crop_box(H, W, b, RATIO)
                ledger[(fid, t)] = {"status": "ok", "output_path": f"{t}.png"}
                crops[f"{t}.png"] = src[cy1:cy2, cx1:cx2].copy()
            elif kind == "failsafe":
                ledger[(fid, t)] = {"status": "failsafe", "output_path": None}
        result, _ = compose_frame(out, fid, faces, ledger, lambda rec: crops.get(rec["output_path"]), RATIO, last)
        for t, b in boxes.items():
            assert result[t] in ("generated", "reused", "filled")
            assert box_changed(out, src, b) >= MIN_CHANGED_FRACTION, (fid, t, result[t])
