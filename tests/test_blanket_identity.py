"""BLANKET backend: which face of the identity image is the swap source (2026-10-30).

The identity image is the context crop with the face inpainted, so it can hold real
neighbours. The backend sends the inpainted face's box (the candidate crop's box) to the swap
server, which takes the face that overlaps it; for identities cached before the box was
stored, the box is recovered from the candidate the cache entry names.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.pipeline.phase2_generate.models.blanket.backend import candidate_box


def cand(frame: int, box) -> SimpleNamespace:
    return SimpleNamespace(frame_id=frame, box_in_crop=box)


CANDS = [cand(10, (5, 5, 25, 25)), cand(42, (6, 7, 30, 31)), cand(99, (1, 2, 3, 4))]


def test_box_from_the_candidate_frame():
    assert candidate_box(CANDS, {"candidate_frame": 42, "attempt": 0}) == [6.0, 7.0, 30.0, 31.0]


def test_box_from_the_attempt_when_the_frame_is_unknown():
    assert candidate_box(CANDS, {"attempt": 2}) == [1.0, 2.0, 3.0, 4.0]


def test_no_box_when_nothing_matches():
    assert candidate_box(CANDS, {"candidate_frame": 7, "attempt": 5}) is None
    assert candidate_box([], {}) is None
