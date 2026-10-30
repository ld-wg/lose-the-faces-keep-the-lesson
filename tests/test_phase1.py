"""Stage 1: the ByteTrack-style tracker and the offline recall post-pass (fill.py)."""

from __future__ import annotations

from src.pipeline.contracts import Face, Frame
from src.pipeline.phase1_detect.detector import Detection
from src.pipeline.phase1_detect.fill import count_sources, fill, track_gaps
from src.pipeline.phase1_detect.tracker import FaceTracker, _iou


def det(x: float, conf: float = 0.9, y: float = 10.0, size: float = 20.0) -> Detection:
    return Detection(x, y, x + size, y + size, conf)


# --- tracker ---------------------------------------------------------------------------------

def test_iou():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_track_keeps_its_id_while_moving():
    tr = FaceTracker(min_hits=1)
    ids = {t.track_id for x in range(0, 40, 2) for t in tr.update([det(float(x))])}
    assert ids == {1}


def test_confirmation_after_min_hits():
    tr = FaceTracker(min_hits=3)
    assert tr.update([det(0)]) == []
    assert tr.update([det(1)]) == []
    assert [t.track_id for t in tr.update([det(2)])] == [1]


def test_low_confidence_extends_but_never_starts_a_track():
    tr = FaceTracker(min_hits=1)
    assert tr.update([det(0, conf=0.3)]) == []          # below high_thresh: no new track
    tr.update([det(50, conf=0.9)])
    out = tr.update([det(51, conf=0.3)])               # second-stage match keeps the track
    assert [t.track_id for t in out] == [1]


def test_lost_track_survives_max_missed_then_is_dropped():
    tr = FaceTracker(min_hits=1, max_missed=2)
    tr.update([det(0)])
    tr.update([])
    tr.update([])
    assert [t.track_id for t in tr.update([det(1)])] == [1]
    for _ in range(3):
        tr.update([])
    assert [t.track_id for t in tr.update([det(1)])] == [2]


def test_two_faces_two_tracks():
    tr = FaceTracker(min_hits=1)
    for x in range(5):
        out = tr.update([det(float(x)), det(100.0 + x)])
    assert sorted(t.track_id for t in out) == [1, 2]


# --- recall post-pass ------------------------------------------------------------------------

def frames_with(track_frames: dict[int, list[int]], n: int) -> list[Frame]:
    frames = [Frame(frame_id=i) for i in range(n)]
    for tid, fids in track_frames.items():
        for f in fids:
            frames[f].faces.append(Face(track_id=tid, box=(float(f), 0.0, f + 10.0, 10.0), confidence=0.9))
    return frames


def test_gaps_are_interpolated_linearly():
    gaps = track_gaps(frames_with({1: [0, 4]}, 5))
    assert sorted(gaps) == [1, 2, 3]
    assert gaps[2][0]["box"] == (2.0, 0.0, 12.0, 10.0)
    assert gaps[1][0]["length"] == 3


def test_gaps_longer_than_max_gap_stay_open():
    assert track_gaps(frames_with({1: [0, 10]}, 11), max_gap=5) == {}


def test_fill_sources():
    frames = frames_with({1: [5, 6, 9], 2: [20]}, 25)
    frames, confirmed = fill(frames, 25, min_hits=3, max_gap=30, dilate=2)
    assert confirmed == {1}
    counts = count_sources(frames)
    assert counts == {"detector": 3, "interpolated": 2, "dilated": 4, "unconfirmed": 1}
    by_frame = {fr.frame_id: [f.source for f in fr.faces] for fr in frames if fr.faces}
    assert by_frame[3] == ["dilated"] and by_frame[11] == ["dilated"]
    assert by_frame[20] == ["unconfirmed"]


def test_dilation_is_clipped_to_the_video():
    frames, _ = fill(frames_with({1: [0, 1, 2]}, 3), 3, dilate=3)
    assert count_sources(frames) == {"detector": 3}


def test_only_detector_faces_are_detected():
    frames, _ = fill(frames_with({1: [0, 2, 3]}, 4), 4, dilate=0)
    assert [f.detected for fr in frames for f in fr.faces] == [True, False, True, True]
