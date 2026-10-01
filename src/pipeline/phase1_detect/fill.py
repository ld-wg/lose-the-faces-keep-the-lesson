"""Phase 1 recall post-pass — put a box wherever a tracked face almost certainly is.

The tracker only reports a face on frames where the detector found it, so a
hand, an object or a head turn that hides the face for a few frames leaves
those frames with no box, and nothing downstream touches the real face. The
pipeline runs offline, so the whole video's tracks are known before writing
anything; this pass uses that:

    unconfirmed   tracks with fewer than `min_hits` detections keep their
                  faces (emitted from the first hit), marked "unconfirmed":
                  hidden in the final video, but no synthetic identity is
                  generated for what may be a false positive
    interpolated  every gap inside a confirmed track (at most the tracker's
                  `max_missed`, 30 frames) gets a linearly interpolated box
    dilated       `dilate` frames before a confirmed track's first detection
                  and after its last, with the edge box

Measured on video-demo-3.mov before this pass: 519 in-track gap frames and
~64 pre-confirmation frames showed a real face (research/next-steps/
exposure-audit-2026-09-30.md). Literature and parameter choices:
research/stages/identification-occlusion.md (ByteTrack interpolates gaps up
to 30 frames; FaceOff dilates by a few frames).

Only "detector" faces carry landmarks and reach the generator; the others
are hidden by compose_video.py's fail-safe.
"""

from __future__ import annotations

from collections import defaultdict

from ..contracts import Face, Frame


def track_gaps(frames: list[Frame], tracks: set[int] | None = None, max_gap: int | None = None) -> dict[int, list[dict]]:
    """frame_id -> boxes interpolated inside each track's gaps:
    [{track_id, box, index (1-based position in the gap), length}]."""
    seen: dict[int, dict[int, tuple]] = defaultdict(dict)  # track -> frame -> box
    for fr in frames:
        for face in fr.faces:
            if face.detected and (tracks is None or face.track_id in tracks):
                seen[face.track_id][fr.frame_id] = tuple(face.box)
    gaps: dict[int, list[dict]] = defaultdict(list)
    for tid, boxes in seen.items():
        ids = sorted(boxes)
        for a, b in zip(ids, ids[1:]):
            n = b - a - 1
            if n == 0 or (max_gap is not None and n > max_gap):
                continue
            for k in range(1, n + 1):
                t = k / (n + 1)
                box = tuple((1 - t) * p + t * q for p, q in zip(boxes[a], boxes[b]))
                gaps[a + k].append({"track_id": tid, "box": box, "index": k, "length": n})
    return gaps


def fill(frames: list[Frame], num_frames: int, *, min_hits: int = 3, max_gap: int = 30,
         dilate: int = 3) -> tuple[list[Frame], set[int]]:
    """Returns the frames with the post-pass faces added, and the confirmed track ids."""
    first: dict[int, tuple[int, tuple]] = {}
    last: dict[int, tuple[int, tuple]] = {}
    hits: dict[int, int] = defaultdict(int)
    for fr in frames:
        for face in fr.faces:
            hits[face.track_id] += 1
            first.setdefault(face.track_id, (fr.frame_id, tuple(face.box)))
            last[face.track_id] = (fr.frame_id, tuple(face.box))
    confirmed = {t for t, n in hits.items() if n >= min_hits}

    by_id = {fr.frame_id: fr for fr in frames}
    for fr in frames:
        for face in fr.faces:
            if face.track_id not in confirmed:
                face.source = "unconfirmed"

    added: dict[int, list[Face]] = defaultdict(list)
    for fid, items in track_gaps(frames, confirmed, max_gap).items():
        added[fid] += [Face(track_id=g["track_id"], box=g["box"], confidence=0.0, source="interpolated")
                       for g in items]
    for tid in confirmed:
        (f0, box0), (f1, box1) = first[tid], last[tid]
        for k in range(1, dilate + 1):
            if f0 - k >= 0:
                added[f0 - k].append(Face(track_id=tid, box=box0, confidence=0.0, source="dilated"))
            if f1 + k < num_frames:
                added[f1 + k].append(Face(track_id=tid, box=box1, confidence=0.0, source="dilated"))

    for fid, faces in added.items():
        if fid in by_id:
            by_id[fid].faces.extend(faces)
    return frames, confirmed


def count_sources(frames: list[Frame]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for fr in frames:
        for face in fr.faces:
            counts[face.source] += 1
    return dict(counts)
