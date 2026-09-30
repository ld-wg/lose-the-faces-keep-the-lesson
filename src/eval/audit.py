"""Exposure audit — where does a real face stay visible, and why?

Classifies every face instance of a video into one outcome and draws them on
the video, so detection misses and generation failures can be told apart:

    generated   Phase 1 found it and Phase 2 replaced it
    failed      Phase 1 found it, Phase 2 fell back to passthrough (the real
                face is shown); the label carries the backend's reason
    detected    Phase 1 found it (no Phase 2 folder given)
    gap         the track lost the face mid-track: no box in this frame, so
                nothing downstream touches it. The box is interpolated between
                the observations on either side of the gap.

Faces Phase 1 never found at all are only visible in the video (no box on a
face). The tracker also hides the first `min_hits - 1` frames of every track;
that count is reported as an estimate, since Phase 1 does not record them.

Usage:
    python -m src.eval.audit --phase1-dir runs/demo3 --out runs/audit-demo3 \\
        [--phase2-dir runs/phase2-blanket-demo3-p2-b1.2] [--video path] [--background output.mp4]

Outputs (in --out): audit.json, audit.md, audit.mp4. No face crop or embedding
is written; the video is the source footage with boxes on it.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from ..pipeline.contracts import Frame, Manifest

MIN_HITS = 3  # src/pipeline/phase1_detect/tracker.py FaceTracker default

# BGR
COLORS = {"generated": (80, 200, 60), "failed": (40, 40, 235), "detected": (235, 160, 40), "gap": (0, 215, 255)}
SHORT = {"identity_unusable": "identity", "identity_no_face": "identity", "swap_no_face": "swap-no-face",
         "swap_iou_rejected": "swap-iou", "skipped_no_landmarks": "no-landmarks"}


def load_phase1(phase1_dir: Path) -> list[Frame]:
    with (phase1_dir / "detections.jsonl").open() as f:
        return [Frame.from_json(line) for line in f]


def load_ledger(phase2_dir: Path) -> dict[tuple[int, int], tuple[str, str | None]]:
    """(frame_id, track_id) -> (status, reason)."""
    out = {}
    with (phase2_dir / "generation.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            out[(r["frame_id"], r["track_id"])] = (r["status"], r.get("reason"))
    return out


def track_gaps(frames: list[Frame]) -> dict[int, list[dict]]:
    """frame_id -> interpolated boxes of tracks that are alive but unmatched there."""
    seen: dict[int, dict[int, tuple]] = collections.defaultdict(dict)  # track -> frame -> box
    for fr in frames:
        for face in fr.faces:
            seen[face.track_id][fr.frame_id] = tuple(face.box)
    gaps: dict[int, list[dict]] = collections.defaultdict(list)
    for tid, boxes in seen.items():
        ids = sorted(boxes)
        for a, b in zip(ids, ids[1:]):
            n = b - a - 1
            for k in range(1, n + 1):
                t = k / (n + 1)
                box = tuple((1 - t) * p + t * q for p, q in zip(boxes[a], boxes[b]))
                gaps[a + k].append({"track_id": tid, "box": box, "index": k, "length": n})
    return gaps


def classify(frames, ledger, gaps) -> tuple[list[list[dict]], dict]:
    per_frame, per_track = [], collections.defaultdict(collections.Counter)
    for fr in frames:
        items = []
        for face in fr.faces:
            if ledger is None:
                outcome, reason = "detected", None
            else:
                status, reason = ledger.get((fr.frame_id, face.track_id), ("missing", None))
                outcome = "generated" if status == "ok" else "failed"
                reason = None if outcome == "generated" else (reason or status)
            items.append({"track_id": face.track_id, "box": tuple(face.box), "conf": face.confidence,
                          "outcome": outcome, "reason": reason})
            per_track[face.track_id][outcome if reason is None else f"failed:{reason}"] += 1
        for g in gaps.get(fr.frame_id, []):
            items.append({**g, "outcome": "gap", "reason": None})
            per_track[g["track_id"]]["gap"] += 1
        per_frame.append(items)
    return per_frame, per_track


def summarize(per_frame, per_track, gaps) -> dict:
    counts = collections.Counter()
    reasons = collections.Counter()
    for items in per_frame:
        for it in items:
            counts[it["outcome"]] += 1
            if it["reason"]:
                reasons[it["reason"]] += 1
    gap_lengths = collections.Counter()
    for items in gaps.values():
        for g in items:
            if g["index"] == 1:
                gap_lengths[g["length"]] += 1
    found = counts["generated"] + counts["failed"] + counts["detected"]
    exposed = counts["failed"] + counts["gap"]
    return {
        "frames": len(per_frame),
        "tracks": len(per_track),
        "detected": found,
        "outcomes": dict(counts),
        "failed_reasons": dict(reasons.most_common()),
        "gaps": {"count": sum(gap_lengths.values()), "frames": counts["gap"],
                 "lengths": dict(sorted(gap_lengths.items()))},
        "pre_confirmation_estimate": (MIN_HITS - 1) * len(per_track),
        "exposed_known": exposed,
        "exposed_share_of_known_faces": round(exposed / (found + counts["gap"]), 4) if found else None,
        "tracks_detail": {str(t): dict(c) for t, c in sorted(per_track.items())},
    }


def write_markdown(summary: dict, path: Path, phase2: bool) -> None:
    o = summary["outcomes"]
    known = summary["detected"] + summary["gaps"]["frames"]
    lines = [
        "# Exposure audit", "",
        f"{summary['frames']} frames, {summary['tracks']} tracks, {summary['detected']} detected face instances.", "",
        "| Outcome | Face instances | Share of known faces | Real face visible? |",
        "|---|---|---|---|",
    ]
    rows = [("generated", "no"), ("failed", "**yes**"), ("detected", "—"), ("gap", "**yes**")]
    for k, visible in rows:
        if o.get(k):
            lines.append(f"| {k} | {o[k]} | {o[k] / known:.3f} | {visible} |")
    lines += ["", f"Pre-confirmation frames hidden by the tracker (estimate, {MIN_HITS - 1} per track): "
              f"{summary['pre_confirmation_estimate']}. Faces never detected are not counted; check the video.", ""]
    if phase2 and summary["failed_reasons"]:
        lines += ["Generation failures by reason: " +
                  ", ".join(f"`{r}` {n}" for r, n in summary["failed_reasons"].items()), ""]
    lines += ["Gap lengths (frames: count): " +
              ", ".join(f"{k}: {v}" for k, v in summary["gaps"]["lengths"].items()), "",
              "| Track | " + " | ".join(["generated", "failed", "gap"] if phase2 else ["detected", "gap"]) + " | failure reasons |",
              "|---|" + "---|" * (4 if phase2 else 3)]
    for t, c in summary["tracks_detail"].items():
        failed = {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("failed:")}
        cells = ([c.get("generated", 0), sum(failed.values()), c.get("gap", 0)] if phase2
                 else [c.get("detected", 0), c.get("gap", 0)])
        lines.append(f"| {t} | " + " | ".join(map(str, cells)) + " | " +
                     ", ".join(f"{SHORT.get(k, k)} {v}" for k, v in failed.items()) + " |")
    path.write_text("\n".join(lines) + "\n")


def dashed_rect(img, p1, p2, color, thickness=2, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for xa, ya, xb, yb in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = int(np.hypot(xb - xa, yb - ya))
        for s in range(0, length, 2 * dash):
            e = min(s + dash, length)
            a = (int(xa + (xb - xa) * s / length), int(ya + (yb - ya) * s / length))
            b = (int(xa + (xb - xa) * e / length), int(ya + (yb - ya) * e / length))
            cv2.line(img, a, b, color, thickness)


def label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw(frame, frame_id, items, phase2: bool):
    counts = collections.Counter(it["outcome"] for it in items)
    for it in items:
        x1, y1, x2, y2 = map(int, it["box"])
        color = COLORS[it["outcome"]]
        if it["outcome"] == "gap":
            dashed_rect(frame, (x1, y1), (x2, y2), color)
            text = f"T{it['track_id']} lost {it['index']}/{it['length']}"
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"T{it['track_id']} " + (SHORT.get(it["reason"], it["reason"]) if it["reason"] else f"{it.get('conf', 0):.2f}")
        label(frame, text, x1, y1 - 2, color)
    keys = ["generated", "failed", "gap"] if phase2 else ["detected", "gap"]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (20, 20, 20), -1)
    x = 10
    cv2.putText(frame, f"frame {frame_id}", (x, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    x += 140
    for k in keys:
        cv2.rectangle(frame, (x, 9), (x + 14, 23), COLORS[k], -1)
        t = f"{k} {counts.get(k, 0)}"
        cv2.putText(frame, t, (x + 20, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        x += 40 + cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]


def main() -> None:
    p = argparse.ArgumentParser(description="Exposure audit: detection gaps vs generation failures, drawn on the video")
    p.add_argument("--phase1-dir", required=True)
    p.add_argument("--phase2-dir", default=None, help="Phase 2 output (generation.jsonl); omit for Phase 1 only")
    p.add_argument("--video", default=None, help="Source video (default: tracks.json video.source)")
    p.add_argument("--background", default=None,
                   help="Video to draw on instead of the source, e.g. the anonymized output.mp4")
    p.add_argument("--out", required=True)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    phase1_dir, out = Path(args.phase1_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = load_phase1(phase1_dir)
    ledger = load_ledger(Path(args.phase2_dir)) if args.phase2_dir else None
    gaps = track_gaps(frames)
    per_frame, per_track = classify(frames, ledger, gaps)
    summary = summarize(per_frame, per_track, gaps)
    summary["inputs"] = {"phase1_dir": str(phase1_dir), "phase2_dir": args.phase2_dir}
    (out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, out / "audit.md", ledger is not None)
    print((out / "audit.md").read_text())

    if args.no_video:
        return
    source = args.background or args.video or Manifest.load(phase1_dir / "tracks.json").video.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        p.error(f"could not open {source} (pass --video)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out / "audit.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr, items in zip(frames, per_frame):
        ok, frame = cap.read()
        if not ok:
            break
        draw(frame, fr.frame_id, items, ledger is not None)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"wrote {out / 'audit.mp4'}")


if __name__ == "__main__":
    main()
