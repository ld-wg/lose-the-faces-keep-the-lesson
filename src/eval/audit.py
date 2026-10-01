"""Exposure audit — where does a real face stay visible, and why?

Classifies every face box of a video into one outcome and draws them on the
video, so detection misses, generation failures and the fail-safe can be
told apart:

    generated   Phase 2 replaced the face
    reused      hidden by the track's last generated face (compose fail-safe)
    filled      hidden by a neutral fill (compose fail-safe)
    failed      not generated, and no composed video to say what hid it
                (with --phase2-dir but no compose.jsonl); the label says why
    exposed     composed with --failsafe off: the real face is shown
    detected    Phase 1 box, no Phase 2 folder given (post-pass boxes are
                labelled with their source)
    gap         the track has no box at all in this frame (Phase 1 without
                the recall post-pass): the box is interpolated between the
                observations on either side, and nothing hides the face

Faces Phase 1 never found at all are only visible in the video (no box on a
face).

Usage:
    python -m src.eval.audit --phase1-dir runs/demo3 --out runs/audit-demo3 \\
        [--phase2-dir runs/phase2-blanket-demo3] [--video path] [--background output.mp4]

Outputs (in --out): audit.json, audit.md, audit.mp4. No face crop or embedding
is written; the video is the source footage (or --background) with boxes on it.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from ..pipeline.contracts import Frame, Manifest
from ..pipeline.phase1_detect.fill import track_gaps


# BGR
COLORS = {"generated": (80, 200, 60), "reused": (200, 220, 90), "filled": (170, 170, 170),
          "failed": (40, 40, 235), "exposed": (40, 40, 235), "detected": (235, 160, 40),
          "gap": (0, 215, 255)}
EXPOSED = ("failed", "exposed", "gap")
SHORT = {"identity_unusable": "identity", "identity_no_face": "identity", "swap_no_face": "swap-no-face",
         "swap_iou_rejected": "swap-iou", "skipped_no_landmarks": "no-landmarks"}


def load_phase1(phase1_dir: Path) -> list[Frame]:
    with (phase1_dir / "detections.jsonl").open() as f:
        return [Frame.from_json(line) for line in f]


def load_jsonl(path: Path) -> dict[tuple[int, int], dict]:
    with path.open() as f:
        return {(r["frame_id"], r["track_id"]): r for r in map(json.loads, f)}


def classify(frames, ledger, composed) -> list[list[dict]]:
    """Every face box with its outcome, plus the track gaps nothing covers."""
    gaps = track_gaps(frames)
    per_frame = []
    for fr in frames:
        items, boxed = [], {f.track_id for f in fr.faces}
        for face in fr.faces:
            key = (fr.frame_id, face.track_id)
            reason = None if face.detected else face.source
            if composed is not None and key in composed:
                outcome = composed[key]["outcome"]
            elif ledger is not None:
                rec = ledger.get(key, {"status": "missing"})
                outcome = "generated" if rec["status"] == "ok" else "failed"
                if outcome == "failed" and face.detected:
                    reason = rec.get("reason") or rec["status"]
            else:
                outcome = "detected"
            if outcome in ("reused", "filled", "exposed") and face.detected and ledger is not None:
                rec = ledger.get(key)
                reason = (rec.get("reason") or rec["status"]) if rec else None
            items.append({"track_id": face.track_id, "box": tuple(face.box), "conf": face.confidence,
                          "source": face.source, "outcome": outcome, "reason": reason})
        for g in gaps.get(fr.frame_id, []):
            if g["track_id"] not in boxed:
                items.append({**g, "source": None, "outcome": "gap", "reason": None})
        per_frame.append(items)
    return per_frame


def summarize(per_frame) -> dict:
    outcomes, reasons, sources = collections.Counter(), collections.Counter(), collections.Counter()
    per_track: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    gap_lengths = collections.Counter()
    for items in per_frame:
        for it in items:
            outcomes[it["outcome"]] += 1
            per_track[it["track_id"]][it["outcome"]] += 1
            if it["source"]:
                sources[it["source"]] += 1
            if it["reason"] and it["outcome"] != "generated":
                reasons[it["reason"]] += 1
                per_track[it["track_id"]][f"why:{it['reason']}"] += 1
            if it["outcome"] == "gap" and it["index"] == 1:
                gap_lengths[it["length"]] += 1
    known = sum(outcomes.values())
    exposed = sum(outcomes[k] for k in EXPOSED)
    return {
        "frames": len(per_frame),
        "tracks": len(per_track),
        "face_boxes": sum(sources.values()),
        "sources": dict(sources),
        "outcomes": dict(outcomes),
        "not_generated_reasons": dict(reasons.most_common()),
        "gaps": {"count": sum(gap_lengths.values()), "frames": outcomes["gap"],
                 "lengths": dict(sorted(gap_lengths.items()))},
        "exposed_known": exposed,
        "exposed_share_of_known_faces": round(exposed / known, 4) if known else None,
        "tracks_detail": {str(t): dict(c) for t, c in sorted(per_track.items())},
    }


def write_markdown(summary: dict, path: Path) -> None:
    o = summary["outcomes"]
    known = sum(o.values())
    lines = [
        "# Exposure audit", "",
        f"{summary['frames']} frames, {summary['tracks']} tracks, {summary['face_boxes']} face boxes "
        f"(by source: {summary['sources']}).", "",
        "| Outcome | Face instances | Share of known faces | Real face visible? |",
        "|---|---|---|---|",
    ]
    for k in COLORS:
        if o.get(k):
            visible = "**yes**" if k in EXPOSED else ("—" if k == "detected" else "no")
            lines.append(f"| {k} | {o[k]} | {o[k] / known:.3f} | {visible} |")
    lines += ["", f"Known faces with the real face visible: **{summary['exposed_known']}** "
              f"({summary['exposed_share_of_known_faces']}). Faces never detected are not counted; "
              "check the video.", ""]
    if summary["not_generated_reasons"]:
        lines += ["Not generated, by reason: " +
                  ", ".join(f"`{r}` {n}" for r, n in summary["not_generated_reasons"].items()), ""]
    if summary["gaps"]["count"]:
        lines += ["Uncovered track gaps (frames: count): " +
                  ", ".join(f"{k}: {v}" for k, v in summary["gaps"]["lengths"].items()), ""]
    shown = [k for k in COLORS if o.get(k)]
    lines += ["| Track | " + " | ".join(shown) + " | not generated because |", "|---|" + "---|" * (len(shown) + 1)]
    for t, c in summary["tracks_detail"].items():
        why = {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("why:")}
        lines.append(f"| {t} | " + " | ".join(str(c.get(k, 0)) for k in shown) + " | " +
                     ", ".join(f"{SHORT.get(k, k)} {v}" for k, v in why.items()) + " |")
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


def draw(frame, frame_id, items, keys):
    counts = collections.Counter(it["outcome"] for it in items)
    for it in items:
        x1, y1, x2, y2 = map(int, it["box"])
        color = COLORS[it["outcome"]]
        if it["outcome"] == "gap":
            dashed_rect(frame, (x1, y1), (x2, y2), color)
            text = f"T{it['track_id']} lost {it['index']}/{it['length']}"
        else:
            if it["source"] == "detector":
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                dashed_rect(frame, (x1, y1), (x2, y2), color, thickness=1, dash=5)
            detail = SHORT.get(it["reason"], it["reason"]) if it["reason"] else f"{it['conf']:.2f}"
            text = f"T{it['track_id']} {detail}" if it["outcome"] in ("generated", "detected") else \
                f"T{it['track_id']} {it['outcome']} {detail}"
        label(frame, text, x1, y1 - 2, color)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(frame, f"frame {frame_id}", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    x = 150
    for k in keys:
        cv2.rectangle(frame, (x, 9), (x + 14, 23), COLORS[k], -1)
        t = f"{k} {counts.get(k, 0)}"
        cv2.putText(frame, t, (x + 20, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        x += 40 + cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]


def main() -> None:
    p = argparse.ArgumentParser(description="Exposure audit: what hides (or does not hide) every face, drawn on the video")
    p.add_argument("--phase1-dir", required=True)
    p.add_argument("--phase2-dir", default=None,
                   help="Phase 2 output (generation.jsonl; compose.jsonl too, if the video was composed)")
    p.add_argument("--compose", default=None, help="compose.jsonl to use (default: <phase2-dir>/compose.jsonl)")
    p.add_argument("--video", default=None, help="Source video (default: tracks.json video.source)")
    p.add_argument("--background", default=None,
                   help="Video to draw on instead of the source, e.g. the anonymized output.mp4")
    p.add_argument("--out", required=True)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    phase1_dir, out = Path(args.phase1_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = load_phase1(phase1_dir)
    ledger = load_jsonl(Path(args.phase2_dir) / "generation.jsonl") if args.phase2_dir else None
    compose_path = Path(args.compose) if args.compose else (
        Path(args.phase2_dir) / "compose.jsonl" if args.phase2_dir else None)
    composed = load_jsonl(compose_path) if compose_path and compose_path.is_file() else None
    per_frame = classify(frames, ledger, composed)
    summary = summarize(per_frame)
    summary["inputs"] = {"phase1_dir": str(phase1_dir), "phase2_dir": args.phase2_dir,
                         "compose": str(compose_path) if composed is not None else None}
    (out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, out / "audit.md")
    print((out / "audit.md").read_text())

    if args.no_video:
        return
    keys = [k for k in COLORS if summary["outcomes"].get(k)]
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
        draw(frame, fr.frame_id, items, keys)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"wrote {out / 'audit.mp4'}")


if __name__ == "__main__":
    main()
