"""Exposure ground truth: how many visible faces does Phase 1 never box?

The fail-closed composition hides every face Phase 1 knows about, so the
only real face a viewer can still see is one with no box at all. That
residual needs a person to look, so this module only prepares and scores the
looking:

    make   samples every `--every`-th frame and draws every box of
           detections.jsonl (any source) with a number, writes the frames and
           a CSV to fill in, one row per frame:
               frame_id, boxes, false_boxes, missed
           `false_boxes`: drawn boxes on no face; `missed`: visible faces
           (any pose, partly occluded included) with no box.
    rate   reads the filled CSV: residual miss rate = missed / (boxes −
           false_boxes + missed), with a 95% bootstrap interval over frames.

The sampled frames show real people: they stay on serra1 or the author's
machine under results/, gitignored, are never published, and are deleted
once the CSV is filled.

Usage:
    python -m src.eval.miss_sheet make --phase1-dir DET --out DIR [--every 15]
    python -m src.eval.miss_sheet rate DIR/annotations.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ..pipeline.contracts import Frame, Manifest

COLORS = {"detector": (0, 255, 0), "interpolated": (0, 215, 255), "dilated": (255, 160, 0),
          "unconfirmed": (200, 0, 200)}


def make(args) -> None:
    phase1_dir, out = Path(args.phase1_dir), Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    with (phase1_dir / "detections.jsonl").open() as f:
        frames = {fr.frame_id: fr for fr in map(Frame.from_json, f)}
    wanted = sorted(fid for fid in frames if fid % args.every == args.offset)
    cap = cv2.VideoCapture(args.video or Manifest.load(phase1_dir / "tracks.json").video.source)
    rows, fid = [], 0
    while wanted and fid <= wanted[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        if fid in wanted:
            faces = frames[fid].faces
            for i, face in enumerate(faces, 1):
                x1, y1, x2, y2 = map(int, face.box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[face.source], 2)
                cv2.putText(frame, str(i), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[face.source], 2)
            cv2.putText(frame, f"frame {fid}: {len(faces)} boxes. Count visible faces with NO box.", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(out / "frames" / f"{fid:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append({"frame_id": fid, "boxes": len(faces), "false_boxes": "", "missed": ""})
        fid += 1
    cap.release()
    with (out / "annotations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "boxes", "false_boxes", "missed"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} frames in {out / 'frames'}; fill {out / 'annotations.csv'}")


def rate(args) -> None:
    with open(args.csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["missed"] != ""]
    if not rows:
        raise SystemExit("no annotated rows (fill the missed column)")
    boxes = np.array([int(r["boxes"]) - int(r["false_boxes"] or 0) for r in rows], float)
    missed = np.array([int(r["missed"]) for r in rows], float)
    est = missed.sum() / max(boxes.sum() + missed.sum(), 1)
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(rows), (2000, len(rows)))
    boots = missed[pick].sum(1) / np.maximum(boxes[pick].sum(1) + missed[pick].sum(1), 1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    result = {"frames": len(rows), "visible_faces": int(boxes.sum() + missed.sum()), "missed": int(missed.sum()),
              "miss_rate": round(float(est), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Exposure ground truth: faces Phase 1 never boxed")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="sample frames with every box drawn, plus a CSV to fill")
    m.add_argument("--phase1-dir", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--video", default=None)
    m.add_argument("--every", type=int, default=15, help="sample one frame in this many")
    m.add_argument("--offset", type=int, default=7, help="which frame of each group (avoids frame 0)")
    r = sub.add_parser("rate", help="residual miss rate from a filled CSV")
    r.add_argument("csv")
    r.add_argument("--out", default=None, help="write the result as JSON here")
    args = p.parse_args()
    make(args) if args.cmd == "make" else rate(args)


if __name__ == "__main__":
    main()
