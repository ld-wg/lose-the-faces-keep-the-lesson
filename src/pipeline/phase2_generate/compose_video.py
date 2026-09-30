"""Phase 2 post-processing — build the final video, and never let a known face through.

`run.py` writes one crop PNG per (frame, track) (`generated/<track_id>/
<frame_id>.png`): the context-padded crop with the face replaced, whose
background pixels are bit-identical to the source (blending happened inside
`run.py`, lossless PNG). This module reads the source video and, per frame:

1. **Generated faces.** Pastes each generated crop back at the rectangle
   `run.py`'s `crop_box()` cut it from. Only the pixels the generator changed
   are pasted. Before 2026-09-30 the whole crop was pasted, and a later
   crop's real background overwrote part of an earlier neighbour's generated
   face: 432 of 2139 generated faces on video-demo-3.mov (20%), 125 of them
   over at least half the box.
2. **Fail-safe (fail closed).** Every other face box of `detections.jsonl`
   is hidden. That covers a generation failure (passthrough), a Phase 1
   post-pass box (gap, dilation, unconfirmed track) or no ledger row:
   - `reused`: the same track's last generated face, if it is at most
     `--reuse-frames` old, resized into the current box (BLANKET's own video
     pipeline also reuses the last successful swap);
   - `filled`: otherwise, a neutral gray ellipse.

   Both are pasted with a feathered ellipse inscribed in the box expanded
   `--expand` times; the default 1.5 covers the whole detection box. Fills
   go last, so nothing pasted afterwards can uncover them. Blur and
   pixelation are not used: both are reversible or attackable
   (research/stages/identification-occlusion.md).

Faces Phase 1 never found cannot be hidden here; that is Phase 1's recall
(`phase1_detect/fill.py`).

Writes the video and `compose.jsonl` next to it: one line per face box,
{frame_id, track_id, source, status, outcome}, where outcome is
generated / reused / filled, or `exposed` with `--failsafe off` (diagnostic
only: the pre-2026-09-30 behavior for failures).

Usage:
    python -m src.pipeline.phase2_generate.compose_video \\
        --phase1-dir runs/demo --phase2-dir runs/phase2-ganon-real \\
        --out runs/phase2-ganon-real/output.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# allow running as a module from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CONFIG  # noqa: E402,F401

from .cropping import crop_box  # noqa: E402
from ..contracts import Frame, Manifest  # noqa: E402

FILL_BGR = (128, 128, 128)


def _load_ledger(ledger_path: Path) -> dict[tuple[int, int], dict]:
    """(frame_id, track_id) -> the ledger record."""
    with ledger_path.open() as f:
        return {(r["frame_id"], r["track_id"]): r for r in map(json.loads, f)}


def _region(box, expand: float, w: int, h: int):
    """The ellipse inscribed in the box expanded `expand` times, and the frame
    region holding it plus its feathered edge (clipped to the frame), with the
    ellipse in region coordinates: (cx, cy, half-width, half-height, feather)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * expand / 2, (y2 - y1) * expand / 2
    feather = max(1.0, 0.1 * min(hw, hh))
    pad = 3 * feather  # the soft edge fades out inside the region, not at its border
    rx1, ry1 = max(0, int(cx - hw - pad)), max(0, int(cy - hh - pad))
    rx2, ry2 = min(w, int(np.ceil(cx + hw + pad))), min(h, int(np.ceil(cy + hh + pad)))
    return (rx1, ry1, rx2, ry2), (cx - rx1, cy - ry1, hw, hh, feather)


def _alpha(shape, ellipse) -> np.ndarray:
    """Feathered ellipse: 1 inside, soft only outside the ellipse's edge."""
    cx, cy, hw, hh, feather = ellipse
    mask = np.zeros(shape, np.float32)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (max(1, int(hw)), max(1, int(hh))),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return np.clip(2 * mask, 0, 1)[..., None]


def _blend(frame: np.ndarray, region, ellipse, content: np.ndarray) -> None:
    rx1, ry1, rx2, ry2 = region
    if rx2 <= rx1 or ry2 <= ry1:
        return
    dst = frame[ry1:ry2, rx1:rx2]
    a = _alpha(dst.shape[:2], ellipse)
    dst[:] = (a * content + (1 - a) * dst).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 — final video: generated faces, every other face hidden")
    p.add_argument("--phase1-dir", type=str, required=True, help="Phase 1 output dir")
    p.add_argument("--phase2-dir", type=str, required=True,
                   help="Phase 2 output dir (from run.py --out) — needs generation.jsonl, "
                        "run_manifest.json, and generated/")
    p.add_argument("--video", type=str, default=None,
                   help="Source video path (default: manifest.video.source from tracks.json)")
    p.add_argument("--out", type=str, default=None,
                   help="Output video path (default: <phase2-dir>/output.mp4); compose.jsonl goes next to it")
    p.add_argument("--failsafe", choices=("on", "off"), default="on",
                   help="off leaves failed faces real (diagnostic only)")
    p.add_argument("--reuse-frames", type=int, default=10,
                   help="Reuse the track's last generated face if at most this many frames old")
    p.add_argument("--expand", type=float, default=1.5,
                   help="Hidden region: ellipse inscribed in the box expanded this much (>= 1.42 covers the box)")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many frames (testing)")
    args = p.parse_args()

    phase1_dir = Path(args.phase1_dir)
    phase2_dir = Path(args.phase2_dir)
    out_path = Path(args.out) if args.out else phase2_dir / "output.mp4"

    manifest = Manifest.load(phase1_dir / "tracks.json")
    run_manifest = json.loads((phase2_dir / "run_manifest.json").read_text())
    context_ratio = run_manifest["context_ratio"]
    ledger = _load_ledger(phase2_dir / "generation.jsonl")

    video_path = args.video or manifest.video.source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        p.error(f"could not open source video: {video_path} (pass --video to override)")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # same fallback as phase1_detect/run.py's --preview
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_face: dict[int, tuple[int, np.ndarray]] = {}  # track -> (frame_id, final pixels of its region)
    outcomes: Counter = Counter()
    num_frames = 0

    with (phase1_dir / "detections.jsonl").open() as jf, (out_path.parent / "compose.jsonl").open("w") as cf:
        for line in jf:
            if args.limit is not None and num_frames >= args.limit:
                break
            frame_rec = Frame.from_json(line)
            num_frames += 1

            ret, frame = cap.read()
            if not ret:
                break  # source video ended before detections.jsonl did — same gap run.py logs
            source = frame.copy()
            fid = frame_rec.frame_id
            result: dict[int, str] = {}

            # 1. generated faces: only the pixels the generator changed
            for face in frame_rec.faces:
                rec = ledger.get((fid, face.track_id))
                if rec is None or rec["status"] != "ok" or not rec.get("output_path"):
                    continue
                crop_img = cv2.imread(str(phase2_dir / rec["output_path"]))
                cx1, cy1, cx2, cy2 = crop_box(h, w, face.box, context_ratio)
                if crop_img is None or crop_img.shape[:2] != (cy2 - cy1, cx2 - cx1):
                    continue  # missing or mismatched crop: left to the fail-safe below
                changed = np.any(crop_img != source[cy1:cy2, cx1:cx2], axis=2)
                frame[cy1:cy2, cx1:cx2][changed] = crop_img[changed]
                result[face.track_id] = "generated"

            # 2. fail-safe: reuse first, fills last
            pending = [f for f in frame_rec.faces if f.track_id not in result]
            if args.failsafe == "on":
                for face in pending:
                    prev = last_face.get(face.track_id)
                    if prev is not None and fid - prev[0] <= args.reuse_frames:
                        region, ellipse = _region(face.box, args.expand, w, h)
                        rx1, ry1, rx2, ry2 = region
                        if rx2 > rx1 and ry2 > ry1:
                            _blend(frame, region, ellipse, cv2.resize(prev[1], (rx2 - rx1, ry2 - ry1)))
                            result[face.track_id] = "reused"
                for face in pending:
                    if face.track_id not in result:
                        region, ellipse = _region(face.box, args.expand, w, h)
                        rx1, ry1, rx2, ry2 = region
                        _blend(frame, region, ellipse, np.full((ry2 - ry1, rx2 - rx1, 3), FILL_BGR, np.uint8))
                        result[face.track_id] = "filled"

            # 3. remember each generated face as it finally looks (neighbours already hidden)
            for face in frame_rec.faces:
                if result.get(face.track_id) == "generated":
                    (rx1, ry1, rx2, ry2), _ = _region(face.box, args.expand, w, h)
                    if rx2 > rx1 and ry2 > ry1:
                        last_face[face.track_id] = (fid, frame[ry1:ry2, rx1:rx2].copy())

            for face in frame_rec.faces:
                outcome = result.get(face.track_id, "exposed")
                outcomes[outcome] += 1
                rec = ledger.get((fid, face.track_id))
                cf.write(json.dumps({"frame_id": fid, "track_id": face.track_id, "source": face.source,
                                     "status": rec["status"] if rec else None, "outcome": outcome}) + "\n")
            writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote {out_path}: {num_frames} frames, face boxes {dict(outcomes)}")


if __name__ == "__main__":
    main()
