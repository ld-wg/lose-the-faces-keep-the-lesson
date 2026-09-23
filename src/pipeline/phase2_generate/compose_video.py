"""Phase 2 post-processing — paste generated crops back into a full video.

`run.py` only ever writes per-(frame, track) crop PNGs (`generated/<track_id>/
<frame_id>.png`) — each one already contains the *whole* processed crop
(background + composited face, or an unmodified passthrough), not just the
face region. This module does the "later compositing stage" `run.py`'s own
docstring refers to: read every frame of the source video, and for each
face `run.py` produced a crop for, paste that exact crop back at the same
pixel rectangle it was originally cut from (`run.py`'s `crop_box()` — same
formula, imported directly, not reimplemented) — then write the whole
sequence out as a video. Frames/faces `run.py` skipped (no crop written,
e.g. `skipped_no_identity`/`skipped_no_frame`) are left as the untouched
source frame.

No re-blending happens here: blending already happened once, inside
`run.py`'s per-crop generation (`models/_compositing.py`), producing a
crop whose background pixels are already bit-identical to the source and
whose face region is already composited. Pasting that whole crop back is
just placing already-finished pixels at their known location.

Usage:
    python -m src.pipeline.phase2_generate.compose_video \\
        --phase1-dir runs/demo --phase2-dir runs/phase2-ganon-real \\
        --out runs/phase2-ganon-real/output.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# allow running as a module from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CONFIG  # noqa: E402

from .run import crop_box  # noqa: E402
from ..contracts import Frame, Manifest  # noqa: E402


def _load_ledger(ledger_path: Path) -> dict[tuple[int, int], str]:
    """(frame_id, track_id) -> output_path, for every ledger line that has one
    (i.e. `run.py` actually wrote a crop — excludes `skipped_no_identity`/
    `skipped_no_frame`, which have `output_path: null`)."""
    lookup: dict[tuple[int, int], str] = {}
    with ledger_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["output_path"] is not None:
                lookup[(rec["frame_id"], rec["track_id"])] = rec["output_path"]
    return lookup


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 — paste generated crops back into a full video")
    p.add_argument("--phase1-dir", type=str, required=True, help="Phase 1 output dir")
    p.add_argument("--phase2-dir", type=str, required=True,
                   help="Phase 2 output dir (from run.py --out) — needs generation.jsonl, "
                        "run_manifest.json, and generated/")
    p.add_argument("--video", type=str, default=None,
                   help="Source video path (default: manifest.video.source from tracks.json)")
    p.add_argument("--out", type=str, default=None,
                   help="Output video path (default: <phase2-dir>/output.mp4)")
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

    jsonl_path = phase1_dir / "detections.jsonl"
    num_frames = 0
    num_pasted = 0

    with jsonl_path.open() as jf:
        for line in jf:
            if args.limit is not None and num_frames >= args.limit:
                break
            frame_rec = Frame.from_json(line)
            num_frames += 1

            ret, frame = cap.read()
            if not ret:
                break  # source video ended before detections.jsonl did — same gap run.py logs

            for face in frame_rec.faces:
                output_path = ledger.get((frame_rec.frame_id, face.track_id))
                if output_path is None:
                    continue  # run.py skipped this one — leave the source frame untouched here
                crop_img = cv2.imread(str(phase2_dir / output_path))
                if crop_img is None:
                    continue  # crop file missing/unreadable — leave untouched rather than crash
                cx1, cy1, cx2, cy2 = crop_box(h, w, face.box, context_ratio)
                if crop_img.shape[:2] != (cy2 - cy1, cx2 - cx1):
                    continue  # shape mismatch (e.g. frame size changed) — skip, don't corrupt the frame
                frame[cy1:cy2, cx1:cx2] = crop_img
                num_pasted += 1

            writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote {out_path}: {num_frames} frames, {num_pasted} crop(s) pasted")


if __name__ == "__main__":
    main()
