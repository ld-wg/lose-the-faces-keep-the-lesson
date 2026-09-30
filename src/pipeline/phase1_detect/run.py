"""Phase 1 runner — face detection + tracking on video or images.

Produces the pipeline contract output: per-frame [frame_id, track_id, box, conf]
plus optional face crops (padded) for the Phase 2 generation stage.

Usage:
    # Video
    python -m src.pipeline.phase1_detect.run --input lecture.mp4 --out runs/phase1

    # Webcam (live preview)
    python -m src.pipeline.phase1_detect.run --webcam

    # Image directory
    python -m src.pipeline.phase1_detect.run --input imgs/ --out runs/phase1

    # Swap detector backend (research/stages/identification.md, Tier 0.5)
    python -m src.pipeline.phase1_detect.run --input lecture.mp4 --out runs/scrfd34 --model scrfd-34gf
    python -m src.pipeline.phase1_detect.run --input lecture.mp4 --out runs/yolov2 --model yolo-facev2-s

Outputs (in --out dir) — the Phase 1 -> Phase 2 contract, see src/pipeline/contracts.py:
    detections.jsonl   one JSON object per frame: {frame_id, tracks:[{track_id, box, conf, landmarks, crop_path}]}
    tracks.json         whole-video per-track manifest: seed, first/last frame, representative crop
                        (video/webcam mode only — image-directory mode has no persistent
                        track identity to seed-lock, so it writes detections.jsonl only)
    crops/             optional padded face crops named {frame:06d}_{track_id}.jpg
    preview.mp4        optional annotated video (if --preview; not part of the Phase 2 contract)

Recall post-pass (video mode, on by default; --no-fill turns it off): tracks are
emitted from their first detection, and `fill.py` adds boxes where a tracked face
almost certainly is (gaps, a few frames around each track, too-short tracks),
each marked with its `source`. `--conf` defaults to 0.1 so the tracker's
low-confidence association (0.1-0.5) receives detections; new tracks still start
only at >= 0.5. The pre-2026-09-30 output is `--no-fill --conf 0.5`: SCRFD-10GF
silently cut at 0.5 until then (FaceAnalysis.prepare's det_thresh default).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# allow running as a module from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CONFIG  # noqa: E402

from .detector import FaceDetector  # noqa: E402
from .fill import count_sources, fill  # noqa: E402
from .models import DEFAULT_WEIGHTS_FILENAME, MODEL_NAMES  # noqa: E402
from .tracker import FaceTracker  # noqa: E402
from ..contracts import Face, Frame, Identity, Manifest, Video, derive_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def draw_tracks(frame: np.ndarray, tracks) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"ID{t.track_id} {t.confidence:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, f"Faces: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    return out


# BGR per Face.source; post-pass boxes are drawn thinner, in their own color
SOURCE_COLORS = {"detector": (0, 255, 0), "interpolated": (0, 215, 255),
                 "dilated": (255, 160, 0), "unconfirmed": (200, 0, 200)}


def draw_faces(frame: np.ndarray, faces: list[Face]) -> np.ndarray:
    out = frame.copy()
    for f in faces:
        x1, y1, x2, y2 = map(int, f.box)
        color = SOURCE_COLORS[f.source]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 if f.detected else 1)
        text = f"ID{f.track_id} {f.confidence:.2f}" if f.detected else f"ID{f.track_id} {f.source}"
        cv2.putText(out, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    counts = {k: sum(1 for f in faces if f.source == k) for k in SOURCE_COLORS}
    cv2.putText(out, "  ".join(f"{k} {n}" for k, n in counts.items() if n), (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
    return out


def write_preview(video_path: str, frames: list[Frame], out_path: Path) -> None:
    """Second pass over the video: every face of detections.jsonl, drawn by source."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(draw_faces(frame, fr.faces))
    cap.release()
    writer.release()


def run_video(args, detector: FaceDetector, tracker: FaceTracker) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    if args.save_crops:
        crops_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(0 if args.webcam else str(args.input))
    if not cap.isOpened():
        logger.error(f"Could not open input: {args.input or 'webcam'}")
        sys.exit(1)

    jsonl_path = out_dir / "detections.jsonl"
    video_source = "webcam" if args.webcam else str(args.input)
    frame_id = 0
    frame_w = frame_h = 0
    t0 = time.time()
    total_faces = 0
    frames: list[Frame] = []   # whole video in memory (geometry only), written after the post-pass
    track_stats: dict[int, dict] = {}  # track_id -> running manifest fields

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_h, frame_w = frame.shape[:2]

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        total_faces += len(tracks)

        observations = []
        for t in tracks:
            crop_path = None
            if args.save_crops:
                x1 = max(0, int(t.box[0]) - args.pad)
                y1 = max(0, int(t.box[1]) - args.pad)
                x2 = min(frame_w, int(t.box[2]) + args.pad)
                y2 = min(frame_h, int(t.box[3]) + args.pad)
                crop = frame[y1:y2, x1:x2]
                crop_name = f"{frame_id:06d}_{t.track_id}.jpg"
                cv2.imwrite(str(crops_dir / crop_name), crop)
                crop_path = f"crops/{crop_name}"

            landmarks = [tuple(p) for p in t.landmarks.tolist()] if t.landmarks is not None else None
            observations.append(Face(
                track_id=t.track_id, box=t.box, confidence=t.confidence,
                landmarks=landmarks, crop_path=crop_path,
            ))

            # accumulate whole-video manifest fields for this track
            stats = track_stats.setdefault(t.track_id, {
                "first_frame": frame_id, "last_frame": frame_id, "num_observations": 0,
                "best_confidence": -1.0, "best_crop": None,
            })
            stats["last_frame"] = frame_id
            stats["num_observations"] += 1
            if t.confidence > stats["best_confidence"]:
                stats["best_confidence"] = t.confidence
                stats["best_crop"] = crop_path

        frames.append(Frame(frame_id=frame_id, faces=observations))

        if args.webcam:
            cv2.imshow(f"Phase 1 — {args.model} + ByteTrack", draw_tracks(frame, tracks))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1
        if frame_id % 100 == 0:
            logger.info(f"frame {frame_id}: {len(tracks)} active tracks")

    cap.release()
    if args.webcam:
        cv2.destroyAllWindows()

    confirmed = set(track_stats)
    if args.fill and not args.webcam:
        frames, confirmed = fill(frames, frame_id, min_hits=args.min_hits, max_gap=args.max_gap,
                                 dilate=args.dilate)
        logger.info(f"recall post-pass: {count_sources(frames)} face boxes by source, "
                    f"{len(confirmed)}/{len(track_stats)} tracks confirmed")

    # pipeline contract output (see src/pipeline/contracts.py)
    with jsonl_path.open("w") as jf:
        for fr in frames:
            jf.write(fr.to_json() + "\n")

    # whole-video track manifest (see src/pipeline/contracts.py) — the seed-locking
    # contract Phase 2 needs, derived entirely from what we already tracked above.
    # Only confirmed tracks get an identity; unconfirmed faces are hidden downstream.
    manifest = Manifest(
        video=Video(source=video_source, width=frame_w, height=frame_h,
                    fps=cap.get(cv2.CAP_PROP_FPS) or 0.0, frame_count=frame_id),
        identities=[
            Identity(
                track_id=tid, seed=derive_seed(video_source, tid),
                first_frame=s["first_frame"], last_frame=s["last_frame"],
                num_observations=s["num_observations"],
                representative_crop=s["best_crop"],
                representative_confidence=round(s["best_confidence"], 3),
            )
            for tid, s in track_stats.items() if tid in confirmed
        ],
    )
    manifest_path = out_dir / "tracks.json"
    manifest.save(manifest_path)

    if args.preview and not args.webcam:
        write_preview(str(args.input), frames, out_dir / "preview.mp4")

    dt = time.time() - t0
    fps = frame_id / dt if dt > 0 else 0
    logger.info(f"Done: {frame_id} frames, {total_faces} face-detections, "
                f"{len(confirmed)} unique tracks, {fps:.1f} fps")
    logger.info(f"Output: {jsonl_path}, {manifest_path}")


def run_images(args, detector: FaceDetector) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted([p for p in Path(args.input).iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not img_paths:
        logger.error(f"No images found in {args.input}")
        sys.exit(1)

    jsonl_path = out_dir / "detections.jsonl"
    with jsonl_path.open("w") as jf:
        for i, p in enumerate(img_paths):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            detections = detector.detect(frame)
            record = {
                "frame_id": i,
                "image": p.name,
                "tracks": [
                    {"track_id": -1, "box": [round(v, 1) for v in d.box],
                     "conf": round(d.confidence, 3)}
                    for d in detections
                ],
            }
            jf.write(json.dumps(record) + "\n")
            logger.info(f"{p.name}: {len(detections)} faces")
    logger.info(f"Output: {jsonl_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 — face detection + tracking")
    p.add_argument("--input", type=str, help="Video file or image directory")
    p.add_argument("--webcam", action="store_true", help="Use webcam (live preview)")
    p.add_argument("--out", type=str, default=str(CONFIG.runs_dir / "phase1"),
                   help="Output directory")
    p.add_argument("--conf", type=float, default=0.1,
                   help="Detection confidence threshold. Detections below the tracker's 0.5 only "
                        "extend existing tracks (ByteTrack's low-confidence association)")
    p.add_argument("--det-size", type=int, default=640, help="Detector input size")
    p.add_argument("--pad", type=int, default=32, help="Crop padding (px) for Phase 2")
    p.add_argument("--save-crops", action="store_true", help="Save padded face crops")
    p.add_argument("--preview", action="store_true", help="Write annotated preview.mp4")
    p.add_argument("--model", type=str, default="scrfd-10gf", choices=MODEL_NAMES,
                   help="Detector backend (research/stages/identification.md, Tier 0.5)")
    p.add_argument("--weights", type=str, default=None,
                   help="Weights file for --model (default: CONFIG.weights_dir/<model default filename>; "
                        "unused for scrfd-10gf, which resolves its pack by name)")
    p.add_argument("--no-fill", dest="fill", action="store_false",
                   help="Skip the recall post-pass (fill.py): emit tracks only after --min-hits, "
                        "no gap/dilation boxes (the pre-2026-09-30 output, with --conf 0.5)")
    p.add_argument("--min-hits", type=int, default=3, help="Detections before a track is confirmed")
    p.add_argument("--max-gap", type=int, default=30, help="Longest in-track gap to interpolate (frames)")
    p.add_argument("--dilate", type=int, default=3, help="Frames added before/after each confirmed track")
    args = p.parse_args()

    if not args.webcam and not args.input:
        p.error("provide --input or --webcam")

    weights = Path(args.weights) if args.weights else None
    if weights is None and args.model in DEFAULT_WEIGHTS_FILENAME:
        weights = CONFIG.weights_dir / DEFAULT_WEIGHTS_FILENAME[args.model]

    detector = FaceDetector(model=args.model, weights=weights,
                             conf_threshold=args.conf, det_size=(args.det_size, args.det_size))
    # With the post-pass, every matched track is emitted from its first hit and
    # confirmation happens afterwards (fill.py); without it, the tracker's own gate.
    tracker = FaceTracker(min_hits=1 if args.fill and not args.webcam else args.min_hits)

    if args.input and Path(args.input).is_dir():
        run_images(args, detector)
    else:
        run_video(args, detector, tracker)


if __name__ == "__main__":
    main()
