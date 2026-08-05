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

Outputs (in --out dir):
    detections.jsonl   one JSON object per frame: {frame_id, tracks:[{track_id, box, conf}]}
    crops/             optional padded face crops named {frame:06d}_{track_id}.jpg
    preview.mp4        optional annotated video (if --preview)
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
from .tracker import FaceTracker  # noqa: E402

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

    writer = None
    if args.preview and not args.webcam:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_dir / "preview.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    jsonl_path = out_dir / "detections.jsonl"
    frame_id = 0
    t0 = time.time()
    total_faces = 0

    with jsonl_path.open("w") as jf:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            total_faces += len(tracks)

            # pipeline contract output
            record = {
                "frame_id": frame_id,
                "tracks": [
                    {"track_id": t.track_id, "box": [round(v, 1) for v in t.box],
                     "conf": round(t.confidence, 3)}
                    for t in tracks
                ],
            }
            jf.write(json.dumps(record) + "\n")

            # optional padded crops for Phase 2
            if args.save_crops:
                h, w = frame.shape[:2]
                for t in tracks:
                    x1 = max(0, int(t.box[0]) - args.pad)
                    y1 = max(0, int(t.box[1]) - args.pad)
                    x2 = min(w, int(t.box[2]) + args.pad)
                    y2 = min(h, int(t.box[3]) + args.pad)
                    crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(str(crops_dir / f"{frame_id:06d}_{t.track_id}.jpg"), crop)

            if args.webcam or writer is not None:
                vis = draw_tracks(frame, tracks)
                if args.webcam:
                    cv2.imshow("Phase 1 — RetinaFace + ByteTrack", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if writer is not None:
                    writer.write(vis)

            frame_id += 1
            if frame_id % 100 == 0:
                logger.info(f"frame {frame_id}: {len(tracks)} active tracks")

    cap.release()
    if writer is not None:
        writer.release()
    if args.webcam:
        cv2.destroyAllWindows()

    dt = time.time() - t0
    fps = frame_id / dt if dt > 0 else 0
    logger.info(f"Done: {frame_id} frames, {total_faces} face-detections, "
                f"{tracker._next_id - 1} unique tracks, {fps:.1f} fps")
    logger.info(f"Output: {jsonl_path}")


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
    p.add_argument("--conf", type=float, default=0.3,
                   help="Detection confidence threshold (low = high recall)")
    p.add_argument("--det-size", type=int, default=640, help="Detector input size")
    p.add_argument("--pad", type=int, default=32, help="Crop padding (px) for Phase 2")
    p.add_argument("--save-crops", action="store_true", help="Save padded face crops")
    p.add_argument("--preview", action="store_true", help="Write annotated preview.mp4")
    args = p.parse_args()

    if not args.webcam and not args.input:
        p.error("provide --input or --webcam")

    detector = FaceDetector(conf_threshold=args.conf, det_size=(args.det_size, args.det_size))
    tracker = FaceTracker()

    if args.input and Path(args.input).is_dir():
        run_images(args, detector)
    else:
        run_video(args, detector, tracker)


if __name__ == "__main__":
    main()
