"""Phase 2 runner — synthetic face generation over a Phase 1 run's output.

Reads the Phase 1 -> Phase 2 contract (`src/pipeline/contracts.py`):
`detections.jsonl` + `tracks.json`. Face pixels come from the *source video*
directly (`Manifest.video.source`), not from Phase 1's saved debug crops —
this runner cuts its own crop around each `Face.box`, sized proportionally
to the box via the backend's own `DEFAULT_CONTEXT_RATIO` (see `models/`).

Why not reuse Phase 1's `--save-crops` output: those crops use a small
fixed pad (32px) meant for visual debugging, not for feeding a generator
that expects a specific portrait framing convention (see
`models/ciagan/NOTICE.md`'s oversized-face bug writeup). Deriving the crop
straight from `Face.box` + the source video sidesteps that mismatch
entirely and drops the `--save-crops` requirement on the Phase 1 run that
feeds this one — one less flag to remember.

Usage:
    # Real run (needs a converted CIAGAN checkpoint + dlib shape predictor —
    # see models/ciagan/NOTICE.md)
    python -m src.pipeline.phase2_generate.run --phase1-dir runs/phase1 --out runs/phase2

    # Smoke test: no real checkpoint needed, output is NOT real anonymization
    python -m src.pipeline.phase2_generate.run --phase1-dir runs/phase1 --out runs/phase2-smoke --random-init

Outputs (in --out dir):
    generated/<track_id>/<frame_id:06d>.png   one PNG per (frame, track) — the
                                               box-proportional context crop,
                                               lossless (this is the pixel
                                               data a later compositing stage
                                               consumes, not a debug artifact).
    generation.jsonl    one ledger line per processed item: status is "ok",
                        "skipped_no_landmarks" (dlib found nothing usable —
                        passthrough of the original crop, unmodified),
                        "skipped_no_frame" (the source video ended before this
                        frame_id) or "skipped_no_identity" (track_id missing
                        from tracks.json — a Phase 1 data-integrity gap).
    run_manifest.json   run-level summary: config + per-identity frame counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

# allow running as a module from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CONFIG  # noqa: E402

from .generator import FaceGenerator  # noqa: E402
from .models import (  # noqa: E402
    DEFAULT_CONTEXT_RATIO,
    DEFAULT_DLIB_PREDICTOR_FILENAME,
    DEFAULT_IMG_SIZE,
    DEFAULT_SEGMENTATION_WEIGHTS_FILENAME,
    DEFAULT_WEIGHTS_FILENAME,
    MODEL_NAMES,
)
from ..contracts import Frame, Manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _context_crop(frame, box: tuple[float, float, float, float], context_ratio: float):
    """Cut a crop around `box`, padded proportionally to its own size.

    Unlike a fixed-pixel pad, this scales with how big the face already is
    in the frame — needed so the crop's framing stays proportionally
    consistent across near/far faces (see `models/ciagan/NOTICE.md` for why
    a fixed small pad broke CIAGAN's own portrait-crop assumption).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = context_ratio * box_w, context_ratio * box_h
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    return frame[cy1:cy2, cx1:cx2]


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 — synthetic face generation")
    p.add_argument("--phase1-dir", type=str, required=True, help="Phase 1 output dir")
    p.add_argument("--out", type=str, default=str(CONFIG.runs_dir / "phase2"),
                   help="Output directory")
    p.add_argument("--video", type=str, default=None,
                   help="Source video path (default: manifest.video.source from tracks.json — "
                        "override if that path no longer resolves, e.g. the file moved)")
    p.add_argument("--model", type=str, default="ciagan", choices=MODEL_NAMES,
                   help="Generation backend")
    p.add_argument("--weights", type=str, default=None,
                   help="Weights file for --model (default: CONFIG.weights_dir/<model default filename>; "
                        "unused with --random-init)")
    p.add_argument("--dlib-predictor", type=str, default=None,
                   help="ciagan: path to shape_predictor_68_face_landmarks.dat "
                        "(default: CONFIG.weights_dir/<default filename>)")
    p.add_argument("--segmentation-weights", type=str, default=None,
                   help="ganonymization (required)/ciagan with --refine-mask (opt-in): path to the "
                        "head-segmentation checkpoint (default: CONFIG.weights_dir/<default filename>, "
                        "see models/DEFAULT_SEGMENTATION_WEIGHTS_FILENAME)")
    p.add_argument("--context-ratio", type=float, default=None,
                   help="Crop padding as a multiple of the detected box's own width/height "
                        "(default: model-specific, see models/DEFAULT_CONTEXT_RATIO)")
    p.add_argument("--num-classes", type=int, default=1200,
                   help="ciagan: identity classes in the loaded checkpoint")
    p.add_argument("--img-size", type=int, default=None,
                   help="Network resolution (default: model-specific, see models/DEFAULT_IMG_SIZE — "
                        "ciagan is fixed at 128, ganonymization at 512, neither is a free knob)")
    p.add_argument("--portrait-scale", type=float, default=1.0,
                   help="ciagan: correction factor on the checkpoint's built-in CelebA-portrait "
                        "crop radius, calibrated for this project's footage (see "
                        "models/ciagan/NOTICE.md) — not a per-video tunable, don't change casually")
    p.add_argument("--align-rotation", action=argparse.BooleanOptionalAction, default=True,
                   help="ganonymization: level the crop by eye-line angle before landmark extraction "
                        "(default: on — see models/ganonymization/backend.py's module docstring)")
    p.add_argument("--refine-mask", action=argparse.BooleanOptionalAction, default=False,
                   help="ciagan: intersect its composite mask with a real head-segmentation model's "
                        "output, to clean up (not expand) the seam — opt-in, see models/ciagan/NOTICE.md")
    p.add_argument("--ctx-id", type=int, default=0, help="0 for GPU/MPS, -1 for CPU")
    p.add_argument("--random-init", action="store_true",
                   help="Smoke test: random generator weights, output is NOT real anonymization")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many frames (testing)")
    args = p.parse_args()

    phase1_dir = Path(args.phase1_dir)
    out_dir = Path(args.out)
    generated_dir = out_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(exist_ok=True)

    manifest = Manifest.load(phase1_dir / "tracks.json")
    identities = {ident.track_id: ident for ident in manifest.identities}

    video_path = args.video or manifest.video.source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        p.error(f"could not open source video: {video_path} (pass --video to override)")

    context_ratio = args.context_ratio
    if context_ratio is None:
        context_ratio = DEFAULT_CONTEXT_RATIO.get(args.model, 1.0)

    img_size = args.img_size if args.img_size is not None else DEFAULT_IMG_SIZE.get(args.model, 512)

    weights = Path(args.weights) if args.weights else None
    if weights is None and not args.random_init and args.model in DEFAULT_WEIGHTS_FILENAME:
        weights = CONFIG.weights_dir / DEFAULT_WEIGHTS_FILENAME[args.model]

    dlib_predictor = Path(args.dlib_predictor) if args.dlib_predictor else None
    if dlib_predictor is None and args.model in DEFAULT_DLIB_PREDICTOR_FILENAME:
        dlib_predictor = CONFIG.weights_dir / DEFAULT_DLIB_PREDICTOR_FILENAME[args.model]

    # Shared between ganonymization (required) and ciagan (opt-in via
    # --refine-mask) — see models/DEFAULT_SEGMENTATION_WEIGHTS_FILENAME.
    needs_segmentation = args.model == "ganonymization" or (args.model == "ciagan" and args.refine_mask)
    segmentation_weights = Path(args.segmentation_weights) if args.segmentation_weights else None
    if segmentation_weights is None and needs_segmentation:
        segmentation_weights = CONFIG.weights_dir / DEFAULT_SEGMENTATION_WEIGHTS_FILENAME

    backend_kwargs: dict = {"random_init": args.random_init}
    if args.model == "ciagan":
        backend_kwargs.update(
            num_classes=args.num_classes, img_size=img_size,
            dlib_predictor=dlib_predictor, portrait_scale=args.portrait_scale,
            refine_mask=args.refine_mask,
            segmentation_weights=segmentation_weights if args.refine_mask else None,
        )
    elif args.model == "ganonymization":
        backend_kwargs.update(
            img_size=img_size, segmentation_weights=segmentation_weights,
            align_rotation=args.align_rotation,
        )

    generator = FaceGenerator(model=args.model, weights=weights, ctx_id=args.ctx_id, **backend_kwargs)
    if args.random_init:
        logger.warning("--random-init: output is NOT real anonymization, smoke test only")

    if not args.random_init and args.model in DEFAULT_WEIGHTS_FILENAME and weights is None:
        p.error(f"--weights is required for --model {args.model} unless --random-init is set")

    if not args.random_init and needs_segmentation and segmentation_weights is None:
        p.error(f"--segmentation-weights is required for --model {args.model}"
                + (" with --refine-mask" if args.model == "ciagan" else "") + " unless --random-init is set")

    stats: dict[int, dict] = {}  # track_id -> counts
    jsonl_path = phase1_dir / "detections.jsonl"
    ledger_path = out_dir / "generation.jsonl"
    t0 = time.time()
    num_frames = 0

    with jsonl_path.open() as jf, ledger_path.open("w") as lf:
        for line in jf:
            if args.limit is not None and num_frames >= args.limit:
                break
            frame_rec = Frame.from_json(line)
            num_frames += 1

            ret, video_frame = cap.read()
            if not ret:
                for face in frame_rec.faces:
                    lf.write(json.dumps({
                        "frame_id": frame_rec.frame_id, "track_id": face.track_id,
                        "status": "skipped_no_frame", "output_path": None,
                        "model": args.model, "seed": None, "identity_class": None,
                    }) + "\n")
                continue

            for face in frame_rec.faces:
                identity = identities.get(face.track_id)
                s = stats.setdefault(face.track_id, {
                    "seed": identity.seed if identity else None,
                    "num_frames_generated": 0, "num_frames_passthrough": 0,
                    "num_frames_skipped_no_identity": 0,
                })

                if identity is None:
                    # track_id present in detections.jsonl but missing from
                    # tracks.json's Identity list — a data-integrity gap in
                    # the Phase 1 output, not a video-read failure.
                    s["num_frames_skipped_no_identity"] += 1
                    lf.write(json.dumps({
                        "frame_id": frame_rec.frame_id, "track_id": face.track_id,
                        "status": "skipped_no_identity", "output_path": None,
                        "model": args.model, "seed": None, "identity_class": None,
                    }) + "\n")
                    continue

                crop = _context_crop(video_frame, face.box, context_ratio)
                out_track_dir = generated_dir / str(face.track_id)
                out_track_dir.mkdir(exist_ok=True)
                out_path = out_track_dir / f"{frame_rec.frame_id:06d}.png"

                out_img = generator.generate(crop, identity.seed)
                if out_img is None:
                    status = "skipped_no_landmarks"
                    s["num_frames_passthrough"] += 1
                    cv2.imwrite(str(out_path), crop)
                else:
                    status = "ok"
                    s["num_frames_generated"] += 1
                    cv2.imwrite(str(out_path), out_img)

                lf.write(json.dumps({
                    "frame_id": frame_rec.frame_id, "track_id": face.track_id,
                    "status": status, "output_path": str(out_path.relative_to(out_dir)),
                    "model": args.model, "seed": identity.seed,
                    "identity_class": generator.identity_class(identity.seed),
                }) + "\n")

            if num_frames % 100 == 0:
                logger.info(f"frame {num_frames}")

    cap.release()
    dt = time.time() - t0
    fps = num_frames / dt if dt > 0 else 0
    passthrough_total = sum(s["num_frames_passthrough"] for s in stats.values())

    def _jsonable(v):
        return str(v) if isinstance(v, Path) else v

    backend_config = {k: _jsonable(v) for k, v in backend_kwargs.items()
                       if k not in {"random_init", "img_size"}}  # already surfaced at top level

    run_manifest = {
        "phase1_dir": str(phase1_dir), "video": video_path, "model": args.model,
        "weights": str(weights) if weights else None, "random_init": args.random_init,
        "context_ratio": context_ratio, "img_size": img_size,
        "backend_config": backend_config,
        "identities": [
            {"track_id": tid, "seed": s["seed"],
             "identity_class": generator.identity_class(s["seed"]) if s["seed"] is not None else None,
             "num_frames_generated": s["num_frames_generated"],
             "num_frames_passthrough": s["num_frames_passthrough"],
             "num_frames_skipped_no_identity": s["num_frames_skipped_no_identity"]}
            for tid, s in stats.items()
        ],
        "num_frames": num_frames, "fps": round(fps, 2),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))

    logger.info(f"Done: {num_frames} frames, {len(stats)} tracks, {fps:.1f} fps")
    if passthrough_total:
        logger.warning(
            f"{passthrough_total} frame(s) fell back to passthrough (no usable "
            "landmarks) — unmodified original crop written, see generation.jsonl"
        )
    logger.info(f"Output: {ledger_path}, {out_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
