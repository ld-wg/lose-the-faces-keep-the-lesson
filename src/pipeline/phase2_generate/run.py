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
                                               data `compose_video.py`
                                               consumes to build a full
                                               output video, not a debug
                                               artifact).
    generation.jsonl    one ledger line per processed item: status is "ok",
                        "skipped_no_landmarks" (dlib found nothing usable —
                        passthrough of the original crop, unmodified),
                        "skipped_no_frame" (the source video ended before this
                        frame_id), "skipped_no_identity" (track_id missing
                        from tracks.json — a Phase 1 data-integrity gap) or
                        "failsafe" (not generated, hidden by compose_video.py:
                        a Phase 1 post-pass box, `reason` = its source; or a
                        detection below --generate-min-conf, `reason` =
                        "low_conf") or "gated" (generated, but --privacy-gate
                        found it still too close to the real face; the crop
                        is kept for evaluation, compose_video.py hides it).
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

from .context import FaceContext  # noqa: E402
from .cropping import context_crop as _context_crop, crop_box  # noqa: E402,F401 — re-exported
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
from ..identity.aggregate import MODES as AGGREGATION_MODES  # noqa: E402
from ..identity.align import shift as shift_landmarks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
                        "(default: on — real-video A/B test showed it's a strict coverage superset, "
                        "see models/ganonymization/backend.py's module docstring)")
    p.add_argument("--refine-mask", action=argparse.BooleanOptionalAction, default=None,
                   help="ciagan: intersect its composite mask with a real head-segmentation model's "
                        "output, to clean up (not expand) the seam — opt-in (default: off), see "
                        "models/ciagan/NOTICE.md. blanket: composite with a full-head segmentation "
                        "mask instead of BLANKET's own face-only footprint — default-on for blanket "
                        "(not just a seam cleanup there — it's how this project addresses BLANKET's "
                        "own documented weak-identity-suppression limitation, see "
                        "models/blanket/NOTICE.md). Unset (neither flag passed) resolves to "
                        "model-specific defaults below.")
    p.add_argument("--min-detection-confidence", type=float, default=0.3,
                   help="ganonymization: MediaPipe FaceMesh detection threshold (default: 0.3, not "
                        "upstream's 0.5 — real-video sweep found 0.3 improves coverage but going "
                        "lower, e.g. 0.1-0.2, makes it WORSE than the 0.5 baseline, non-monotonically; "
                        "see models/ganonymization/NOTICE.md)")
    p.add_argument("--enhance-detection-input", action=argparse.BooleanOptionalAction, default=True,
                   help="ganonymization: CLAHE-boost the (detection-only) input to the final FaceMesh "
                        "pass, to help detect small/blurry faces — never reaches the generator's own "
                        "input (default: on — real-video test showed a real coverage gain, see "
                        "models/ganonymization/backend.py's _enhance_for_detection())")
    p.add_argument("--blanket-repo", type=str, default=None,
                   help="blanket: path to a checkout of the sibling blanket-anonymizer-bridge repo "
                        "(with .venv-identity/.venv-swap already set up) — required unless --random-init. "
                        "See models/blanket/NOTICE.md")
    p.add_argument("--blanket-identity-python", type=str, default=None,
                   help="blanket: python interpreter for the IdentityGenerator venv "
                        "(default: <blanket-repo>/.venv-identity/bin/python)")
    p.add_argument("--blanket-swap-python", type=str, default=None,
                   help="blanket: python interpreter for the FaceSwapper venv "
                        "(default: <blanket-repo>/.venv-swap/bin/python)")
    p.add_argument("--blanket-server-timeout", type=float, default=900.0,
                   help="blanket: seconds to wait for either external server to become ready/respond "
                        "before raising. Covers the WHOLE per-call round trip, not just startup — a "
                        "cold SDXL+2 ControlNets+refiner pipeline load plus generation plus refinement "
                        "took >180s and triggered a real client-side timeout on a shared/contended GPU "
                        "(verified 2026-09-24, see models/blanket/NOTICE.md); 900s default leaves real "
                        "margin, not just covering the one observed case")
    p.add_argument("--identity-prepass", action=argparse.BooleanOptionalAction, default=False,
                   help="run the track-level identity pre-pass first (src/pipeline/identity/): "
                        "per-track real-identity estimate and best-quality seed crops, passed to the "
                        "backend as context. Off by default so the frozen baselines stay reproducible")
    p.add_argument("--aggregation", choices=AGGREGATION_MODES, default="quality_mean",
                   help="identity pre-pass: how a track's frames are combined (see identity/aggregate.py)")
    p.add_argument("--seed-candidates", type=int, default=5,
                   help="identity pre-pass: best-quality real crops kept per track")
    p.add_argument("--ema-alpha-floor", type=float, default=0.8,
                   help="identity pre-pass, --aggregation ema_adaptive: α_f (0.8 was best on "
                        "video-demo-2; Deep OC-SORT's 0.95 stays anchored to the first frame, see "
                        "src/eval/NOTICE.md)")
    p.add_argument("--blanket-max-identity-attempts", type=int, default=3,
                   help="blanket with --identity-prepass: seed candidates tried per track before "
                        "giving up (each attempt is one SDXL generation)")
    p.add_argument("--blanket-identity-cache", type=str, default=None,
                   help="blanket with --identity-prepass: directory persisting generated identities "
                        "(and failures) across runs, so experiment arms share identical identities")
    p.add_argument("--blanket-swap-face-detector-score", type=float, default=None,
                   help="blanket: FaceFusion's face_detector_score in the swap stage (BLANKET ships 0.5)")
    p.add_argument("--blanket-identity-gpu", type=str, default=None,
                   help="blanket: CUDA_VISIBLE_DEVICES for the SDXL identity server (default: inherit). "
                        "On a shared GPU, put it on a different device than the swap server")
    p.add_argument("--blanket-swap-gpu", type=str, default=None,
                   help="blanket: CUDA_VISIBLE_DEVICES for the FaceFusion swap server (default: inherit)")
    p.add_argument("--blanket-swap-mode", choices=("native", "none", "track"), default="native",
                   help="blanket: identity push in the swap embedding (contribution P2). native = "
                        "BLANKET unchanged (fixed push away from the current frame's real face); none = "
                        "no push; track = push away from the pre-pass's track-level real identity "
                        "(needs --identity-prepass)")
    p.add_argument("--blanket-push-beta", type=float, default=0.35,
                   help="blanket --blanket-swap-mode track: push strength (0.35 = the magnitude of "
                        "BLANKET's own native push)")
    p.add_argument("--ciagan-push", choices=("none", "track"), default="none",
                   help="ciagan: P3, optimize the identity code per frame away from the pre-pass's "
                        "track identity (needs --identity-prepass)")
    p.add_argument("--ciagan-push-steps", type=int, default=10, help="ciagan --ciagan-push: Adam steps per frame")
    p.add_argument("--ciagan-push-lr", type=float, default=0.5, help="ciagan --ciagan-push: Adam step size on the logits")
    p.add_argument("--ciagan-push-tau", type=float, default=0.2,
                   help="ciagan --ciagan-push: hinge threshold on ArcFace cos(output, real); stops below it")
    p.add_argument("--ciagan-push-lambda", type=float, default=1.0,
                   help="ciagan --ciagan-push: weight of ||code − seed one-hot||² (keeps the pseudonym)")
    p.add_argument("--generate-min-conf", type=float, default=0.0,
                   help="detections below this confidence are not generated, only hidden by the "
                        "fail-safe (status failsafe, reason low_conf); 0 generates every detection")
    p.add_argument("--privacy-gate", type=float, default=None, metavar="TAU",
                   help="reject a generated face whose ArcFace cosine to the real face is still >= TAU "
                        "(status gated; the fail-safe hides it). Typically the in-domain ArcFace "
                        "threshold at FAR 1%% from E0. Off by default")
    p.add_argument("--censor-mode", choices=("blur", "mosaic"), default="blur",
                   help="censor: blur or mosaic of the detected box (E2 baseline)")
    p.add_argument("--mosaic-blocks", type=int, default=8,
                   help="censor --censor-mode mosaic: cells across the box's shorter side")
    p.add_argument("--blanket-p1-scale", type=float, default=0.0,
                   help="blanket: P1, ArcFace guidance in the SDXL identity generation away from the track's "
                        "real identity; step size on the predicted clean latent (0 = off)")
    p.add_argument("--blanket-p1-tau", type=float, default=0.2,
                   help="blanket P1: hinge threshold on ArcFace cos(identity, real); stops pushing below it")
    p.add_argument("--blanket-p1-window", type=int, default=3,
                   help="blanket P1: guide the last N denoising steps of the base pass (0 = all; 7 run)")
    p.add_argument("--blanket-detection", choices=("phase1", "upstream"), default="phase1",
                   help="blanket: which face detection the swap uses. phase1 = this project's own "
                        "detection (box and 5 points) for the target face, plus FaceFusion's lenient "
                        "detector for identity images yolo_face rejects; upstream = BLANKET's own "
                        "yolo_face re-detection and IoU filter everywhere (runs before 2026-09-30)")
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

    # --refine-mask has no single global default: ciagan defaults off (pure
    # seam cleanup, opt-in), blanket defaults ON (the only way this project
    # addresses BLANKET's own documented weak-identity-suppression
    # limitation — see models/blanket/NOTICE.md). Explicit --refine-mask/
    # --no-refine-mask always wins regardless of model.
    refine_mask = args.refine_mask
    if refine_mask is None:
        refine_mask = args.model == "blanket"

    weights = Path(args.weights) if args.weights else None
    if weights is None and not args.random_init and args.model in DEFAULT_WEIGHTS_FILENAME:
        weights = CONFIG.weights_dir / DEFAULT_WEIGHTS_FILENAME[args.model]

    dlib_predictor = Path(args.dlib_predictor) if args.dlib_predictor else None
    if dlib_predictor is None and args.model in DEFAULT_DLIB_PREDICTOR_FILENAME:
        dlib_predictor = CONFIG.weights_dir / DEFAULT_DLIB_PREDICTOR_FILENAME[args.model]

    # Shared between ganonymization (required), ciagan (opt-in via
    # --refine-mask) and blanket (default-on, see models/blanket/NOTICE.md) —
    # see models/DEFAULT_SEGMENTATION_WEIGHTS_FILENAME.
    needs_segmentation = (
        args.model == "ganonymization"
        or (args.model == "ciagan" and refine_mask)
        or (args.model == "blanket" and refine_mask)
    )
    segmentation_weights = Path(args.segmentation_weights) if args.segmentation_weights else None
    if segmentation_weights is None and needs_segmentation:
        segmentation_weights = CONFIG.weights_dir / DEFAULT_SEGMENTATION_WEIGHTS_FILENAME

    blanket_repo = Path(args.blanket_repo) if args.blanket_repo else None
    blanket_identity_python = Path(args.blanket_identity_python) if args.blanket_identity_python else None
    blanket_swap_python = Path(args.blanket_swap_python) if args.blanket_swap_python else None

    backend_kwargs: dict = {"random_init": args.random_init}
    if args.model == "ciagan":
        backend_kwargs.update(
            num_classes=args.num_classes, img_size=img_size,
            dlib_predictor=dlib_predictor, portrait_scale=args.portrait_scale,
            refine_mask=refine_mask,
            segmentation_weights=segmentation_weights if refine_mask else None,
            push_mode=args.ciagan_push, push_steps=args.ciagan_push_steps, push_lr=args.ciagan_push_lr,
            push_tau=args.ciagan_push_tau, push_lambda=args.ciagan_push_lambda,
        )
        if args.ciagan_push == "track" and not args.identity_prepass:
            p.error("--ciagan-push track needs --identity-prepass (the push target is the "
                    "pre-pass's track-level real identity)")
    elif args.model == "ganonymization":
        backend_kwargs.update(
            img_size=img_size, segmentation_weights=segmentation_weights,
            align_rotation=args.align_rotation,
            min_detection_confidence=args.min_detection_confidence,
            enhance_detection_input=args.enhance_detection_input,
        )
    elif args.model == "censor":
        backend_kwargs.update(mode=args.censor_mode, mosaic_blocks=args.mosaic_blocks)
    elif args.model == "blanket":
        backend_kwargs.update(
            bridge_repo=blanket_repo,
            identity_python=blanket_identity_python,
            swap_python=blanket_swap_python,
            server_timeout=args.blanket_server_timeout,
            refine_mask=refine_mask,
            segmentation_weights=segmentation_weights if refine_mask else None,
            max_identity_attempts=args.blanket_max_identity_attempts,
            identity_cache_dir=Path(args.blanket_identity_cache) if args.blanket_identity_cache else None,
            swap_face_detector_score=args.blanket_swap_face_detector_score,
            identity_gpu=args.blanket_identity_gpu,
            swap_gpu=args.blanket_swap_gpu,
            swap_mode=args.blanket_swap_mode,
            push_beta=args.blanket_push_beta,
            detection=args.blanket_detection,
            p1_scale=args.blanket_p1_scale, p1_tau=args.blanket_p1_tau, p1_window=args.blanket_p1_window,
        )
        if args.blanket_p1_scale > 0 and not args.identity_prepass:
            p.error("--blanket-p1-scale needs --identity-prepass (the guidance target is the track identity)")
        if args.blanket_swap_mode == "track" and not args.identity_prepass:
            p.error("--blanket-swap-mode track needs --identity-prepass (the push target is the "
                    "pre-pass's track-level real identity)")

    generator = FaceGenerator(model=args.model, weights=weights, ctx_id=args.ctx_id, **backend_kwargs)
    if args.random_init:
        logger.warning("--random-init: output is NOT real anonymization, smoke test only")

    if not args.random_init and args.model in DEFAULT_WEIGHTS_FILENAME and weights is None:
        p.error(f"--weights is required for --model {args.model} unless --random-init is set")

    if not args.random_init and needs_segmentation and segmentation_weights is None:
        suffix = " with --refine-mask" if args.model == "ciagan" else ""
        p.error(f"--segmentation-weights is required for --model {args.model}{suffix} unless --random-init is set")

    if not args.random_init and args.model == "blanket" and blanket_repo is None:
        p.error("--blanket-repo is required for --model blanket unless --random-init is set "
                "(path to a blanket-anonymizer-bridge checkout — see models/blanket/NOTICE.md)")

    stats: dict[int, dict] = {}  # track_id -> counts
    jsonl_path = phase1_dir / "detections.jsonl"
    ledger_path = out_dir / "generation.jsonl"
    t0 = time.time()
    num_frames = 0

    prepass = None
    if args.identity_prepass:
        from ..identity.embedder import ArcFaceEmbedder, PoseEstimator
        from ..identity.prepass import run_prepass

        prepass = run_prepass(
            jsonl_path, video_path, identities, context_ratio,
            embedder=ArcFaceEmbedder(ctx_id=args.ctx_id), pose_estimator=PoseEstimator(ctx_id=args.ctx_id),
            mode=args.aggregation, k_candidates=args.seed_candidates, limit=args.limit,
            ema_alpha_floor=args.ema_alpha_floor,
        )
        # The pre-pass's ONNX sessions are unreferenced now; collect them so
        # their GPU memory is back before the generator loads (shared GPUs).
        import gc
        gc.collect()
    tracks = prepass.tracks if prepass is not None else {}

    gate_embedder = None
    if args.privacy_gate is not None:
        from ..identity.aggregate import normalize
        from ..identity.embedder import ArcFaceEmbedder

        _arcface = ArcFaceEmbedder(ctx_id=args.ctx_id)

        def gate_embedder(image, landmarks):
            # in memory only: the two embeddings are compared and dropped (LGPD)
            return normalize(_arcface.embed(image, landmarks))

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
                    "num_frames_skipped_no_identity": 0, "num_frames_failsafe": 0,
                    "passthrough_reasons": {},
                })

                if not face.detected:
                    # A box from Phase 1's recall post-pass (gap, dilation, too-short
                    # track): no landmarks, maybe no visible face. Not generated;
                    # compose_video.py's fail-safe hides it.
                    s["num_frames_failsafe"] += 1
                    lf.write(json.dumps({
                        "frame_id": frame_rec.frame_id, "track_id": face.track_id,
                        "status": "failsafe", "reason": face.source, "output_path": None,
                        "model": args.model, "seed": identity.seed if identity else None,
                        "identity_class": None,
                    }) + "\n")
                    continue

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

                if face.confidence < args.generate_min_conf:
                    s["num_frames_failsafe"] += 1
                    lf.write(json.dumps({
                        "frame_id": frame_rec.frame_id, "track_id": face.track_id,
                        "status": "failsafe", "reason": "low_conf", "output_path": None,
                        "model": args.model, "seed": identity.seed, "identity_class": None,
                    }) + "\n")
                    continue

                fh, fw = video_frame.shape[:2]
                cx1, cy1, cx2, cy2 = crop_box(fh, fw, face.box, context_ratio)
                crop = video_frame[cy1:cy2, cx1:cx2]
                x1, y1, x2, y2 = face.box
                context = FaceContext(
                    frame_id=frame_rec.frame_id,
                    box_in_crop=(x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
                    landmarks_in_crop=shift_landmarks(face.landmarks, cx1, cy1) if face.landmarks else None,
                    track=tracks.get(face.track_id),
                )
                out_track_dir = generated_dir / str(face.track_id)
                out_track_dir.mkdir(exist_ok=True)
                out_path = out_track_dir / f"{frame_rec.frame_id:06d}.png"

                out_img = generator.generate(crop, identity.seed, context=context)
                reason = None
                if out_img is None:
                    status = "skipped_no_landmarks"
                    reason = generator.last_skip_reason
                    s["num_frames_passthrough"] += 1
                    key = reason or "unspecified"
                    s["passthrough_reasons"][key] = s["passthrough_reasons"].get(key, 0) + 1
                    cv2.imwrite(str(out_path), crop)
                else:
                    status = "ok"
                    if gate_embedder is not None and face.landmarks:
                        cos = float(gate_embedder(video_frame, face.landmarks)
                                    @ gate_embedder(out_img, context.landmarks_in_crop))
                        if cos >= args.privacy_gate:
                            status, reason = "gated", "privacy_gate"
                            s["num_frames_gated"] = s.get("num_frames_gated", 0) + 1
                    if status == "ok":
                        s["num_frames_generated"] += 1
                    cv2.imwrite(str(out_path), out_img)

                record = {
                    "frame_id": frame_rec.frame_id, "track_id": face.track_id,
                    "status": status, "reason": reason, "output_path": str(out_path.relative_to(out_dir)),
                    "model": args.model, "seed": identity.seed,
                    "identity_class": generator.identity_class(identity.seed),
                }
                if generator.last_push is not None:
                    record["push"] = generator.last_push  # scalars only (cosines, steps)
                lf.write(json.dumps(record) + "\n")

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
        "generate_min_conf": args.generate_min_conf, "privacy_gate": args.privacy_gate,
        "backend_config": backend_config,
        "identities": [
            {"track_id": tid, "seed": s["seed"],
             "identity_class": generator.identity_class(s["seed"]) if s["seed"] is not None else None,
             "num_frames_generated": s["num_frames_generated"],
             "num_frames_passthrough": s["num_frames_passthrough"],
             "num_frames_skipped_no_identity": s["num_frames_skipped_no_identity"],
             "num_frames_failsafe": s["num_frames_failsafe"],
             "num_frames_gated": s.get("num_frames_gated", 0),
             "passthrough_reasons": s["passthrough_reasons"]}
            for tid, s in stats.items()
        ],
        "num_frames": num_frames, "fps": round(fps, 2),
        # No embeddings here — TrackIdentity.summary() is scalars and frame ids only (LGPD).
        "identity_prepass": None if prepass is None else {
            "aggregation": args.aggregation, "seed_candidates": args.seed_candidates,
            "ema_alpha_floor": args.ema_alpha_floor, **prepass.summary(),
        },
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
