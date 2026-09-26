"""Privacy/utility evaluation of Phase 2 runs (contribution plan, Step 2).

Reads one Phase 1 run (the real faces) and any number of Phase 2 runs over
it (`generation.jsonl`, `run_manifest.json`, `generated/`), and scores
every face observation with two recognizers side by side: ArcFace
`w600k_r50` (the model later stages guide against) and the held-out
FaceNet evaluator (`heldout.py`, never used for guidance).

Pairing: the real face is embedded from the full source frame with Phase
1's own 5-point landmarks. The anonymized face is the generated crop at
`output_path`, aligned with landmarks re-detected on it by SCRFD (the
detection whose box best overlaps the expected one); if nothing is
re-detected, Phase 1's landmarks shifted into the crop are used and the
observation is flagged — the re-detection rate is itself reported
(post-anonymization detectability).

Metrics, per run and per recognizer:
    coverage      — anonymized / total, plus status and `reason` counts
    cos           — cosine(real, anonymized), per observation
    verified      — share with cos above the in-domain threshold
    rank1         — closed-set re-identification: does the anonymized face
                    match its own track's real gallery (leave-one-out) best
                    among all tracks of the video?
    consistency   — mean pairwise cosine among one track's anonymized faces
    flicker       — mean cosine between consecutive anonymized frames
    expression    — (--expression) 106-point landmark error, real vs
                    anonymized, after similarity alignment, in inter-ocular units
Privacy metrics come in two versions: over anonymized observations only,
and over all observations with passthrough counted as a leak (the real
face is what the output shows). Without the second, a backend that skips
hard faces would look more private than one that anonymizes all of them.

In-domain threshold: genuine pairs are real frames of one track at least
5 frames apart; impostor pairs are real frames of two tracks visible in
the same frame (certainly different people, which avoids counting a
re-entering person as an impostor). Threshold at `--far` (default 1%),
reported with its true-accept rate — that TAR is the sanity check that a
recognizer works on this footage at all.

Scalars only: no embedding is written (LGPD).

Usage:
    python -m src.eval.evaluate --phase1-dir runs/demo2 \\
        --phase2-dir runs/phase2-ciagan-demo2 --phase2-dir runs/phase2-blanket-demo2 \\
        --out runs/eval/demo2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..pipeline.contracts import Frame, Manifest
from ..pipeline.identity import align
from ..pipeline.identity.aggregate import normalize
from ..pipeline.identity.embedder import ArcFaceEmbedder, Landmark106
from ..pipeline.phase2_generate.cropping import crop_box
from .heldout import FaceNetEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RECOGNIZERS = ("arcface", "facenet")
MIN_GENUINE_GAP = 5          # frames between genuine-pair members
MAX_PAIRS_PER_TRACK = 50     # threshold calibration sampling
MAX_CONSISTENCY_PAIRS = 200


@dataclass
class Run:
    name: str
    dir: Path
    model: str
    context_ratio: float
    ledger: dict[tuple[int, int], dict]


@dataclass
class RealObs:
    track_id: int
    frame_id: int
    emb: dict[str, np.ndarray]


@dataclass
class AnonObs:
    track_id: int
    frame_id: int
    status: str
    reason: Optional[str]
    anonymized: bool
    redetected: Optional[bool]
    emb: dict[str, np.ndarray]
    expression: Optional[float] = None


def load_run(phase2_dir: Path) -> Run:
    manifest = json.loads((phase2_dir / "run_manifest.json").read_text())
    ledger = {}
    with (phase2_dir / "generation.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            ledger[(rec["frame_id"], rec["track_id"])] = rec
    return Run(name=phase2_dir.name, dir=phase2_dir, model=manifest["model"],
               context_ratio=manifest["context_ratio"], ledger=ledger)


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Redetector:
    """SCRFD-10GF (buffalo_l's detector, the same one Phase 1 runs) on anonymized crops."""

    def __init__(self, ctx_id: int = 0, det_size: int = 320, min_iou: float = 0.3):
        self.ctx_id, self.det_size, self.min_iou = ctx_id, det_size, min_iou
        self._app = None

    def match(self, image_bgr: np.ndarray, expected_box) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self._app is None:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
            self._app.prepare(ctx_id=self.ctx_id, det_size=(self.det_size, self.det_size))
        best, best_iou = None, self.min_iou
        for face in self._app.get(image_bgr):
            iou = _iou(face.bbox, expected_box)
            if iou >= best_iou and face.kps is not None:
                best, best_iou = face, iou
        return (best.bbox, best.kps) if best is not None else None


def _expression_error(real_106: np.ndarray, anon_106: np.ndarray, iod: float) -> float:
    m = align.fit_similarity(anon_106, real_106)
    mapped = anon_106 @ m[:, :2].T + m[:, 2]
    return float(np.linalg.norm(mapped - real_106, axis=1).mean() / max(iod, 1e-6))


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _mean(values) -> Optional[float]:
    v = [x for x in values if x is not None]
    return _r(np.mean(v)) if v else None


def calibrate(reals: list[RealObs], far: float, rng: np.random.Generator) -> dict:
    by_track: dict[int, list[RealObs]] = defaultdict(list)
    for r in reals:
        by_track[r.track_id].append(r)
    genuine_pairs, impostor_pairs = [], []
    for obs in by_track.values():
        pairs = [(a, b) for i, a in enumerate(obs) for b in obs[i + 1:]
                 if abs(a.frame_id - b.frame_id) >= MIN_GENUINE_GAP]
        if len(pairs) > MAX_PAIRS_PER_TRACK:
            pairs = [pairs[k] for k in rng.choice(len(pairs), MAX_PAIRS_PER_TRACK, replace=False)]
        genuine_pairs.extend(pairs)
    frames = {t: {o.frame_id for o in obs} for t, obs in by_track.items()}
    tids = sorted(by_track)
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            if not frames[a] & frames[b]:
                continue
            for _ in range(MAX_PAIRS_PER_TRACK):
                impostor_pairs.append((by_track[a][rng.integers(len(by_track[a]))],
                                       by_track[b][rng.integers(len(by_track[b]))]))
    out = {}
    for rec in RECOGNIZERS:
        g = np.array([float(a.emb[rec] @ b.emb[rec]) for a, b in genuine_pairs])
        imp = np.array([float(a.emb[rec] @ b.emb[rec]) for a, b in impostor_pairs])
        thr = float(np.quantile(imp, 1.0 - far)) if imp.size else float("nan")
        out[rec] = {"threshold": _r(thr), "far": far, "tar": _r((g >= thr).mean()) if g.size else None,
                    "genuine_mean": _r(g.mean()) if g.size else None,
                    "impostor_mean": _r(imp.mean()) if imp.size else None,
                    "n_genuine": int(g.size), "n_impostor": int(imp.size)}
    return out


def run_metrics(obs: list[AnonObs], real_by_key: dict[tuple[int, int], RealObs],
                calibration: dict, rng: np.random.Generator) -> dict:
    tids = sorted({r.track_id for r in real_by_key.values()})
    tindex = {t: i for i, t in enumerate(tids)}
    sums = {rec: np.zeros((len(tids), 512), dtype=np.float64) for rec in RECOGNIZERS}
    counts = Counter()
    for r in real_by_key.values():
        counts[r.track_id] += 1
        for rec in RECOGNIZERS:
            sums[rec][tindex[r.track_id]] += r.emb[rec]

    n_total = len(obs)
    anon = [o for o in obs if o.anonymized]
    out: dict = {
        "coverage": _r(len(anon) / n_total) if n_total else None,
        "n_total": n_total, "n_anonymized": len(anon),
        "status_counts": dict(Counter(o.status for o in obs)),
        "reason_counts": dict(Counter(o.reason for o in obs if o.reason)),
        "redetection_rate": _mean(float(o.redetected) for o in anon),
        "expression_error": _mean(o.expression for o in anon),
        "recognizers": {},
    }
    for rec in RECOGNIZERS:
        thr = calibration[rec]["threshold"]
        per_obs = []
        for o in obs:
            real = real_by_key[(o.frame_id, o.track_id)].emb[rec]
            cos = float(o.emb[rec] @ real)
            rank1 = None
            if counts[o.track_id] > 1:  # leave-one-out gallery needs another frame of the track
                gallery = sums[rec].copy()
                gallery[tindex[o.track_id]] -= real
                gallery /= np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
                rank1 = float(np.argmax(gallery @ o.emb[rec]) == tindex[o.track_id])
            per_obs.append((o, cos, rank1))

        def summarize(rows):
            return {"n": len(rows),
                    "cos_mean": _mean(c for _, c, _ in rows),
                    "verified_rate": _mean(float(c >= thr) for _, c, _ in rows) if thr is not None else None,
                    "rank1_rate": _mean(r for _, _, r in rows)}

        by_track = defaultdict(list)
        for o in anon:
            by_track[o.track_id].append(o)
        consistency = []
        flicker = []
        for track_obs in by_track.values():
            track_obs.sort(key=lambda x: x.frame_id)
            pairs = [(a, b) for i, a in enumerate(track_obs) for b in track_obs[i + 1:]]
            if len(pairs) > MAX_CONSISTENCY_PAIRS:
                pairs = [pairs[k] for k in rng.choice(len(pairs), MAX_CONSISTENCY_PAIRS, replace=False)]
            if pairs:
                consistency.append(np.mean([float(a.emb[rec] @ b.emb[rec]) for a, b in pairs]))
            flicker.extend(float(a.emb[rec] @ b.emb[rec]) for a, b in zip(track_obs, track_obs[1:])
                           if b.frame_id - a.frame_id == 1)

        per_track = {}
        for t in tids:
            rows = [row for row in per_obs if row[0].track_id == t]
            if rows:
                per_track[str(t)] = {"n": len(rows), "n_anonymized": sum(r[0].anonymized for r in rows),
                                     **{k: v for k, v in summarize(rows).items() if k != "n"}}
        out["recognizers"][rec] = {
            "anonymized_only": summarize([row for row in per_obs if row[0].anonymized]),
            "all_passthrough_as_leak": summarize(per_obs),
            "consistency_mean": _mean(consistency),
            "flicker_consecutive_cos": _mean(flicker),
            "per_track": per_track,
        }
    return out


def _markdown(report: dict) -> str:
    cal = report["calibration"]
    lines = [f"# Evaluation — {report['phase1_dir']}", "",
             "In-domain thresholds (FAR {:.0%}): ".format(cal["facenet"]["far"])
             + ", ".join(f"{r} τ={cal[r]['threshold']} TAR={cal[r]['tar']}" for r in RECOGNIZERS), "",
             "| run | model | coverage | FaceNet cos (anon) | FaceNet cos (all) | FaceNet verified (all) "
             "| FaceNet rank-1 (all) | ArcFace cos (all) | ArcFace rank-1 (all) | FaceNet consistency "
             "| FaceNet flicker | re-detected | expression err |",
             "|" + "---|" * 13]
    for name, m in report["runs"].items():
        fn, arc = m["recognizers"]["facenet"], m["recognizers"]["arcface"]
        lines.append(" | ".join(str(x) for x in [
            f"| {name}", m["model"], m["coverage"], fn["anonymized_only"]["cos_mean"],
            fn["all_passthrough_as_leak"]["cos_mean"], fn["all_passthrough_as_leak"]["verified_rate"],
            fn["all_passthrough_as_leak"]["rank1_rate"], arc["all_passthrough_as_leak"]["cos_mean"],
            arc["all_passthrough_as_leak"]["rank1_rate"], fn["consistency_mean"],
            fn["flicker_consecutive_cos"], m["redetection_rate"], m["expression_error"],
        ]) + " |")
    lines += ["", "Lower cos / verified / rank-1 = more private. Higher consistency = one stable "
              "pseudonymous identity per track. Lower expression error = expression better kept."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Phase 2 runs against their Phase 1 source")
    p.add_argument("--phase1-dir", required=True)
    p.add_argument("--phase2-dir", action="append", required=True, help="repeatable")
    p.add_argument("--video", default=None, help="default: tracks.json's video.source")
    p.add_argument("--out", default=None, help="output dir (default: runs/eval/<phase1-dir name>)")
    p.add_argument("--far", type=float, default=0.01)
    p.add_argument("--facenet-weights", default=None)
    p.add_argument("--expression", action="store_true", help="also compute the 106-landmark expression proxy")
    p.add_argument("--ctx-id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="stop after this many frames (testing)")
    args = p.parse_args()

    phase1_dir = Path(args.phase1_dir)
    manifest = Manifest.load(phase1_dir / "tracks.json")
    identities = {i.track_id for i in manifest.identities}
    runs = [load_run(Path(d)) for d in args.phase2_dir]
    if len({r.name for r in runs}) != len(runs):
        p.error("--phase2-dir basenames must be distinct (they name the runs in the report)")
    out_dir = Path(args.out) if args.out else Path("runs") / "eval" / phase1_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    arcface = ArcFaceEmbedder(ctx_id=args.ctx_id)
    facenet = FaceNetEmbedder(Path(args.facenet_weights) if args.facenet_weights else None, ctx_id=args.ctx_id)
    redetector = Redetector(ctx_id=args.ctx_id)
    lmk106 = Landmark106(ctx_id=args.ctx_id) if args.expression else None

    def embed(image, landmarks) -> dict[str, np.ndarray]:
        return {"arcface": normalize(arcface.embed(image, landmarks)), "facenet": facenet.embed(image, landmarks)}

    video = args.video or manifest.video.source
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        p.error(f"could not open source video: {video}")

    t0 = time.time()
    reals: list[RealObs] = []
    real_by_key: dict[tuple[int, int], RealObs] = {}
    per_run: dict[str, list[AnonObs]] = {r.name: [] for r in runs}
    num_frames = 0
    with (phase1_dir / "detections.jsonl").open() as jf:
        for line in jf:
            if args.limit is not None and num_frames >= args.limit:
                break
            frame_rec = Frame.from_json(line)
            num_frames += 1
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            for face in frame_rec.faces:
                if face.track_id not in identities or not face.landmarks:
                    continue
                key = (frame_rec.frame_id, face.track_id)
                real = RealObs(face.track_id, frame_rec.frame_id, embed(frame, face.landmarks))
                reals.append(real)
                real_by_key[key] = real
                lm = np.asarray(face.landmarks, dtype=np.float32)
                iod = float(np.linalg.norm(lm[0] - lm[1]))
                real_106 = lmk106.landmarks(frame, face.box) if lmk106 else None

                for run in runs:
                    rec = run.ledger.get(key)
                    if rec is None:
                        continue  # this run never reached the observation (e.g. --limit)
                    status, reason = rec["status"], rec.get("reason")
                    anon_img = None
                    if status == "ok" and rec.get("output_path"):
                        anon_img = cv2.imread(str(run.dir / rec["output_path"]))
                    cx1, cy1, cx2, cy2 = crop_box(h, w, face.box, run.context_ratio)
                    if anon_img is None or anon_img.shape[:2] != (cy2 - cy1, cx2 - cx1):
                        if status == "ok":
                            status = "ok_unreadable"
                        per_run[run.name].append(AnonObs(face.track_id, frame_rec.frame_id, status, reason,
                                                         anonymized=False, redetected=None, emb=real.emb))
                        continue
                    x1, y1, x2, y2 = face.box
                    expected = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
                    match = redetector.match(anon_img, expected)
                    if match is not None:
                        box_a, lm_a = match
                    else:
                        box_a, lm_a = expected, align.shift(face.landmarks, cx1, cy1)
                    expression = None
                    if lmk106 is not None:
                        anon_106 = lmk106.landmarks(anon_img, box_a) + np.array([cx1, cy1], dtype=np.float32)
                        expression = _expression_error(real_106, anon_106, iod)
                    per_run[run.name].append(AnonObs(
                        face.track_id, frame_rec.frame_id, status, reason, anonymized=True,
                        redetected=match is not None, emb=embed(anon_img, lm_a), expression=expression,
                    ))
            if num_frames % 50 == 0:
                logger.info(f"evaluate: frame {num_frames}")
    cap.release()

    rng = np.random.default_rng(0)
    calibration = calibrate(reals, args.far, rng)
    report = {
        "phase1_dir": str(phase1_dir), "video": video, "num_frames": num_frames,
        "num_real_observations": len(reals),
        "evaluator": {"facenet_margin": facenet.margin, "redetect_det_size": redetector.det_size,
                      "min_genuine_gap_frames": MIN_GENUINE_GAP},
        "calibration": calibration,
        "runs": {r.name: {"model": r.model, "dir": str(r.dir), "context_ratio": r.context_ratio,
                          **run_metrics(per_run[r.name], real_by_key, calibration, rng)}
                 for r in runs},
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2))
    (out_dir / "eval.md").write_text(_markdown(report))
    logger.info(f"wrote {out_dir / 'eval.json'} and eval.md")
    print(_markdown(report))


if __name__ == "__main__":
    main()
