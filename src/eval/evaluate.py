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
    rank5         — the same, own track among the 5 best
    privacy gain  — 1 − rank1(anonymized) / rank1(real vs real, the same
                    leave-one-out protocol): the share of re-identification removed
    utility       — (--utility, comma list), real vs anonymized:
                    expression  106-point landmark error after similarity
                                alignment, in inter-ocular units
                    pose        head-pose MAE (pitch, yaw, roll), degrees
                    attributes  apparent-age error (years), gender agreement
                    emotion     agreement of the 8-class expression label (HSEmotion)
Privacy metrics come in two versions: over anonymized observations only,
and over all observations with passthrough counted as a leak (the real
face is what the output shows). Without the second, a backend that skips
hard faces would look more private than one that anonymizes all of them.
A third, `final_video`, needs the run's compose.jsonl and scores what the
viewer sees after the fail-safe: a `reused` face is the track's last
generated face, a `filled` one shows no face (never re-identified), an
`exposed` one the real face.

In-domain threshold: genuine pairs are real frames of one track at least
5 frames apart; impostor pairs are real frames of two tracks visible in
the same frame (certainly different people, which avoids counting a
re-entering person as an impostor). Threshold at `--far` (default 1%),
reported with its true-accept rate — that TAR is the sanity check that a
recognizer works on this footage at all.

Noise floor (--noise-floor): the same metrics between consecutive real
frames of a track, as a pseudo-run `_noise_floor`: what "no change" looks
like for each probe on this footage (E0).

`--obs-out` writes one CSV row per (run, frame, track) with every per-face
scalar plus the covariates for stratification (face size, detection
confidence, real |yaw|), so experiments can bootstrap over tracks and pair
arms on the same faces.

Scalars only: no embedding is written (LGPD).

Usage:
    python -m src.eval.evaluate --phase1-dir runs/demo2 \\
        --phase2-dir runs/phase2-ciagan-demo2 --phase2-dir runs/phase2-blanket-demo2 \\
        --out runs/eval/demo2
"""

from __future__ import annotations

import argparse
import csv
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
from ..pipeline.identity.embedder import ArcFaceEmbedder, AttributeEstimator, Landmark106, PoseEstimator
from ..pipeline.phase2_generate.cropping import crop_box
from .heldout import FaceNetEmbedder
from .probes import EmotionClassifier, pose_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RECOGNIZERS = ("arcface", "facenet")
UTILITY = ("expression", "pose", "attributes", "emotion")
UTILITY_METRICS = ("expression", "pose_err", "age_err", "gender_agree", "emotion_agree")
NOISE_FLOOR = "_noise_floor"
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
    compose: Optional[dict[tuple[int, int], str]] = None  # compose.jsonl outcome per (frame, track)


@dataclass
class RealObs:
    track_id: int
    frame_id: int
    emb: dict[str, np.ndarray]
    probes: dict                 # utility probe outputs of the real face (in memory only)
    face_px: float = 0.0         # sqrt(box area)
    det_conf: float = 0.0
    abs_yaw: Optional[float] = None


@dataclass
class AnonObs:
    track_id: int
    frame_id: int
    status: str
    reason: Optional[str]
    anonymized: bool
    redetected: Optional[bool]
    emb: dict[str, np.ndarray]
    utility: dict = None         # UTILITY_METRICS -> value, anonymized observations only


def load_run(phase2_dir: Path) -> Run:
    manifest = json.loads((phase2_dir / "run_manifest.json").read_text())
    ledger = {}
    with (phase2_dir / "generation.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            ledger[(rec["frame_id"], rec["track_id"])] = rec
    compose = None
    if (phase2_dir / "compose.jsonl").is_file():
        with (phase2_dir / "compose.jsonl").open() as f:
            compose = {(r["frame_id"], r["track_id"]): r["outcome"] for r in map(json.loads, f)}
    return Run(name=phase2_dir.name, dir=phase2_dir, model=manifest["model"],
               context_ratio=manifest["context_ratio"], ledger=ledger, compose=compose)


def composition(phase2_dir: Path) -> Optional[dict]:
    """What the final video shows at every face box (compose_video.py's compose.jsonl):
    `generated`, `reused` (the track's last generated face) or `filled` (neutral
    fill), by Face.source. None if the run was never composed into a video."""
    path = phase2_dir / "compose.jsonl"
    if not path.is_file():
        return None
    outcomes: Counter = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            outcomes[r["outcome"]] += 1
            by_source[r["source"]][r["outcome"]] += 1
    n = sum(outcomes.values())
    return {"n_boxes": n, "outcomes": dict(outcomes),
            "shares": {k: _r(v / n) for k, v in outcomes.items()},
            "by_source": {k: dict(v) for k, v in by_source.items()}}


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


class Probes:
    """The utility probes asked for with --utility, run on one face (real or anonymized)."""

    def __init__(self, names: set[str], ctx_id: int, emotion_model: Optional[Path]):
        self.names = names
        self.lmk106 = Landmark106(ctx_id=ctx_id) if "expression" in names else None
        self.pose = PoseEstimator(ctx_id=ctx_id) if names & {"pose"} else None
        self.attrs = AttributeEstimator(ctx_id=ctx_id) if "attributes" in names else None
        self.emotion = EmotionClassifier(emotion_model, ctx_id=ctx_id) if "emotion" in names else None

    def run(self, image: np.ndarray, box) -> dict:
        out = {}
        if self.lmk106:
            out["lmk106"] = self.lmk106.landmarks(image, box)
        if self.pose:
            out["pose"] = self.pose.pose(image, box)
        if self.attrs:
            out["gender"], out["age"] = self.attrs.attributes(image, box)
        if self.emotion:
            out["emotion"] = int(np.argmax(self.emotion.probabilities(image, box)))
        return out


def compare(real: dict, anon: dict, iod: float, offset=(0.0, 0.0)) -> dict:
    """Utility metrics between two probe outputs. `offset` moves the anonymized
    106 landmarks (crop coordinates) into the real frame's."""
    out = {}
    if "lmk106" in real and "lmk106" in anon:
        out["expression"] = _expression_error(real["lmk106"], anon["lmk106"] + np.asarray(offset, np.float32), iod)
    if "pose" in real and "pose" in anon:
        out["pose_err"] = pose_error(real["pose"], anon["pose"])
    if "age" in real and "age" in anon:
        out["age_err"] = float(abs(real["age"] - anon["age"]))
        out["gender_agree"] = float(real["gender"] == anon["gender"])
    if "emotion" in real and "emotion" in anon:
        out["emotion_agree"] = float(real["emotion"] == anon["emotion"])
        # the labels themselves, for a chance-corrected agreement (Cohen's kappa): raw
        # agreement is dominated by the base rate (most classroom faces read as neutral)
        out["emotion_real"], out["emotion_anon"] = int(real["emotion"]), int(anon["emotion"])
    return out


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _mean(values) -> Optional[float]:
    v = [x for x in values if x is not None]
    return _r(np.mean(v)) if v else None


def calibrate(reals: list[RealObs], far: float, rng: np.random.Generator,
              scores: Optional[list[dict]] = None) -> dict:
    """In-domain thresholds per recognizer. When `scores` is given, every
    genuine / impostor similarity is appended to it (scalars only) for plotting."""
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
        if scores is not None:
            scores.extend({"recognizer": rec, "kind": "genuine", "score": round(float(s), 5)} for s in g)
            scores.extend({"recognizer": rec, "kind": "impostor", "score": round(float(s), 5)} for s in imp)
    return out


class Gallery:
    """Per-track sums of the real embeddings, for closed-set re-identification
    with a leave-one-out gallery (the probe's own real frame is removed)."""

    def __init__(self, real_by_key: dict[tuple[int, int], RealObs]):
        self.tids = sorted({r.track_id for r in real_by_key.values()})
        self.tindex = {t: i for i, t in enumerate(self.tids)}
        self.sums = {rec: np.zeros((len(self.tids), 512), dtype=np.float64) for rec in RECOGNIZERS}
        self.counts = Counter()
        for r in real_by_key.values():
            self.counts[r.track_id] += 1
            for rec in RECOGNIZERS:
                self.sums[rec][self.tindex[r.track_id]] += r.emb[rec]

    def rank(self, rec: str, probe: np.ndarray, real: RealObs) -> Optional[int]:
        """1-based rank of the true track for `probe`; None if the track has one frame only."""
        if self.counts[real.track_id] <= 1:
            return None
        gallery = self.sums[rec].copy()
        gallery[self.tindex[real.track_id]] -= real.emb[rec]
        gallery /= np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
        sims = gallery @ probe
        return int((sims > sims[self.tindex[real.track_id]]).sum()) + 1


def real_baseline(real_by_key: dict[tuple[int, int], RealObs], gallery: Gallery) -> dict:
    """Rank-1 / rank-5 of the real faces themselves: the re-identification an
    anonymizer has to remove (Privacy Gain's denominator)."""
    out = {}
    for rec in RECOGNIZERS:
        ranks = [gallery.rank(rec, r.emb[rec], r) for r in real_by_key.values()]
        ranks = [k for k in ranks if k is not None]
        out[rec] = {"rank1_rate": _mean(float(k == 1) for k in ranks),
                    "rank5_rate": _mean(float(k <= 5) for k in ranks), "n": len(ranks)}
    return out


FILLED_RANK = 10 ** 6  # a filled box shows no face: ranked last, never verified


def final_embeddings(obs: list[AnonObs], compose: dict) -> dict[tuple[int, int], tuple[str, Optional[dict]]]:
    """(outcome, embeddings shown) per observation in the composed video."""
    out, last = {}, {}
    for o in sorted(obs, key=lambda x: x.frame_id):
        key = (o.frame_id, o.track_id)
        outcome = compose.get(key, "exposed")
        if outcome == "generated" and o.anonymized:
            out[key] = (outcome, o.emb)
            last[o.track_id] = o.emb
        elif outcome == "reused" and o.track_id in last:
            out[key] = (outcome, last[o.track_id])
        elif outcome in ("reused", "filled"):
            out[key] = ("filled", None)
        else:  # exposed, or generated per compose but unreadable here: the real face
            out[key] = ("exposed", o.emb)
    return out


def run_metrics(obs: list[AnonObs], real_by_key: dict[tuple[int, int], RealObs],
                calibration: dict, rng: np.random.Generator, gallery: Gallery, baseline: dict,
                out_rows: Optional[list[dict]] = None, run_name: str = "",
                compose: Optional[dict] = None) -> dict:
    tids = gallery.tids
    final = final_embeddings(obs, compose) if compose is not None else None

    n_total = len(obs)
    anon = [o for o in obs if o.anonymized]
    out: dict = {
        "coverage": _r(len(anon) / n_total) if n_total else None,
        "n_total": n_total, "n_anonymized": len(anon),
        "status_counts": dict(Counter(o.status for o in obs)),
        "reason_counts": dict(Counter(o.reason for o in obs if o.reason)),
        "redetection_rate": _mean(float(o.redetected) for o in anon if o.redetected is not None),
        "expression_error": _mean(o.utility.get("expression") for o in anon if o.utility),
        "utility": {m: _mean(o.utility.get(m) for o in anon if o.utility) for m in UTILITY_METRICS},
        "recognizers": {},
    }
    obs_rows = {(o.frame_id, o.track_id): {} for o in obs}
    for rec in RECOGNIZERS:
        thr = calibration[rec]["threshold"]
        per_obs = []
        for o in obs:
            real = real_by_key[(o.frame_id, o.track_id)]
            cos = float(o.emb[rec] @ real.emb[rec])
            # the noise floor's probe (the next real frame) is in its own track's gallery: no rank
            rank = None if run_name == NOISE_FLOOR else gallery.rank(rec, o.emb[rec], real)
            per_obs.append((o, cos, rank))
            obs_rows[(o.frame_id, o.track_id)].update({
                f"{rec}_cos": round(cos, 5), f"{rec}_rank": rank,
                f"{rec}_verified": int(cos >= thr) if thr is not None else None,
                f"real_{rec}_rank": gallery.rank(rec, real.emb[rec], real)})
            if final is not None:
                outcome, shown = final[(o.frame_id, o.track_id)]
                fcos = None if shown is None else float(shown[rec] @ real.emb[rec])
                obs_rows[(o.frame_id, o.track_id)].update({
                    "outcome": outcome,
                    f"{rec}_final_rank": (None if gallery.counts[o.track_id] <= 1 else
                                          FILLED_RANK if shown is None else gallery.rank(rec, shown[rec], real)),
                    f"{rec}_final_verified": 0 if fcos is None or thr is None else int(fcos >= thr)})

        def summarize(rows):
            ranks = [k for _, _, k in rows if k is not None]
            r1 = _mean(float(k == 1) for k in ranks)
            base = baseline[rec]["rank1_rate"]
            return {"n": len(rows),
                    "cos_mean": _mean(c for _, c, _ in rows),
                    "verified_rate": _mean(float(c >= thr) for _, c, _ in rows) if thr is not None else None,
                    "rank1_rate": r1,
                    "rank5_rate": _mean(float(k <= 5) for k in ranks),
                    "privacy_gain": _r(1 - r1 / base) if r1 is not None and base else None}

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
        final_view = None
        if final is not None:
            fr = [obs_rows[(o.frame_id, o.track_id)] for o in obs]
            franks = [r[f"{rec}_final_rank"] for r in fr if r[f"{rec}_final_rank"] is not None]
            f1 = _mean(float(k == 1) for k in franks)
            base = baseline[rec]["rank1_rate"]
            final_view = {"n": len(fr), "verified_rate": _mean(r[f"{rec}_final_verified"] for r in fr),
                          "rank1_rate": f1, "rank5_rate": _mean(float(k <= 5) for k in franks),
                          "privacy_gain": _r(1 - f1 / base) if f1 is not None and base else None,
                          "outcomes": dict(Counter(r["outcome"] for r in fr))}
        out["recognizers"][rec] = {
            "anonymized_only": summarize([row for row in per_obs if row[0].anonymized]),
            "all_passthrough_as_leak": summarize(per_obs),
            "final_video": final_view,
            "consistency_mean": _mean(consistency),
            "flicker_consecutive_cos": _mean(flicker),
            "per_track": per_track,
        }
    if out_rows is not None:   # (not `rows`: the per-track summaries above reuse that name)
        for o in obs:
            real = real_by_key[(o.frame_id, o.track_id)]
            out_rows.append({"run": run_name, "frame_id": o.frame_id, "track_id": o.track_id,
                         "status": o.status, "reason": o.reason or "", "anonymized": int(o.anonymized),
                         "redetected": "" if o.redetected is None else int(o.redetected),
                         "face_px": round(real.face_px, 1), "det_conf": round(real.det_conf, 3),
                         "abs_yaw": "" if real.abs_yaw is None else round(real.abs_yaw, 1),
                         **obs_rows[(o.frame_id, o.track_id)],
                         **{m: _r((o.utility or {}).get(m)) for m in UTILITY_METRICS},
                         "emotion_real": (o.utility or {}).get("emotion_real", ""),
                         "emotion_anon": (o.utility or {}).get("emotion_anon", "")})
    return out


def _markdown(report: dict) -> str:
    cal = report["calibration"]
    base = report["real_baseline"]
    lines = [f"# Evaluation — {report['phase1_dir']}", "",
             "In-domain thresholds (FAR {:.0%}): ".format(cal["facenet"]["far"])
             + ", ".join(f"{r} τ={cal[r]['threshold']} TAR={cal[r]['tar']}" for r in RECOGNIZERS),
             "Real faces against their own gallery (what anonymization must remove): "
             + ", ".join(f"{r} rank-1 {base[r]['rank1_rate']} rank-5 {base[r]['rank5_rate']}" for r in RECOGNIZERS), "",
             "| run | model | coverage | FaceNet cos (all) | FaceNet verified (all) | FaceNet rank-1 (all) "
             "| FaceNet rank-5 (all) | FaceNet privacy gain | FaceNet rank-1 (final video) | ArcFace rank-1 (all) | FaceNet consistency "
             "| FaceNet flicker | re-detected | " + " | ".join(UTILITY_METRICS) + " |",
             "|" + "---|" * (13 + len(UTILITY_METRICS))]
    for name, m in report["runs"].items():
        fn, arc = m["recognizers"]["facenet"], m["recognizers"]["arcface"]
        leak = fn["all_passthrough_as_leak"]
        lines.append(" | ".join(str(x) for x in [
            f"| {name}", m["model"], m["coverage"], leak["cos_mean"], leak["verified_rate"], leak["rank1_rate"],
            leak["rank5_rate"], leak["privacy_gain"], (fn["final_video"] or {}).get("rank1_rate"),
            arc["all_passthrough_as_leak"]["rank1_rate"],
            fn["consistency_mean"], fn["flicker_consecutive_cos"], m["redetection_rate"],
            *(m["utility"][k] for k in UTILITY_METRICS),
        ]) + " |")
    lines += ["", "Lower cos / verified / rank-1 = more private. Higher consistency = one stable "
              "pseudonymous identity per track. Lower expression / pose / age error and higher agreement = "
              f"behavior better kept; `{NOISE_FLOOR}` is the same metrics between consecutive real frames."]
    composed = {name: m["composition"] for name, m in report["runs"].items() if m.get("composition")}
    if composed:
        lines += ["", "Final video, every face box (detector and post-pass): what replaced the real face.", "",
                  "| run | boxes | generated | reused | filled |", "|---|---|---|---|---|"]
        for name, c in composed.items():
            sh = c["shares"]
            lines.append(f"| {name} | {c['n_boxes']} | {sh.get('generated', 0)} | {sh.get('reused', 0)} "
                         f"| {sh.get('filled', 0)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Phase 2 runs against their Phase 1 source")
    p.add_argument("--phase1-dir", required=True)
    p.add_argument("--phase2-dir", action="append", default=[], help="repeatable")
    p.add_argument("--calibration-only", action="store_true",
                   help="only the real faces: in-domain thresholds and TAR per recognizer, no Phase 2 run")
    p.add_argument("--scores-out", default=None,
                   help="write every genuine / impostor similarity to this CSV (scalars only, no embeddings)")
    p.add_argument("--obs-out", default=None,
                   help="write one row per (run, frame, track) with every per-face scalar to this CSV")
    p.add_argument("--video", default=None, help="default: tracks.json's video.source")
    p.add_argument("--out", default=None, help="output dir (default: runs/eval/<phase1-dir name>)")
    p.add_argument("--far", type=float, default=0.01)
    p.add_argument("--facenet-weights", default=None)
    p.add_argument("--utility", default="",
                   help=f"comma list of utility probes: {','.join(UTILITY)} (or 'all')")
    p.add_argument("--expression", action="store_true", help="same as adding 'expression' to --utility")
    p.add_argument("--emotion-model", default=None, help="HSEmotion ONNX (default: weights/fer/, downloaded)")
    p.add_argument("--noise-floor", action="store_true",
                   help=f"also score consecutive real frames of each track as the pseudo-run {NOISE_FLOOR}")
    p.add_argument("--ctx-id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="stop after this many frames (testing)")
    args = p.parse_args()

    if not args.phase2_dir and not args.calibration_only:
        p.error("give at least one --phase2-dir, or --calibration-only")
    utility = set(UTILITY) if args.utility == "all" else {u for u in args.utility.split(",") if u}
    if args.expression:
        utility.add("expression")
    if utility - set(UTILITY):
        p.error(f"unknown --utility {sorted(utility - set(UTILITY))}; choose from {UTILITY}")
    phase1_dir = Path(args.phase1_dir)
    manifest = Manifest.load(phase1_dir / "tracks.json")
    identities = {i.track_id for i in manifest.identities}
    runs = [load_run(Path(d)) for d in args.phase2_dir]
    if len({r.name for r in runs}) != len(runs) or NOISE_FLOOR in {r.name for r in runs}:
        p.error("--phase2-dir basenames must be distinct (they name the runs in the report)")
    out_dir = Path(args.out) if args.out else Path("runs") / "eval" / phase1_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    arcface = ArcFaceEmbedder(ctx_id=args.ctx_id)
    facenet = FaceNetEmbedder(Path(args.facenet_weights) if args.facenet_weights else None, ctx_id=args.ctx_id)
    redetector = Redetector(ctx_id=args.ctx_id)
    probes = Probes(utility, args.ctx_id, Path(args.emotion_model) if args.emotion_model else None)
    yaw_estimator = probes.pose or PoseEstimator(ctx_id=args.ctx_id)  # |yaw| is a stratification covariate

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
    if args.noise_floor:
        per_run[NOISE_FLOOR] = []
    iods: dict[tuple[int, int], float] = {}
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
                # Detector faces only: post-pass boxes (fill.py) have no landmarks and are
                # hidden by compose_video.py's fail-safe, counted in `composition` below.
                if face.track_id not in identities or not face.detected or not face.landmarks:
                    continue
                key = (frame_rec.frame_id, face.track_id)
                x1, y1, x2, y2 = face.box
                real_probes = probes.run(frame, face.box)
                pose = real_probes.get("pose") or yaw_estimator.pose(frame, face.box)
                real = RealObs(face.track_id, frame_rec.frame_id, embed(frame, face.landmarks), real_probes,
                               face_px=float(np.sqrt(max(0.0, (x2 - x1) * (y2 - y1)))),
                               det_conf=float(face.confidence), abs_yaw=abs(pose[1]))
                reals.append(real)
                real_by_key[key] = real
                lm = np.asarray(face.landmarks, dtype=np.float32)
                iod = float(np.linalg.norm(lm[0] - lm[1]))
                iods[key] = iod

                prev = real_by_key.get((frame_rec.frame_id - 1, face.track_id))
                if args.noise_floor and prev is not None:
                    # "Anonymized" = this track's real face one frame later, scored against the earlier one.
                    per_run[NOISE_FLOOR].append(AnonObs(
                        face.track_id, prev.frame_id, "real_next", None, anonymized=True, redetected=None,
                        emb=real.emb, utility=compare(prev.probes, real.probes, iods[(prev.frame_id, face.track_id)])))

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
                    expected = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
                    match = redetector.match(anon_img, expected)
                    if match is not None:
                        box_a, lm_a = match
                    else:
                        box_a, lm_a = expected, align.shift(face.landmarks, cx1, cy1)
                    per_run[run.name].append(AnonObs(
                        face.track_id, frame_rec.frame_id, status, reason, anonymized=True,
                        redetected=match is not None, emb=embed(anon_img, lm_a),
                        utility=compare(real_probes, probes.run(anon_img, box_a), iod, offset=(cx1, cy1)),
                    ))
            if num_frames % 50 == 0:
                logger.info(f"evaluate: frame {num_frames}")
    cap.release()

    rng = np.random.default_rng(0)
    scores: Optional[list[dict]] = [] if args.scores_out else None
    calibration = calibrate(reals, args.far, rng, scores)
    if scores is not None:
        with open(args.scores_out, "w") as f:
            f.write("recognizer,kind,score\n")
            f.writelines(f"{s['recognizer']},{s['kind']},{s['score']}\n" for s in scores)
    gallery = Gallery(real_by_key)
    baseline = real_baseline(real_by_key, gallery)
    obs_rows: Optional[list[dict]] = [] if args.obs_out else None
    run_info = {r.name: r for r in runs}
    report = {
        "phase1_dir": str(phase1_dir), "video": video, "num_frames": num_frames,
        "num_real_observations": len(reals),
        "evaluator": {"facenet_margin": facenet.margin, "redetect_det_size": redetector.det_size,
                      "min_genuine_gap_frames": MIN_GENUINE_GAP, "utility": sorted(utility)},
        "calibration": calibration,
        "real_baseline": baseline,
        "runs": {name: {"model": run_info[name].model if name in run_info else "real",
                        "dir": str(run_info[name].dir) if name in run_info else None,
                        "context_ratio": run_info[name].context_ratio if name in run_info else None,
                        **run_metrics(obs, real_by_key, calibration, rng, gallery, baseline, obs_rows, name,
                                      run_info[name].compose if name in run_info else None),
                        "composition": composition(run_info[name].dir) if name in run_info else None}
                 for name, obs in per_run.items()},
        "seconds": round(time.time() - t0, 1),
    }
    if obs_rows is not None:
        keys: list[str] = []
        for r in obs_rows:
            keys += [k for k in r if k not in keys]
        with open(args.obs_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(obs_rows)
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2))
    (out_dir / "eval.md").write_text(_markdown(report), encoding="utf-8")
    logger.info(f"wrote {out_dir / 'eval.json'} and eval.md")
    print(_markdown(report))


if __name__ == "__main__":
    main()
