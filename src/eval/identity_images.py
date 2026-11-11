"""Identity images — how close each synthetic identity is to the real person it replaces.

A BLANKET identity image is generated once per track and only its ArcFace
embedding reaches the swap. This scores every cached identity image against
the track's real identity (the same C2 estimate the pipeline pushes away
from): ArcFace cos(identity image, real track identity). With two caches
made for the same tracks (e.g. without and with P1 guidance), the per-track
difference is reported paired. When a cache records P1's per-step cosines
(`p1_guided_cos` in its index.json, written by the BLANKET backend), they are
summarized too; `--guidance-log` recovers them from a run log for caches made
before the backend recorded them (lines `[identity_server] P1 seed S:
guided-step cos [...]`, the last line per seed being the accepted attempt).

Scalars only: no embedding is written (LGPD).

Usage:
    python -m src.eval.identity_images --phase1-dir runs/demo1 --video demo.mov \\
        --cache plain=cache/demo1/blanket --cache p1=cache/demo1/blanket-p1-c0.1 --out identity_images.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Optional

import cv2

from ..pipeline.identity.aggregate import normalize
from ..pipeline.identity.embedder import ArcFaceEmbedder, PoseEstimator
from ..pipeline.identity.prepass import load_identities, run_prepass
from ..pipeline.phase1_detect.detector import FaceDetector

logger = logging.getLogger(__name__)
GUIDED = re.compile(r"P1 seed (\d+): guided-step cos (\[[^\]]*\])")


def guided_from_log(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = GUIDED.search(line)
        if m:
            out[m.group(1)] = json.loads(m.group(2))   # later lines win: the accepted attempt
    return out


def summary(values: list[float], below: Optional[float] = None) -> dict:
    if not values:
        return {"n": 0}
    out = {"n": len(values), "mean": round(statistics.mean(values), 4), "median": round(statistics.median(values), 4)}
    if below is not None:
        out[f"share_below_{below}"] = round(sum(v < below for v in values) / len(values), 4)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase1-dir", type=Path, required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--cache", action="append", required=True, metavar="LABEL=DIR",
                   help="an identity cache (index.json + images); two or more are compared paired by seed")
    p.add_argument("--aggregation", default="quality_mean", help="C2 mode of the real identity (as the pipeline)")
    p.add_argument("--context-ratio", type=float, default=0.8)
    p.add_argument("--det-conf", type=float, default=0.25, help="detector threshold on the identity images")
    p.add_argument("--tau", type=float, default=0.2, help="reference threshold reported as a share below it")
    p.add_argument("--guidance-log", type=Path, help="run log to recover P1 guided-step cosines from")
    p.add_argument("--ctx-id", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    caches = dict(c.split("=", 1) for c in args.cache)
    _, identities = load_identities(args.phase1_dir)
    embedder = ArcFaceEmbedder(ctx_id=args.ctx_id)
    prepass = run_prepass(args.phase1_dir / "detections.jsonl", args.video, identities, args.context_ratio,
                          embedder=embedder, pose_estimator=PoseEstimator(ctx_id=args.ctx_id),
                          mode=args.aggregation, k_candidates=0)
    # in memory only, dropped when the process exits (LGPD)
    real = {str(identities[t].seed): normalize(tr.result.embedding)
            for t, tr in prepass.tracks.items() if tr.result.embedding is not None}
    detector = FaceDetector("scrfd-10gf", conf_threshold=args.det_conf, ctx_id=args.ctx_id)
    log_cos = guided_from_log(args.guidance_log) if args.guidance_log else {}

    per_cache: dict[str, dict[str, float]] = {}
    report: dict = {"aggregation": args.aggregation, "tracks_with_identity": len(real), "caches": {}}
    for label, directory in caches.items():
        index = json.loads((Path(directory) / "index.json").read_text())
        entries, guided_cache = index["entries"], "p1" in index.get("settings", {})
        cos, guided, no_face, guided_source = {}, [], 0, {"index": 0, "log": 0}
        for seed, entry in entries.items():
            if entry.get("status") != "ok":
                continue
            g, src = entry.get("p1_guided_cos"), "index"
            if g is None and guided_cache and seed in log_cos:
                g, src = log_cos[seed], "log"
            if g:
                guided.append(g)
                guided_source[src] += 1
            if seed not in real:
                continue
            image = cv2.imread(entry["path"])
            faces = detector.detect(image) if image is not None else []
            if not faces:
                no_face += 1
                continue
            face = max(faces, key=lambda d: d.area)
            cos[seed] = float(normalize(embedder.embed(image, face.landmarks)) @ real[seed])
        per_cache[label] = cos
        report["caches"][label] = {
            "identity_cos": summary(list(cos.values()), args.tau),
            "no_face_in_identity": no_face,
            **({"guided": {"first_step": summary([g[0] for g in guided], args.tau),
                           "last_step": summary([g[-1] for g in guided], args.tau),
                           "source": guided_source, "per_identity": guided}} if guided else {}),
            "per_track": {seed: round(c, 4) for seed, c in cos.items()},   # seed -> cos, for pooling
        }

    labels = list(caches)
    report["paired"] = {}
    for other in labels[1:]:
        a, b = per_cache[labels[0]], per_cache[other]
        seeds = sorted(set(a) & set(b))
        diff = [b[s] - a[s] for s in seeds]
        report["paired"][f"{other}-vs-{labels[0]}"] = {
            **summary(diff), "share_lower": round(sum(d < 0 for d in diff) / len(diff), 4) if diff else None}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info(f"identity images: {json.dumps(report)}")


if __name__ == "__main__":
    main()
