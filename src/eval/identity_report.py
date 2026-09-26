"""Identity report — checks the identity pre-pass on real footage before
anything is built on it (contribution plan, Step 1).

Answers, with data from one Phase 1 run:
  1. pose sanity   — yaw/pitch distributions (confirms degrees, plausible range)
  2. spread        — per track, how well each frame agrees with the rest
                     (cosine to the leave-one-out mean of the track)
  3. quality       — Spearman correlation between that agreement and each
                     quality term, the combined score, and the raw embedding
                     norm (plan D3: which terms earn their place)
  4. outliers      — outlier frame ranges and suspected tracker ID switches
  5. cross-track   — similarity of co-occurring tracks (certainly different
                     people: a calibration floor) and of disjoint tracks that
                     look alike (re-link candidates, reported only)
  6. modes         — which aggregation mode best represents the person in
                     frames it did not see: each track split into even/odd
                     observations, estimate from one half, mean cosine to
                     the other; plus the same as a function of how many
                     frames the estimate has seen (plan D1/D2 evidence)

Scalars only — no embedding is written (LGPD).

Usage:
    python -m src.eval.identity_report --phase1-dir runs/demo2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np

from ..pipeline.identity.aggregate import MODES, Observation, aggregate, normalize
from ..pipeline.identity.embedder import ArcFaceEmbedder, PoseEstimator
from ..pipeline.identity.prepass import load_identities, run_prepass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONVERGENCE_NS = (1, 2, 3, 5, 10, 20, 40)


def _r(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, nd)


def _stats(values) -> dict:
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": _r(v.mean()), "min": _r(v.min()),
            "p5": _r(np.percentile(v, 5)), "median": _r(np.median(v)),
            "p95": _r(np.percentile(v, 95)), "max": _r(v.max())}


def _spearman(a, b):
    from scipy.stats import spearmanr

    if len(a) < 3:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant input -> nan, reported as null
        rho = spearmanr(a, b).statistic
    return _r(rho)


def _loo_agreement(obs: list[Observation]) -> np.ndarray:
    units = np.stack([o.unit for o in obs])
    total = units.sum(axis=0)
    return np.array([float(u @ normalize(total - u)) for u in units])


def _features(o: Observation) -> dict:
    return {"det": o.terms.det, "size": o.terms.size, "pose": o.terms.pose, "quality": o.quality,
            "embedding_norm": o.embedding_norm, "confidence": o.confidence,
            "abs_yaw": abs(o.yaw) if o.yaw is not None else np.nan}


def main() -> None:
    p = argparse.ArgumentParser(description="Validate the identity pre-pass on a Phase 1 run")
    p.add_argument("--phase1-dir", required=True)
    p.add_argument("--video", default=None, help="default: tracks.json's video.source")
    p.add_argument("--out", default=None, help="default: <phase1-dir>/identity_report.json")
    p.add_argument("--ctx-id", type=int, default=0)
    p.add_argument("--outlier-cos", type=float, default=0.2)
    p.add_argument("--link-cos", type=float, default=0.3,
                   help="report disjoint-in-time track pairs above this cosine as re-link candidates")
    p.add_argument("--no-pose", action="store_true", help="skip the 3D-68 pose model (pose term = 1)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    phase1_dir = Path(args.phase1_dir)
    video_source, identities = load_identities(phase1_dir)
    video = args.video or video_source
    out_path = Path(args.out) if args.out else phase1_dir / "identity_report.json"

    t0 = time.time()
    res = run_prepass(
        phase1_dir / "detections.jsonl", video, identities, context_ratio=0.8,
        embedder=ArcFaceEmbedder(ctx_id=args.ctx_id),
        pose_estimator=None if args.no_pose else PoseEstimator(ctx_id=args.ctx_id),
        k_candidates=0, keep_observations=True, limit=args.limit, outlier_cos=args.outlier_cos,
    )
    obs_by_track = {t: sorted(o, key=lambda x: x.frame_id) for t, o in res.observations.items()}
    all_obs = [o for obs in obs_by_track.values() for o in obs]
    if not all_obs:
        p.error("the pre-pass produced no observations (no tracks with landmarks?)")
    report: dict ={"phase1_dir": str(phase1_dir), "video": video, "prepass": res.summary(),
                    "prepass_seconds": round(time.time() - t0, 1)}

    # 1. pose sanity
    report["pose"] = {"yaw_deg": _stats(o.yaw for o in all_obs if o.yaw is not None)}

    # 2. spread + 3. quality correlations
    spread, pooled_agree, pooled_feats, within = {}, [], [], {k: [] for k in _features(all_obs[0])}
    for tid, obs in obs_by_track.items():
        if len(obs) < 3:
            continue
        agree = _loo_agreement(obs)
        spread[str(tid)] = {"n": len(obs), **{k: v for k, v in _stats(agree).items() if k != "n"}}
        if len(obs) >= 5:
            feats = [_features(o) for o in obs]
            pooled_agree.extend(agree)
            pooled_feats.extend(feats)
            if len(obs) >= 10:
                for k in within:
                    rho = _spearman(agree, [f[k] for f in feats])
                    if rho is not None:
                        within[k].append(rho)
    report["spread"] = {"per_track": spread,
                        "all_frames": _stats(a for s in obs_by_track.values() if len(s) >= 3
                                             for a in _loo_agreement(s))}
    report["quality_vs_agreement"] = {
        "pooled_spearman": {k: _spearman(pooled_agree, [f[k] for f in pooled_feats]) for k in within},
        "within_track_mean_spearman": {k: _r(np.mean(v)) if v else None for k, v in within.items()},
        "within_track_n": {k: len(v) for k, v in within.items()},
        "note": "positive = the feature predicts agreement with the rest of the track",
    }

    # 4. outliers
    report["outliers"] = {str(t.track_id): {"num_observations": t.num_observations,
                                            "num_outliers": len(t.result.outlier_frames),
                                            "outlier_frame_ranges": t.summary()["outlier_frame_ranges"],
                                            "id_switch_suspected": t.result.id_switch_suspected}
                          for t in res.tracks.values() if t.result.outlier_frames}

    # 5. cross-track
    centers = {tid: aggregate(obs, "quality_mean", outlier_cos=args.outlier_cos).embedding
               for tid, obs in obs_by_track.items()}
    frame_sets = {tid: {o.frame_id for o in obs} for tid, obs in obs_by_track.items()}
    tids = sorted(centers)
    cooccur, relink = [], []
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            c = float(centers[a] @ centers[b])
            if frame_sets[a] & frame_sets[b]:
                cooccur.append(c)
            elif c >= args.link_cos:
                relink.append({"tracks": [a, b], "cos": _r(c),
                               "spans": [[min(frame_sets[a]), max(frame_sets[a])],
                                         [min(frame_sets[b]), max(frame_sets[b])]]})
    report["cross_track"] = {"cooccurring_pairs_cos": _stats(cooccur),
                             "relink_candidates": sorted(relink, key=lambda r: -r["cos"])}

    # 6. modes on held-out halves, and convergence
    per_mode = {m: [] for m in MODES}
    conv = {m: {n: [] for n in CONVERGENCE_NS} for m in MODES}
    for obs in obs_by_track.values():
        if len(obs) < 6:
            continue
        half_a, half_b = obs[0::2], obs[1::2]
        units_b = np.stack([o.unit for o in half_b])
        for m in MODES:
            est = aggregate(half_a, m, outlier_cos=args.outlier_cos).embedding
            per_mode[m].append(float((units_b @ est).mean()))
            for n in CONVERGENCE_NS:
                if len(half_a) >= n:
                    est_n = aggregate(half_a[:n], m, outlier_cos=args.outlier_cos).embedding
                    conv[m][n].append(float((units_b @ est_n).mean()))
    report["modes_heldout_half"] = {
        "per_mode_mean_cos": {m: _r(np.mean(v)) if v else None for m, v in per_mode.items()},
        "num_tracks": len(per_mode["mean"]),
        "convergence_mean_cos": {m: {str(n): _r(np.mean(v)) if v else None for n, v in d.items()}
                                 for m, d in conv.items()},
        "convergence_num_tracks": {str(n): len(conv["mean"][n]) for n in CONVERGENCE_NS},
    }

    out_path.write_text(json.dumps(report, indent=2))
    logger.info(f"wrote {out_path}")
    logger.info(f"held-out-half cosine by mode: {report['modes_heldout_half']['per_mode_mean_cos']}")
    logger.info(f"quality vs agreement (pooled): {report['quality_vs_agreement']['pooled_spearman']}")


if __name__ == "__main__":
    main()
