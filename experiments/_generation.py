"""Shared steps of the generation experiments (E2, E3): one arm = one Phase 2
configuration. Detect once per video, anonymize every arm, compose its final
video (fail-closed), evaluate all arms in one call so they share the real
embeddings, calibration and faces, then summarize with track-bootstrap
intervals and paired comparisons.

An arm is a dict from config.toml. `name` and `backend` are required; every
other key maps to a `src.pipeline.phase2_generate.run` flag (ARM_FLAGS).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from _common import Experiment, Video, kappa_ci, load_obs, obs_ci, paired_obs, privacy_gain_ci, strata

BRIDGE = Path(os.environ.get("BLANKET_REPO", Path.home() / "projects" / "blanket-anonymizer-bridge"))

ARM_FLAGS = {
    "mode": "--censor-mode", "mosaic_blocks": "--mosaic-blocks",
    "identity_prepass": "--identity-prepass", "aggregation": "--aggregation",
    "seed_candidates": "--seed-candidates", "ema_alpha_floor": "--ema-alpha-floor",
    "max_identity_attempts": "--blanket-max-identity-attempts", "swap_mode": "--blanket-swap-mode",
    "push_beta": "--blanket-push-beta", "detection": "--blanket-detection",
    "ciagan_push": "--ciagan-push", "ciagan_push_steps": "--ciagan-push-steps",
    "ciagan_push_tau": "--ciagan-push-tau", "ciagan_push_lambda": "--ciagan-push-lambda",
    "generate_min_conf": "--generate-min-conf", "privacy_gate": "--privacy-gate",
    "p1_scale": "--blanket-p1-scale", "p1_tau": "--blanket-p1-tau", "p1_window": "--blanket-p1-window",
}
NOT_FLAGS = {"name", "backend", "label", "identity_cache"}

# Labels used in the paper's tables (short, so seven columns fit 160 mm).
METRICS = {
    "facenet_final.rank1": "FaceNet R1", "facenet_final.rank5": "FaceNet R5",
    "arcface_final.rank1": "ArcFace R1", "facenet_final.verified": "Verified",
    "expression": "Expr.", "pose_err": "Pose (°)", "emotion_agree": "Emotion", "age_err": "Age (y)",
    "gender_agree": "Gender",
}


def ctx_args() -> list[str]:
    """Device of the generation step's own process. On a shared GPU, BLANKET's
    main process runs on CPU (CTX_ID=-1) next to its two GPU servers."""
    return ["--ctx-id", os.environ["CTX_ID"]] if "CTX_ID" in os.environ else []


def eval_ctx_args() -> list[str]:
    """Device of the evaluator, which runs after the generation servers exit:
    the GPU by default (EVAL_CTX_ID, 0); on CPU a BLANKET-sized video takes hours."""
    return ["--ctx-id", os.environ.get("EVAL_CTX_ID", "0")]


def detect(exp: Experiment, video: Video, det: dict, out: Path) -> Path:
    exp.module(f"detect {video.name}", "src.pipeline.phase1_detect.run", "--input", video.path, "--out", out,
               "--model", det["model"], "--conf", det["conf"], "--det-size", det["det_size"])
    return out


def arm_flags(arm: dict, cache: Optional[Path]) -> list[Any]:
    unknown = set(arm) - set(ARM_FLAGS) - NOT_FLAGS
    if unknown:
        raise ValueError(f"arm {arm['name']}: unknown keys {sorted(unknown)} (see ARM_FLAGS)")
    flags: list[Any] = []
    for key, value in arm.items():
        if key in NOT_FLAGS:
            continue
        if isinstance(value, bool):
            flags += [ARM_FLAGS[key]] if value else []
        else:
            flags += [ARM_FLAGS[key], value]
    if arm["backend"] == "blanket":
        # a cold SDXL + 2 ControlNets + refiner load plus the first generation passed 900 s on
        # serra1's shared machine (2026-10-05): the RPC limit covers the whole round trip
        flags += ["--blanket-repo", BRIDGE, "--blanket-server-timeout", os.environ.get("BLANKET_TIMEOUT", "2400")]
        for env, flag in (("IDENTITY_GPU", "--blanket-identity-gpu"), ("SWAP_GPU", "--blanket-swap-gpu")):
            if env in os.environ:
                flags += [flag, os.environ[env]]
        if cache is not None and arm.get("identity_prepass"):
            flags += ["--blanket-identity-cache", cache]
    return flags


def anonymize(exp: Experiment, video: Video, det_dir: Path, arm: dict, out: Path,
              cache: Optional[Path] = None) -> Path:
    """Phase 2 for one arm, then its final video (compose.jsonl next to it)."""
    exp.module(f"anonymize {video.name} {arm['name']}", "src.pipeline.phase2_generate.run",
               "--phase1-dir", det_dir, "--video", video.path, "--model", arm["backend"], "--out", out,
               *arm_flags(arm, Path(arm["identity_cache"]) if "identity_cache" in arm else cache), *ctx_args())
    exp.module(f"compose {video.name} {arm['name']}", "src.pipeline.phase2_generate.compose_video",
               "--phase1-dir", det_dir, "--phase2-dir", out, "--video", video.path, "--out", out / "output.mp4")
    return out


def evaluate(exp: Experiment, video: Video, det_dir: Path, arm_dirs: Sequence[Path], ev: dict, out: Path,
             noise_floor: bool = True) -> tuple[dict, list[dict]]:
    """One evaluate call for every arm of a video. Returns (eval.json, per-face rows with video=)."""
    utility = [u for u in ev["utility"] if u != "gaze"]   # gaze: a stated limitation (DECISIONS.md 10)
    exp.module(f"evaluate {video.name}", "src.eval.evaluate", "--phase1-dir", det_dir, "--video", video.path,
               *[a for d in arm_dirs for a in ("--phase2-dir", d)], "--far", ev["far"],
               "--utility", ",".join(utility), *(["--noise-floor"] if noise_floor else []),
               "--obs-out", out / "obs.csv", "--out", out, *eval_ctx_args())
    report = json.loads((out / "eval.json").read_text())
    return report, load_obs(out / "obs.csv", video=video.name)


def by_run(rows: Sequence[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["run"], []).append(r)
    return out


def cell(rows: Sequence[dict], metric: str, ev: dict):
    if metric == "emotion_kappa":
        est, half, _ = kappa_ci(rows, n=ev["bootstrap"], seed=ev["seed"])
    else:
        est, half, _ = obs_ci(rows, metric, n=ev["bootstrap"], seed=ev["seed"])
    return None if est is None else (est, half)


def arm_row(label: str, rows: Sequence[dict], metrics: Sequence[str], ev: dict) -> list:
    return [label, *(cell(rows, m, ev) for m in metrics)]


def privacy_gain_cell(rows: Sequence[dict], ev: dict, rec: str = "facenet_final"):
    """Privacy Gain of what the viewer sees: 1 − rank1(final video) / rank1(real)."""
    # privacy_gain_ci compares <rec>.rank1 with real_<base>.rank1; the final view shares the real baseline
    base = rec.replace("_final", "")
    shifted = [{**r, f"real_{rec}_rank": r.get(f"real_{base}_rank")} for r in rows]
    est, half = privacy_gain_ci(shifted, rec, n=ev["bootstrap"], seed=ev["seed"])
    return None if est is None else (est, half)


def metric_keys(exp: Experiment, prefix: str, rows: Sequence[dict], metrics: Sequence[str], ev: dict) -> None:
    """`\\result{<prefix>/<metric>}` and `.../<metric>.ci` for every metric of one arm."""
    for m in metrics:
        c = cell(rows, m, ev)
        if c is not None and c[0] == c[0]:   # skip NaN (e.g. kappa with one class only)
            exp.metric(f"{prefix}/{m}", round(c[0], 3))
            exp.metric(f"{prefix}/{m}.ci", round(c[1], 3))


def paired_rows(rows_by_arm: dict[str, list[dict]], reference: str, metric: str,
                ev: dict) -> list[list]:
    """Each arm against the reference on the same faces: Δ ± CI, whether the interval
    excludes zero, the track-level sign-flip p, and the number of faces."""
    ref = rows_by_arm[reference]
    out = []
    for name, rows in rows_by_arm.items():
        if name == reference:
            continue
        d = paired_obs(rows, ref, metric, n=ev["bootstrap"], seed=ev["seed"])
        out.append([name, None if d["diff"] is None else (d["diff"], d["half"]),
                    "yes" if d["excludes_zero"] else "no", d["p"], d["n"]])
    return out


def strata_rows(rows_by_arm: dict[str, list[dict]], key: str, edges: Sequence[float], metric: str,
                ev: dict, labels: dict[str, str]) -> tuple[list[str], list[list]]:
    bins = [lab for lab, _ in strata(next(iter(rows_by_arm.values())), key, edges)]
    out = []
    for name, rows in rows_by_arm.items():
        out.append([labels.get(name, name), *(cell(sub, metric, ev) for _, sub in strata(rows, key, edges))])
    return bins, out
