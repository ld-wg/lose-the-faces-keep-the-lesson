"""E2 — generators in their original form. Design and outputs: README.md in this folder.

    uv run experiments/e2_generation/run.py [--video NAME ...] [--fresh]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import SERIES, Experiment, kappa_ci, obs_ci  # noqa: E402
from _generation import (anonymize, arm_row, by_run, cell, detect, evaluate,  # noqa: E402
                         metric_keys, privacy_gain_cell, strata_rows)

NOISE = "_noise_floor"
# emotion: Cohen's kappa (chance-corrected); raw agreement is kept in the metrics but rewards
# methods whose output reads as the classroom's dominant class (neutral)
UTILITY = ["expression", "pose_err", "emotion_kappa", "age_err", "gender_agree", "redetected"]
# Over all detector faces: `facenet.rank1` is the "passthrough as a leak" view (a failed
# face is scored with its real embedding), `facenet_final.*` what the composed video shows.
KEYS = ["anonymized", "facenet.rank1", "facenet.rank5", "facenet_final.rank1", "facenet_final.rank5",
        "facenet_final.verified", "arcface.rank1", "arcface_final.rank1"]
GENERATED_KEYS = ["facenet.rank1", "facenet.rank5", "arcface.rank1", "facenet.cos", "emotion_agree", *UTILITY]


with Experiment(__file__) as exp:
    cfg = exp.config
    ev, arms = cfg["eval"], cfg["arms"]
    labels = {a["name"]: a.get("label", a["name"]) for a in arms}
    pooled: dict[str, list[dict]] = {}
    consistency: dict[str, list[float]] = {}
    fps: dict[str, list[float]] = {}

    for video in exp.videos():
        out = exp.results / video.name
        det_dir = detect(exp, video, cfg["detect"], out / "detect")
        dirs = [anonymize(exp, video, det_dir, arm, out / arm["name"], exp.cache / video.name / arm["name"])
                for arm in arms]
        report, rows = evaluate(exp, video, det_dir, dirs, ev, out / "eval")
        for name, arm_rows in by_run(rows).items():
            pooled.setdefault(name, []).extend(arm_rows)
            metric_keys(exp, f"e2/{video.name}/{name}", arm_rows, KEYS, ev)
            metric_keys(exp, f"e2/{video.name}/{name}/generated", [r for r in arm_rows if r["anonymized"]],
                        GENERATED_KEYS, ev)
            if name != NOISE:
                c = report["runs"][name]["recognizers"]["facenet"]["consistency_mean"]
                if c is not None:
                    consistency.setdefault(name, []).append(c)
                fps.setdefault(name, []).append(json.loads((out / name / "run_manifest.json").read_text())["fps"])

    names = [a["name"] for a in arms]
    for name, rows in pooled.items():
        metric_keys(exp, f"e2/pooled/{name}", rows, KEYS, ev)
        metric_keys(exp, f"e2/pooled/{name}/generated", [r for r in rows if r["anonymized"]], GENERATED_KEYS, ev)
        pg = privacy_gain_cell(rows, ev)
        if pg:
            exp.metric(f"e2/pooled/{name}/facenet_final.privacy-gain", round(pg[0], 3))

    # Main table: coverage and FaceNet (held-out) rank-1 in the three views, pooled over videos.
    exp.table("main", ["Arm", "Coverage", "R1 gen.", "R1 all", "R1 final", "PG final", "Verified"], [
        [labels[n], cell(pooled[n], "anonymized", ev), cell([r for r in pooled[n] if r["anonymized"]], "facenet.rank1", ev),
         cell(pooled[n], "facenet.rank1", ev), cell(pooled[n], "facenet_final.rank1", ev),
         privacy_gain_cell(pooled[n], ev), cell(pooled[n], "facenet_final.verified", ev)]
        for n in names], best={"R1 final": "min", "Verified": "min"}, align="lrrrrrr")

    # FN = FaceNet (held-out), AF = ArcFace (guidance space); the caption says so
    exp.table("privacy", ["Arm", "FN R1", "FN R5", "AF R1", "AF R5", "FN cos", "Consist."], [
        [labels[n], cell(pooled[n], "facenet_final.rank1", ev), cell(pooled[n], "facenet_final.rank5", ev),
         cell(pooled[n], "arcface_final.rank1", ev), cell(pooled[n], "arcface_final.rank5", ev),
         cell([r for r in pooled[n] if r["anonymized"]], "facenet.cos", ev),
         round(sum(consistency[n]) / len(consistency[n]), 3) if consistency.get(n) else None]
        for n in names], best={"FN R1": "min", "AF R1": "min"}, align="lrrrrrr")

    utility_rows = [arm_row(labels[n], [r for r in pooled[n] if r["anonymized"]], UTILITY, ev) for n in names]
    if NOISE in pooled:
        utility_rows.append(arm_row("Real, next frame", pooled[NOISE], UTILITY, ev))
    exp.table("utility", ["Arm", "Expr.", "Pose (°)", "Emotion κ", "Age (y)", "Gender", "Re-det."], utility_rows,
              best={"Expr.": "min", "Pose (°)": "min", "Emotion κ": "max", "Age (y)": "min", "Gender": "max"},
              align="lrrrrrr")

    exp.table("cost", ["Arm", "Frames / s", "Videos"], [
        [labels[n], round(sum(fps[n]) / len(fps[n]), 3), len(fps[n])] for n in names if fps.get(n)], align="lrr")

    # Robustness: FaceNet rank-1 on generated faces by face size, |yaw| and detection confidence.
    strata_cfg = cfg["eval"]["strata"]
    generated = {n: [r for r in pooled[n] if r["anonymized"]] for n in names}
    for key, title in (("face_px", "Face size (px)"), ("abs_yaw", "|Yaw| (°)"), ("det_conf", "Detection conf.")):
        edges = strata_cfg[{"face_px": "face_size_px", "abs_yaw": "abs_yaw_deg", "det_conf": "det_conf"}[key]]
        bins, rows = strata_rows(generated, key, edges, "facenet.rank1", ev, labels)
        exp.table(f"strata-{key.replace('_', '-')}", [title, *bins], rows, align="l" + "r" * len(bins))

    # Trade-off figure: what the viewer sees (FaceNet rank-1, final video) vs emotion agreement.
    fig_rows = []
    with exp.figure("tradeoff", rows=fig_rows, height_in=2.6) as fig:
        ax = fig.subplots()
        for i, n in enumerate(names):
            x, xh, _ = obs_ci(pooled[n], "facenet_final.rank1")
            y, yh, _ = kappa_ci([r for r in pooled[n] if r["anonymized"]], n=ev["bootstrap"], seed=ev["seed"])
            if x is None or y is None:
                continue
            ax.errorbar(x, y, xerr=xh, yerr=yh, fmt="o", color=SERIES[i % len(SERIES)], capsize=2, label=labels[n])
            fig_rows.append({"arm": n, "facenet_final_rank1": round(x, 4), "ci_x": round(xh, 4),
                             "emotion_kappa": round(y, 4), "ci_y": round(yh, 4)})
        if NOISE in pooled:
            y, _, _ = kappa_ci(pooled[NOISE], n=ev["bootstrap"], seed=ev["seed"])
            if y is not None:
                ax.axhline(y, color="#0b0b0b", linewidth=0.8, linestyle=(0, (3, 2)), label="real, next frame")
        ax.set_xlabel("FaceNet rank-1, final video (lower = more private)")
        ax.set_ylabel("emotion agreement (Cohen's κ)")
        ax.legend(frameon=False, fontsize=7, loc="best")
