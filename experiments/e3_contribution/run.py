"""E3 — with and without our ideas. Design and outputs: README.md in this folder.

    uv run experiments/e3_contribution/run.py [--video NAME ...] [--fresh]
    E3_PARTS=e3c,e3d uv run experiments/e3_contribution/run.py     # a subset of the parts

Parts: e3a C2 identity estimate · e3b coverage (seed candidates, Phase 1 detection)
· e3c P2 strength · e3d P2 target · e3e fail-closed policies · e3f P3 (CIAGAN)
· e3g P1 (SDXL guidance, on top of P2; run only after the bridge's tools/p1_spike.py says go).
Arms with identical settings run once and are shared between parts; BLANKET
identities are frozen per video in cache/ so every BLANKET arm swaps the same faces.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import SERIES, Experiment, obs_ci  # noqa: E402
from _generation import (anonymize, by_run, cell, detect, eval_ctx_args, evaluate,  # noqa: E402
                         metric_keys, paired_rows)

PARTS = [p for p in os.environ.get("E3_PARTS", "e3a,e3b,e3c,e3d,e3e,e3f,e3g").split(",") if p]
GEN = lambda rows: [r for r in rows if r["anonymized"]]  # noqa: E731
KEYS = ["facenet.rank1", "facenet.rank5", "arcface.rank1", "facenet_final.rank1", "facenet_final.verified",
        "anonymized", "expression", "pose_err", "emotion_agree", "emotion_kappa", "age_err", "gender_agree"]


def reason_share(rows, reason: str):
    v = [1.0 if r.get("reason") == reason else 0.0 for r in rows]
    return round(sum(v) / len(v), 3) if v else None


with Experiment(__file__) as exp:
    cfg, ev = exp.config, exp.config["eval"]
    base = cfg["blanket"]
    parts: dict[str, dict[str, str]] = {}          # part -> arm label -> run dir name
    pooled: dict[str, dict[str, list[dict]]] = {}  # part -> run dir name -> rows
    identity: dict[str, list[dict]] = {}           # e3a: video -> identity report

    for video in exp.videos():
        out = exp.results / video.name
        det_dir = detect(exp, video, cfg["detect"], out / "detect")
        cache = exp.cache / video.name / "blanket"
        done: dict[tuple, Path] = {}

        def run(arm: dict) -> str:
            """Anonymize once per distinct setting; returns the run dir's name."""
            key = tuple(sorted((k, str(v)) for k, v in arm.items() if k != "name"))
            if key not in done:
                done[key] = anonymize(exp, video, det_dir, arm, out / arm["name"], cache)
            return done[key].name

        def part_eval(part: str, arms: dict[str, str]) -> None:
            dirs = sorted({out / d for d in arms.values()})
            _, rows = evaluate(exp, video, det_dir, dirs, ev, out / f"eval-{part}", noise_floor=False)
            parts[part] = arms
            for name, run_rows in by_run(rows).items():
                pooled.setdefault(part, {}).setdefault(name, []).extend(run_rows)
                for label, d in arms.items():
                    if d == name:
                        metric_keys(exp, f"{part}/{video.name}/{label}", run_rows, KEYS, ev)

        if "e3a" in PARTS:
            a = cfg["e3a"]
            report = out / "e3a" / "identity_report.json"
            exp.module(f"identity report {video.name}", "src.eval.identity_report", "--phase1-dir", det_dir,
                       "--video", video.path, "--out", report, "--outlier-cos", a["outlier_cos"],
                       "--link-cos", a["link_cos"], "--ema-alphas", *a["ema_alpha_floors"], *eval_ctx_args())
            identity[video.name] = json.loads(report.read_text())

        if "e3b" in PARTS:
            arms = {}
            for arm in cfg["e3b"]["arms"]:   # in order: upstream candidates freeze identities before phase1 reuses them
                arms[arm["name"]] = run({**arm, "name": f"b-{arm['name']}", "backend": "blanket"})
            part_eval("e3b", arms)

        p2 = lambda beta, agg, extra=None: {**base, "swap_mode": "track", "push_beta": beta,  # noqa: E731
                                            "aggregation": agg, **(extra or {})}
        # one folder per distinct (beta, target) setting, shared by e3c/e3d/e3e/e3g
        p2_name = lambda beta, agg, extra=None: (f"p2-b{beta}-{agg}"  # noqa: E731
                                                 + (f"-a{extra['ema_alpha_floor']}" if extra and "ema_alpha_floor" in extra else ""))
        if "e3c" in PARTS:
            c = cfg["e3c"]
            arms = {m: run({**base, "name": f"c-{m}", "swap_mode": m, "aggregation": c["aggregation"]})
                    for m in c["modes"]}
            for beta in c["betas"]:
                arms[f"β {beta}"] = run({**p2(beta, c["aggregation"]), "name": p2_name(beta, c["aggregation"])})
            part_eval("e3c", arms)

        if "e3d" in PARTS:
            d = cfg["e3d"]
            arms = {}
            for agg in d["aggregations"]:
                extra = {"ema_alpha_floor": d["ema_alpha_floor"]} if agg == "ema_adaptive" else {}
                arms[agg] = run({**p2(d["beta"], agg, extra), "name": p2_name(d["beta"], agg, extra)})
            part_eval("e3d", arms)

        if "e3e" in PARTS:
            e = cfg["e3e"]
            tau = None
            arms = {}
            for arm in e["arms"]:
                extra = {k: v for k, v in arm.items() if k != "name"}
                if extra.get("privacy_gate") == "arcface":
                    if tau is None:   # the video's in-domain ArcFace threshold at FAR 1%, as in E0
                        cal = out / "calibration"
                        exp.module(f"calibrate {video.name}", "src.eval.evaluate", "--phase1-dir", det_dir,
                                   "--video", video.path, "--calibration-only", "--far", ev["far"], "--out", cal,
                                   *eval_ctx_args())
                        tau = json.loads((cal / "eval.json").read_text())["calibration"]["arcface"]["threshold"]
                        exp.metric(f"e3e/{video.name}/gate-tau", tau)
                    extra["privacy_gate"] = tau
                if extra.get("privacy_gate", 0) is None:
                    exp.note(f"e3e/{video.name}/gate", "skipped: no in-domain ArcFace threshold (no impostor pairs)")
                    continue
                name = p2_name(e["beta"], e["aggregation"]) if not extra else f"e-{arm['name']}-{e['aggregation']}"
                arms[arm["name"]] = run({**p2(e["beta"], e["aggregation"], extra), "name": name})
            part_eval("e3e", arms)

        if "e3g" in PARTS:
            g = cfg["e3g"]
            arms = {"P2": run({**p2(g["beta"], g["aggregation"]), "name": p2_name(g["beta"], g["aggregation"])})}
            for scale in g["scales"]:
                # guided identities are a different population: their own frozen cache
                arms[f"P1 c {scale} + P2"] = run({
                    **p2(g["beta"], g["aggregation"], {"p1_scale": scale, "p1_tau": g["tau"], "p1_window": g["window"]}),
                    "name": f"g-p1-c{scale}-{g['aggregation']}",
                    "identity_cache": str(exp.cache / video.name / f"blanket-p1-c{scale}")})
            part_eval("e3g", arms)

        if "e3f" in PARTS:
            f = cfg["e3f"]
            arms = {"ciagan": run({"name": "f-ciagan", "backend": "ciagan"})}
            for tau in f["taus"]:
                arms[f"P3 τ {tau}"] = run({"name": f"f-p3-tau{tau}", "backend": "ciagan", "identity_prepass": True,
                                           "aggregation": f["aggregation"], "ciagan_push": "track",
                                           "ciagan_push_steps": f["steps"], "ciagan_push_lambda": f["lambda"],
                                           "ciagan_push_tau": tau})
            part_eval("e3f", arms)

    # ---- tables, pooled over videos -------------------------------------------------
    def rows_of(part: str) -> dict[str, list[dict]]:
        return {label: pooled[part][d] for label, d in parts[part].items()}

    if identity:
        modes = cfg["e3a"]["modes"]
        n = {v: r["modes_heldout_half"]["num_tracks"] for v, r in identity.items()}
        total = sum(n.values()) or 1
        def wmean(key, m):  # noqa: E301
            vals = [(r["modes_heldout_half"][key][m], n[v]) for v, r in identity.items()
                    if r["modes_heldout_half"][key].get(m) is not None]
            return round(sum(x * w for x, w in vals) / sum(w for _, w in vals), 3) if vals else None
        exp.table("identity", ["Mode", "Held-out cos", "Quality-weighted cos"],
                  [[m, wmean("per_mode_mean_cos", m), wmean("per_mode_quality_weighted_cos", m)] for m in modes],
                  best={"Held-out cos": "max", "Quality-weighted cos": "max"}, align="lrr")
        for m in modes:
            exp.metric(f"e3a/pooled/{m}.heldout-cos", wmean("per_mode_mean_cos", m))
        exp.metric("e3a/pooled/tracks", total)
        floors = [r["cross_track"]["cooccurring_pairs_cos"].get("mean") for r in identity.values()]
        floors = [x for x in floors if x is not None]
        if floors:
            exp.metric("e3a/pooled/cross-track-cos-mean", round(sum(floors) / len(floors), 3))

    for part in parts:
        for label, rows in rows_of(part).items():
            metric_keys(exp, f"{part}/pooled/{label}", rows, KEYS, ev)

    if "e3b" in parts:
        exp.table("coverage", ["Arm", "Coverage", "Identity unusable", "Swap no face", "R1 final", "Expr.", "Emotion κ"], [
            [label, cell(rows, "anonymized", ev), reason_share(rows, "identity_unusable"), reason_share(rows, "swap_no_face"),
             cell(rows, "facenet_final.rank1", ev), cell(GEN(rows), "expression", ev), cell(GEN(rows), "emotion_kappa", ev)]
            for label, rows in rows_of("e3b").items()], best={"Coverage": "max", "R1 final": "min"}, align="lrrrrrr")

    def strength_table(part: str, name: str, reference: str) -> None:
        rows = rows_of(part)
        gen = {k: GEN(v) for k, v in rows.items()}
        delta = {r[0]: r[1] for r in paired_rows(gen, reference, "facenet.rank1", ev)}
        exp.table(name, ["Arm", "FaceNet R1", "ArcFace R1", f"Δ vs {reference}", "Expr.", "Emotion κ", "Pose (°)"], [
            [label, cell(g, "facenet.rank1", ev), cell(g, "arcface.rank1", ev), delta.get(label),
             cell(g, "expression", ev), cell(g, "emotion_kappa", ev), cell(g, "pose_err", ev)]
            for label, g in gen.items()], best={"FaceNet R1": "min", "ArcFace R1": "min"}, align="lrrrrrr")
        for r in paired_rows(gen, reference, "facenet.rank1", ev):
            if r[1] is not None:
                exp.metric(f"{part}/pooled/{r[0]}/delta-facenet.rank1", round(r[1][0], 3))
                exp.metric(f"{part}/pooled/{r[0]}/delta-facenet.rank1.ci", round(r[1][1], 3))
                exp.metric(f"{part}/pooled/{r[0]}/delta-excludes-zero", r[2])

    if "e3c" in parts:
        strength_table("e3c", "p2-strength", cfg["e3c"]["reference"])
        curve = []
        with exp.figure("p2-curve", rows=curve, height_in=2.4) as fig:
            ax_p, ax_u = fig.subplots(1, 2)
            gen = {k: GEN(v) for k, v in rows_of("e3c").items()}
            betas = cfg["e3c"]["betas"]
            for i, (rec, lab) in enumerate((("facenet", "FaceNet (held-out)"), ("arcface", "ArcFace (guidance)"))):
                pts = [(b, *obs_ci(gen[f"β {b}"], f"{rec}.rank1")[:2]) for b in betas]
                ax_p.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o",
                              color=SERIES[i], capsize=2, label=lab)
                ref, _, _ = obs_ci(gen[cfg["e3c"]["reference"]], f"{rec}.rank1")
                ax_p.axhline(ref, color=SERIES[i], linewidth=0.8, linestyle=(0, (3, 2)))
                curve += [{"recognizer": rec, "beta": b, "rank1": round(y, 4), "ci": round(h, 4)} for b, y, h in pts]
            ax_p.set_xlabel("push strength β")
            ax_p.set_ylabel("rank-1 (generated faces)")
            ax_p.legend(frameon=False, fontsize=7)
            pts = [(b, *obs_ci(gen[f"β {b}"], "expression")[:2]) for b in betas]
            ax_u.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o",
                          color=SERIES[2], capsize=2)
            ax_u.set_xlabel("push strength β")
            ax_u.set_ylabel("expression error")
            curve += [{"recognizer": "expression", "beta": b, "rank1": round(y, 4), "ci": round(h, 4)} for b, y, h in pts]

    if "e3d" in parts:
        strength_table("e3d", "p2-target", cfg["e3d"]["reference"])

    if "e3e" in parts:
        rows = rows_of("e3e")
        ref = cfg["e3e"]["reference"]
        delta = {r[0]: r[1] for r in paired_rows(rows, ref, "facenet_final.rank1", ev)}
        hidden = lambda rs: round(sum(r.get("outcome") in ("reused", "filled") for r in rs) / len(rs), 3) if rs else None  # noqa: E731
        exp.table("policies", ["Policy", "R1 final", "Verified", "Hidden", f"Δ R1 vs {ref}", "Expr.", "Emotion κ"], [
            [label, cell(r, "facenet_final.rank1", ev), cell(r, "facenet_final.verified", ev), hidden(r),
             delta.get(label), cell(GEN(r), "expression", ev), cell(GEN(r), "emotion_kappa", ev)]
            for label, r in rows.items()], best={"R1 final": "min", "Verified": "min"}, align="lrrrrrr")

    if "e3f" in parts:
        strength_table("e3f", "p3", cfg["e3f"]["reference"])

    if "e3g" in parts:
        strength_table("e3g", "p1", "P2")
