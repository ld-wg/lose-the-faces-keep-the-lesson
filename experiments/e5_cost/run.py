"""E5 — cost of the final pipeline. Design and outputs: README.md in this folder.

    uv run experiments/e5_cost/run.py [--video NAME ...] [--resume] [--fresh]

Each arm is the final pipeline with one cost factor changed (main process
device, GFPGAN, how many frames are generated). Arms run on E3's frozen
BLANKET identities and are scored on the composed video itself (the evaluator's
--video-view): a face reused across frames is measured as it looks, and motion
error says whether it froze or jittered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import SERIES, Experiment, obs_ci, paired_kappa_diff  # noqa: E402
from _generation import anonymize, by_run, cell, detect, eval_ctx_args, evaluate, metric_keys, paired_rows  # noqa: E402

PIPELINE_KEYS = ("identity_cache_from", "reference", "arms")   # [pipeline] entries that are not arm flags
# what the viewer sees (--video-view), FaceNet = held-out
VIDEO = ["facenet_video.rank1", "facenet_video.verified", "arcface_video.rank1", "expression_video",
         "pose_err_video", "emotion_kappa_video", "gender_agree_video", "motion_err_video"]
STAGES = {"client.head_segmentation": "Seg.", "server.enhance_face": "GFPGAN", "server.swap_face": "Swap",
          "server.create_faces": "Landm."}
GROUPS = {"ref": 0, "ref-cpu": 0, "enh-off": 1, "enh-96": 1, "every2": 2, "every3": 2,
          "adapt-0.05": 3, "adapt-0.08": 3, "fast": 4}


with Experiment(__file__) as exp:
    cfg, ev = exp.config, exp.config["eval"]
    pipe = cfg["pipeline"]
    base = {k: v for k, v in pipe.items() if k not in PIPELINE_KEYS}
    arms_cfg = pipe["arms"]
    ref = pipe["reference"]
    pooled: dict[str, list[dict]] = {}
    cost: dict[str, dict] = {}   # arm -> summed loop seconds, generate calls, detector faces, profile stages
    per_video: dict[str, dict[str, tuple[float, int]]] = {}   # arm -> video -> (loop seconds, detector faces)

    for video in exp.videos():
        out = exp.results / video.name
        det_dir = detect(exp, video, cfg["detect"], out / "detect")
        cache = exp.cache.parent / pipe["identity_cache_from"] / video.name / "blanket"

        # the privacy gate's threshold: the video's in-domain ArcFace FAR 1% threshold (as E3e)
        cal = out / "calibration"
        if not (cal / "eval.json").is_file():
            exp.module(f"calibrate {video.name}", "src.eval.evaluate", "--phase1-dir", det_dir, "--video",
                       video.path, "--calibration-only", "--far", ev["far"], "--out", cal, *eval_ctx_args())
        tau = json.loads((cal / "eval.json").read_text())["calibration"]["arcface"]["threshold"]
        if tau is None:   # no impostor pairs: the final pipeline cannot be built for this video (as E3e)
            exp.note(f"e5/{video.name}", "skipped: no in-domain ArcFace threshold for the privacy gate")
            continue
        exp.metric(f"e5/{video.name}/gate-tau", tau)

        dirs = {}
        for arm in arms_cfg:
            if video.name not in arm.get("videos", [video.name]):
                continue
            spec = {**base, **{k: v for k, v in arm.items() if k != "videos"}, "privacy_gate": tau,
                    "identity_cache": str(cache), "profile": True}
            spec.setdefault("main_gpu", True)
            dirs[arm["name"]] = anonymize(exp, video, det_dir, spec, out / arm["name"])

        _, rows = evaluate(exp, video, det_dir, list(dirs.values()), ev, out / "eval", noise_floor=False,
                           video_view=True)
        for name, run_rows in by_run(rows).items():
            pooled.setdefault(name, []).extend(run_rows)
            metric_keys(exp, f"e5/{video.name}/{name}", run_rows, VIDEO, ev)
            m = json.loads((dirs[name] / "run_manifest.json").read_text())
            c = cost.setdefault(name, {"seconds": 0.0, "calls": 0, "faces": 0, "stages": {}})
            c["seconds"] += m["loop_seconds"]
            per_video.setdefault(name, {})[video.name] = (m["loop_seconds"], len(run_rows))
            c["calls"] += m["generate_calls"]
            c["faces"] += len(run_rows)          # detector faces with a real reference
            prof = dirs[name] / "profile.json"
            if prof.is_file():
                for stage, s in json.loads(prof.read_text()).items():
                    acc = c["stages"].setdefault(stage, [0.0, 0])
                    acc[0] += s["total_s"]
                    acc[1] += s["n"]

    names = [a["name"] for a in arms_cfg if a["name"] in pooled]
    proj = cfg["projection"]

    def ms_per_face(name: str) -> float:
        return 1000 * cost[name]["seconds"] / cost[name]["faces"]

    def class_hours(name: str) -> float:
        """GPU hours per hour of class at the projection's frame rate and faces per frame."""
        return ms_per_face(name) / 1000 * proj["fps"] * proj["faces_per_frame"]

    def stage_ms(name: str, stage: str):
        s = cost[name]["stages"].get(stage)
        return round(1000 * s[0] / s[1], 1) if s and s[1] else None

    def share(rows, pred):
        return round(sum(1 for r in rows if pred(r)) / len(rows), 3) if rows else None

    def hidden(rows):   # shown as a gray fill: generation failed or was rejected, and nothing to reuse
        return share(rows, lambda r: r.get("video_outcome") == "filled")

    def gated(rows):    # generated, then rejected by the privacy gate
        return share(rows, lambda r: r.get("status") == "gated")

    for name in names:
        rows = pooled[name]
        metric_keys(exp, f"e5/pooled/{name}", rows, VIDEO, ev)
        exp.metric(f"e5/pooled/{name}/hidden-share", hidden(rows))
        exp.metric(f"e5/pooled/{name}/gated-share", gated(rows))
        exp.metric(f"e5/pooled/{name}/ms-per-face", round(ms_per_face(name), 1))
        exp.metric(f"e5/pooled/{name}/class-hours", round(class_hours(name), 1))
        exp.metric(f"e5/pooled/{name}/generated-share", round(cost[name]["calls"] / cost[name]["faces"], 3))
        for stage, label in STAGES.items():
            v = stage_ms(name, stage)
            if v is not None:
                exp.metric(f"e5/pooled/{name}/stage-{label.lower().rstrip('.')}-ms", v)
    speedup: dict[str, float] = {}
    if ref in names:
        # on the videos both arms ran (ref-cpu runs on a subset): faces differ in size and density
        for name in names:
            if name != ref:
                common = set(per_video[name]) & set(per_video[ref])
                ms = {a: sum(per_video[a][v][0] for v in common) / sum(per_video[a][v][1] for v in common)
                      for a in (name, ref)}
                speedup[name] = round(ms[ref] / ms[name], 2)
                exp.metric(f"e5/pooled/{name}/speedup", speedup[name])
                exp.note(f"e5/speedup/{name}/videos", sorted(common))

    # Cost: what each arm does and spends, per detector face (skipped faces included).
    # (ms / face pools each arm's own videos; the speed-up compares with ref on the videos both ran)
    exp.table("cost", ["Arm", "Generated", "ms / face", *STAGES.values(), "× ref", "h / class-h"], [
        [n, round(cost[n]["calls"] / cost[n]["faces"], 2), round(ms_per_face(n)),
         *(stage_ms(n, s) for s in STAGES), speedup.get(n, 1.0 if n == ref else None), round(class_hours(n), 1)]
        for n in names], best={"ms / face": "min", "h / class-h": "min"}, align="lrrrrrrrr")

    # What the viewer sees: privacy, utility and motion on the composed video.
    exp.table("video", ["Arm", "FN R1", "Verified", "Hidden", "Expr.", "Pose (°)", "Emotion κ", "Motion"], [
        [n, cell(pooled[n], "facenet_video.rank1", ev), cell(pooled[n], "facenet_video.verified", ev), hidden(pooled[n]),
         *(cell(pooled[n], m, ev) for m in ("expression_video", "pose_err_video", "emotion_kappa_video", "motion_err_video"))]
        for n in names], best={"FN R1": "min", "Motion": "min", "Expr.": "min", "Emotion κ": "max"}, align="lrrrrrrr")

    # Paired against the reference on the same faces: privacy, expression and motion.
    if ref in names:
        rows_by_arm = {n: pooled[n] for n in names}
        paired = {m: {r[0]: r for r in paired_rows(rows_by_arm, ref, m, ev)}
                  for m in ("facenet_video.rank1", "facenet_video.verified", "expression_video", "motion_err_video")}
        kappa = {n: paired_kappa_diff(pooled[n], pooled[ref], "emotion_real_video", "emotion_video",
                                      n=ev["bootstrap"], seed=ev["seed"]) for n in names if n != ref}
        for n, k in kappa.items():
            if k["diff"] is not None:
                exp.metric(f"e5/pooled/{n}/delta-kappa", round(k["diff"], 3))
                exp.metric(f"e5/pooled/{n}/delta-kappa.ci", round(k["half"], 3))
                exp.metric(f"e5/pooled/{n}/delta-kappa.excl", "yes" if k["excludes_zero"] else "no")
        # Δ and whether the paired interval excludes zero, as "Δ*" (excludes) or "Δ" (does not)
        def mark(c, ex):
            return None if c is None else (f"{c[0]:+.2f}" + ("*" if ex in ("yes", True) else ""))
        exp.table("paired", ["Arm", "Δ FN R1", "Δ Verified", "Δ Expr.", "Δ Motion", "Δ Emotion κ"], [
            [n, *(mark(paired[m][n][1], paired[m][n][2]) for m in paired),
             mark((kappa[n]["diff"], kappa[n]["half"]) if kappa[n]["diff"] is not None else None, kappa[n]["excludes_zero"])]
            for n in names if n != ref], align="lrrrrr")
        short = {"facenet_video.rank1": "fn-r1", "facenet_video.verified": "fn-ver",
                 "expression_video": "expr", "motion_err_video": "motion"}   # keys fit a line
        for m, by_arm in paired.items():
            for n, r in by_arm.items():
                if r[1] is not None:
                    exp.metric(f"e5/pooled/{n}/delta-{short[m]}", round(r[1][0], 3))
                    exp.metric(f"e5/pooled/{n}/delta-{short[m]}.ci", round(r[1][1], 3))
                    exp.metric(f"e5/pooled/{n}/delta-{short[m]}.excl", r[2])

    # Trade-off figure: compute per hour of class against motion error and re-identification.
    fig_rows = []
    with exp.figure("tradeoff", rows=fig_rows, height_in=2.6) as fig:
        ax_m, ax_p = fig.subplots(1, 2)
        for n in names:
            x = class_hours(n)
            color = SERIES[GROUPS.get(n, 5) % len(SERIES)]
            for ax, metric in ((ax_m, "motion_err_video"), (ax_p, "facenet_video.rank1")):
                y, h, _ = obs_ci(pooled[n], metric, n=ev["bootstrap"], seed=ev["seed"])
                if y is None:
                    continue
                ax.errorbar(x, y, yerr=h, fmt="o", color=color, capsize=2, markersize=4)
                ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
                fig_rows.append({"arm": n, "class_hours": round(x, 2), "metric": metric,
                                 "value": round(y, 4), "ci": round(h, 4)})
        for ax in (ax_m, ax_p):
            ax.set_xscale("log")
            ax.set_xlabel("GPU hours per hour of class")
        ax_m.set_ylabel("motion error (inter-ocular)")
        ax_p.set_ylabel("FaceNet rank-1, composed video")
