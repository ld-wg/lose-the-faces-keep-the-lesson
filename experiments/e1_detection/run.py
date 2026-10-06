"""E1 — face detectors. Design and outputs: README.md in this folder.

    uv run experiments/e1_detection/run.py [--video NAME ...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))   # the repo root, for src.eval
from _common import Experiment  # noqa: E402
from src.eval.detection_stats import track_stats  # noqa: E402

LABELS = {"scrfd-10gf": "SCRFD-10GF", "scrfd-34gf": "SCRFD-34GF", "yolo-facev2-s": "YOLO-FaceV2-s"}


def ctx() -> list[str]:
    return ["--ctx-id", os.environ.get("EVAL_CTX_ID", "0")]


with Experiment(__file__) as exp:
    cfg = exp.config
    det, wf, models = cfg["detect"], cfg["widerface"], cfg["arms"]["models"]
    fppi = wf["fppi"]

    bench = {}
    for model in models:
        out = exp.results / "widerface" / model
        exp.module(f"widerface {model}", "src.eval.widerface", "--model", model, "--out", out,
                   "--det-size", det["det_size"], "--conf-at", det["conf"], "--fppi", *fppi, "--iou", wf["iou"],
                   "--latency-warmup", wf["latency_warmup"], "--latency-images", wf["latency_images"], *ctx())
        r = json.loads((out / "widerface.json").read_text())
        bench[model] = r
        for s in ("easy", "medium", "hard"):
            exp.metric(f"e1/widerface/{model}/ap-{s}", r["ap"][s])
        op = r["operating_points"]
        for k in ("precision", "recall", "f1"):
            exp.metric(f"e1/widerface/{model}/{k}", op[k])
        for target, rec in op["recall_at_fppi"].items():
            exp.metric(f"e1/widerface/{model}/recall-at-fppi-{target}", rec)
        exp.metric(f"e1/widerface/{model}/latency-ms", r["latency_ms"]["median"])

    exp.table("widerface-ap", ["Detector", "AP easy", "AP medium", "AP hard", "Latency (ms)", "Params (M)"], [
        [LABELS[m], r["ap"]["easy"], r["ap"]["medium"], r["ap"]["hard"], r["latency_ms"]["median"],
         round(r["params"] / 1e6, 2) if r.get("params") else None] for m, r in bench.items()],
        best={"AP easy": "max", "AP medium": "max", "AP hard": "max", "Latency (ms)": "min"}, align="lrrrrr")
    conf = det["conf"]
    exp.table("widerface-recall", ["Detector", f"P @ {conf}", f"R @ {conf}", f"F1 @ {conf}",
                                   *[f"R @ FPPI {t}" for t in fppi]], [
        [LABELS[m], r["operating_points"]["precision"], r["operating_points"]["recall"], r["operating_points"]["f1"],
         *[r["operating_points"]["recall_at_fppi"][str(t)] for t in fppi]] for m, r in bench.items()],
        best={f"R @ {conf}": "max", **{f"R @ FPPI {t}": "max" for t in fppi}}, align="l" + "r" * (3 + len(fppi)))

    rows = []
    for video in exp.videos():
        for model in models:
            out = exp.results / video.name / model
            exp.module(f"detect {video.name} {model}", "src.pipeline.phase1_detect.run", "--input", video.path,
                       "--out", out, "--model", model, "--conf", det["conf"], "--det-size", det["det_size"])
            st = track_stats(out, cfg["videos"]["short_track_frames"])
            for k in ("faces_per_frame", "tracks", "median_track_frames", "short_track_share", "fps"):
                exp.metric(f"e1/{video.name}/{model}/{k.replace('_', '-')}", st[k])
            rows.append([video.name, LABELS[model], st["faces_per_frame"], st["tracks"],
                         int(st["median_track_frames"]) if st["median_track_frames"] is not None else None,
                         st["short_track_share"], st["fps"]])
    # Median: track length in frames; Short: share of tracks under videos.short_track_frames
    exp.table("videos", ["Video", "Detector", "Faces/frame", "Tracks", "Median", "Short", "FPS"],
              rows, decimals=2, align="llrrrrr")
