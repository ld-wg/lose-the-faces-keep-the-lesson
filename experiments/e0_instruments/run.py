"""E0 — instrument validation. Design and outputs: README.md in this folder.

    uv run experiments/e0_instruments/run.py [--video NAME ...]
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import SERIES, Experiment  # noqa: E402

RECOGNIZERS = {"facenet": "FaceNet", "arcface": "ArcFace"}   # roles go in the caption: held-out / guidance
BRIDGE = Path(os.environ.get("BLANKET_REPO", Path.home() / "projects" / "blanket-anonymizer-bridge"))


def ctx_args() -> list[str]:
    return ["--ctx-id", os.environ["CTX_ID"]] if "CTX_ID" in os.environ else []


def mechanism_check(exp: Experiment, crops: Path, faces: int) -> None:
    """FaceFusion's recognizer vs buffalo_l's, and inswapper's emap, in the bridge's swap venv."""
    python = BRIDGE / ".venv-swap" / "bin" / "python"
    if not python.is_file():
        exp.note("mechanism", f"skipped: no swap environment at {python}")
        return
    env = {"CUDA_VISIBLE_DEVICES": os.environ["SWAP_GPU"]} if "SWAP_GPU" in os.environ else None
    out = exp.results / "mechanism.json"
    exp.sh("mechanism check", python, BRIDGE / "tools" / "check_embedding_space.py",
           "--images", f"{crops}/*.jpg", "--limit", faces, "--json", out, env=env, cwd=BRIDGE)
    m = json.loads(out.read_text())
    exp.metric("e0/mechanism/recognizer-cos-mean", m["recognizer_equivalence_cos"]["mean"])
    exp.metric("e0/mechanism/recognizer-cos-min", m["recognizer_equivalence_cos"]["min"])
    exp.metric("e0/mechanism/emap-orthogonality", m["emap_orthogonality"])
    exp.metric("e0/mechanism/emap-direction-cos-mean", m["emap_direction_cos"]["mean"])
    exp.metric("e0/mechanism/faces", m["faces"])
    exp.table("mechanism", ["Check", "Value"], [
        ["Recognizer equivalence, mean cosine", m["recognizer_equivalence_cos"]["mean"]],
        ["Recognizer equivalence, minimum cosine", m["recognizer_equivalence_cos"]["min"]],
        ["emap distance from orthogonal", m["emap_orthogonality"]],
        ["emap direction change, mean cosine", m["emap_direction_cos"]["mean"]],
        ["Faces checked", m["faces"]],
    ], align="lr")


with Experiment(__file__) as exp:
    cfg = exp.config
    det, far, bins = cfg["detect"], cfg["recognizers"]["far"], cfg["recognizers"]["histogram_bins"]
    rows, scores = [], []

    for i, video in enumerate(exp.videos()):
        out = exp.results / video.name
        mechanism_here = cfg["mechanism"]["enabled"] and i == 0
        exp.module(f"detect {video.name}", "src.pipeline.phase1_detect.run",
                   "--input", video.path, "--out", out / "detect", "--model", det["model"],
                   "--conf", det["conf"], "--det-size", det["det_size"],
                   *(["--save-crops"] if mechanism_here else []))
        exp.module(f"calibrate {video.name}", "src.eval.evaluate",
                   "--phase1-dir", out / "detect", "--video", video.path, "--calibration-only",
                   "--far", far, "--out", out / "eval", "--scores-out", out / "eval" / "scores.csv", *ctx_args())

        calibration = json.loads((out / "eval" / "eval.json").read_text())["calibration"]
        for rec, label in RECOGNIZERS.items():
            c = calibration[rec]
            rows.append([video.name, label, c["threshold"], c["tar"], c["genuine_mean"], c["impostor_mean"],
                         f"{c['n_genuine']}/{c['n_impostor']}"])
            exp.metric(f"e0/{video.name}/{rec}.threshold", c["threshold"])
            exp.metric(f"e0/{video.name}/{rec}.tar", c["tar"])
            exp.metric(f"e0/{video.name}/{rec}.genuine-mean", c["genuine_mean"])
            exp.metric(f"e0/{video.name}/{rec}.impostor-mean", c["impostor_mean"])
        with open(out / "eval" / "scores.csv") as f:
            scores += [{"video": video.name, **r} for r in csv.DictReader(f)]

        if mechanism_here:
            crops = out / "detect" / "crops"
            try:
                mechanism_check(exp, crops, cfg["mechanism"]["faces"])
            finally:
                shutil.rmtree(crops, ignore_errors=True)   # real faces: never kept

    if cfg["noise_floor"]["enabled"]:
        raise NotImplementedError("utility noise floors need the utility probes (see README)")
    exp.note("noise_floor", "pending: needs the utility probes")

    exp.table("recognizers",
              ["Video", "Recognizer", "Threshold", "TAR", "Genuine", "Impostor", "Pairs"], rows, align="llrrrrr")

    # Genuine vs impostor similarity, pooled over videos, with each video's threshold.
    hist_rows = []
    with exp.figure("scores", rows=hist_rows, height_in=2.2) as fig:
        axes = fig.subplots(1, len(RECOGNIZERS), sharey=False)
        for ax, (rec, label) in zip(axes, RECOGNIZERS.items()):
            edges = np.linspace(-0.2, 1.0, bins + 1)
            for kind, color in (("genuine", SERIES[0]), ("impostor", SERIES[1])):
                s = np.array([float(r["score"]) for r in scores if r["recognizer"] == rec and r["kind"] == kind])
                density, _ = np.histogram(s, bins=edges, density=True)
                ax.stairs(density, edges, color=color, linewidth=1.6,
                          label="genuine (same track)" if kind == "genuine" else "impostor (co-visible tracks)")
                hist_rows += [{"recognizer": rec, "kind": kind, "bin_left": round(float(a), 4),
                               "bin_right": round(float(b), 4), "density": round(float(d), 5)}
                              for a, b, d in zip(edges[:-1], edges[1:], density)]
            for i, r in enumerate(r for r in rows if r[1] == label):
                ax.axvline(r[2], color="#0b0b0b", linewidth=0.8, linestyle=(0, (3, 2)),
                           label="threshold at FAR 1% (one per video)" if i == 0 else "_nolegend_")
            ax.set_title(label)
            ax.set_xlabel("cosine similarity")
        axes[0].set_ylabel("density")
        # One legend above both panels, clear of the data and the threshold lines.
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
