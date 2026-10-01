"""Shared machinery for the paper experiments (see experiments/README.md).

A run.py uses it like this:

    from _common import Experiment

    with Experiment(__file__) as exp:              # results/ is deleted and recreated
        for video in exp.videos():                 # config.toml inputs, or --video
            exp.sh("detect", "uv", "run", "python", "-m", "src.pipeline.phase1_detect.run",
                   "--input", video.path, "--out", exp.results / video.name / "detect")
        exp.metric("e0/demo2/facenet.tar", 0.692)  # -> metrics.json and \\result{} in results.tex
        exp.table("recognizers", columns, rows)    # -> recognizers.md and latex/tab-e0-recognizers.tex
        with exp.figure("scores", rows=data) as fig:   # -> latex/fig-e0-scores.pdf + .csv
            ax = fig.subplots()
            ...

It provides a clean `results/`, `run.log`, `commands.sh` (every command, in
order), `provenance.json`, `metrics.json` and the LaTeX artifacts in
`results/latex/`, in the formats `paper/main.tex` uses: booktabs tabulars only,
vector PDF figures in a serif font, and a `\\result{key}` table that stops the
compilation on an unknown key. On failure, `provenance.json` records the
traceback and no LaTeX is written, so a broken run can never be exported to the
paper.

Stdlib + numpy only at import time; matplotlib is imported when a figure is drawn.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import logging
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VIDEOS_DIR = Path(os.environ.get("VIDEOS_DIR", REPO / "data" / "videos"))
PENDING_EXIT = 2          # run_all.sh reports it as "pending" and continues

# Figure style: SBC template, 12pt, single column (\linewidth = 16 cm).
FIG_WIDTH_IN = 6.3
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e1e0d9"

log = logging.getLogger("experiment")


@dataclass(frozen=True)
class Video:
    name: str
    path: Path


def resolve_video(name_or_path: str) -> Video:
    """A path to an existing file, or a short name found as data/videos/<name>.*"""
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return Video(p.stem, p.resolve())
    hits = sorted(VIDEOS_DIR.glob(f"{name_or_path}.*"))
    if not hits:
        known = sorted({h.stem for h in VIDEOS_DIR.glob("*")}) if VIDEOS_DIR.is_dir() else []
        raise SystemExit(f"no video '{name_or_path}': not a file, and not in {VIDEOS_DIR}/ (known: {known})")
    return Video(name_or_path, hits[0].resolve())


def pending(run_file: str, missing: Sequence[str]) -> None:
    """Exit before touching results/ when components the experiment needs do not exist yet."""
    exp = Path(run_file).resolve().parent.name
    print(f"{exp}: pending — missing components:", file=sys.stderr)
    for m in missing:
        print(f"  - {m}", file=sys.stderr)
    print(f"see experiments/{exp}/README.md", file=sys.stderr)
    sys.exit(PENDING_EXIT)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(*args: str, cwd: Path = REPO) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, used = (s.strip() for s in line.split(","))
        gpus.append({"index": int(idx), "name": name, "memory_total_mib": int(total), "memory_used_mib": int(used)})
    return gpus


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "torch", "onnxruntime-gpu", "onnxruntime", "insightface", "opencv-python", "matplotlib"):
        with contextlib.suppress(md.PackageNotFoundError):
            out[pkg] = md.version(pkg)
    return out


# ---------------------------------------------------------------------------
# LaTeX


_LATEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                  "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                  # inputenc's utf8 (the SBC template) has no Greek or math symbols
                  "β": r"$\beta$", "τ": r"$\tau$", "Δ": r"$\Delta$", "≥": r"$\geq$", "≤": r"$\leq$",
                  "±": r"$\pm$", "×": r"$\times$", "°": r"\textdegree{}"}

_KEY_SUBS = {"β": "b", "τ": "tau", "Δ": "d", "≥": "ge", "≤": "le", "°": "deg"}


def result_key(key: str) -> str:
    """ASCII form of a \\result key: letters, digits and . / _ + - only."""
    key = "".join(_KEY_SUBS.get(c, c) for c in key)
    return re.sub(r"[^A-Za-z0-9./_+-]+", "-", key).strip("-")


def latex_escape(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(c, c) for c in str(text))


def _fmt(value: Any, decimals: int) -> str:
    """One cell: numbers at fixed decimals, (value, ci) as value ± ci, None as a dash."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, tuple):
        v, ci = value
        return f"{_fmt(v, decimals)}\\,{{\\scriptsize$\\pm${ci:.{decimals}f}}}" if ci is not None else _fmt(v, decimals)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return latex_escape(value)


def _plain(value: Any, decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, tuple):
        v, ci = value
        return f"{_plain(v, decimals)} ± {ci:.{decimals}f}" if ci is not None else _plain(v, decimals)
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _numeric(value: Any) -> Optional[float]:
    v = value[0] if isinstance(value, tuple) else value
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(float(v)) else None


# ---------------------------------------------------------------------------
# Statistics


def bootstrap_ci(values: Sequence[float], groups: Optional[Sequence[Any]] = None, *, n: int = 2000,
                 seed: int = 0, level: float = 0.95, stat: Callable = np.mean) -> tuple[float, float, float]:
    """(estimate, low, high). Resamples whole groups (tracks) when given, so
    frames of one track are never treated as independent."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = [np.array([i]) for i in range(v.size)]
    else:
        g = np.asarray(groups)
        idx = [np.flatnonzero(g == k) for k in np.unique(g)]
    boots = []
    for _ in range(n):
        pick = rng.integers(0, len(idx), len(idx))
        boots.append(stat(np.concatenate([v[idx[i]] for i in pick])))
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(stat(v)), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], groups: Optional[Sequence[Any]] = None,
                     **kwargs) -> tuple[float, float, float]:
    """Interval for mean(a − b) over paired observations (same faces in two arms)."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), groups, **kwargs)


def mcnemar_p(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Exact two-sided McNemar test for paired binary outcomes (e.g. per-face rank-1 hits)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ---------------------------------------------------------------------------
# Per-face observations (src.eval.evaluate --obs-out): one row per (run, frame, track)

def load_obs(path: Path, **extra: Any) -> list[dict]:
    """Rows of an --obs-out CSV with numbers parsed; `extra` fields (e.g. video=...)
    are added to every row so videos can be pooled."""
    def parse(v: str):
        if v == "":
            return None
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    with open(path, newline="") as f:
        return [{**{k: parse(v) for k, v in r.items()}, **extra} for r in csv.DictReader(f)]


def obs_value(row: dict, metric: str) -> Optional[float]:
    """`facenet.rank1` / `.rank5` / `.cos` / `.verified`, `real_facenet.rank1`, or a
    utility column (`expression`, `pose_err`, `age_err`, `gender_agree`, `emotion_agree`)."""
    if "." in metric:
        rec, what = metric.split(".", 1)
        if what in ("rank1", "rank5"):
            rank = row.get(f"{rec}_rank")
            return None if rank is None else float(rank <= (1 if what == "rank1" else 5))
        v = row.get(f"{rec}_{what}")
    else:
        v = row.get(metric)
    return None if v is None else float(v)


def _track_sums(rows: Sequence[dict], metric: str) -> tuple[list, np.ndarray, np.ndarray]:
    sums: dict[Any, list[float]] = {}
    for r in rows:
        v = obs_value(r, metric)
        if v is not None:
            s = sums.setdefault((r.get("video"), r["track_id"]), [0.0, 0])
            s[0] += v
            s[1] += 1
    keys = list(sums)
    return keys, np.array([sums[k][0] for k in keys]), np.array([sums[k][1] for k in keys], dtype=float)


def obs_ci(rows: Sequence[dict], metric: str, *, n: int = 2000, seed: int = 0,
           level: float = 0.95) -> tuple[Optional[float], Optional[float], int]:
    """(mean, half-width of the 95% interval, n faces), resampling whole tracks."""
    keys, s, c = _track_sums(rows, metric)
    if not keys or c.sum() == 0:
        return None, None, 0
    est = float(s.sum() / c.sum())
    pick = np.random.default_rng(seed).integers(0, len(keys), (n, len(keys)))
    boots = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1)
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return est, float((hi - lo) / 2), int(c.sum())


def privacy_gain_ci(rows: Sequence[dict], rec: str = "facenet", *, n: int = 2000,
                    seed: int = 0) -> tuple[Optional[float], Optional[float]]:
    """1 − rank1(anonymized) / rank1(real), with a track-bootstrap half-width,
    over the faces that have both ranks."""
    rows = [r for r in rows if obs_value(r, f"{rec}.rank1") is not None
            and obs_value(r, f"real_{rec}.rank1") is not None]
    keys_a, sa, ca = _track_sums(rows, f"{rec}.rank1")
    keys_b, sb, cb = _track_sums(rows, f"real_{rec}.rank1")
    if not keys_a or keys_a != keys_b or sb.sum() == 0:
        return None, None
    est = 1 - (sa.sum() / ca.sum()) / (sb.sum() / cb.sum())
    pick = np.random.default_rng(seed).integers(0, len(keys_a), (n, len(keys_a)))
    boots = 1 - (sa[pick].sum(1) / np.maximum(ca[pick].sum(1), 1)) / np.maximum(
        sb[pick].sum(1) / np.maximum(cb[pick].sum(1), 1), 1e-12)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(est), float((hi - lo) / 2)


def paired_obs(rows_a: Sequence[dict], rows_b: Sequence[dict], metric: str, *, n: int = 2000,
               seed: int = 0) -> dict:
    """Arm A minus arm B on the faces both scored (joined on video, frame, track):
    mean difference with a track-bootstrap 95% interval, plus McNemar's p for
    binary metrics (rank1, rank5, verified, agreements)."""
    key = lambda r: (r.get("video"), r["frame_id"], r["track_id"])  # noqa: E731
    b_by = {key(r): r for r in rows_b}
    diffs, xa, xb = [], [], []
    for r in rows_a:
        other = b_by.get(key(r))
        va, vb = obs_value(r, metric), obs_value(other, metric) if other else None
        if va is not None and vb is not None:
            diffs.append({"video": r.get("video"), "track_id": r["track_id"], "d": va - vb})
            xa.append(va)
            xb.append(vb)
    est, half, m = obs_ci(diffs, "d", n=n, seed=seed)
    binary = bool(xa) and set(xa) | set(xb) <= {0.0, 1.0}
    return {"diff": est, "half": half, "n": m,
            "excludes_zero": est is not None and half is not None and abs(est) > half,
            "mcnemar_p": mcnemar_p(np.array(xa) > 0.5, np.array(xb) > 0.5) if binary else None}


def strata(rows: Sequence[dict], key: str, edges: Sequence[float]) -> list[tuple[str, list[dict]]]:
    """Rows split into [edges[i], edges[i+1]) bins of a covariate (face_px, abs_yaw, det_conf)."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        label = f"{lo:g}–{hi:g}" if hi < 1e5 else f"≥{lo:g}"
        out.append((label, [r for r in rows if r.get(key) is not None and lo <= r[key] < hi]))
    return out


# ---------------------------------------------------------------------------
# The experiment context


class Experiment:
    def __init__(self, run_file: str, argv: Optional[Sequence[str]] = None):
        self.dir = Path(run_file).resolve().parent
        self.name = self.dir.name                      # e.g. "e0_instruments"
        self.id = self.name.split("_", 1)[0]           # e.g. "e0"
        self.results = self.dir / "results"
        self.latex = self.results / "latex"
        self.cache = REPO / "cache" / self.name
        self.config = tomllib.loads((self.dir / "config.toml").read_text())

        p = argparse.ArgumentParser(description=f"{self.name}: see experiments/{self.name}/README.md")
        p.add_argument("--video", action="append", default=None,
                       help="override config.toml inputs (repeatable): short name or path")
        p.add_argument("--fresh", action="store_true", help="also clear this experiment's cache/")
        self.args = p.parse_args(argv)

        self._metrics: dict[str, Any] = {}
        self._steps: list[dict] = []
        self._videos: dict[str, dict] = {}
        self._notes: dict[str, Any] = {}
        self._t0 = 0.0
        self._started = ""

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Experiment":
        if self.results.exists():
            shutil.rmtree(self.results)
        self.latex.mkdir(parents=True)
        if self.args.fresh and self.cache.exists():
            shutil.rmtree(self.cache)
        self._t0 = time.time()
        self._started = dt.datetime.now().astimezone().isoformat(timespec="seconds")

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
        for h in (logging.FileHandler(self.results / "run.log"), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            root.addHandler(h)
        (self.results / "commands.sh").write_text(
            f"#!/usr/bin/env bash\n# Commands run by experiments/{self.name}/run.py, {self._started}.\n"
            f"# Re-run any step by hand from the repository root.\nset -euo pipefail\ncd {shlex.quote(str(REPO))}\n")
        log.info(f"{self.name}: results -> {self.results}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        ok = exc_type is None
        if ok:
            self._write_latex_results()
        (self.results / "metrics.json").write_text(json.dumps(self._metrics, indent=2, sort_keys=True))
        self._write_provenance("done" if ok else "failed",
                               None if ok else "".join(traceback.format_exception(exc_type, exc, tb)))
        log.info(f"{self.name}: {'done' if ok else 'FAILED'} in {time.time() - self._t0:.0f}s")
        return False

    # -- inputs ------------------------------------------------------------

    def videos(self) -> list[Video]:
        names = self.args.video or self.config["inputs"]["videos"]
        out = []
        for name in names:
            v = resolve_video(name)
            if v.name not in self._videos:
                log.info(f"input {v.name}: {v.path} (hashing)")
                self._videos[v.name] = {"path": str(v.path), "sha256": _sha256(v.path)}
            out.append(v)
        return out

    # -- running components --------------------------------------------------

    def sh(self, step: str, *cmd: Any, env: Optional[dict] = None, cwd: Path = REPO) -> None:
        """Run one command, streaming its output to the console and run.log, and
        append it to commands.sh. Fails the experiment on a non-zero exit."""
        cmd = [str(c) for c in cmd]
        env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
        line = (env_prefix + " " if env_prefix else "") + shlex.join(cmd)
        with open(self.results / "commands.sh", "a") as f:
            f.write(f"\n# {step}\n{line}\n")
        log.info(f"[{step}] {line}")
        t = time.time()
        proc = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **{k: str(v) for k, v in (env or {}).items()}},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with open(self.results / "run.log", "a") as runlog:
            for out_line in proc.stdout:
                sys.stdout.write(out_line)
                runlog.write(out_line)
        rc = proc.wait()
        self._steps.append({"step": step, "command": line, "seconds": round(time.time() - t, 1), "exit": rc})
        if rc != 0:
            raise RuntimeError(f"step '{step}' failed with exit code {rc}: {line}")

    def module(self, step: str, module: str, *args: Any, env: Optional[dict] = None) -> None:
        """Run a component: `uv run python -m <module> <args>` from the repo root."""
        self.sh(step, "uv", "run", "python", "-m", module, *args, env=env)

    def note(self, key: str, value: Any) -> None:
        """Free-form facts for provenance.json (e.g. a step skipped and why)."""
        self._notes[key] = value

    # -- outputs -------------------------------------------------------------

    def metric(self, key: str, value: Any) -> None:
        """A number the paper may cite: metrics.json, and \\result{key} in results.tex
        (key made ASCII: β 1.2 -> b-1.2)."""
        self._metrics[result_key(key)] = value

    def table(self, name: str, columns: Sequence[str], rows: Iterable[Sequence[Any]], *,
              decimals: int = 3, best: Optional[dict[str, str]] = None, align: Optional[str] = None) -> None:
        """A table as results/<name>.md and a booktabs tabular in latex/tab-<id>-<name>.tex.
        Cells: numbers, strings, None, or (value, ci_halfwidth). `best` maps a column
        to "min" or "max"; that column's best value is set in bold."""
        rows = [list(r) for r in rows]
        bold: set[tuple[int, int]] = set()
        for col, how in (best or {}).items():
            j = list(columns).index(col)
            vals = [(i, _numeric(r[j])) for i, r in enumerate(rows)]
            vals = [(i, v) for i, v in vals if v is not None]
            if vals:
                target = (min if how == "min" else max)(v for _, v in vals)
                bold |= {(i, j) for i, v in vals if v == target}
        align = align or "l" + "r" * (len(columns) - 1)

        tex = [f"% generated by experiments/{self.name}/run.py — do not edit",
               f"\\begin{{tabular}}{{{align}}}", "\\toprule",
               " & ".join(latex_escape(c) for c in columns) + " \\\\", "\\midrule"]
        for i, r in enumerate(rows):
            cells = [_fmt(v, decimals) for v in r]
            cells = [f"\\textbf{{{c}}}" if (i, j) in bold else c for j, c in enumerate(cells)]
            tex.append(" & ".join(cells) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}", ""]
        (self.latex / f"tab-{self.id}-{name}.tex").write_text("\n".join(tex))

        md_lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for i, r in enumerate(rows):
            cells = [_plain(v, decimals) for v in r]
            cells = [f"**{c}**" if (i, j) in bold else c for j, c in enumerate(cells)]
            md_lines.append("| " + " | ".join(cells) + " |")
        (self.results / f"{name}.md").write_text("\n".join(md_lines) + "\n")

    @contextlib.contextmanager
    def figure(self, name: str, rows: Sequence[dict], *, height_in: float = 2.4, width_in: float = FIG_WIDTH_IN):
        """Yields a matplotlib Figure styled for the paper; on exit saves
        latex/fig-<id>-<name>.pdf and the plotted data as latex/fig-<id>-<name>.csv."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        style = {"font.family": "serif", "font.serif": ["STIXGeneral", "STIX Two Text", "DejaVu Serif"],
                 "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                 "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                 "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                 "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                 "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                 "axes.prop_cycle": matplotlib.cycler(color=SERIES), "lines.linewidth": 1.8,
                 "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02}
        with plt.rc_context(style):
            fig = plt.figure(figsize=(width_in, height_in))
            yield fig
            fig.savefig(self.latex / f"fig-{self.id}-{name}.pdf")
            plt.close(fig)
        keys: list[str] = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(self.latex / f"fig-{self.id}-{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # -- private -------------------------------------------------------------

    def _write_latex_results(self) -> None:
        sha = _git("rev-parse", "--short", "HEAD") or "unknown"
        lines = [f"% generated by experiments/{self.name}/run.py, {self._started}, git {sha} — do not edit",
                 "\\makeatletter",
                 "\\providecommand\\result[1]{\\@ifundefined{result@#1}"
                 "{\\PackageError{results}{Unknown result key `#1'}"
                 "{Run the experiment, then experiments/export_paper.sh.}}{\\@nameuse{result@#1}}}"]
        for key, value in sorted(self._metrics.items()):
            if _numeric(value) is None and not isinstance(value, str):
                continue
            lines.append(f"\\@namedef{{result@{key}}}{{{_fmt(value, 3)}}}")
        lines += ["\\makeatother", ""]
        (self.latex / "results.tex").write_text("\n".join(lines))

    def _write_provenance(self, status: str, error: Optional[str]) -> None:
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        if dirty:
            (self.results / "code.diff").write_text(_git("diff", "HEAD") or "")
        bridge = Path(os.environ.get("BLANKET_REPO", Path.home() / "projects" / "blanket-anonymizer-bridge"))
        prov = {
            "experiment": self.name,
            "status": status,
            "error": error,
            "argv": sys.argv,
            "started": self._started,
            "seconds": round(time.time() - self._t0, 1),
            "config": self.config,
            "inputs": self._videos,
            "steps": self._steps,
            "notes": self._notes,
            "git": {"sha": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                    "dirty": dirty},
            "bridge": {"path": str(bridge), "sha": _git("rev-parse", "HEAD", cwd=bridge) if bridge.is_dir() else None},
            "host": platform.node(),
            "gpus": _gpus(),
            "env": {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "IDENTITY_GPU", "SWAP_GPU", "CTX_ID",
                                                "BLANKET_REPO") if k in os.environ},
            "versions": _versions(),
        }
        (self.results / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
