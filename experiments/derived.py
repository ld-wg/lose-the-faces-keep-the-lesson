"""Analyses derived from the exported \\result keys, no data needed (advisor review, 2026-10-29).

    python3 experiments/derived.py            # run by experiments/export_paper.sh

Reads paper/generated/e*_*/results.tex and writes paper/generated/derived/:
- results.tex: \\result{derived/...} keys
  - chance level per video: 1/N, N = tracks of the video's gallery;
  - rank-1 per video, as a multiple of chance, and its macro mean (videos weighted
    equally; the pooled values weight them by faces, so demo3 counts 54%);
  - leave-one-video-out choice of the P2 strength β: chosen on the other two videos
    (face-weighted mean of FaceNet rank-1), scored on the held-out one. It tests whether
    E3c's choice of β survives outside the videos it was made on.
- tab-derived-chance.tex, tab-derived-lovo.tex (+ pt/ copies via tables_pt.py).

Added with the 2026-10-30 critical review (research/next-steps/analise-critica-banca-2026-10-29.md):
- privacy x coverage frontier: the final-video rank-1 counts a gray fill as never
  re-identified, so a method that hides most faces wins it (the released BLANKET shows 28%).
  Each method as a pair (share of faces with a synthetic face, rank-1 among them);
  tab-derived-frontier.tex and pt/fig-derived-frontier.tex.
- geometry of the P2 push: with c = cos(e_synth, e_real) (E3g identity images),
  normalize(e_synth - beta e_real) has cosine (1 - beta c)/n with the synthetic identity and
  (c - beta)/n with the real one, n = sqrt(1 + beta^2 - 2 beta c). beta = c removes the real
  component; beyond it the target goes towards the antipode; tab-derived-p2geom.tex.
- the verification target (<= 5% at FAR 1%) against each 95% interval.
- Portuguese pgfplots versions of the E3 and E5 figures from their exported CSVs
  (pt/fig-e3-p2-curve.tex, pt/fig-e5-tradeoff.tex): no server needed.

Track counts come from the e3/<video>/tracks keys (E3 exports them from tracks.json); for
runs older than that, from experiments/derived.toml, measured from the same tracks.json files.
Standard library only.
"""

from __future__ import annotations

import csv
import math
import re
import sys
import tomllib
from pathlib import Path

from tables_pt import LABELS

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper" / "generated"
OUT = GENERATED / "derived"
VIDEOS = ("demo1", "demo2", "demo3")
BETAS = ("b-0.2", "b-0.35", "b-0.5", "b-0.8", "b-1.2", "b-1.6")

# rank-1 series: (key part, label, key pattern with {v} for the video)
SERIES = [
    ("blur-final", "blur, final video", "e2/{v}/blur/facenet_final.rank1"),
    ("ciagan-final", "CIAGAN, final video", "e2/{v}/ciagan/facenet_final.rank1"),
    ("blanket-final", "BLANKET (released), final video", "e2/{v}/blanket/facenet_final.rank1"),
    ("native-generated", "BLANKET native mix, generated", "e3c/{v}/native/facenet.rank1"),
    ("p2-1.2-generated", "P2 $\\beta$ 1.2, generated", "e3c/{v}/b-1.2/facenet.rank1"),
    ("p2-1.6-generated", "P2 $\\beta$ 1.6, generated", "e3c/{v}/b-1.6/facenet.rank1"),
    ("gate-final", "final pipeline (gate), final video", "e3e/{v}/gate/facenet_final.rank1"),
    ("ref-video", "final pipeline, composed video", "e5/{v}/ref/facenet_video.rank1"),
    ("adapt-video", "final pipeline + adaptive, composed video", "e5/{v}/adapt-0.08/facenet_video.rank1"),
]


def read_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    for f in sorted(GENERATED.glob("e*_*/results.tex")):
        for k, v in re.findall(r"\\@namedef\{result@([^}]*)\}\{([^}]*)\}", f.read_text(encoding="utf-8")):
            keys[k] = v
    return keys


def num(keys: dict[str, str], key: str) -> float | None:
    try:
        return float(keys[key])
    except (KeyError, ValueError):
        return None


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> int:
    keys = read_keys()
    fallback = tomllib.loads((Path(__file__).with_name("derived.toml")).read_text())
    tracks = {v: num(keys, f"e3/{v}/tracks") or fallback["tracks"][v] for v in VIDEOS}
    from_keys = all(num(keys, f"e3/{v}/tracks") for v in VIDEOS)
    # face counts per video for the leave-one-video-out weights: detector faces of E3 (derived.toml)
    faces = {v: fallback["faces"][v] for v in VIDEOS}

    out: dict[str, str] = {}
    for v in VIDEOS:
        out[f"derived/{v}/tracks"] = str(int(tracks[v]))
        out[f"derived/{v}/chance-rank1"] = fmt(1 / tracks[v])

    chance_rows = []
    for name, label, pat in SERIES:
        vals = {v: num(keys, pat.format(v=v)) for v in VIDEOS}
        if any(x is None for x in vals.values()):
            continue
        mult = {v: vals[v] * tracks[v] for v in VIDEOS}
        macro = sum(vals.values()) / len(VIDEOS)
        for v in VIDEOS:
            out[f"derived/{name}/{v}/rank1"] = fmt(vals[v])
            out[f"derived/{name}/{v}/x-chance"] = fmt(mult[v], 1)
        out[f"derived/{name}/macro"] = fmt(macro)
        out[f"derived/{name}/x-chance-min"] = fmt(min(mult.values()), 1)
        out[f"derived/{name}/x-chance-max"] = fmt(max(mult.values()), 1)
        chance_rows.append([label, *(f"{fmt(vals[v], 2)} ({fmt(mult[v], 1)}$\\times$)" for v in VIDEOS),
                            fmt(macro, 2)])

    lovo_rows = []
    for held in VIDEOS:
        others = [v for v in VIDEOS if v != held]

        def score(arm: str) -> float:
            return sum(num(keys, f"e3c/{o}/{arm}/facenet.rank1") * faces[o] for o in others) / sum(faces[o] for o in others)

        if any(num(keys, f"e3c/{v}/{a}/facenet.rank1") is None for v in VIDEOS for a in (*BETAS, "native")):
            break
        chosen = min(BETAS, key=score)
        native = num(keys, f"e3c/{held}/native/facenet.rank1")
        got = num(keys, f"e3c/{held}/{chosen}/facenet.rank1")
        oracle = min(BETAS, key=lambda a: num(keys, f"e3c/{held}/{a}/facenet.rank1"))
        beta = chosen.removeprefix("b-")
        out[f"derived/lovo/{held}/beta"] = beta
        out[f"derived/lovo/{held}/rank1"] = fmt(got)
        out[f"derived/lovo/{held}/native"] = fmt(native)
        out[f"derived/lovo/{held}/delta"] = f"{got - native:+.3f}"
        out[f"derived/lovo/{held}/oracle-beta"] = oracle.removeprefix("b-")
        out[f"derived/lovo/{held}/oracle-rank1"] = fmt(num(keys, f"e3c/{held}/{oracle}/facenet.rank1"))
        lovo_rows.append([held, beta, fmt(native, 3), fmt(got, 3), f"{got - native:+.3f}",
                          f"{oracle.removeprefix('b-')} ({fmt(num(keys, f'e3c/{held}/{oracle}/facenet.rank1'), 3)})"])

    frontier_rows = frontier(keys, out)
    geom_rows = p2_geometry_rows(keys, out)
    verification_targets(keys, out)

    OUT.mkdir(parents=True, exist_ok=True)
    note = ("tracks from e3/<video>/tracks keys" if from_keys
            else "tracks from experiments/derived.toml (tracks.json, 2026-10-21); E3 exports them from its next run")
    lines = ["% generated by experiments/derived.py — do not edit; " + note, "\\makeatletter",
             "\\providecommand\\result[1]{\\@ifundefined{result@#1}"
             "{\\PackageError{results}{Unknown result key `#1'}"
             "{Run the experiment, then experiments/export_paper.sh.}}{\\@nameuse{result@#1}}}"]
    lines += [f"\\@namedef{{result@{k}}}{{{v}}}" for k, v in sorted(out.items())]
    lines += ["\\makeatother", ""]
    (OUT / "results.tex").write_text("\n".join(lines), encoding="utf-8")

    def table(name: str, columns: list[str], rows: list[list[str]], align: str) -> None:
        tex = ["% generated by experiments/derived.py — do not edit", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
               " & ".join(columns) + " \\\\", "\\midrule", *(" & ".join(r) + " \\\\" for r in rows),
               "\\bottomrule", "\\end{tabular}", ""]
        (OUT / f"tab-derived-{name}.tex").write_text("\n".join(tex), encoding="utf-8")

    table("chance", ["Series", *(f"{v} (N={int(tracks[v])})" for v in VIDEOS), "Macro"], chance_rows, "lrrrr")
    table("lovo", ["Held-out video", "$\\beta$ chosen", "R1 native", "R1 chosen", "$\\Delta$", "Best $\\beta$ (R1)"],
          lovo_rows, "lrrrrr")
    table("frontier", ["Method", "View", "Synthetic share", "R1 among them"],
          [[r["label"], r["view"], fmt(r["share"]), fmt(r["rank1"])] for r in frontier_rows], "llrr")
    table("p2geom", ["$\\beta$", "cos($\\hat e$, synth.)", "cos($\\hat e$, real)", "FaceNet R1", "ArcFace R1"],
          geom_rows, "rrrrr")
    figures = {"fig-derived-frontier.tex": frontier_figure(frontier_rows),
               "fig-e3-p2-curve.tex": p2_curve_figure(keys),
               "fig-e5-tradeoff.tex": e5_tradeoff_figure()}
    (OUT / "pt").mkdir(exist_ok=True)
    for name, tex in figures.items():
        if tex:
            (OUT / "pt" / name).write_text(tex, encoding="utf-8")
    print(f"derived: {len(out)} keys, {len(chance_rows)} chance rows, {len(lovo_rows)} LOVO rows, "
          f"{len(frontier_rows)} frontier points, {sum(1 for t in figures.values() if t)} pt figures ({note})")
    return 0


# ---------------------------------------------------------------------------------------------
# 2026-10-30 critical review


def frontier(keys: dict[str, str], out: dict[str, str]) -> list[dict]:
    """(share of faces with a synthetic face, FaceNet rank-1 among them) per method.
    E2/E3: the generated faces (what the method replaced). E5: the composed video's shown
    faces, exact because a filled box is never rank-1: rank-1 / (1 - filled share)."""
    points = []
    for name, label in (("blur", "blur"), ("mosaic", "mosaic"), ("ciagan", "CIAGAN"),
                        ("ganonymization", "GANonymization"), ("blanket", "BLANKET (released)")):
        share, r1 = num(keys, f"e2/pooled/{name}/anonymized"), num(keys, f"e2/pooled/{name}/generated/facenet.rank1")
        if share is not None and r1 is not None:
            points.append({"key": f"e2-{name}", "label": label, "view": "generated", "share": share, "rank1": r1})
    for arm, label in (("native", "BLANKET native mix, Stage 1 detection"), ("b-1.2", "P2 $\\beta$ 1.2"),
                       ("b-1.6", "P2 $\\beta$ 1.6")):
        share, r1 = num(keys, f"e3c/pooled/{arm}/anonymized"), num(keys, f"e3c/pooled/{arm}/facenet.rank1")
        if share is not None and r1 is not None:
            points.append({"key": f"e3c-{arm}", "label": label, "view": "generated", "share": share, "rank1": r1})
    # once E3 is re-evaluated (shown / facenet_shown.rank1 keys), the final pipeline's own point
    share, r1 = num(keys, "e3e/pooled/gate/shown"), num(keys, "e3e/pooled/gate/facenet_shown.rank1")
    if share is not None and r1 is not None:
        points.append({"key": "e3e-gate", "label": "final pipeline (gate)", "view": "final video, shown",
                       "share": share, "rank1": r1})
    for arm, label in (("ref", "final pipeline"), ("adapt-0.08", "final pipeline + adaptive"),
                       ("enh-off", "final pipeline, GFPGAN off")):
        hidden, r1 = num(keys, f"e5/pooled/{arm}/hidden-share"), num(keys, f"e5/pooled/{arm}/facenet_video.rank1")
        if hidden is not None and r1 is not None and hidden < 1:
            points.append({"key": f"e5-{arm}", "label": label, "view": "composed video, shown",
                           "share": 1 - hidden, "rank1": r1 / (1 - hidden)})
    for p in points:
        out[f"derived/frontier/{p['key']}/share"] = fmt(p["share"])
        out[f"derived/frontier/{p['key']}/rank1"] = fmt(p["rank1"])
    return points


def geometry(c: float, beta: float) -> tuple[float, float]:
    """cos of normalize(e_synth - beta e_real) with e_synth and with e_real, cos(e_synth, e_real) = c."""
    n = math.sqrt(1 + beta * beta - 2 * beta * c)
    return (1 - beta * c) / n, (c - beta) / n


def p2_geometry_rows(keys: dict[str, str], out: dict[str, str]) -> list[list[str]]:
    c = num(keys, "e3g/pooled/identity-cos.plain")
    if c is None:
        return []
    out["derived/p2geom/c"] = fmt(c)
    out["derived/p2geom/orthogonal-beta"] = fmt(c, 2)
    rows = []
    for beta in (0.0, c, 0.35, 0.5, 0.8, 1.2, 1.6):
        cs, cr = geometry(c, beta)
        arm = "none" if beta == 0 else f"b-{beta:g}" if beta != c else None
        fn = num(keys, f"e3c/pooled/{arm}/facenet.rank1") if arm else None
        af = num(keys, f"e3c/pooled/{arm}/arcface.rank1") if arm else None
        if beta != c and beta > 0:
            out[f"derived/p2geom/b-{beta:g}/cos-synth"] = fmt(cs, 2)
            out[f"derived/p2geom/b-{beta:g}/cos-real"] = fmt(cr, 2)
        rows.append([f"{beta:.2f}" + (" ($\\perp$)" if beta == c else ""), fmt(cs, 2), fmt(cr, 2),
                     fmt(fn) if fn is not None else "--", fmt(af) if af is not None else "--"])
    return rows


def verification_targets(keys: dict[str, str], out: dict[str, str], target: float = 0.05) -> None:
    """Whether the 95% interval of the verification rate lies below, above or around the target."""
    for name, key in (("e3e-gate", "e3e/pooled/gate/facenet_final.verified"),
                      ("e5-ref", "e5/pooled/ref/facenet_video.verified")):
        v, h = num(keys, key), num(keys, key + ".ci")
        if v is None or h is None:
            continue
        lo, hi = max(0.0, v - h), v + h
        out[f"derived/target/{name}/verified-low"] = fmt(lo)
        out[f"derived/target/{name}/verified-high"] = fmt(hi)
        out[f"derived/target/{name}/verdict"] = ("met" if hi < target else "missed" if lo > target
                                                 else "undecided")


# --- Portuguese pgfplots figures (thesis only) ------------------------------------------------

PALETTE = ["2A78D6", "EB6834", "1BAF7A", "EDA100", "E87BA4", "008300", "4A3AA7", "E34948"]
HEADER = ("% generated by experiments/derived.py — do not edit; Portuguese, pgfplots\n"
          + "".join(f"\\definecolor{{lftl{i}}}{{HTML}}{{{c}}}\n" for i, c in enumerate(PALETTE)))
AXIS = ("/pgf/number format/use comma, /pgf/number format/1000 sep={}, tick label style={font=\\footnotesize}, "
        "label style={font=\\footnotesize}, title style={font=\\footnotesize}, axis lines*=left, "
        "grid=major, grid style={draw=black!10}")


def lab(text: str) -> str:
    return LABELS.get(text, text)


def frontier_figure(points: list[dict]) -> str:
    if not points:
        return ""
    groups = {"e2-blur": 0, "e2-mosaic": 0, "e2-ciagan": 1, "e2-ganonymization": 1, "e2-blanket": 2,
              "e3c-native": 2, "e3c-b-1.2": 3, "e3c-b-1.6": 3, "e3e-gate": 4, "e5-ref": 4,
              "e5-adapt-0.08": 4, "e5-enh-off": 5}
    names = ["censura", "GAN", "BLANKET", "P2", "pipeline final", "pipeline final sem GFPGAN"]
    # labelled points: (key, anchor); the censor/CIAGAN cluster at full coverage gets one label
    labels = {"e2-ganonymization": ("GANonymization", "west"), "e2-blanket": ("BLANKET liberado", "west"),
              "e3c-native": ("mix nativo", "east"), "e3c-b-1.2": ("P2 $\\beta$\\,1,2", "south west"),
              "e3c-b-1.6": ("P2 $\\beta$\\,1,6", "north west"), "e5-ref": ("pipeline final", "east"),
              "e5-enh-off": ("sem GFPGAN", "east"), "e3e-gate": ("pipeline final (E3)", "east")}
    lines = [HEADER, "\\begin{tikzpicture}", "\\begin{axis}[width=0.8\\linewidth, height=6.4cm, " + AXIS + ",",
             "  xmin=0.2, xmax=1.18, ymin=0, ymax=0.6, xlabel={rostos com face sintética (parcela)},",
             "  ylabel={FaceNet rank-1 entre eles}, legend style={font=\\scriptsize, draw=none, at={(0.5,-0.3)}, anchor=north},",
             "  legend columns=3, legend cell align=left]"]
    for g, name in enumerate(names):
        pts = [p for p in points if groups.get(p["key"]) == g]
        if not pts:
            continue
        coords = " ".join(f"({p['share']:.4f},{p['rank1']:.4f})" for p in pts)
        lines.append(f"\\addplot[only marks, mark=*, mark size=2.2pt, color=lftl{g}] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{name}}}")
    for p in points:
        if p["key"] in labels:
            text, anchor = labels[p["key"]]
            lines.append(f"\\node[anchor={anchor}, font=\\scriptsize, inner sep=3pt] at "
                         f"(axis cs:{p['share']:.4f},{p['rank1']:.4f}) {{{text}}};")
    cluster = [p for p in points if p["key"] in ("e2-blur", "e2-mosaic", "e2-ciagan")]
    if cluster:
        y = sum(p["rank1"] for p in cluster) / len(cluster)
        lines.append(f"\\node[anchor=east, font=\\scriptsize, inner sep=4pt] at (axis cs:1.0,{y:.4f}) "
                     "{blur, mosaico, CIAGAN};")
    lines += ["\\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def p2_curve_figure(keys: dict[str, str]) -> str:
    rows = read_csv(GENERATED / "e3_contribution" / "fig-e3-p2-curve.csv")
    if not rows:
        return ""

    def series(rec):
        return [(float(r["beta"]), float(r["rank1"]), float(r["ci"])) for r in rows if r["recognizer"] == rec]

    def plot(rec, color, legend=None):
        coords = " ".join(f"({b:g},{y:.4f}) +- (0,{h:.4f})" for b, y, h in series(rec))
        s = (f"\\addplot[color={color}, mark=*, mark size=1.6pt, error bars/.cd, y dir=both, y explicit] "
             f"coordinates {{{coords}}};")
        return s + (f"\n\\addlegendentry{{{legend}}}" if legend else "")

    def native(key, color):
        v = num(keys, key)
        return ("" if v is None else
                f"\\addplot[color={color}, dashed, forget plot, domain=0:1.7, samples=2] {{{v:.4f}}};")

    return "\n".join([
        HEADER, "\\begin{tikzpicture}",
        "\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=2.3cm}, width=0.44\\linewidth, "
        "height=5cm, " + AXIS + ", xlabel={força do empurrão $\\beta$}, xmin=0, xmax=1.7]",
        "\\nextgroupplot[ylabel={rank-1 (rostos gerados)}, ymin=0, legend style={font=\\scriptsize, draw=none, "
        "fill=none}, legend pos=south west]",
        plot("facenet", "lftl0", "FaceNet (avaliador)"), plot("arcface", "lftl1", "ArcFace (otimizado)"),
        native("e3c/pooled/native/facenet.rank1", "lftl0"), native("e3c/pooled/native/arcface.rank1", "lftl1"),
        "\\nextgroupplot[ylabel={erro de expressão (IOD)}, ymin=0, scaled y ticks=false, yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2}]",
        plot("expression", "lftl2"), native("e3c/pooled/native/expression", "lftl2"),
        "\\end{groupplot}", "\\end{tikzpicture}", ""])


def e5_tradeoff_figure() -> str:
    rows = read_csv(GENERATED / "e5_cost" / "fig-e5-tradeoff.csv")
    if not rows:
        return ""
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    group = {"ref": 0, "ref-cpu": 0, "enh-off": 1, "enh-96": 1, "every2": 2, "every3": 2,
             "adapt-0.05": 3, "adapt-0.08": 3, "fast": 4}
    # symbolic coordinates must be plain: ASCII ids on the axis, Portuguese labels as tick labels
    names = {a: "arm" + re.sub(r"[^A-Za-z0-9]", "", a) for a in arms}
    coords_y = ",".join(names[a] for a in arms)
    ticklabels = ",".join("{" + lab(a) + "}" for a in arms)
    panels = [("class_hours", "GPU-h / hora de aula"), ("facenet_video.rank1", "FaceNet rank-1"),
              ("emotion_kappa_video", "$\\kappa$ emoção"), ("motion_err_video", "erro de movimento")]
    lines = [HEADER, "\\begin{tikzpicture}",
             "\\begin{groupplot}[group style={group size=4 by 1, horizontal sep=0.35cm, yticklabels at=edge left}, "
             "width=0.255\\linewidth, height=6cm, scaled x ticks=false, xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2}, " + AXIS + ", y dir=reverse, "
             f"symbolic y coords={{{coords_y}}}, ytick={{{coords_y}}}, yticklabels={{{ticklabels}}}, "
             "enlarge y limits=0.06]"]
    for metric, title in panels:
        lines.append(f"\\nextgroupplot[title={{{title}}}" + (", xmin=0" if metric == "class_hours" else "") + "]")
        for a in arms:
            r = next((x for x in rows if x["arm"] == a and x["metric"] == metric), None)
            if r is None:
                continue
            color = f"lftl{group.get(a, 5)}"
            v = float(r["value"])
            if metric == "class_hours":
                lines.append(f"\\addplot[xbar, bar width=4pt, fill={color}, draw=none] coordinates {{({v:.2f},{names[a]})}};")
            else:
                h = float(r["ci"] or 0)
                lines.append(f"\\addplot[only marks, mark=*, mark size=1.6pt, color={color}, error bars/.cd, "
                             f"x dir=both, x explicit] coordinates {{({v:.4f},{names[a]}) +- ({h:.4f},0)}};")
    lines += ["\\end{groupplot}", "\\end{tikzpicture}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
