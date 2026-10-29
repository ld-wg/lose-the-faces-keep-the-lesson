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

Track counts come from the e3/<video>/tracks keys (E3 exports them from tracks.json); for
runs older than that, from experiments/derived.toml, measured from the same tracks.json files.
Standard library only.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

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
    print(f"derived: {len(out)} keys, {len(chance_rows)} chance rows, {len(lovo_rows)} LOVO rows ({note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
