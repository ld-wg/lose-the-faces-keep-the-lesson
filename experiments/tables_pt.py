"""Portuguese copies of the exported tables, for the thesis (paper/tcc-latex).

    python3 experiments/tables_pt.py          # run by experiments/export_paper.sh

For every paper/generated/<experiment>/tab-*.tex, writes
paper/generated/<experiment>/pt/tab-*.tex with the header row and the row labels
translated, internal arm names replaced by descriptive ones, and decimal commas.
The English originals stay as they are (the SBC article draft uses them).

Standard library only: it rewrites text, it does not recompute anything.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper" / "generated"

HEADERS = {
    "$\\Delta$ Emotion $\\kappa$": "$\\Delta$ $\\kappa$ emoção", "$\\Delta$ Motion": "$\\Delta$ Movim.",
    "$\\Delta$ R1 vs p2": "$\\Delta$ R1 vs P2", "$\\Delta$ Verified": "$\\Delta$ Verif.",
    "$\\Delta$ vs ciagan": "$\\Delta$ vs CIAGAN", "$\\Delta$ vs first": "$\\Delta$ vs 1º quadro",
    "$\\Delta$ vs native": "$\\Delta$ vs nativo", "$\\times$ ref": "$\\times$ ref.",
    "AP easy": "AP fácil", "AP hard": "AP difícil", "AP medium": "AP médio", "Age (y)": "Idade (anos)",
    "Arm": "Braço", "Check": "Verificação", "Coverage": "Cobertura", "Detection conf.": "Conf. de detecção",
    "Emotion $\\kappa$": "$\\kappa$ emoção", "FPS": "Quadros/s", "Face size (px)": "Tamanho do rosto (px)",
    "Faces/frame": "Rostos/quadro", "Frames / s": "Quadros/s", "Gen.": "Ger.", "Gender": "Gênero",
    "Genuine": "Genuínos", "Held-out cos": "Cos. held-out", "Hidden": "Ocultos",
    "Reused": "Reutilizados", "Identity unusable": "Ident. inutilizável", "Impostor": "Impostores", "Latency (ms)": "Latência (ms)",
    "Median": "Mediana", "Metric (consecutive real frames)": "Métrica (quadros reais consecutivos)",
    "Mode": "Modo", "Motion": "Movim.", "PG final": "Red. rel. R1", "Rel. red.": "Red. rel. R1", "Pairs": "Pares",
    "Params (M)": "Parâm. (M)", "Policy": "Política", "Quality-weighted cos": "Cos. ponderado",
    "R1 all": "R1 todos", "R1 gen.": "R1 ger.", "Recognizer": "Reconhecedor", "Short": "Curtos",
    "Swap": "Troca", "Swap no face": "Troca sem rosto", "Threshold": "Limiar", "Value": "Valor",
    "Verified": "Verificados", "Video": "Vídeo", "Videos": "Vídeos", "ms/face": "ms/rosto",
    # experiments/derived.py
    "Series": "Série", "Macro": "Macro", "Held-out video": "Vídeo de fora", "$\\beta$ chosen": "$\\beta$ escolhido",
    "R1 native": "R1 nativo", "R1 chosen": "R1 escolhido", "Best $\\beta$ (R1)": "Melhor $\\beta$ (R1)",
    "Method": "Método", "View": "Visão", "Synthetic share": "Com face sintética", "R1 among them": "R1 entre eles",
    "cos($\\hat e$, synth.)": "cos($\\hat e$, sint.)", "cos($\\hat e$, real)": "cos($\\hat e$, real)",
    "FaceNet R1": "R1 FaceNet", "ArcFace R1": "R1 ArcFace",
}

LABELS = {
    # arms (internal names -> descriptive)
    "blanket": "BLANKET", "ciagan": "CIAGAN", "ganonymization": "GANonymization", "mosaic": "mosaico",
    "native": "mix nativo", "none": "sem mix", "candidates": "candidatos", "phase1": "detecção do Estágio 1",
    "p2": "P2", "hide-low": "oculta conf. < 0,5", "gate": "privacy gate",
    "first": "primeiro quadro", "best": "melhor quadro", "mean": "média", "quality\\_mean": "média ponderada",
    "ema\\_adaptive": "EMA adaptativa", "ref": "referência", "ref-cpu": "referência (CPU)",
    "enh-off": "sem GFPGAN", "enh-96": "GFPGAN $\\geq$ 96 px", "every2": "1 a cada 2", "every3": "1 a cada 3",
    "adapt-0.05": "adaptativa 0,05", "adapt-0.08": "adaptativa 0,08", "fast": "rápida",
    "Real, next frame": "Real, quadro seguinte", "P1 c 0.1 + P2": "P1 (0,1) + P2",
    # E0 rows
    "Age difference (years)": "Diferença de idade (anos)", "ArcFace cosine": "Cosseno ArcFace",
    "Emotion agreement": "Concordância de emoção",
    "Emotion agreement (Cohen's $\\kappa$)": "Concordância de emoção ($\\kappa$ de Cohen)",
    "Expression error": "Erro de expressão", "FaceNet cosine": "Cosseno FaceNet",
    "Faces checked": "Rostos verificados", "Gender agreement": "Concordância de gênero",
    "Head-pose error (\\textdegree{})": "Erro de pose (\\textdegree{})",
    "Recognizer equivalence, mean cosine": "Equivalência dos reconhecedores, cosseno médio",
    "Recognizer equivalence, minimum cosine": "Equivalência dos reconhecedores, cosseno mínimo",
    "emap direction change, mean cosine": "Mudança de direção pela emap, cosseno médio",
    "emap distance from orthogonal": "Distância da emap à ortogonal",
    # experiments/derived.py series
    "blur, final video": "blur, vídeo final", "CIAGAN, final video": "CIAGAN, vídeo final",
    "BLANKET (released), final video": "BLANKET (código liberado), vídeo final",
    "BLANKET native mix, generated": "BLANKET mix nativo, gerados",
    "P2 $\\beta$ 1.2, generated": "P2 $\\beta$ 1,2, gerados", "P2 $\\beta$ 1.6, generated": "P2 $\\beta$ 1,6, gerados",
    "final pipeline (gate), final video": "pipeline final (gate), vídeo final",
    "final pipeline, composed video": "pipeline final, vídeo composto",
    "final pipeline + adaptive, composed video": "pipeline final + adaptativa, vídeo composto",
    "BLANKET (released)": "BLANKET (código liberado)",
    "BLANKET native mix, Stage 1 detection": "BLANKET mix nativo (Estágio 1)",
    "final pipeline": "pipeline final", "final pipeline + adaptive": "pipeline final + adaptativa",
    "final pipeline, GFPGAN off": "pipeline final sem GFPGAN", "final pipeline (gate)": "pipeline final (gate)",
}

CELLS = {"yes": "sim", "no": "não", "generated": "gerados", "final video, shown": "vídeo final",
         "composed video, shown": "vídeo composto"}

# a decimal number not preceded by a letter, digit, dot or backslash (keeps \hspace{5pt}, keys)
DECIMAL = re.compile(r"(?<![A-Za-z0-9.\\])([+-]?\d*)\.(\d+)")


def decimal_comma(text: str) -> str:
    return DECIMAL.sub(lambda m: f"{m.group(1)}{{,}}{m.group(2)}", text)


def translate(tex: str) -> str:
    lines = tex.splitlines()
    out, state = [], "preamble"
    for line in lines:
        if line.startswith("% generated by"):
            out.append(line + " (pt: experiments/tables_pt.py)")
            continue
        if line == "\\toprule":
            state = "header"
            out.append(line)
            continue
        if line == "\\midrule":
            state = "body"
            out.append(line)
            continue
        if line.startswith("\\bottomrule"):
            state = "end"
            out.append(line)
            continue
        if state in ("header", "body") and line.endswith("\\\\"):
            cells = [c.strip() for c in line[:-2].rstrip().split(" & ")]
            if state == "header":
                cells = [HEADERS.get(c, c) for c in cells]
            else:
                cells[0] = LABELS.get(cells[0], cells[0])
                cells = [CELLS.get(c, c) for c in cells]
            out.append(decimal_comma(" & ".join(cells)) + " \\\\")
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if tex.endswith("\n") else "")


def main() -> int:
    written = 0
    for tab in sorted(GENERATED.glob("*/tab-*.tex")):
        dest = tab.parent / "pt" / tab.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(translate(tab.read_text(encoding="utf-8")), encoding="utf-8")
        written += 1
    print(f"tables_pt: {written} tables -> paper/generated/*/pt/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
