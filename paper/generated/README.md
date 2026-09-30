# paper/generated/

LaTeX artifacts produced by the paper experiments (`experiments/`), copied here
by `experiments/export_paper.sh`. **Do not edit by hand**: rerun the experiment
and export again. `SOURCES.md` records which run (commit, date, input hashes)
each experiment's files come from.

Only aggregate artifacts land here: booktabs tabulars, vector figures with their
data, and the `\result{}` keys. Never faces or frames.

| File | Use in `main.tex` |
|---|---|
| `results.tex` | `\input{generated/results}` once in the preamble: defines `\result{key}` for every exported experiment |
| `<experiment>/tab-*.tex` | `\input{generated/<experiment>/tab-…}` inside a `table` environment; the caption and label stay in the paper |
| `<experiment>/fig-*.pdf` | `\includegraphics[width=\linewidth]{generated/<experiment>/fig-…}`; the `.csv` next to it holds the plotted data |

`\result{key}` stops the compilation with an error when the key does not exist,
so a number cited in the text cannot silently lose its source. The keys are
listed in each experiment's `results.tex` and `metrics.json`.
