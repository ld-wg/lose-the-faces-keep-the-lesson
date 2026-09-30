#!/usr/bin/env bash
# Compile every generated table, figure and \result{} key in a document with the
# paper's page geometry (SBC template: article, 12pt, 160 mm text width), and
# fail if anything does not compile or overflows the text width.
#
#   experiments/check_latex.sh                  # every experiment's results/latex/
#   experiments/check_latex.sh paper/generated  # what the paper will include
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=${1:-$root/experiments}
command -v pdflatex >/dev/null || { echo "pdflatex not found" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mapfile -t tables < <(find "$src" -path '*latex*' -name 'tab-*.tex' -o -path '*generated*' -name 'tab-*.tex' | sort -u)
mapfile -t figures < <(find "$src" -path '*latex*' -name 'fig-*.pdf' -o -path '*generated*' -name 'fig-*.pdf' | sort -u)
mapfile -t results < <(find "$src" -mindepth 2 -name 'results.tex' | sort -u)
[[ ${#tables[@]} -gt 0 || ${#figures[@]} -gt 0 ]] || { echo "nothing to check under $src" >&2; exit 1; }

{
  echo '\documentclass[12pt]{article}'
  echo '\usepackage{booktabs}'
  echo '\usepackage{graphicx,url}'
  echo '\usepackage[utf8]{inputenc}'
  echo '\setlength{\textwidth}{160mm}\setlength{\oddsidemargin}{0mm}\setlength{\evensidemargin}{0mm}'
  for r in "${results[@]}"; do echo "\\input{$r}"; done
  echo '\begin{document}'
  for r in "${results[@]}"; do
    sed -n 's/^\\@namedef{result@\(.*\)}{.*}$/\1/p' "$r" | while read -r key; do
      echo "\\noindent\\texttt{\\detokenize{$key}} = \\result{$key}\\par"
    done
  done
  for t in "${tables[@]}"; do
    echo "\\begin{table}[ht]\\centering\\caption{\\detokenize{$(basename "$t")}}\\input{$t}\\end{table}"
  done
  for f in "${figures[@]}"; do
    echo "\\begin{figure}[ht]\\centering\\includegraphics[width=\\linewidth]{$f}\\caption{\\detokenize{$(basename "$f")}}\\end{figure}"
  done
  echo '\end{document}'
} > "$tmp/check.tex"

if ! (cd "$tmp" && pdflatex -interaction=nonstopmode -halt-on-error check.tex > check.log 2>&1); then
  grep -m5 -A3 '^!' "$tmp/check.log" >&2
  echo "✗ LaTeX does not compile" >&2
  exit 1
fi
if grep -q 'Overfull \\hbox' "$tmp/check.log"; then
  grep 'Overfull \\hbox' "$tmp/check.log" >&2
  echo "✗ something is wider than the 160 mm text width" >&2
  exit 1
fi
echo "✓ ${#tables[@]} tables, ${#figures[@]} figures and every \\result{} key compile within 160 mm"
