# Paper experiments

The test plan for the paper. Every number the paper reports comes from one of
the experiments below. Each experiment is a self-contained folder: its
`README.md` states the question and the design, `config.toml` fixes every
parameter and input, and `run.py` deletes the previous results and regenerates
all of them inside the folder. `run_all.sh` regenerates everything.

Decisions behind this plan, including what the paper drops and which
limitations it declares: [`DECISIONS.md`](DECISIONS.md).

## Research question and what answers it

> Can faces in classroom recordings be replaced by synthetic, demographically
> similar ones so that **identity is suppressed** while **expression, gaze and
> affect are preserved** well enough for pedagogical analysis?

| Objective / hypothesis | Experiment | Primary measure |
|---|---|---|
| O1 — high-recall detection in classrooms | E1 | recall at fixed false positives (WIDER FACE), track statistics on our videos |
| H(a), O2 — identity suppression | E2, E3c | rank-1 re-identification by the held-out recognizer (FaceNet) |
| H(b), O3 — expression, gaze and affect preserved | E2, E3c | emotion agreement, expression / pose / gaze error, against the noise floor from E0 |
| Demographically similar surrogate | E2, E3 | age error and gender agreement |
| O4 — compute cost | E1, E2 | latency / fps and peak GPU memory |
| Contribution C2 — track-level real identity | E3a | cosine to unseen frames of the same track |
| Contribution C1 / P2 — identity push in the swap | E3c, E3d | rank-1 with vs without the push, at equal utility |
| Measurement validity | E0 | TAR at FAR 1%, utility noise floors |

## Experiments

| ID | Folder | Question | Arms | Status |
|---|---|---|---|---|
| E0 | [`e0_instruments`](e0_instruments) | Do the recognizers and utility probes measure what we claim on this footage? | FaceNet, ArcFace; utility probes | recognizer validation runnable; noise floors pending the utility probes |
| E1 | [`e1_detection`](e1_detection) | Which detector gives the best recall for its cost, and how does it behave in a classroom? | SCRFD-10GF, SCRFD-34GF, YOLO-FaceV2-s | skeleton; needs the WIDER FACE evaluator and track statistics |
| E2 | [`e2_generation`](e2_generation) | What is the privacy–utility trade-off of each generator family, against censorship? | blur, mosaic, CIAGAN, GANonymization, BLANKET (original) | skeleton; needs the censorship backend and the utility probes |
| E3 | [`e3_contribution`](e3_contribution) | Do our ideas improve privacy without costing utility? | C2 modes; BLANKET ± seed candidates; P2 push strength β; P2 target | skeleton; components exist, run.py pending |

Inputs: three classroom videos, `demo1` (239 frames, 15 tracks), `demo2` (244
frames, 24 tracks) and `demo3` (355 frames, 32 tracks), plus WIDER FACE val for
E1. Unless E1 says otherwise, E2 and E3 use SCRFD-10GF detections.

## Metrics

Grouped by the question they answer. Definitions and implementation notes:
`research/stages/40-evaluation.md` (local vault) and `src/eval/`.

| Group | Question | Metrics |
|---|---|---|
| Coverage | Is every face anonymized? | share anonymized; passthrough causes |
| Privacy | Can the person still be recognized? | **rank-1** (primary) and rank-5, verification at FAR 1%, cosine to the real face, Privacy Gain; FaceNet (held-out) next to ArcFace (the space the push is computed in); two views: anonymized faces only, and all faces with passthrough counted as a leak |
| Pseudonym | Does each person keep one stable synthetic identity? | within-track consistency |
| Temporal | Does the face flicker? | consecutive-frame similarity |
| Utility | Is the behavioral signal kept? | emotion agreement, expression error (106 landmarks), head-pose error, gaze error, age error, gender agreement, re-detection |
| Cost | Can it run on standard hardware? | latency / fps, peak GPU memory |
| Validity | Is the instrument valid here? | TAR at FAR 1% on real pairs; noise floor of every utility metric between consecutive real frames |

Every metric is reported per video and pooled, with 95% bootstrap confidence
intervals over tracks. Arms that share faces and identities are compared with
paired tests (paired bootstrap; McNemar for per-face rank-1).

## Conclusion criteria (fixed before running)

- **Privacy improves** when held-out rank-1 drops and the paired confidence
  interval against the reference arm excludes zero.
- **Suppression** (aspirational, to confirm with the advisor): verification at
  FAR 1% ≤ 5% and rank-1 ≤ 2× chance. Exploratory runs already show BLANKET + P2
  above this (rank-1 ≈ 0.3), and the paper will say so.
- **Utility is preserved** when emotion agreement and the expression / pose /
  gaze errors stay within the confidence interval of the reference arm, read
  against the E0 noise floor.
- **The final pipeline** is the best privacy–utility trade-off, with coverage as
  a constraint: a missed face is a leak.

## Running

```bash
uv sync --extra phase2-ganonymization --extra eval --extra experiments
uv run experiments/e0_instruments/run.py            # one experiment
uv run experiments/e0_instruments/run.py --video demo2   # override its inputs
experiments/run_all.sh                              # all, in order; stops at the first failure
experiments/run_all.sh --only e0,e2
experiments/export_paper.sh                         # copy LaTeX artifacts into paper/generated/
```

Each run **deletes `results/` and regenerates it**. Expensive artifacts that are
safe to reuse live outside it, in `cache/`: the frozen SDXL identities of
BLANKET, keyed by video hash and settings. Reusing them also controls SDXL's
run-to-run non-determinism. `--fresh` clears the experiment's cache too.

Machine-specific settings come from the environment, never from `config.toml`:
`BLANKET_REPO`, `IDENTITY_GPU`, `SWAP_GPU`, `CTX_ID` (on serra1: `1`, `0`, `-1`).
Videos resolve from `data/videos/<name>.*`.

## What a results folder contains

```
eN_name/results/            # gitignored, regenerated on every run
  run.log                   # everything the run printed
  commands.sh               # every command executed, in order: rerun any step by hand
  provenance.json           # git SHA (+ diff if dirty), bridge commit, host, GPUs, versions,
                            # video sha256, timings, status
  metrics.json              # every number the experiment produces, by key
  <table>.md                # human-readable tables
  latex/                    # what the paper uses (aggregate numbers only, no faces)
    tab-eN-<name>.tex       # booktabs tabular only; caption and label stay in the paper
    fig-eN-<name>.pdf/.csv  # vector figure + the data it plots
    results.tex             # \result{key} values for numbers cited in the text
  <video>/<arm>/...         # component outputs (detections, frames, videos)
```

## Paper artifacts

`export_paper.sh` copies each `results/latex/` into `paper/generated/<experiment>/`,
which **is** versioned: results stay out of git, but the aggregate artifacts the
paper cites are part of the paper. The paper uses them as:

```latex
% preamble, once
\input{generated/results}            % loads every experiment's \result{} keys

% a table: the caption and label stay in the paper
\begin{table}[ht]\centering
  \caption{P2 push strength vs re-identification.}\label{tab:p2-strength}
  \input{generated/e3_contribution/tab-e3-p2-strength}
\end{table}

% a figure
\includegraphics[width=\linewidth]{generated/e3_contribution/fig-e3-p2-curve}

% a number in the text: a key that does not exist stops the compilation
FaceNet rank-1 falls to \result{e3c/demo2/beta1.2/facenet.rank1}.
```

Formats match `paper/main.tex` (SBC template, 12pt, single column, `booktabs`
and `graphicx`, no `pgfplots`/`siunitx`):
- **Tables** have three decimals, CIs as `0.284\,{\scriptsize$\pm$0.031}`, the
  best value per column in bold, and at most about seven columns.
- **Figures** are PDF with a serif font sized for `\linewidth`.
- **Example frames are never generated automatically.** Even anonymized frames
  show real people in the background, so they are a manual, consent-checked
  decision.

## Cost on serra1 (GPUs shared with another job)

| Experiment | Estimate |
|---|---|
| E0 | ~30 min |
| E1 | ~1 h |
| E2 | ~8 h (SDXL identities for three videos) |
| E3 | ~15 h with one GPU per stream |

## Quick, single-step runs

Outside the experiments, `bin/` runs one step at a time with short commands.
Useful for exploring and debugging; the experiments never depend on it.

```bash
det=$(bin/detect demo3 | tail -1)
bin/anonymize "$det" blanket --blanket-swap-mode track --blanket-push-beta 1.2
bin/video results/<anonymize-folder>
bin/eval  results/<anonymize-folder> [more folders]
bin/ls
```
