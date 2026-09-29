# E1 — Face detectors

**Question.** Which detector gives the best recall for its cost, and how does it
behave on classroom footage? (Objective 1: no face goes unnoticed, because a
missed face is a leak.)

## Design

- **Arms:** SCRFD-10GF (current default, InsightFace `buffalo_l`), SCRFD-34GF,
  YOLO-FaceV2-s.
- **Fixed:** the same tracker (ByteTrack-style), confidence threshold 0.3, and
  input size 640.

**On WIDER FACE val** (annotated; the standard protocol):
- AP on the easy / medium / hard splits, with the official evaluation protocol
  (IoU 0.5, the hard set includes the easy and medium faces);
- precision, recall and F1 at the operating threshold (0.3);
- recall at fixed false positives per image (0.01, 0.1), the headline number
  for a privacy pipeline;
- latency per image and peak GPU memory on serra1; parameters and GFLOPs.

**On our three videos** (no annotations, so operational statistics only):
- faces per frame, number of tracks, median track length;
- share of tracks shorter than 10 frames, a fragmentation proxy (ID-switch rate
  needs ground truth and is a declared limitation);
- frames per second.

## Outputs (`results/`)

| File | Content |
|---|---|
| `latex/tab-e1-widerface.tex` | AP easy/medium/hard, P/R/F1 @ 0.3, recall @ FPPI, latency, memory, params, GFLOPs |
| `latex/tab-e1-videos.tex` | per video and detector: faces/frame, tracks, median length, short-track share, fps |
| `latex/fig-e1-pr.pdf` (+ `.csv`) | precision–recall curves on the hard split |
| `latex/results.tex` | `\result{e1/<detector>/wider.hard.ap}`, `…recall.fppi0.1`, … |

## Status

Pending two components:
- `src/eval/widerface.py`, a port of the official protocol. The dataset is on
  serra1 at `CONFIG.widerface_root`.
- `src/eval/detection_stats.py`, the track statistics.

The three detector backends exist. `run.py` exits with "pending" until both
components land.

**Validation:** SCRFD-10GF must reproduce its published hard AP (83.05) within
a small tolerance before any other number here is trusted.
