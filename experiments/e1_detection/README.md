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
| `latex/tab-e1-widerface-ap.tex` | official AP easy/medium/hard, latency, params |
| `latex/tab-e1-widerface-recall.tex` | P/R/F1 at 0.3 and recall at FPPI 0.01 / 0.1 (hard set) |
| `latex/tab-e1-videos.tex` | per video and detector: faces/frame, tracks, median length, short-track share, fps |
| `latex/results.tex` | `\result{e1/<detector>/wider.hard.ap}`, `…recall.fppi0.1`, … |

## Status

Run on serra1 2026-10-05 (19 min); exported to `paper/generated/e1_detection/`.
SCRFD-10GF hard AP 0.813, within 1.8 points of the published 83.05.
- `src/eval/widerface.py` ports the official protocol. The ground-truth
  `.mat` files are fetched if missing.
- `src/eval/detection_stats.py` gives the track statistics.

**Validation:** SCRFD-10GF's hard AP should land near its published 83.05,
at 640 px.

