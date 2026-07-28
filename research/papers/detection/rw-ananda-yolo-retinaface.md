# Ananda, Nugroho & Ardiyanto — Multi-Face Detection in Classrooms

**Citation:** G. F. Ananda, H. A. Nugroho, I. Ardiyanto, ICVEE 2024 (`gf-ananda`). DOI: 10.1109/ICVEE63912.2024.10823781

## Summary

Compared **four** detectors — Haar Cascade, MTCNN, YOLOFace, RetinaFace — on a **collected classroom dataset** (fluctuating lighting, diverse student postures), bridging the gap left by standard benchmarks (LFW, FDDB, WIDER FACE). Pre-trained models used off-the-shelf, no fine-tuning.

## Key findings

| Method | Precision | Recall | Inference (s) |
|--------|-----------|--------|---------------|
| Haar Cascade | 0.89–0.93 | **0.25–0.34** | **0.54–0.80 (fastest)** |
| MTCNN | 0.97–1.00 | 0.84–0.97 | 4.27–10.08 |
| **YOLOFace** | **1.00 (all)** | 0.97–1.00 | **2.15–3.33** |
| **RetinaFace** | 0.97–1.00 | **1.00 (all)** | **9.15–12.73 (slowest)** |

- **YOLOFace**: perfect precision all scenarios, best speed/accuracy balance → ideal for real-time
- **RetinaFace**: perfect recall all scenarios, best in low light/occlusion, but slowest → best when recall critical
- **Haar Cascade**: fast but misses most faces (recall ~0.3) → inadequate for classrooms
- Dataset: 4 scenarios, 32–40 students, 152 lux + one 40-lux low-light scenario; T4 GPU; IoU 0.5–0.75

## Relevance to us

- **Directly validates our core premise**: standard benchmarks don't capture classroom conditions — citable justification for evaluating on real classroom footage
- **Gives a classroom eval protocol**: vary headcount + illuminance (incl. ~40 lux low-light), report P/R/F1 + inference with IoU≥0.5 — we can mirror this
- Confirms accuracy-vs-latency trade-off; supports our "non-real-time → accuracy > latency" stance
- **Caveat:** their "YOLOFace" is YOLOv3-based, NOT YOLOv8 — supports YOLO family generally; our YOLOv8n choice still needs our own validation (see [[50-experiments]])

## Decision

Adopt their 4-scenario (headcount + low-light) protocol for our detection eval. Offline pipeline → can afford RetinaFace-level recall on hard frames; YOLOv8n for throughput.

## Links

- Related work: [[20-related-work]]
- Our detection: [[30-method]], [[50-experiments]]

#phase1/detection
