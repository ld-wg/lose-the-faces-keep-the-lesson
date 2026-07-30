# Stage: Identification (Face Detection)

> **⚠️ See [[identification-refined]] for the updated opinion.** The "done" status below is challenged: YOLOv8n (mAP50 0.616) is architecturally behind SCRFD-34G (0.853 hard AP) and RetinaFace (perfect classroom recall), and adult-trained detectors are age-biased against our child subjects.

First pipeline stage: localize every face per frame with high recall.

## Goal

Per-frame bounding boxes + confidence + track IDs. Priority: **high recall + stable boxes** in classroom conditions (occlusions, board turns, motion). Real-time NOT required.

## Status: ✅ Complete

- YOLOv8n on WIDER FACE, SAM optimizer
- mAP50 0.616, mAP50-95 0.327
- Weights: `weights/face_detector_sam_best.pt`
- Details: [[50-experiments]]

## Outputs

- Per-frame face crops + bounding boxes with IDs/confidence
- Input API for the generative stage
- Censorship baseline (blur/mosaic/black box) — usable ethically on its own

## Family

See [[detection]] for the detection family overview.

## Open questions

- YOLOv9m upgrade? Classroom-specific eval? (see [[60-open-questions]])

## Links

- Method: [[30-method]]
- Next stage: [[generation]]

#phase1/detection #stage
