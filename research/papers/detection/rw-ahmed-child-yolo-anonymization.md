# Ahmed et al. — Enhanced Child Face Anonymization (YOLOv5/v8/v9 + blur)

**Citation:** H. A. Ahmed, M. H. Ahmed, J. Majidpour, S. M. Omer, 2025, *Neural Computing and Applications* 37:19793–19816 (`hunar-ahmed`). DOI: 10.1007/s00521-025-11424-x

## Summary

Real-time child-face **detection + Gaussian blur** (no generation). Custom 296-image child-face dataset (Unsplash/Pexels/Pixabay, hand-labeled, single class `Child`; adults present but unlabeled to force child/adult discrimination). Benchmarks YOLOv5/v8/v9 in S/M sizes at 50/100/150 epochs, then blurs detections with a (401,401) Gaussian kernel.

## Key findings

- **YOLOv9-M @ 150ep best**: mAP@0.5 = 0.963, P = 0.94, R = 0.901, mAP@0.5–95 = 0.746; statistically > v5 (p=0.0003), > v8 (p=0.0001)
- YOLOv8 worst of the three — "newer ≠ better"
- **External set (20 imgs): big drop** — v9-M mAP@0.5 → 0.889, P → 0.796; main false positives = adults misclassified as children
- **Hard conditions (groups, occlusion, low light, motion blur): collapse** — v9-M R = 0.34, mAP@0.5 = 0.57
- **Latency**: v9 ≈ 122 ms/frame vs v5-S 50 ms — 2–3× slower, threatens real-time

## Relevance to us

- **Phase 1 only** — says nothing about generation; does NOT address the Kung et al. child-face synthesis gap
- Child-specific fine-tuning works even with ~300 images; adopt their "label only children, include adults" trick for our classroom set (teacher = adult to suppress)
- YOLOv9-M is a candidate to replace our YOLOv8n (offline batch → latency OK); better child/adult discrimination
- Hard-condition recall collapse (R=0.34) validates our temporal-smoothing plan over per-frame detection
- Their blur = our **utility floor baseline** (zero expression/gaze/affect preservation)

## Limitations

- Blur destroys all utility — no expression/gaze/affect preservation, no identity synthesis
- Tiny dataset (296 img), not released; no age-stratified or fairness metrics
- Single-frame detection; no temporal consistency
- Slow (YOLOv9) for real-time; vulnerable to adversarial attacks (authors note)

## Links

- Related work: [[20-related-work]]
- Detection family: [[families/detection]]
- Our detection: [[30-method]], [[50-experiments]]
- Classroom detection comparison: [[rw-ananda-yolo-retinaface]]

#phase1/detection #children #baseline #yolo
