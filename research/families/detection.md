# Face Detection (Identification Stage)

Family overview of detection approaches for the pipeline's first stage.

## Papers

- [[rw-ananda-yolo-retinaface]] — YOLOFace vs RetinaFace on classroom data
- [[rw-ahmed-yolov9-classroom]] — YOLOv9 > v5/v8 on children's faces

## Takeaways

- YOLO family: efficient single-pass detection, near-perfect precision/recall on classroom data
- RetinaFace: SOTA accuracy + occlusion resilience, but slowest
- YOLOv9 (medium): clear gains over v5/v8 in classroom-like conditions
- **Non-real-time use case → accuracy > marginal latency**

## Our implementation

- YOLOv8n on WIDER FACE, SAM optimizer (see [[50-experiments]])
- mAP50 0.616, mAP50-95 0.327

## Open questions

- Upgrade to YOLOv9m for higher recall? (see [[60-open-questions]])
- Evaluate on classroom-specific data vs only WIDER FACE?

## Links

- Method: [[30-method]]
- Experiments: [[50-experiments]]

#phase1/detection #family/detection
