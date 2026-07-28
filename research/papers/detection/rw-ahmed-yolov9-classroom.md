# ⚠️ UNVERIFIED — Ahmed — YOLOv9 for classroom face detection

> **SOURCE STATUS: UNVERIFIED — do not cite in paper.**
> The PDF at `~/Documents/tese/papers/Comparative analysis of deep learning image detection algorithms.pdf` was expected to be this paper but is actually **Srivastava et al. 2021** (SSD vs Faster R-CNN vs YOLOv3 on MS COCO — generic objects, no faces/children/YOLOv9). The claims below are **not confirmed against any PDF in our library**. The real YOLOv9-for-children paper may be mis-filed or read elsewhere. Locate the correct source before citing.
>
> **Note:** A *different*, verified paper — Ahmed et al. 2025 "Enhanced child face anonymization" (YOLOv5/v8/v9 + blur) — IS in our library and covers YOLOv9-M for child faces. See [[rw-ahmed-child-yolo-anonymization]].

**Citation:** Ahmed, 2025 (`ahmed`) — ⚠️ unverified

## Summary (unverified)

Compared YOLOv9 to YOLOv5 and YOLOv8 on a custom, manually labeled dataset of children's faces (from public websites).

## Key findings (unverified)

- YOLOv9 (esp. medium model): clear gains in precision, recall, overall detection quality in classroom-like conditions

## Relevance to us

- For non-real-time, accuracy > marginal latency → YOLOv9's higher fidelity supports reliable face localization without sacrificing downstream expression fidelity
- **Open question:** should we upgrade from YOLOv8n to YOLOv9m? (see [[60-open-questions]])
- **Partially answered by verified source:** [[rw-ahmed-child-yolo-anonymization]] confirms YOLOv9-M > v8 for child faces (mAP@0.5 0.963)

## Links

- Related work: [[20-related-work]]
- Our detection: [[30-method]], [[50-experiments]]
- Verified alternative: [[rw-ahmed-child-yolo-anonymization]]

#phase1/detection #open-question #unverified
