# Experiments — Optimizer Study (Phase 1)

## Setup

- **Model:** YOLOv8n (nano)
- **Dataset:** WIDER FACE (32k images, 394k faces)
- **Epochs:** 30, batch 16, imgsz 640
- **Goal:** compare optimizers for face detection

## Results

| Optimizer | mAP50 | mAP50-95 | Notes |
|-----------|-------|----------|-------|
| SGD (baseline) | 0.608 | 0.319 | Fast, stable |
| Lion | 0.580 | 0.306 | 4× slower, needs tuning |
| **SAM** | **0.616** | **0.327** | Best generalization |

## Training configs

**Lion:** lr0 0.004, weight-decay 0.02, warmup 2 epochs, patience 0
**SAM:** rho 0.03, lr0 0.001, weight-decay 0.0005, warmup 2 epochs, patience 0, AMP disabled (FP32), grad clipping

Full justifications: `docs/optimizer_training_notes.md`

## Implementation

- Custom SAM + Lion: `src/identify/optimizers/custom_optimizers.py`
- Custom trainer: `src/identify/optimizers/custom_trainer.py`
- SAM does full two-step perturbation per batch (first_step → recompute loss at w+ε → second_step + base.step())
- Guide: `docs/custom_optimizers_guide.md`

## Weights

- `weights/face_detector_sgd_best.pt`
- `weights/face_detector_sam_best.pt`

## Raw logs

- `experiments/results_baseline_sgd.csv`
- `experiments/results_lion.csv`
- `experiments/results_sam.csv`

## Observations

- SAM's sharpness-aware minimization → best generalization (highest mAP50 + mAP50-95)
- Lion underperformed + 4× slower — sign-based updates may need more tuning for detection
- SAM ~2× step time vs SGD (double forward/backward)

## Open questions

- [ ] Would YOLOv9m improve detection further? (see [[rw-ahmed-yolov9-classroom]])
- [ ] Is 30 epochs enough? Longer training change ranking?
- [ ] Should we evaluate on a classroom-specific set, not just WIDER FACE?

## Links

- Method: [[30-method]]
- Related work: [[rw-ananda-yolo-retinaface]], [[rw-ahmed-yolov9-classroom]]

#phase1/detection #evaluation
