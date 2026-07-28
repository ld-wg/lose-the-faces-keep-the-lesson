# Research Notes: Lose the Faces, Keep the Lesson

## Overview

Expression-preserving face de-identification for educational research. Pipeline: detect faces → replace with synthetic faces → preserve expressions/gaze/affect.

## Phase 1: Face Detection (Complete)

### YOLOv8 on WIDER FACE

Trained YOLOv8n on WIDER FACE dataset with different optimizers:

| Optimizer | mAP50 | mAP50-95 | Notes |
|-----------|-------|----------|-------|
| SGD (baseline) | 0.608 | 0.319 | Fast, stable |
| Lion | 0.580 | 0.306 | 4x slower, needs tuning |
| **SAM** | **0.616** | **0.327** | Best generalization |

- Dataset: WIDER FACE (32k images, 394k faces)
- Model: YOLOv8n (nano)
- Epochs: 30
- Batch size: 16
- Image size: 640x640

### Key References for Detection

- YOLOv8 (Ultralytics): https://github.com/ultralytics/ultralytics
- WIDER FACE benchmark: http://shuoyang1213.me/WIDERFACE/
- RetinaFace comparison: [G. F. Ananda and Ardiyanto 2024]
- YOLOv9 for classroom: [Ahmed 2025]

## Phase 2: Expression-Preserving De-identification (TODO)

### Candidate Approaches

1. **GAN-based**:
   - GANonymization [Hellmann 2024] — landmark-based, preserves emotion
   - CIAGAN [Maximov et al. 2020] — conditional identity anonymization
   - DeepPrivacy2 [Hukkelås and Lindseth 2023] — full-body anonymization
   - G2Face [Yang 2024] — geometry-aware identity control

2. **Diffusion-based**:
   - ReferenceNet + Stable Diffusion [Kung 2025] — pose/expression coherent
   - Gradient injection [Wang 2025] — adversarial identity manipulation

### Evaluation Metrics

**Privacy**:
- Embedding-space similarity (before/after)
- Verification ASR (attack success rate)
- Identification Rank-N-T
- Commercial API confidence

**Utility**:
- Categorical emotion agreement
- 3DMM coefficient L2 distance
- Pose/gaze quaternion angular distance
- PSNR, SSIM, FID

### Temporal Consistency

- Reuse previous synthetic face as reference for next frame
- Gradient injection for identity guidance
- Frame-to-frame smoothing

## Datasets

- **WIDER FACE**: Face detection training (used)
- **AffectNet**: Expression evaluation (referenced)
- **CK+**: Expression sequences (referenced)
- **CelebA-HQ / FFHQ**: Generator training (referenced)
