# Lose the Faces, Keep the Lesson

Expression-preserving face de-identification for educational research. Detects faces in classroom videos and replaces them with synthetic, demographically similar ones while preserving expressions, gaze, and affect.

## Pipeline

```
Input Video
    ↓
[identify]  Face Detection — YOLOv8 on WIDER FACE
    ↓ per-frame bounding boxes + confidence
[generate]  Synthetic Face Generation — GAN/Diffusion (Phase 2)
    ↓ synthetic face conditioned on expression + pose
[inpaint]   Face Replacement — compositing + temporal smoothing (Phase 2)
    ↓
Output Video (de-identified, expression-preserved)
```

## Project Structure

```
├── src/
│   ├── identify/              # Face identification network
│   │   ├── train_face_detector.py
│   │   ├── dataset_utils.py
│   │   ├── webcam_detect.py
│   │   └── optimizers/        # Custom optimizers (Lion, SAM)
│   ├── generate/              # Synthetic face generation (Phase 2)
│   └── inpaint/               # Face replacement/compositing (Phase 2)
├── weights/                   # Trained model weights
│   ├── yolov8n.pt
│   ├── face_detector_sgd_best.pt
│   └── face_detector_sam_best.pt
├── docs/                      # Paper, notes, guides
├── paper/                     # LaTeX source
├── experiments/               # Training result CSVs
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
```

Train face detector:

```bash
cd src/identify
python train_face_detector.py --fraction 1.0 --epochs 30 --optimizer sam
```

Run webcam detection:

```bash
python src/identify/webcam_detect.py --model weights/face_detector_sam_best.pt
```

## Face Detection Results (WIDER FACE, 30 epochs)

| Optimizer | mAP50 | mAP50-95 |
|-----------|-------|----------|
| SGD       | 0.608 | 0.319    |
| Lion      | 0.580 | 0.306    |
| SAM       | 0.616 | 0.327    |

## Roadmap

- [x] Face detection with YOLOv8 on WIDER FACE
- [x] Optimizer comparison (SGD, Lion, SAM)
- [ ] GAN/diffusion-based face generation
- [ ] Face inpainting and temporal consistency
- [ ] Privacy/utility evaluation

## References

- WIDER FACE: http://shuoyang1213.me/WIDERFACE/
- YOLOv8: https://github.com/ultralytics/ultralytics
