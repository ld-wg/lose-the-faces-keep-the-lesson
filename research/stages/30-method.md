# Method / Pipeline Design

## Pipeline

```
Input Video
    ↓
[identify]  Face Detection — YOLOv8 on WIDER FACE
    ↓ per-frame bounding boxes + confidence + track IDs
[generate]  Synthetic Face Generation — GAN/Diffusion (Phase 2)
    ↓ synthetic face conditioned on expression + pose
[inpaint]   Face Replacement — compositing + temporal smoothing (Phase 2)
    ↓
Output Video (de-identified, expression-preserved)
```

## Stage 1 — Detection ✅

- YOLOv8n trained on WIDER FACE (32k images, 394k faces), 30 epochs
- Optimizer study: SGD / Lion / SAM → SAM best (see [[50-experiments]])
- Output: per-frame bounding boxes, confidence, track IDs
- Also delivers a **censorship baseline** (blur/mosaic/black box) usable ethically on its own
- Priority: high recall + stable boxes; real-time NOT required

## Stage 2 — Generation ⬜ (not started)

Candidate approaches (see [[20-related-work]]):

**GAN-based:**
- GANonymization ([[rw-hellmann-ganonymization]]) — landmark carrier, best emotion preservation
- CIAGAN — conditional identity anonymization
- DeepPrivacy2 — full-body
- G2Face ([[rw-yang-g2face]]) — geometry-aware identity control

**Diffusion-based:**
- ReferenceNet + SD2.1 ([[rw-kung-referencenet]]) — pose/expression coherent, anonymization knob
- Gradient injection ([[rw-wang-gradient-injection]]) — adversarial identity guidance

## Stage 3 — Temporal consistency ⬜ (not started)

Planned approach:
- Reuse previous frame's synthetic face as reference for next frame
- Combine with gradient injection to guide identity toward target while keeping facial structure
- **Critical challenge:** visual continuity of facial features over time

## Open design decisions

- [ ] GAN vs diffusion for primary generator (see [[60-open-questions]])
- [ ] How to condition on expression/pose (landmarks? 3DMM? driving image?)
- [ ] Temporal mechanism: naive reuse vs. gradient injection vs. other
- [ ] One synthetic identity per person across whole video, or per-segment?
- [ ] Demographic similarity: how to control + measure

## Links

- Problem: [[10-problem]]
- Evaluation: [[40-evaluation]]
- Decisions: [[70-decisions]]
- Source: `paper/main.tex` §Methodology, §Expected Results

#phase1/detection #phase2/generation #phase2/inpainting
