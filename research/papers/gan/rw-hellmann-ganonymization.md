# Hellmann et al. — GANonymization

**Citation:** Hellmann, Mertes, Benouis, Hustinx, Hsieh, Conati, Krawitz, André — ACM TOMM 21(1), Dec 2024 (`hellmann`). DOI: 10.1145/3641107

## Summary

**Expression-first** stance. 4-stage pipeline: RetinaFace extract/align (512×512) → U-Net background removal → MediaPipe **478 3D landmarks** projected to 2D → **pix2pix** re-synthesis (trained on CelebA, 25 epochs). Beats DeepPrivacy2 on most emotions; **comparable to CIAGAN** (CIAGAN better on Fear/Happy/Sadness). Passes Facenet512 embedding cosine-distance threshold (0.3), though **less aggressively** than baselines (0.71 vs 0.81/0.93).

## Evaluation

Datasets paired with anonymized counterparts (GANonymization, DeepPrivacy2, CIAGAN):
- AffectNet (~0.4M in-the-wild, 8 emotions)
- CK+ (593 sequences @ 30fps)
- FACES (2,052 high-quality frontal)

Two emotion-eval settings:
- **Inference** — classifier trained on original, tested on anonymized (class-prob distance)
- **Training** — classifier trained on anonymized, F1 vs original (AffectNet 0.58→0.37, CK+ 0.99→0.69, FACES 0.97→0.81)

**Facial-trait removal** (CelebA, 40 attrs): removes Bald/Gray Hair (100%), hats/mustache (>97%); **preserves Smiling (only 4.7% removed)**, Young, Heavy Makeup, Wearing Lipstick — latter two reveal CelebA training bias.

## Relevance to us

- Dense-landmark carrier = strong emotion preservation → good for pedagogical affect analysis
- **Reuse dual emotion-eval protocol** (inference class-prob distance + training F1 drop) → [[40-evaluation]]
- **Add facial-trait removal as fairness metric** → [[40-evaluation]]
- Treat **CIAGAN as equal baseline**, not lesser
- Keep our stronger privacy metrics (ASR/Rank-N-T) — their embedding-distance threshold is weaker

## Limitations

- Detection failure → "average face" fallback (expression lost entirely)
- Fear/Happy confusion (Happy/Surprise → Fear)
- CelebA demographic bias (Heavy Makeup/Lipstick preserved)
- Image-only (no temporal consistency) — our differentiator

## Links

- Related work: [[20-related-work]]
- Method: [[30-method]]
- Evaluation: [[40-evaluation]]

#phase2/generation #utility
