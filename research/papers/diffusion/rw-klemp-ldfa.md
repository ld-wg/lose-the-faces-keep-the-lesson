# Klemp et al. — LDFA (Latent Diffusion Face Anonymization)

**Citation:** Klemp, Rösch, Wagner, Quehl, Lauer — 2023, arXiv:2302.08931 (`klemp`)

## Summary

Two-stage, **training-free** pipeline: RetinaFace (conf 0.4, high recall) → per-face crop with 32 px context padding → **off-the-shelf Stable Diffusion 2 Inpainting** (no prompt, CFG 1, 50 steps) re-synthesizes each face. No face-specific fine-tuning. Shows general LDMs match specialized GAN anonymizers (DeepPrivacy/2) and beat them on post-anonymization face-detection mAP (0.675 vs 0.566/0.521 on Cityscapes), especially for small faces. Graceful on false positives (inpaints plausible non-face content).

## Evaluation

- Cityscapes (3,765 faces; 3,214 small)
- Downstream segmentation (Mask2Former): ΔIoU_rel −0.24% person, −0.64% rider ≈ GAN methods, better than naive
- Face re-detection mAP after anonymization (utility = "still a face")
- Embedding L2 distance (VGG-Face) as anonymization level; notes realism↔unrecognizability trade-off
- NoA (number anonymized): coverage metric — missed detections = privacy leak

## Relevance to us

- **Zero-shot diffusion baseline** for our experiments; validated skeleton for our pipeline (high-recall detect → pad → per-face inpaint)
- Concrete starting hyperparameters (threshold 0.4, 32 px pad, 512², 50 steps)
- Two metrics to adopt: post-anonymization detection mAP + anonymization coverage (NoA) → [[40-evaluation]]
- Its core weakness = our contribution: **no expression/gaze/affect conditioning at all** — motivates ReferenceNet-style conditioning ([[rw-kung-referencenet]]) or gradient guidance ([[rw-wang-gradient-injection]])

## Limitations

- No identity/expression/gaze preservation — face fully re-synthesized unconditionally
- Blurry/deformed inpaintings on tiny faces (<32² px); skin-tone mismatch around glasses
- Image-only, no temporal modeling; embedding-L2-only privacy eval (no ASR/Rank-N-T)
- Domain is street scenes (small, distant faces), not classroom close-ups

## Links

- Related work: [[20-related-work]]
- Method: [[30-method]]
- Evaluation: [[40-evaluation]]

#phase2/generation #privacy #utility
