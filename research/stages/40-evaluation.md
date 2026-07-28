# Evaluation Plan

Dual-objective: **privacy** (identity suppression) × **utility** (behavioral signal preservation).

## Privacy metrics

| Metric | Definition | Source |
|--------|-----------|--------|
| Embedding similarity | Cosine similarity of face embeddings before/after de-ID | — |
| Verification ASR | Fraction of de-identified faces still verified as target | [[rw-wang-gradient-injection]] |
| Identification Rank-N-T | Target appears in top-N of closed-set gallery | [[rw-wang-gradient-injection]] |
| API confidence | Confidence scores from commercial FR services | [[rw-wang-gradient-injection]] |
| Privacy Gain (PG) | $(1-R_p)-(1-R_o)$ — recognition drop after enhancement | [[rw-meden-bpet-survey]] |
| PIC | Privacy-gain Identity-loss Coefficient — joint privacy+utility | [[rw-meden-bpet-survey]] |
| Anonymization coverage (NoA) | Fraction of faces actually anonymized (missed = leak) | [[rw-klemp-ldfa]] |
| **Held-out verifier** | Evaluate with a DIFFERENT FR model than used in training | [[rw-wu-ppgan]] |
| **Parrot/imitation attack** | Retrain matcher on de-ID data, then measure Rank-1/ASR | [[rw-meden-bpet-survey]] |

## Utility metrics

| Metric | Definition | Source |
|--------|-----------|--------|
| Categorical emotion agreement | Emotion classification agreement before/after | [[rw-hellmann-ganonymization]] |
| 3DMM coefficient L2 | L2 distance over 3DMM shape/expression coefficients | [[rw-hellmann-ganonymization]], [[rw-kung-referencenet]] |
| Pose/gaze distance | Quaternion angular distance | [[rw-hellmann-ganonymization]], [[rw-kung-referencenet]] |
| PSNR / SSIM / FID | Image quality (⚠️ poor proxy for generative methods) | [[rw-wang-gradient-injection]], [[rw-kung-referencenet]] |
| Face-specific IQA | Face image quality assessment | [[rw-kung-referencenet]] |
| Post-anonymization detection mAP | Face still detectable after de-ID | [[rw-klemp-ldfa]] |
| Facial-trait removal (fairness) | CelebA 40-attr suppression; check demographic bias | [[rw-hellmann-ganonymization]] |

> **Correction (2026-07-28):** 3DMM L2 and pose/gaze were misattributed to Wang (gradient injection). Wang only uses PSNR/SSIM/FID. 3DMM/pose/gaze come from GANonymization + ReferenceNet. Fixed.

## Baselines

- **Censorship baseline:** blur / mosaic / black box (utility floor — destroys all affect) — see [[rw-ahmed-child-yolo-anonymization]]
- **Zero-shot diffusion baseline:** LDFA (off-the-shelf SD2 inpainting, no training) — see [[rw-klemp-ldfa]]
- Compare privacy + utility of generative approach vs. both baselines

## Fairness

- Stratified analysis by demographics + scene conditions
- Cannot guarantee across all subgroups (dataset limitations) — see [[60-open-questions]]

## Reproducibility & ethics

- Documented configs, seeds, scripts
- Synthetic exemplars for illustration
- Consent, storage policies, auditability

## Open questions

- [ ] What thresholds define "acceptable" identity suppression / emotion drift?
- [ ] Which FR backbone(s) for embedding similarity + attacks?
- [ ] Which emotion classifier for agreement?
- [ ] Human-rated pedagogical analyzability: protocol? raters? rubric?

## Links

- Method: [[30-method]]
- Open questions: [[60-open-questions]]
- Source: `paper/main.tex` §Methodology

#evaluation #privacy #utility
