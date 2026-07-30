# Phase 1 (Identification) — A Refined Opinion

Where our face-detection stage stands, what the evidence actually says, and what we should do about it. This supersedes the "YOLOv8n is done" assumption.

---

## The uncomfortable truth

We treated Phase 1 as **solved**. It isn't. Our YOLOv8n-SAM baseline hits **mAP50 = 0.616** on WIDER FACE — and every assumption behind that choice is now contradicted by the evidence:

1. **We optimized for the wrong thing.** We picked a *nano* model for speed. But our pipeline is **offline** — latency buys us nothing. The only currency that matters is **recall**, because in a privacy pipeline **a missed face is a leaked identity.** A false positive is recoverable downstream; a false negative is not.
2. **Our detector is architecturally behind.** SCRFD-34G reaches **85.29 AP on WIDER FACE hard split**; RetinaFace hit **perfect recall (1.00) across all four classroom scenarios** in Ananda et al. (ICVEE 2024), including low light and occlusion. Our 0.616 isn't an optimizer problem — it's a model-class problem.
3. **Our subjects are children, and detectors are age-biased.** Adult-trained detectors collapse on young faces: an off-the-shelf detector scored **7.37% on neonates** (Hausmann 2021), and even a child-tuned YOLOv9-M drops to **R = 0.34** under realistic hard conditions (Ahmed 2025). WIDER FACE is adult-skewed. We are evaluating on the wrong distribution.

---

## What the research says (three fronts)

### A. Alternatives to YOLOv9-nano — yes, better ones exist

| Detector | WIDER FACE hard AP | Why it fits us |
|----------|--------------------|----------------|
| **SCRFD-34G** (ICLR 2022) | **85.29** (val) | Best hard-split accuracy; sample redistribution targets hard/small faces (children in wide shots). Offline-affordable. |
| **SCRFD-10G** | 83.05 | Lighter, still beats YOLOv8m-face. |
| **RetinaFace** (CVPR 2020) | 91.4 (test) | **Perfect classroom recall** (Ananda 2024); landmark supervision helps pose/occlusion. Slowest — irrelevant offline. |
| **YOLO-FaceV2** (2022) | — | Explicitly scale- and occlusion-aware (small faces, hand/toy occlusion) — our two hardest classroom cases. |
| YOLOv8m-face | 84.7 | Best in-family option; easy Ultralytics fine-tune + native tracking. |

**No transformer detector beats SCRFD on WIDER FACE hard.** YOLOv10–v12 have **no published face results** — don't chase novelty. The gap between our YOLOv8n (0.616) and SCRFD-34G (0.853) is architectural.

### B. Low-light detection — enhancement must be *task-driven*, not cosmetic

Classrooms have low-light scenarios (~40 lux, Ananda 2024). The tempting fix — bolt on a low-light enhancer (Zero-DCE/Retinex) — **backfires**: a 2024 study found off-the-shelf enhancement improves *human* viewing but is **inconsistent or harmful for machine detection**.

What works:
- **Fine-tune on low-light data** (highest ROI): DARK FACE + synthetically darkened WIDER FACE (gamma/noise/vignetting — cheap, we already have WIDER FACE).
- **Task-driven enhancement**: FeatEnHancer (plug-and-play, published DARK FACE gain) or GDIP/IA-YOLO-style differentiable preprocessing trained *with the detection loss*.
- **Free offline wins**: multi-exposure/multi-scale TTA, lower confidence threshold (recall ≫ precision for privacy), temporal tracking to recover per-frame misses.
- Benchmark: **DARK FACE** (the low-light face benchmark). ⚠️ It's adult surveillance, not children — our own darkened-classroom eval is essential.

### C. Child face detection — fine-tuning is non-negotiable

- **The domain gap is real and large**: adult-trained → 7.37% on neonates; fine-tuning → 68.7% (Hausmann 2021). Mixed pretrain + child fine-tune → AP50 0.96 (Bin-Obaid 2026). Ahmed got 0.963 mAP with only ~300 child images.
- **No public age-stratified child detection dataset exists.** Real child data is small/private. **Synthetic child faces (ChildGAN, ChildDiffusion) are the practical augmentation route.**
- **Age bias in detection is under-studied** (WIDER-FAIR covers ethnicity/sex, not age) — a gap we can cite and fill.
- **Labeling trick (from Ahmed):** label children, include unlabeled adults → the teacher is suppressed, not detected.

---

## The refined plan for Phase 1

**Detector:** Move off YOLOv8n. Shortlist **SCRFD-34G** (primary — best hard-split recall) and **RetinaFace** (high-recall fallback for hard frames — perfect classroom recall). Keep **YOLOv8m-face** as the easy-integration fallback. Optionally a **cascade**: fast detector + RetinaFace union on low-confidence/track-drop frames.

**Training:** Pretrain on WIDER FACE → **fine-tune on child/classroom data** (the single biggest lever). Augment with **synthetic child faces** (ChildGAN/ChildDiffusion) and **synthetically darkened** frames. Adopt the "label children, include adults" trick.

**Inference (offline, recall-first):** higher input resolution + multi-scale TTA + low confidence threshold; let **ByteTrack** tracking + temporal smoothing reject false positives. This directly attacks the R=0.34 hard-condition collapse.

**Evaluation:** build a small annotated classroom set; report **recall stratified by age band and condition** (lighting, occlusion, headcount) mirroring Ananda's 4-scenario protocol. Optimize **recall at a fixed low FP rate**, not mAP — mAP can hide missed faces. This age-stratified eval is itself a citable contribution (no public benchmark exists).

---

## Open questions

- [ ] SCRFD (mmdetection) vs RetinaFace (InsightFace) vs YOLOv8m (Ultralytics) — integration cost vs recall gain? Prototype all three on the same classroom clips.
- [ ] How much does synthetic child data (ChildGAN/ChildDiffusion) help vs its distribution shift / ethics of child facial data?
- [ ] Does low-light fine-tuning distort facial affect cues downstream (Stage 2 expression preservation)? Validate.
- [ ] Single high-recall detector vs cascade — is the union worth the complexity?

## Links

- Pipeline: [[pipeline]] | Method: [[30-method]] | Experiments: [[50-experiments]]
- Detection family: [[detection]] | Papers: [[rw-ananda-yolo-retinaface]], [[rw-ahmed-child-yolo-anonymization]]
- Open questions: [[60-open-questions]]

#phase1/detection #children #low-light #opinion
