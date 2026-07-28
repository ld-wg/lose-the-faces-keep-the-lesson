# Decision Log

Chronological record of what we decided and why. Newest first.

---

## 2026-07-28 — Paper verification: 10 agents reviewed all PDFs

Spawned one agent per paper to verify notes against actual PDFs in `~/Documents/tese/papers/`. Key outcomes below.

---

## 2026-07-28 — Evaluation: adopt stronger privacy protocol + new metrics

**Decision:** Add to [[40-evaluation]]: Privacy Gain (PG), PIC, anonymization coverage (NoA), post-anonymization detection mAP, facial-trait removal (fairness), **held-out verifier** (evaluate with different FR than training), and a **parrot/imitation attack**.

**Why:** Meden survey ([[rw-meden-bpet-survey]]) justifies the attack-based suite + warns PSNR/SSIM are poor proxies for generative output. PP-GAN ([[rw-wu-ppgan]]) shows held-out-verifier is needed for credible privacy claims. LDFA ([[rw-klemp-ldfa]]) contributes NoA + detection mAP.

**Correction:** 3DMM L2 / pose-gaze were misattributed to Wang — actually from GANonymization + ReferenceNet. Fixed.

---

## 2026-07-28 — Generator: leaning diffusion (ReferenceNet), child-face fairness is gating

**Decision:** Lean toward ReferenceNet ([[rw-kung-referencenet]]) as primary generator; treat CIAGAN as equal GAN baseline to GANonymization. Gating question = child-face degradation.

**Why:** ReferenceNet best-in-table on pose/gaze/expression (our utility axes), knob $d$ = ready-made privacy–utility curve, seed diversity → consistent per-student pseudonym. But it fails on infants/minorities — our subjects are children. See [[gan-vs-diffusion]].

---

## 2026-07-28 — Detection: YOLOv9-M is a candidate; old "Ahmed YOLOv9" note unverified

**Decision:** Consider YOLOv9-M to replace YOLOv8n (offline batch → latency OK). Flag [[rw-ahmed-yolov9-classroom]] as **unverified** (its PDF is actually Srivastava 2021, generic objects). Verified child-face YOLOv9 source = [[rw-ahmed-child-yolo-anonymization]] (YOLOv9-M mAP@0.5 0.963).

**Why:** Ahmed et al. 2025 (verified) shows YOLOv9-M > v8 for child faces with statistical significance; better child/adult discrimination (teacher vs students). Hard-condition recall collapse (R=0.34) validates temporal-smoothing plan.

---

## 2026-07-28 — New papers added to vault

- [[rw-meden-bpet-survey]] — B-PET taxonomy + eval framework (new `survey/` family)
- [[rw-klemp-ldfa]] — zero-shot diffusion baseline (LDFA)
- [[rw-ahmed-child-yolo-anonymization]] — child-face YOLOv9 + blur baseline

---

## 2026-07-27 — Research knowledge base: in-repo markdown, Obsidian-ready

**Decision:** Keep research notes in `research/` inside the repo as plain markdown with `[[wikilinks]]`.

**Why:** Versioned with code, LLM can read/edit directly, git diffs work. Openable as an Obsidian vault for visual graph. Better than a separate vault (decoupled from code) or Obsidian-native-only (graph invisible to LLM).

**Alternatives considered:** Separate Obsidian vault; single giant doc.

---

## 2026-07-27 — Path config: Python module + local JSON + env vars

**Decision:** `src/config.py` singleton; resolution = defaults → `config.local.json` (gitignored) → `PPY_*` env vars.

**Why:** Type-safe, validated, importable anywhere, self-documenting. Local JSON for machine-specific paths, env vars for CI/containers.

**Alternatives considered:** Plain JSON only; `.env` file.

---

## Phase 1 — SAM as primary optimizer for face detection

**Decision:** Use SAM-trained YOLOv8n weights (`face_detector_sam_best.pt`) as the detection backbone.

**Why:** Best generalization — mAP50 0.616, mAP50-95 0.327 (vs SGD 0.608/0.319, Lion 0.580/0.306). See [[50-experiments]].

**Trade-off:** ~2× training time (double forward/backward). Acceptable — training is offline.

---

## Phase 1 — YOLOv8n on WIDER FACE

**Decision:** YOLOv8n (nano) trained on WIDER FACE for face detection.

**Why:** YOLO family efficient + effective on classroom data ([[rw-ananda-yolo-retinaface]]). Nano = compute-efficient, aligns with "accessible on standard hardware" goal.

**Open:** YOLOv9m may be better ([[rw-ahmed-yolov9-classroom]]) — see [[60-open-questions]].

---

## Links

- Open questions: [[60-open-questions]]
- Experiments: [[50-experiments]]

#phase1/detection
