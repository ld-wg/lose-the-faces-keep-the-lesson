# Research Index — Lose the Faces, Keep the Lesson

Expression-preserving face de-identification for educational research.

> Open this folder as an Obsidian vault to see the graph. All links are `[[wikilinks]]`.

## 📁 Papers

**Detection:** [[rw-ananda-yolo-retinaface]], [[rw-ahmed-yolov9-classroom]] ⚠️, [[rw-ahmed-child-yolo-anonymization]]
**GAN:** [[rw-wu-ppgan]], [[rw-yang-g2face]], [[rw-hellmann-ganonymization]]
**Diffusion:** [[rw-kung-referencenet]], [[rw-wang-gradient-injection]], [[rw-klemp-ldfa]]
**Survey:** [[rw-meden-bpet-survey]]

**➕ New (2025–2026, from pipeline validation — notes TODO):** BLANKET (ICDL'25, child-face), Reverse Personalization (WACV'26, Kung successor), AnonNET (ICCV-W'25), NullFace (FG'26), Muştu & Ekenel (DSP'25), FDeID-Toolbox (eval). See [[pipeline]] §6.2.

> ⚠️ [[rw-ahmed-yolov9-classroom]] is **unverified** — its PDF is actually Srivastava 2021 (generic objects). The verified child-face YOLOv9 source is [[rw-ahmed-child-yolo-anonymization]].

## 📁 Families

- [[detection]] — YOLO/RetinaFace for stage 1
- [[gan]] — GAN-based anonymization
- [[diffusion]] — diffusion-based anonymization

## 📁 Stages

- [[10-problem]] — research question, hypothesis, scope
- [[identification]] — stage 1: face detection ✅
- [[generation]] — stage 2: synthetic replacement ⬜
- [[30-method]] — full pipeline design
- [[40-evaluation]] — privacy & utility metrics
- [[50-experiments]] — optimizer study results

## 📁 Open Problems

- [[60-open-questions]] — everything to challenge

## 📁 Next Steps

- [[roadmap]] — immediate actions + Phase 2 kickoff
- [[gan-vs-diffusion]] — the pivotal generator decision
- [[70-decisions]] — decision log

## Overview

- [[20-related-work]] — related work narrative

## Status

| Phase | State | Notes |
|-------|-------|-------|
| 1 — Detection | ✅ Done | YOLOv8n, SAM best (mAP50 0.616) |
| 2 — Generation | ⬜ Not started | `src/generate/`, `src/inpaint/` empty |

## Tags

#phase1/detection #phase2/generation #phase2/inpainting #family/gan #family/diffusion #family/detection #privacy #utility #evaluation #open-question #decision #next-steps #stage
