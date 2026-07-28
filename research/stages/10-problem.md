# Problem Statement

## Research question

Can we replace faces in classroom lecture recordings with synthetic, demographically similar faces such that **identity is suppressed** while **expression, gaze, and affect are preserved** well enough for downstream pedagogical analysis?

## Hypothesis

Diffusion-based face replacement (conditioned on detected expression and pose), applied after high-recall detection, will:
- (a) reduce identity similarity below a strict threshold → low re-identification risk
- (b) preserve emotion/expression cues within acceptable drift for pedagogical analysis

## Motivation

- High-value evidence lives in lecture recordings: authentic interactions, turn-taking, instructional movement
- Same footage exposes identities of children/teenagers/teachers → ethical + legal barriers to large-scale studies and sharing
- Goal: preserve behavioral signal, remove personally identifying facial attributes

## Objectives

1. High-recall face detection under classroom conditions (occlusions, board turns, motion) — no face unnoticed
2. Identity suppression per embedding-based + attack-based privacy metrics
3. Affective/behavioral signal retention per automated emotion agreement + human-rated pedagogical analyzability
4. Accessible, compute-efficient pipeline runnable on standard hardware

## Scope

**In scope:**
- Lecture recordings, frontal cameras
- Audio preserved
- Face-level de-identification as primary target
- Image-level replacement + temporal smoothing

**Out of scope** (acknowledged limitations):
- Full body anonymization
- Lip synchrony for fine-grained speech analysis
- Formal privacy guarantees (differential privacy, provable unlinkability)
- Large-scale cloud orchestration
- Video-native generative model / true sequence modeling / identity locking
- Non-face identifiers: voice, clothing logos, body shape, background

## Risks

- Extreme occlusion
- Demographic fairness in synthetic face generation (tracked empirically, stratified analysis)
- Governance: consent, storage policies, auditability

## Links

- Method: [[30-method]]
- Evaluation: [[40-evaluation]]
- Open questions: [[60-open-questions]]
- Source: `paper/main.tex` §Introduction

#phase1/detection #phase2/generation #privacy #utility
