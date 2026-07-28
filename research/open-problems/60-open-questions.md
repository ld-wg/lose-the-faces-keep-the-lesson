# Open Questions

Everything we want to challenge. Grouped by theme.

## Detection

- [ ] **YOLOv8n vs YOLOv9m?** Ahmed showed v9 > v8 on children's faces ([[rw-ahmed-yolov9-classroom]]). Worth upgrading for higher recall? Trade-off: model size vs accuracy (non-real-time, so accuracy wins?)
- [ ] **Is WIDER FACE representative enough?** Should we evaluate/collect classroom-specific data (UFPR collection with consent)?
- [ ] **Is 30 epochs enough?** Does longer training change the optimizer ranking?
- [ ] **Recall threshold?** What recall is "high enough" that no face goes unnoticed? How do we measure misses in classroom conditions?

## Generation approach

- [ ] **GAN vs diffusion?** GANonymization (landmark, emotion-preserving) vs ReferenceNet diffusion (coherent, tunable). Trade-off: emotion preservation vs identity control + fairness?
- [ ] **How to condition on expression/pose?** Landmarks? 3DMM coefficients? Driving image? What's the minimal sufficient carrier?
- [ ] **One synthetic identity per person across the whole video, or per-segment?** Affects both privacy and temporal consistency.
- [ ] **Demographic similarity: how to control + measure?** What does "demographically similar" mean operationally?

## Temporal consistency

- [ ] **Is naive face reuse + gradient injection enough?** Or do we need explicit video modeling?
- [ ] **How to handle identity locking?** Ensure the same person gets the same synthetic face across frames/scenes?
- [ ] **Flicker?** How to measure + suppress frame-to-frame flicker in the synthetic face?

## Evaluation

- [ ] **Thresholds?** What values define "acceptable" identity suppression / emotion drift?
- [ ] **Which FR backbone(s)** for embedding similarity + attacks? (ArcFace? Multiple?)
- [ ] **Which emotion classifier** for agreement?
- [ ] **Human-rated pedagogical analyzability:** protocol? raters? rubric? How many?
- [ ] **Is attack-based evaluation sufficient** without formal guarantees?

## Fairness & ethics

- [ ] **How to measure + mitigate demographic bias** in synthetic generation? (Kung showed drops for underrepresented groups — [[rw-kung-referencenet]])
- [ ] **Consent + governance:** what's the protocol for UFPR classroom collection?
- [ ] **Non-face identifiers** (voice, clothing, body): acceptable to leave un-anonymized? Disclose how?

## Scope

- [ ] **Is image-level + smoothing defensible** vs. video-native generation? Where's the line for "good enough" temporal consistency?

## Links

- Problem: [[10-problem]]
- Method: [[30-method]]
- Evaluation: [[40-evaluation]]
- Decisions: [[70-decisions]]

#open-question #phase1/detection #phase2/generation #evaluation #privacy #utility
