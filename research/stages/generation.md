# Stage: Generation (Synthetic Face Replacement)

Second pipeline stage: replace each detected face with a non-identifying synthetic surrogate preserving pose, expression, scene coherence.

## Goal

Expression-preserving de-identification via GAN or diffusion. Condition on detected expression + pose.

## Status: ⬜ Not started

`src/generate/` and `src/inpaint/` are empty stubs.

## Candidate families

- [[gan]] — GANonymization, G2Face, PP-GAN
- [[diffusion]] — ReferenceNet, gradient injection

**Decision pending:** [[gan-vs-diffusion]]

## Temporal consistency

- Reuse previous frame's synthetic face as reference
- Combine with gradient injection to guide identity while keeping structure
- **Critical challenge:** visual continuity over time

## Open design decisions

- GAN vs diffusion (primary generator)
- Conditioning signal (landmarks? 3DMM? driving image?)
- One synthetic identity per person across video, or per-segment?
- Demographic similarity: control + measurement

## Links

- Method: [[30-method]]
- Evaluation: [[40-evaluation]]
- Open questions: [[60-open-questions]]

#phase2/generation #phase2/inpainting #stage
