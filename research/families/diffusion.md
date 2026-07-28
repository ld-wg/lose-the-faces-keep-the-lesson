# Diffusion-based Face Anonymization

Family overview of diffusion approaches for expression-preserving de-identification.

## Papers

- [[rw-kung-referencenet]] — ReferenceNet + SD2.1, single anonymization knob
- [[rw-wang-gradient-injection]] — gradient injection for identity manipulation

## Approaches

1. **ReferenceNet (Kung):** latent-diffusion U-Net + ReferenceNet branches mirror the U-Net (inherit SD2.1 weights). Trained on triplets (source, GT, face-swapped driving image). Single knob controls identity distance.
2. **Gradient injection (Wang):** invert source to latent, inject gradients during reverse diffusion to move toward target identity / away from source. Self-attention preserves structure; late truncation improves quality.

## Strengths

- Strong identity suppression with high attribute fidelity
- Controllable anonymization level (knob / gradient scale)
- ReferenceNet: coherent pose, gaze, expression, background

## Weaknesses

- Doesn't always beat StyleGAN-inversion on re-ID
- **Drops for underrepresented groups** (fairness concern, Kung)
- No temporal modeling
- Wang's focus is targeted impersonation, not expression-preserving de-ID

## For our research

- **Best candidate for identity control + coherence:** ReferenceNet
- **Best for temporal consistency idea:** gradient injection (reuse previous face + guide identity)
- See [[gan-vs-diffusion]] for the head-to-head decision

## Links

- Related work: [[20-related-work]]
- Method: [[30-method]]
- Open questions: [[60-open-questions]]

#phase2/generation #family/diffusion
