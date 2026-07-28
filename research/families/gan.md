# GAN-based Face Anonymization

Family overview of GAN approaches for expression-preserving de-identification.

## Papers

- [[rw-wu-ppgan]] — explicit privacy–utility trade-off (verificator + regulator)
- [[rw-yang-g2face]] — 3D geometric prior + identity-aware fusion
- [[rw-hellmann-ganonymization]] — landmark carrier, best emotion preservation

## Evolution

1. **PP-GAN** made the trade-off explicit: verificator (contrastive loss on embeddings) pushes identity away, regulator (SSIM) keeps structure
2. **G2Face** added a 3D geometric prior + identity-aware fusion for higher fidelity/control; reversible "password" path
3. **GANonymization** went expression-first: compress to dense landmarks, re-synthesize from minimal carrier

## Strengths

- Photorealism with controllable identity suppression
- Natural place to encode privacy–utility trade-off in the loss
- GANonymization: strong emotion preservation (good for affect analysis)

## Weaknesses

- No formal privacy guarantees
- No temporal consistency (image-level)
- Dataset/demographic bias
- Computationally heavy, brittle under occlusion/rare poses
- GANonymization relies on landmark sufficiency

## For our research

- **Best candidate for emotion preservation:** GANonymization
- **Best for explicit privacy control:** PP-GAN-style loss
- See [[gan-vs-diffusion]] for the head-to-head decision

## Links

- Related work: [[20-related-work]]
- Method: [[30-method]]
- Open questions: [[60-open-questions]]

#phase2/generation #family/gan
