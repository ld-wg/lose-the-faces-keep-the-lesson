# Related Work — Overview

## Detection

- [[rw-ananda-yolo-retinaface]] — YOLOFace vs RetinaFace on classroom data; YOLO faster, RetinaFace more accurate
- [[rw-ahmed-yolov9-classroom]] — YOLOv9 > v5/v8 on children's faces; accuracy over latency for our use case

**Takeaway:** YOLO family fits our detect-then-anonymize pipeline; accuracy matters more than marginal latency (non-real-time).

## Anonymization evolution

1. **k-Same variants** — early encoder–decoder, k-anonymity-inspired; traded identity suppression vs visual utility
2. **PP-GAN** ([[rw-wu-ppgan]]) — made privacy–utility trade-off explicit: verificator (contrastive loss) pushes identity away, regulator (SSIM) keeps structure
3. **G2Face** ([[rw-yang-g2face]]) — generative prior + 3D geometric prior + identity-aware fusion; reversible "password" path, but no video modeling
4. **GANonymization** ([[rw-hellmann-ganonymization]]) — landmark carrier, best emotion preservation; image-only, relies on landmark sufficiency
5. **ReferenceNet diffusion** ([[rw-kung-referencenet]]) — latent-diffusion + ReferenceNet branches, single anonymization knob; strong identity suppression, no temporal modeling
6. **Gradient injection** ([[rw-wang-gradient-injection]]) — adversarial identity manipulation in latent space; targeted impersonation, not expression-preserving de-ID

## What GANs buy us

✅ Photorealism, controllable identity suppression, natural privacy–utility encoding
❌ No formal guarantees, no temporal consistency, dataset/demographic bias, computationally heavy, brittle under occlusion/rare poses

## What diffusion buys us

✅ Strong identity suppression with high attribute fidelity, controllable anonymization level
❌ Doesn't always beat StyleGAN-inversion on re-ID, drops for underrepresented groups, no temporal modeling

## Datasets referenced

- **Detection:** WIDER FACE (used), LFW, FDDB
- **Expression eval:** AffectNet (~0.4M, 8 emotions), CK+ (593 seq @ 30fps), FACES (2,052 frontal)
- **Generator training:** CelebA-HQ, FFHQ, CelebRef-HQ, LADN

## Links

- Method: [[30-method]]
- Open questions: [[60-open-questions]]
- Source: `paper/main.tex` §Related Works

#phase2/generation #privacy #utility
