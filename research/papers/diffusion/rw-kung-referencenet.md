# Kung et al. — Face Anonymization Made Simple (ReferenceNet + SD)

**Citation:** Kung, Varanka, Saha, Sim, Sebe — WACV 2025 (`kung`). Code: `github.com/hanweikung/face_anon_simple`

## Summary

Latent-diffusion U-Net with **two ReferenceNet branches** (source = identity, driving = pose/expression/background) so the synthesized face stays coherent while a **single anonymization knob $d$** controls distance from source identity. Anonymization feeds the *same image* to both branches and perturbs the source branch.

## Implementation details

- ReferenceNet mirrors U-Net; U-Net + both ReferenceNets inherit Stable Diffusion v2.1 weights
- Trained on CelebRef-HQ, CelebA-HQ, FFHQ with triplets (source, GT, face-swapped "driving" image via InSwapper)
- **Single MSE reconstruction loss only** — no identity loss, no landmarks/masks (key contrast vs GANs)
- Fine-tunes U-Net + only attention blocks in ReferenceNet
- Knob mechanics: $Z'=(1-d)Z$ and $S'=(1-d)S_{cond}+dS_{uncond}$; $d>1$ extrapolates *away* from identity. Operating points: $d{=}1.2$ (utility) / $d{=}1.4$ (privacy)
- **Seed diversity** → different noise = different anonymous identity (enables consistent per-student pseudonym)
- 512×512, 435k steps, DDPM 200 steps, guidance 4.0, 2×A6000

## Results (2,000 test images: 1,000 CelebA-HQ + 1,000 FFHQ)

| Method | Re-ID↓ (C/FF) | Pose↓ | Gaze↓ | Expr↓ | IQA↑ |
|--------|---------------|-------|-------|-------|------|
| DP2 | 0.020/0.046 | 0.140/0.194 | 0.244/0.252 | 10.14/9.61 | 0.459/0.480 |
| **Ours d=1.2** | 0.053/0.098 | **0.048/0.047** | **0.161/0.166** | **8.26/7.77** | 0.701/0.698 |
| **Ours d=1.4** | **0.008**/0.039 | 0.074/0.061 | 0.190/0.206 | 13.13/10.90 | **0.707/0.704** |

**Best pose/gaze/expression preservation at d=1.2**; near-best re-ID at d=1.4; IQA second only to FALCO (1024² native).

## Relevance to us

- **Strongest diffusion candidate** — only one with explicit continuous privacy–utility control; wins on our utility axes (pose, gaze, expression)
- **The knob $d$ IS our privacy–utility trade-off** — inference-time, no retraining; sweep for eval chapter figure
- **Seed diversity answers an open design question**: fix seed per student → consistent pseudonymous identity across frames (helps temporal)
- Image-level baseline fitting classroom needs

## Limitations

- **Fairness failures specifically on infants & ethnic minorities (e.g., Asian individuals)** — critical since our subjects are children; stratified eval mandatory
- Re-ID slightly above FALCO/RiDDLE at matched settings (0.008 vs 0.005 CelebA-HQ at d=1.4)
- No temporal modeling
- 512×512 native; 200 DDPM steps/frame → slow for video

## Links

- Related work: [[20-related-work]]
- Method: [[30-method]]
- Open questions: [[60-open-questions]] (fairness)
- Decision: [[gan-vs-diffusion]]

#phase2/generation #privacy
