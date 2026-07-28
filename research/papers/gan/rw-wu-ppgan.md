# Wu et al. — PP-GAN (Privacy-Protective GAN)

**Citation:** Wu, Yang, Xu, Ling — *J. Comput. Sci. & Technol.* 34(1):47–60, 2019 (`yifan-wu`)

## Summary

Made the privacy–utility trade-off **explicit**. pix2pix cGAN (U-Net generator, PatchGAN discriminator, 128×128 grayscale) steered by two frozen external modules:
- **Verificator** — Siamese Light CNN-9 (256-d embedding), contrastive loss (margin α=2), pair fed as "different identity" → pushes output away from source identity
- **Regulator** — SSIM keeps luminance/contrast/structure

Objective: $L = L_{cGAN} + \lambda_1 L_{verif} + \lambda_2 L_{sim}$. **Key insight:** privacy and structure are *contradictory* inside the GAN loss → must be explicit external losses, not left to adversarial training.

## Evaluation (MORPH male subset, 8 race×age subgroups)

- **Held-out verifier**: de-ID rate measured with a DIFFERENT model (fine-tuned FaceNet, triplet loss) than the training verificator → **93.7–100% de-ID rate**
- **0 identity switches** (IDS check against gallery)
- **Utility**: MTCNN detection rate preserved (~94–99.5%); age-attribute accuracy 16.9%→86.9% with subgroup generators
- **4-way loss ablation**: cGAN / +SSIM / +Verif / +both plotted as de-ID × SSIM

## Relevance to us

- **Template for encoding the trade-off in the loss** — adopt for GAN track; extend regulator beyond SSIM with expression/gaze terms (3DMM, gaze) since SSIM alone doesn't preserve affect
- **Adopt evaluation protocol**: (a) held-out verifier for privacy, (b) de-ID rate + identity-switch rate, (c) detection-rate as utility, (d) 4-way loss ablation → [[40-evaluation]]
- Cite as GAN-side baseline for "explicit trade-off encoding"; contrast: it preserves *attributes* (age/race), we preserve *behavior* (expression/gaze/affect)

## Limitations

- Frontal faces only; fails on pose/occlusion
- Preserves attributes (age/race) but **not expression/gaze/affect** — our differentiator
- No temporal consistency (video = future work), single-image 128×128

## Links

- Related work: [[20-related-work]]
- Evaluation metrics: [[40-evaluation]]

#phase2/generation #privacy #utility
