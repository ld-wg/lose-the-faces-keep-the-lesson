# Wang et al. — DiffAIM (Gradient Injection)

**Citation:** Wang, Hu, Lu, Luo — ACM MM '25 (`wang`). arXiv:2504.21646. Method name: **DiffAIM**.

## Summary

**Active identity manipulation** via gradient injection during reverse diffusion. Edit-friendly DDPM inversion maps source face to latent space; iterative gradient guidance moves features toward target identity while diverging from source. U-Net self-attention preserves facial structure; late-stage truncation enhances visual quality.

**Key design:** divergence is **U-Net-driven, not FR-driven** — naive FR-driven divergence caused artifacts, so they push away from source using U-Net intermediate-block features while pulling toward target using FR models. Adaptive ensemble of white-box FR models (similarity-weighted). $N_a=10$ inner steps, $\lambda=0.1$, truncation $t_s=20/100$.

## Evaluation

- Verification: CelebA-HQ, LADN (grouped source–target pairs); ASR at 0.01 FAR
- Identification: 500 CelebA-HQ probe–gallery pairs; Rank-N-T
- Black-box FR (IRSE50, IR152, FaceNet, MobileFace) + Face++ & Aliyun APIs
- Image quality: PSNR/SSIM/FID; robustness (JPEG/bit-reduction/resize)

## Results

- Verification ASR avg **86.13%** (vs GIFT 81.27%, DiffAM 77.88%) — best
- Identification: +7% Rank-1-T, +15% Rank-5-T over SOTA
- Best image quality: PSNR 27.68, SSIM 0.811, FID 15.56
- API confidence ~72 (Face++) / ~55 (Aliyun)

## Relevance to us

- **Temporal consistency idea:** reuse previous synthetic face + gradient injection to guide identity toward a *consistent synthetic target* across frames while self-attention regularization preserves expression/pose structure (see [[30-method]]). U-Net-driven divergence shows you can push identity without wrecking visual quality.
- Attack-based evaluation protocol (ASR, Rank-N-T, API confidence) → [[40-evaluation]]
- **Gap = our contribution:** no expression/gaze/affect objective, no temporal modeling, no utility metrics. We add those.

## Limitations

- Focus is **targeted impersonation**, not expression-preserving de-ID
- No temporal modeling, no formal privacy/fairness analysis
- Adapting to video needs extra consistency constraints
- ⚠️ Does NOT use 3DMM coefficients or pose/gaze metrics (only PSNR/SSIM/FID) — corrected misattribution in [[40-evaluation]]

## Links

- Related work: [[20-related-work]]
- Method: [[30-method]]
- Evaluation: [[40-evaluation]]

#phase2/generation #phase2/inpainting #privacy
