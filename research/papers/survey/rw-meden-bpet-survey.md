# Meden et al. — Privacy–Enhancing Face Biometrics (B-PET Survey)

**Citation:** Meden, Rot, Terhörst, Damer, Kuijper, Scheirer, Ross, Peer, Štruc — IEEE TIFS 16, 2021 (`meden2021bpet`). DOI: 10.1109/TIFS.2021.3096024

## Summary

Canonical survey of Biometric Privacy-Enhancing Techniques (B-PETs) for face. Taxonomy by pipeline level:
- **Image-level** — obfuscation (mask/filter/transform), adversarial, **synthesis** (k-Same/AAM → GAN/VAE/cGAN). ← our home
- **Representation-level** — template transform / elimination / homomorphic encryption
- **Inference-level** — modified matching (negative face recognition, PE-MIU)

Characterizes B-PETs on 6 axes: input, reversibility, targeted attribute, utility strategy (reduction vs **retention**), guarantees (k-anonymity, DP), target (human/machine).

## Evaluation methodology (Sec. IV) — directly reusable

- Privacy efficiency: verification ROC/EER + CMC on original vs enhanced → **Privacy Gain** $PG=(1-R_p)-(1-R_o)$
- Soft-biometric: **Suppression Rate** $SR=(A_o-A_p)/A_o$
- Joint privacy–utility: **PIC** = attr-error gain − recog-error loss
- Utility proxies: PSNR/SSIM (⚠️ poor proxy for generative methods)
- Attack models: **parrot/imitation**, reconstruction, linkage

## Relevance to us

- Framing/vocabulary for Related Work; positions us as image-level synthesis + utility retention
- Justifies our privacy suite: embedding sim + ASR (verification), Rank-N-T (CMC), API confidence (vanilla matcher)
- **Action:** add PG/PIC metrics + a parrot attack (retrain matcher on de-ID data) to [[40-evaluation]]
- Confirms temporal consistency & demographic fairness as open problems = our niche

## Limitations

- Survey only — no new method; predates diffusion-based de-ID (2021 cutoff)
- k-anonymity guarantees are closed-set/still-image; no video formal guarantees
- Notes PSNR/SSIM unsuitable for generative output

## Links

- Related work: [[20-related-work]]
- Evaluation: [[40-evaluation]]
- Open questions: [[60-open-questions]]

#survey #privacy #evaluation #taxonomy
