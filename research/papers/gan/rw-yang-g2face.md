# Yang et al. — G2Face

**Citation:** Yang, Xu, Xu, Zhang, Qin, Wang, Heng, He — IEEE TIFS 19, 2024 (`h-yang`)

## Summary

Fuses a **StyleGAN2** generative prior with an explicit **3DMM geometric prior** (shape/expression/pose coefficients via D3DFR) and an **identity-aware fusion (IFF) module**. Three modules: GIE (geometry-aware identity extraction), DPIM (dual prior-guided identity manipulation), PE (password extraction). Swaps identity while preserving pose, landmarks, expression.

**"Password" clarified:** NOT user-chosen — it's the original face's **ArcFace embedding binarized to 16,384 bits**, embedded into the anonymized image via steganography-style hiding, recovered by a Password Extractor CNN. Enables reversibility without a pre-defined password.

## Results

- SOTA anonymization on LFW/CelebA-HQ (lowest true-acceptance + identity cosine-sim vs CIAGAN/FIT/RiDDLE/FALCO)
- Best utility on landmark/expression/pose/shape distance
- Best recovery (MAE/LPIPS/SSIM/PSNR); ~real-time inference; trained FFHQ, 256×256

## Relevance to us

- **Adopt the 3DMM geometric prior + identity-aware fusion idea** — concatenating 3DMM shape/expression/pose coefficients into the identity embedding is a proven mechanism for expression/pose preservation
- **Skip the reversibility** — classroom de-ID doesn't need recovery; password/steganography adds unnecessary complexity
- Utility metrics match ours (landmark/expression/pose) — good eval template
- Position our contribution as **temporal/video extension** (its stated weakness)

## Limitations

- Fails on extreme pose/expression/aging (training-data scarcity)
- Higher complexity (FLOPs/params) though inference ~real-time
- **No video modeling** — image-level only

## Links

- Related work: [[20-related-work]]
- Method: [[30-method]]

#phase2/generation
