# Experiment design decisions

The versioned decision record. Each entry says what was decided and why.
Entry numbers are stable: the thesis (`paper/tcc-latex`) and the code cite
them. Superseded entries stay, with a note.

Every number in the thesis comes from the final campaign on constantine
(2026-11, demo2 + demo4, GTX 1080). Older numbers below explain why a decision
was taken; they are not results.

## Scope (2026-09-29)

1. **E0–E3.** E0 validates the instruments. E1 compares detectors, E2 the
   generators in their published form, E3 the generators with and without our
   ideas. E5 (cost) was added later (50).
2. **Only implemented ideas are reported.** C2 (track-level real identity),
   Phase 1 detection in the swap, and P2 (identity push in the swap embedding).
   *Superseded by 43–44:* P1 and P3 were implemented and are reported as
   negative results.
3. **E1 uses WIDER FACE val plus our videos.** The videos have no face
   annotations, so they only give operational statistics. Annotating classroom
   frames was not adopted (cost, and more handling of real faces).
4. **GANonymization stays out of E3.** It has no identity input. It is an E2
   baseline.
5. **Blur and mosaic join E2** as the censorship baseline and the utility
   floor.

## Metrics

6. **Primary privacy recognizer: FaceNet closed-set rank-1.** FaceNet is never
   used to build or tune the pipeline. ArcFace, the space P2 works in, is
   always reported next to it. A gap between them means the method only fools
   its own model.
7. **Rank-5 and Privacy Gain** are reported as well.
8. **Emotion agreement** is a utility metric, because the hypothesis is about
   affect.
9. **Head-pose error** is a utility metric (buffalo_l 3D-68).
10. **Gaze is a stated limitation.** No permissive estimator was validated on
    faces this small.
11. **Age error and gender agreement** (buffalo_l genderage) check demographic
    drift.
12. **Every utility metric has a noise floor** (E0): the same metric between
    consecutive real frames of a track.
13. **PSNR and SSIM are dropped.** They reward pixel similarity, which a
    method that replaces the face should not have.
14. **Statistics.** Per-video and pooled values with 95% bootstrap intervals
    over tracks. Paired comparisons between arms that share faces and
    identities.
15. **Strata.** Face size and head pose (and detection confidence, 41).
    Demographic strata are not possible (20).

## Declared limitations

16. **No parrot attack** (a recognizer retrained on anonymized faces). The
    attacker is naive.
17. **No commercial face-recognition APIs.** Sending minors' faces to third
    parties is not acceptable under the LGPD.
18. **No ID-switch rate.** It needs ground-truth tracks.
19. **No human rating** of pedagogical analyzability.
20. **No demographic fairness analysis.** There are no labels and no consented
    children's footage.

## Conclusion criteria (fixed on 2026-09-29, before the campaign)

21. **Privacy improves** when FaceNet rank-1 drops and the paired interval
    against the reference arm excludes zero.
22. **Suppression target** (aspirational): verification at FAR 1% ≤ 5% and
    rank-1 ≤ 2× chance. Not met; reported as it is.
23. **Utility is preserved** when the utility metrics stay within the
    reference arm's interval, read against the noise floor. This means no
    significant difference, not equivalence.
24. **The final pipeline is chosen after the experiments**, by the
    privacy–utility trade-off, with coverage as a constraint.

## Structure and outputs

25. **One folder per experiment** with `README.md`, `config.toml` and `run.py`.
    Each run regenerates `results/` and records the commands and provenance.
26. **Results stay out of git.** They contain real backgrounds. Only aggregate
    LaTeX keys, tables and figures go to `paper/generated/`.
27. **Paper formats.** Booktabs tables, vector figures with their data as CSV.
    Numbers in the text go through `\result{key}`, which fails the build if the
    key is missing.
28. **Example frames are never generated automatically.** Using one is a
    manual, consent-checked decision.
29. **Frozen SDXL identities are cached**, keyed by video and settings, so
    every arm of an experiment compares the same identities.
30. **`bin/`** has short commands for exploration. The experiments do not
    depend on it.
31. **No renaming of pipeline code** before the defense.

## Recall and fail-safe (2026-09-30)

An audit of one pilot video found that 59% of the faces Phase 1 knew about were
shown real in the output: generation failures, frames where the track lost the
face, and frames before a track was confirmed.

32. **Fail closed.** A face the pipeline knows about is never shown real.
    Composition hides every face box without a valid replacement. It uses the
    track's last generated face if it is at most 10 frames old, and a gray
    ellipse otherwise. Blur and mosaic are not used as fill (reversible).
33. **Only the pixels the generator changed are pasted back.** Pasting the
    whole crop let a neighbour's real background overwrite 20% of generated
    faces.
34. **Phase 1 recall post-pass, on by default.** It fills gaps inside a track
    (up to 30 frames), dilates tracks by 3 frames, and hides short tracks
    without giving them an identity. Only `detector` boxes reach the generator
    and the metrics.
35. **SCRFD-10GF threshold bug fixed.** It cut at 0.5 regardless of the
    setting. The pipeline now runs at 0.1, with ByteTrack's low-confidence
    association; new tracks still start at ≥ 0.5.
36. **BLANKET swaps the face Phase 1 found.** The bridge builds the target from
    Phase 1's box and 5 landmarks, instead of re-detecting with `yolo_face` at
    0.5, which missed small faces. `--blanket-detection upstream` keeps the
    published flow as a reference arm (E3b).
37. **Exposure metrics.** The share of face boxes generated, reused and filled
    (`compose.jsonl`). Faces Phase 1 never found are not measured (no
    annotation).
38. **Known issue:** identity seeds derive from the video's path. Not changed.

## Results campaign design (2026-10-01)

39. **Utility probes:** head pose, age and gender (buffalo_l), emotion over 8
    AffectNet classes (HSEmotion; weights for research use only), 106-landmark
    expression error.
40. **Privacy views.**
    - *Generated only* isolates the generator.
    - *All faces, passthrough as a leak* is the method as published.
    - *Final video* scores what the viewer sees: reused faces as shown, filled
      faces as never re-identified.
41. **Detection-confidence strata** (0.1–0.5 vs ≥ 0.5).
42. **Policies are E3e arms.** `hide-low` hides detections below 0.5. `gate`
    rejects a swap whose ArcFace cosine with the real face reaches the video's
    FAR 1% threshold. The gate uses only ArcFace, so FaceNet stays held out.
43. **P3 implemented** (CIAGAN identity-code optimization against ArcFace).
44. **P1 implemented** (gradient guidance of SDXL's predicted clean latent).
45. **One evaluator call per video or per part**, so arms share the real
    embeddings, the calibration and the faces.
46. **Miss sheets** for annotating faces Phase 1 never found
    (`src/eval/miss_sheet.py`). Not done before the deadline.

## Final pipeline and cost (2026-10-07 to 10-21)

47. **BLANKET's main process ran on CPU in the pilot's E2 and E3**, to leave
    the shared GPU to the bridge servers. Privacy and utility do not depend on
    the device. In the final campaign it runs on the GPU in every experiment
    (`CTX_ID=0`), which is why E5's CPU arm was dropped (72).
48. **Final pipeline** (the E3e `gate` arm): Phase 1 with the post-pass;
    BLANKET with Phase 1 detection, C2 `quality_mean` and P2 β 1.2; the privacy
    gate; fail-closed composition. P1 and P3 are left out.
49. **Diagnostics are `\result` keys**, not hand-copied numbers.
50. **E5, the cost study.** Each arm changes one cost factor of the final
    pipeline: main-process device, GFPGAN (off, ≥ 96 px), fixed sampling, and
    adaptive sampling (regenerate when the landmark error against the last
    generated frame reaches τ, or after 6 frames). Scored on the composed video.
51. **Data protection.** The bridge kept every swapped crop and identity
    debug image. They were deleted, and the bridge now deletes them once read.
52. **E5 changes the final pipeline.** The main process runs on the GPU.
    Adaptive sampling at τ 0.08 is added (about 2.5× faster, no privacy cost,
    smoother motion). GFPGAN stays on: without it, emotion improves but
    verification rises and the gate hides many more faces.

## Wrap-up after the advisor-style review (2026-10-29)

53. **Fail-open path closed.** A swap that changed less than 1% of the box was
    reported as generated. It is now `failsafe` (`unchanged`) and hidden.
    Measured in the final campaign: 8.9% of swaps with Phase 1 detection
    (902 of 10,119).
54. **No crop of a non-replaced face is written.** `experiments/purge.sh`
    deletes what still shows real people after export.
55. **One definition per shared rule** (landmark error, sampling decision,
    reuse window).
56. **Holm–Bonferroni** adjusted p per family, exported next to the raw p.
57. **Derived analyses from exported keys only** (`experiments/derived.py`).
    `experiments/tables_pt.py` writes the Portuguese tables.
58. **New measures:** orientation proxy κ (`orient_kappa`), face-size strata
    for E3e and E5, track counts per video.
59. **Provenance.** `SOURCES.md` records the bridge commit, host, GPU and
    library versions.
60. **Structure.** Composition stays in `phase2_generate/compose_video.py`
    (Stage 3). Unit tests in `tests/` use only synthetic arrays.

## Critical review (2026-10-30)

61. **Primary measure: the privacy × coverage pair.** For each method, the
    share of faces with a synthetic face and FaceNet rank-1 among them.
    Final-video rank-1 alone rewards hiding faces: blur and the released
    BLANKET (most faces hidden) beat the pipeline on it.
62. **New attacks in the evaluator.**
    - `inv_rank`: ranks the least similar identity first, for an attacker who
      knows P2 pushes away.
    - `track_rank`: one decision per student, from the mean shown embedding.
    - `shown_rank`: rank-1 among shown faces; hiding cannot lower it.
    - `identity_audit`: flags pairs of tracks that may be one person.
63. **Framing.** "Expression-preserving" is dropped. The pipeline is described
    as hybrid: diffusion makes the source identity once per track; the pixels
    shown come from inswapper and GFPGAN.
64. **P2 geometry.** At β ≈ c (cosine between the SDXL identity and the real
    person, ≈ 0.35), P2 only removes the real component the identity
    inherited. Above it, the target points toward the real identity's
    antipode, which `inv_rank` measures.
65. **Thesis figures** are pgfplots written by `derived.py`.
66. **Study order, stated in the thesis.** P2 was seen working on 2026-09-26 in
    an exploratory pilot that included demo2. The criteria were fixed on
    2026-09-29. The final pipeline was chosen in October and frozen before
    demo4 was processed.
67. **The swap source is the inpainted face.** The identity image is a context
    crop with real neighbours in it, and BLANKET swapped from the highest-scoring
    face. The swap server now takes the face overlapping `identity_box`
    (IoU ≥ 0.3), else the identity is unusable. Measured: about 2% of identity
    loads used another face before the fix, and about 1% have no overlapping
    face (track hidden).
68. **The diffusion must change the face.** An identity image that changes
    less than 5% of the face box is rejected.
69. **Identity-server placement flags** (`--offload`, attention slicing).
    Same weights, different placement. On the 8 GB GPU only sequential offload
    fits.

## Final campaign (2026-11)

70. **Compute moves to constantine (2026-11-11).** The previous server was
    locked and did not come back.
    - One GTX 1080, 8 GB. Access notes are local-only.
    - Videos: demo2 (same file as in the pilot) and demo4, new. demo4 is the
      only out-of-sample video.
    - Results of the two machines are never compared or pooled.
71. **E3g (P1) swaps on the CPU.** P1 backpropagates through the VAE and
    ArcFace and does not fit next to the swap server on 8 GB. The swap computes
    the same thing on another device; E3g's time is not comparable. `E3_PARTS`
    runs E3 in three steps.
72. **Final review (2026-11-17).**
    - **Mislabelled keys fixed.** Tables labelled "generated faces" used
      all-face rank-1. E3 now exports `<arm>/generated/...`, and `derived.py`
      uses it.
    - **E5 `ref-cpu` dropped.** On constantine it ran on the GPU.
    - **β stays 1.2.** With the new data, FaceNet's drop stops at β 0.8; 1.2 was
      kept so that no hyperparameter is chosen after seeing demo4. No P2 arm is
      significant after Holm.
    - **No GFPGAN skipping in the final pipeline.** Running it only on faces
      ≥ 96 px raises rank-1 among shown faces from about 0.34 to 0.50.
      Frame-to-frame reuse is already adaptive sampling.
    - **The thesis cites only this campaign.**
73. **E3h, swapper comparison (2026-11-17).** The campaign located the leak in
    the swap, so the swapper was varied with everything else fixed.
    - `--blanket-swap-model` reaches the bridge's `--face-swapper-model` (bridge
      `2116af2`). The swap server checks the model it loaded, because BLANKET
      silently falls back to inswapper for models outside its whitelist.
    - Arms: inswapper and simswap_256, each native and with P2 β 1.2.
    - Result: SimSwap leaks more natively (FaceNet rank-1 among swapped faces
      0.58 vs 0.50) and ties with P2 (0.42 vs 0.39, Holm p 0.66). inswapper
      stays.
