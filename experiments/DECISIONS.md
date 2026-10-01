# Experiment design decisions

Decided on 2026-09-29, after reviewing the planned experiments against the
research question (`research/stages/10-problem.md`), the method and evaluation
notes, and what `paper/main.tex` currently promises. Each entry says what was
decided and why. Changes to the paper text that follow from these decisions are
listed at the end; the paper is not edited in this PR.

## Scope

1. **Three experiment families plus instrument validation.** Detectors (E1),
   generators in their original form (E2), and generators with vs without our
   ideas (E3). E0 validates the instruments that every other experiment relies
   on. These map one-to-one onto the paper's research question, objectives and
   hypotheses (see the table in `README.md`).
2. **Only the ideas already implemented enter the paper:** C2 (track-level
   real identity), seed candidates with Phase 1 boxes, and P2 (identity push in
   the swap embedding). P3 (CIAGAN identity-code optimization) and P1 (SDXL
   guidance) become future work. The paper reports measured contributions, not
   planned ones.
3. **E1 uses WIDER FACE val plus our videos.** Our classroom videos have no
   face annotations, so on their own they only give operational statistics.
   WIDER FACE gives AP, precision / recall / F1 and recall at fixed false
   positives under the standard protocol. Annotating classroom frames was
   considered and not adopted (cost, and it means handling more real faces).
4. **GANonymization is not part of E3.** It has no identity input, so our
   ideas do not apply to it. It stays in E2 as a baseline.
5. **A censorship baseline (blur, mosaic) joins E2.** The chronogram promised
   it, and it is the utility floor every generator must beat.

## Metrics

6. **The primary privacy measure is closed-set rank-1 by FaceNet**, a
   recognizer never used to build or tune the pipeline. ArcFace, the space the
   P2 push is computed in, is always reported next to it: a gap between them
   means the method fools only its own model. This is the held-out verifier
   the paper promised.
7. **Rank-5 and Privacy Gain are added.** The paper promised Rank-N
   identification; Privacy Gain is derived from rank-1 at no cost.
8. **Emotion agreement joins the utility metrics.** Hypothesis (b) and
   objective 3 are about affect; the landmark-based expression error alone is a
   proxy. An off-the-shelf facial-expression classifier will be chosen by
   license and accuracy.
9. **Head-pose error joins the utility metrics.** The paper promises pose
   preservation. It uses buffalo_l's 3D-68 model, already in the pipeline, with
   no new dependency.
10. **Gaze error joins the utility metrics if a permissively licensed gaze
    estimator runs reliably.** Gaze is named in the research question.
    Otherwise it is declared a limitation.
11. **Attribute preservation (age error, gender agreement) joins the utility
    metrics.** The research question requires a demographically similar
    surrogate, and the paper promises a check for demographic distortion. It
    uses buffalo_l's age/gender model, with no new dependency. It also measures
    the infant-face bias found in BLANKET's original prompt.
12. **Every utility metric gets a noise floor** (E0): the same metric between
    consecutive real frames of a track. A generator's drift is read against
    what the probe itself fluctuates.
13. **PSNR and SSIM are dropped.** They reward pixel similarity, which a method
    that replaces the face on purpose should not have. Re-detection stands for
    analyzability; face image-quality assessment stays optional.
14. **Statistics:** per-video and pooled values with 95% bootstrap confidence
    intervals over tracks; paired comparisons between arms that share faces
    and identities (paired bootstrap; McNemar for per-face rank-1).
15. **Robustness is stratified by face size and head pose**, the hard
    conditions of a classroom camera, because demographic stratification is not
    possible (next section).

## Declared limitations (what the paper promised and will not measure)

16. **Parrot / imitation attack**, a matcher retrained on anonymized footage.
    It needs an anonymized gallery and its own protocol, outside this work's
    time budget. Declared as a limitation; held-out rank-1 is the attack model
    used.
17. **Commercial face-recognition APIs are dropped** on ethical and LGPD
    grounds: sending minors' faces to third-party services is not acceptable.
    The paper will say so.
18. **ID-switch rate is not measured.** It needs ground-truth tracks. A
    fragmentation proxy (share of tracks shorter than 10 frames) is reported
    instead.
19. **Human-rated pedagogical analyzability is out of scope.** There are no
    raters or protocol. It becomes future work.
20. **No demographic fairness analysis.** There are no demographic labels and
    no consented children's footage. The adult-trained models are a stated
    limitation.

## Conclusion criteria (fixed before running)

21. **Privacy improves** when held-out rank-1 drops and the paired confidence
    interval against the reference arm excludes zero.
22. **Suppression target** (aspirational, to confirm with the advisor):
    verification at FAR 1% ≤ 5% and rank-1 ≤ 2× chance. Exploratory runs show
    BLANKET + P2 at rank-1 ≈ 0.3, above this. If that holds, the paper reports
    it as it is.
23. **Utility is preserved** when emotion agreement and the expression / pose
    / gaze errors stay within the reference arm's confidence interval, read
    against the E0 noise floor.
24. **The final pipeline is chosen after the experiments:** the best
    privacy–utility trade-off, with coverage as a constraint.

## Structure and outputs

25. **Self-contained experiment folders.** Each has `README.md`,
    `config.toml` and `run.py`. `run.py` deletes `results/` and regenerates it,
    recording every command (`commands.sh`) and the provenance
    (`provenance.json`).
26. **Results stay out of git.** They contain component outputs with real
    backgrounds. The aggregate LaTeX artifacts the paper cites are exported to
    `paper/generated/`, which is versioned as part of the paper.
27. **Paper formats match `paper/main.tex`.**
    - Tables are booktabs tabulars only, with the caption and label kept in the
      paper.
    - Figures are vector PDF with the plotted data as CSV.
    - Numbers cited in the text go through `\result{key}`, which stops the
      compilation if the key is missing.
28. **Example frames are never generated automatically.** They show real
    people in the background, so their use is a manual, consent-checked
    decision.
29. **Frozen SDXL identities are cached outside `results/`**, keyed by video
    hash and settings, so every arm of an experiment compares the same
    identities. `--fresh` clears the cache.
30. **`bin/` gives short single-step commands** for exploration. The
    experiments do not depend on it.
31. **No renaming of the pipeline code now.** The components are used as they
    are. Naming and the choice of the default pipeline come after the
    experiments.

## Recall and fail-safe (2026-09-30)

Measured on `video-demo-3.mov` with BLANKET + P2 at β 1.2. **59% of the
faces Phase 1 knew about were shown real in the output**
(`research/next-steps/exposure-audit-2026-09-30.md`):
- 2587 generation failures;
- 519 frames where a track lost the face;
- about 64 frames before each track was confirmed.

The literature review is in `research/stages/identification-occlusion.md`.

32. **Fail closed: a face the pipeline knows about is never shown real.**
    - The final video (`compose_video.py`) hides every face box that was not
      generated. It uses the track's last generated face if that face is at
      most 10 frames old, and a neutral gray ellipse otherwise.
    - Blur and pixelation are not used, because both are reversible or
      attackable.
    - `--failsafe off` exists only for diagnosis.
33. **Only the pixels the generator changed are pasted back.** Pasting the
    whole context crop let a later crop's real background overwrite part of
    an earlier neighbour's generated face. That affected 432 of 2139 generated
    faces (20%), 125 of them over at least half the box.
34. **Phase 1 recall post-pass, on by default (`phase1_detect/fill.py`).**
    - Tracks are emitted from their first detection.
    - Gaps inside a track, up to 30 frames (the tracker's `max_missed`), get
      interpolated boxes.
    - Confirmed tracks are dilated by 3 frames at each end.
    - Tracks with fewer than 3 detections are hidden but get no synthetic
      identity.
    - Every box records its `source`. Only `detector` boxes reach the
      generator and the privacy and utility metrics.
    - `--no-fill` turns the pass off.
35. **SCRFD-10GF now honors its threshold.**
    - Until 2026-09-30 it silently cut at 0.5, because
      `FaceAnalysis.prepare()` defaults `det_thresh` to 0.5 and it was not
      passed. SCRFD-34GF always passed it. **Every earlier number, including
      the first E0 run, used 0.5.**
    - The pipeline default is now `conf = 0.1`, feeding ByteTrack's
      low-confidence association. New tracks still start only at ≥ 0.5.
    - On demo3 this brought 2066 more matched faces and cut the gaps from 586
      to 47 frames. Tracks stopped fragmenting: 22 tracks instead of 32.
    - E0, E2 and E3 use 0.1. E1 keeps its own operating points.
36. **BLANKET swaps the face Phase 1 found.**
    - The bridge builds the target face from Phase 1's box and 5 points with
      FaceFusion's own `create_faces()`. It no longer re-detects with
      `yolo_face` at 0.5, which had missed 1417 faces, and it drops the IoU
      filter, which had rejected 38.
    - Identity images that `yolo_face` rejects are retried with FaceFusion's
      `many` detector at 0.25. That recovered all 13 of the 32 identities
      rejected on demo3; they are small images, 58–269 px.
    - `--blanket-detection upstream` keeps BLANKET's own flow. It becomes a
      reference arm in E3b.
37. **Exposure metrics.**
    - The final video reports the share of face boxes that were generated,
      reused and filled (`compose.jsonl`, `evaluate.py` → `composition`,
      `bin/audit`).
    - With the fail-safe, the exposure of known faces is zero by construction.
      The residual is faces Phase 1 never found, which needs ground truth:
      annotated sampled frames, in the next PR.
    - The privacy metrics keep measuring the generator on detector faces, so
      they stay comparable across arms.
38. **Known issue: identity seeds derive from the video's path string**
    (`derive_seed(video_source, track_id)`). The same video under another path
    gets other seeds and misses the identity cache. This is to be fixed, with
    a content hash, when the caches are next rebuilt.

## Paper text to revise (not done here; `paper/main.tex` has uncommitted edits)

- **Methodology, Stage 3.** Replace gradient injection on a canonical latent
  with what was built and measured: C2 (track-level real identity) and P2
  (identity push in the swap embedding), with the identity fixed per track.
- **Remove Reverse Personalization's attribute conditioning**, which was
  dropped on 2026-09-23. Attribute preservation is now measured instead.
- **Remove commercial FR APIs** (decision 17).
- **State parrot attack, ID-switch rate, human rating and demographic fairness
  as limitations** (decisions 16, 18–20).
- **Update the compute statement.** "16GB RAM, no GPU cluster" no longer
  holds: experiments run on a shared GPU server.
- **Add the conclusion criteria** (decisions 21–24) to the evaluation section.
- **Add `\input{generated/results}` to the preamble**, and use the generated
  tables and figures.
- **Methodology, Stage 1 and composition.**
  - Add the recall post-pass (decision 34) and the fail-closed composition
    (decisions 32–33).
  - State that the detector threshold is 0.1 with ByteTrack's low-confidence
    association (decision 35).
