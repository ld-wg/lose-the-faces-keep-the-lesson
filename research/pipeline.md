# Pipeline — Lose the Faces, Keep the Lesson

End-to-end strategy for expression-preserving face de-identification in classroom videos.

> **Status: VALIDATED against current literature (2026-07-28).** The 3-network merged architecture is sound. One correction: the generator candidate set was outdated — added BLANKET (child-face specialist) and Reverse Personalization (Kung's successor to ReferenceNet). See §6 Conclusions and §7 Final Pipeline.

---

## 1. Overall strategy (original 5-node design view)

```
Input Video (classroom lecture)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ NODE 1 — DETECTION & TRACKING                                │
│ YOLOv8n (SAM) → per-frame boxes + track IDs                 │
│ Output: face crops, boxes, confidence, track IDs            │
└─────────────────────────────────────────────────────────────┘
    │ per-frame face crops + boxes + track IDs
    ▼
┌─────────────────────────────────────────────────────────────┐
│ NODE 2 — EXPRESSION / POSE / GAZE ESTIMATION                 │
│ Extract conditioning signal (landmarks / 3DMM / gaze)       │
│ Output: per-face expression + pose + gaze vector            │
└─────────────────────────────────────────────────────────────┘
    │ conditioning signal
    ▼
┌─────────────────────────────────────────────────────────────┐
│ NODE 3 — SYNTHETIC FACE GENERATION                           │
│ GAN or Diffusion → non-identifying surrogate face           │
│ Conditioned on expression/pose; fixed identity per track    │
│ Output: synthetic face crop                                 │
└─────────────────────────────────────────────────────────────┘
    │ synthetic face
    ▼
┌─────────────────────────────────────────────────────────────┐
│ NODE 4 — INPAINTING / COMPOSITING                            │
│ Blend synthetic face into original frame (seamless)         │
│ Output: composited frame                                    │
└─────────────────────────────────────────────────────────────┘
    │ composited frames
    ▼
┌─────────────────────────────────────────────────────────────┐
│ NODE 5 — TEMPORAL CONSISTENCY                                │
│ Cross-frame identity lock + flicker suppression             │
│ Reuse previous synthetic face + gradient injection          │
│ Output: stable de-identified video                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Output Video (de-identified, expression-preserved) + audio
```

---

## 2. The multi-network question (central design decision)

**Current design = 5 separate networks/stages.** Each is a distinct model with its own weights, inference pass, and failure mode:

| Node | Network(s) | Params (est.) | Concern |
|------|-----------|---------------|---------|
| 1 Detection | YOLOv8n | ~3M | mature ✅ |
| 2 Expression/pose/gaze | MediaPipe / 3DMM / gaze net | ~10–50M | **is this a separate net?** |
| 3 Generation | ReferenceNet+SD2.1 or pix2pix | ~900M (diffusion) / ~50M (GAN) | heaviest |
| 4 Inpainting | blending / Poisson / learned | ~0–10M | classical or learned? |
| 5 Temporal | gradient injection (reuses #3) | reuses #3 | inference-time |

**The question:** Is this multi-network architecture optimal, or should we **merge** some stages?

### Candidate consolidations

- **Merge 2+3:** Some generators (ReferenceNet, G2Face) take expression/pose *implicitly* via a driving image or 3DMM prior — the "estimation" is internal. Could drop a standalone Node 2.
- **Merge 3+4:** End-to-end inpainting generators (CIAGAN, LDFA) generate the face *in place* — no separate compositing step.
- **Merge 1+2:** Some detectors (RetinaFace) already output landmarks → could feed expression estimation directly.
- **Merge 3+5:** Video-native diffusion (future) would do generation + temporal jointly.

### Tension

- **Modularity (current):** each stage swappable, debuggable, benchmarkable. Aligns with "don't invent a new generator, make existing models video-capable."
- **Merging:** fewer error-compounding boundaries, less compute, but couples design choices and may require training a unified model (out of scope?).

**→ This is what the per-node agents will stress-test.**

### ✅ RESOLVED (ARCH-AGENT, 2026-07-28): target = 3 trainable networks + 2 thin stages

The 5-stage diagram above stays as the **design/logging view** (every boundary is benchmarked), but the **deployment view** merges two boundaries away. Both merges are **training-free** — they select pretrained generators whose published behavior already subsumes the merged stage, so "don't invent a new generator" is preserved.

| Stage | Contents | Verdict |
|-------|----------|---------|
| **1. Detection+Tracking** | YOLOv8n + ByteTrack | **Keep separate.** Recall-critical (a miss = privacy leak), mature, output schema is the pipeline contract. |
| ~~2. Expression/pose/gaze~~ | — | **MERGE into 3.** ReferenceNet needs no explicit conditioning network (driving image = implicit conditioning) and still wins pose/gaze/expression. MediaPipe/3DMM survive only as **evaluation probes** (they compute our utility metrics), not pipeline stages. |
| **2. Generation+Compositing** (was 3+4) | ReferenceNet (primary) / CIAGAN (GAN baseline) | **MERGE 4 into 3.** CIAGAN & LDFA generate the face **in place** (end-to-end inpainting); ReferenceNet synthesizes in context via the driving image. Classical blending (alpha-feather/Poisson, ~0 params) remains a fallback utility for the GAN path, not a stage. |
| **3. Temporal consistency** (was 5) | Gradient injection, reuses generator U-Net | **Keep as stage, not a network.** Inference-time; distinct failure modes (drift, identity lock) deserve a separate benchmark. |

**Rejected merges:** 1+2 (RetinaFace landmarks → expression net) only serves the explicit-carrier path we're abandoning and couples the recall-critical detector to a conditioning choice; 3+5 (video-native diffusion) is the right end-state but out of scope for 20 weeks / standard hardware.

**Why merge (robustness, not compute):** Node 2 (~10–50M) and classical compositing (~0) are negligible next to the ~900M diffusion generator at 200 steps/frame — the compute problem is the generator itself (NODE3-AGENT: DDIM/distillation). The real win is deleting the two **silent quality-loss boundaries**: conditioning→generation (landmark failure → average-face fallback destroys affect) and generation→compositing (seam/skin-tone mismatch). The two boundaries that must stay visible — detection recall and temporal identity — remain separate, benchmarkable stages. Modularity is preserved where it matters; merged stages are still individually ablatable via the GAN-vs-diffusion down-select.

---

## 3. Technical details per node

### NODE 1 — Detection & Tracking ✅ (done)
- **Model:** YOLOv8n, SAM optimizer, WIDER FACE
- **Output:** `[frame_id, track_id, x, y, w, h, conf]`
- **Recall priority:** high; offline so latency OK
- **Open:** YOLOv9-M upgrade? ([[rw-ahmed-child-yolo-anonymization]]); tracking = ? (ByteTrack / DeepSORT — currently unspecified!)

### NODE 2 — Expression / Pose / Gaze ⬜
- **Purpose:** produce the conditioning signal for generation
- **Options:** (a) dense landmarks (MediaPipe 478pt, à la GANonymization), (b) 3DMM coefficients (à la G2Face), (c) driving image (à la ReferenceNet — implicit)
- **Open:** is this a separate network or internal to the generator?

### NODE 3 — Generation ⬜
- **Candidates:** ReferenceNet diffusion (leaning) / GANonymization / CIAGAN / G2Face
- **Requirement:** fixed synthetic identity per track ID (seed-lock per student)
- **Open:** GAN vs diffusion ([[gan-vs-diffusion]]); child-face fairness (gating)

### NODE 4 — Inpainting / Compositing ⬜
- **Purpose:** seamless blending of synthetic face into frame
- **Options:** (a) classical (Poisson blending, alpha feathering), (b) learned inpainting (part of generator, à la CIAGAN/LDFA)
- **Open:** separate step or folded into generation?

### NODE 5 — Temporal Consistency ⬜
- **Purpose:** stable identity + no flicker across frames
- **Plan:** reuse previous synthetic face + gradient injection ([[rw-wang-gradient-injection]])
- **Open:** naive reuse vs gradient injection vs video-native; identity locking mechanism

---

## 4. Cross-cutting concerns

- **Fairness:** child faces (Kung fails on infants); stratified eval mandatory
- **Privacy metrics:** ASR, Rank-N-T, API confidence, PG/PIC, NoA, held-out verifier, parrot attack ([[40-evaluation]])
- **Utility metrics:** emotion agreement, 3DMM L2, pose/gaze, detection mAP, trait removal
- **Compute:** standard hardware target; diffusion is heavy (200 DDPM steps/frame)

---

## 5. Node review log

_Agents append findings here per node._

- [x] Node 1 — Detection & Tracking — **reviewed 2026-07-28 (NODE1-AGENT)**

  **Critical gaps**
  1. **No tracker specified.** Output schema promises `track_id` but detection ≠ tracking. Node 3 needs a fixed synthetic identity per track, so tracking is mandatory. → Specify **ByteTrack** (keeps low-conf detections in association → serves high-recall priority; mitigates hard-condition misses). BoT-SORT/DeepSORT are alternatives if ReID helps with occlusion re-acquisition.
  2. **Hard-condition recall collapse unaddressed.** Ahmed: YOLOv9-M R=0.34 under occlusion/groups/low-light. A missed face = privacy leak. → Add mitigations: tracker box interpolation across miss gaps, recall-tuned confidence threshold, high-recall second pass (RetinaFace) on frames with track drops.
  3. **No classroom-specific eval protocol.** "Tuned to classroom-like conditions" is asserted, never operationalized. → Adopt Ananda's 4-scenario protocol (32–40 students, incl. ~40 lux low-light, P/R/F1 @ IoU≥0.5) + Ahmed hard-condition split.

  **Model choice — YOLOv8n questioned**
  - Our YOLOv8n-SAM: mAP50 **0.616** on WIDER FACE ≪ YOLOv9-M 0.963 (child faces, statistically > v8) and RetinaFace R=1.00 in classroom scenarios.
  - Offline pipeline → latency irrelevant → YOLOv8n's speed advantage buys nothing.
  - → Options: (a) upgrade to **YOLOv9-M** fine-tuned on child/classroom faces; (b) **two-stage cascade** YOLOv8n + RetinaFace on low-confidence/hard frames; (c) at minimum, classroom-domain fine-tune of v8n. Decision needed before Phase 1 sign-off.

  **Improvements / smaller fixes**
  - main.tex says "optionally track" — contradicts the design; tracking is required.
  - "identities/track IDs" conflates anonymous track labels with identification — clarify terminology.
  - "Stable boxes" needs metrics: IDF1, ID switches, box jitter (MOTA) — mAP alone doesn't capture temporal stability, and jitter propagates as flicker into Node 5.
  - Chronogram Week 3 "select detector/FR stack" should include tracker selection + tuning.
  - Annotated `paper/main.tex` with `% [NODE1-AGENT]` comments (Methodology, Expected Results, Chronogram) — no text modified.

- [x] Node 2 — Expression/Pose/Gaze — **reviewed 2026-07-28 (NODE2-AGENT).** Verdict: **Node 2 should probably NOT be a separate network** — merge it into the generator. Findings:

  **Architecture (the key question)**
  - ReferenceNet (Kung 2025) — our strongest generator candidate — has **no explicit estimation stage**: it conditions on a face-swapped driving image and still gets **best-in-table pose/gaze/expression** with a single MSE loss. Merging Node 2 into the generator removes an entire failure boundary and a ~10–50M-param model.
  - Counter-argument for a separate Node 2: modularity, inspectable/editable conditioning, generator-agnosticism.
  - → **Recommendation: default to merged (implicit driving-image conditioning); keep a standalone estimator only as an evaluation probe, not as a pipeline stage.** State this as an explicit design decision in Methodology.

  **Conditioning signal — landmarks vs 3DMM vs driving-image**
  - (a) Dense landmarks (GANonymization): cheap/interpretable, strong emotion preservation — **but** detection failure → silent "average face" fallback destroys expression exactly when classrooms are hardest (occlusion, board turns, looking down); Fear/Happy confusion.
  - (b) 3DMM coefficients (G2Face): proven expression/pose preservation via D3DFR coefficients fused into identity embedding; more pose-robust than 2D landmarks, but degrades on extremes and adds an estimator dependency.
  - (c) Driving image (ReferenceNet): no estimator to fail (no Node-2 failure mode), but signal is not inspectable/editable and inherits InSwapper bias.
  - → **Leaning: implicit (c), with 3DMM (b) as fallback if an editable/auditable signal is required.**

  **Gaze — emergent, not explicit**
  - Kung achieves best gaze preservation (0.161/0.166 angular err) with **no gaze module** → gaze emerges from driving-image conditioning. **Do not build a gaze net.** Evaluate gaze as an emergent metric (quaternion/angular distance, already in the utility plan); add explicit conditioning only if the emergent result fails.

  **Failure modes (currently unaddressed — must specify)**
  - Occlusion / extreme pose / child faces degrade MediaPipe & D3DFR; Kung fails on infants (our subjects are children).
  - Need a stated fallback policy: hold-last-valid conditioning, drop to blur/mosaic censorship baseline, or flag for review. **Avoid GANonymization's silent average-face fallback** — it destroys exactly the affect signal we promise to keep.
  - Per-frame confidence gating: replace only when conditioning confidence > threshold, else censor (couples Node 2 to the Week-7 censorship baseline).

  **Affect ≠ structure**
  - SSIM regulators (PP-GAN) preserve only luminance/contrast/structure, **not** affect. If an explicit carrier is used, the loss must include expression/gaze terms (3DMM L2, gaze), not pixel structure alone.

  Annotated `paper/main.tex` Methodology with `% [NODE2-AGENT]` comments (architecture decision, signal choice, gaze, failure modes, affect-vs-structure). No text deleted.

- [x] Node 3 — Generation — **reviewed 2026-07-28.** Findings:
  - **GAN vs diffusion leaning is justified but conditional.** ReferenceNet ([[rw-kung-referencenet]]) wins on our utility axes (best pose/gaze/expression at d=1.2), the knob $d$ is a free inference-time privacy–utility curve, and seed diversity answers the per-track identity question. BUT the child-face fairness failure (infants/ethnic minorities) is a **gating risk**, not a footnote — decision must be empirical, not a priori.
  - **Child-face fairness mitigation (pick one, state up front):** (a) fine-tune ReferenceNet on a child-face dataset; (b) fall back to GAN primary if degradation confirms on our data. Add a fairness smoke test to Chronogram Weeks 8–9 *before* integration.
  - **Fixed identity per track: seed-locking is necessary but not sufficient.** Also need demographic-similarity control (age/gender/ethnicity-matched surrogate) so de-identified children don't get adult-looking faces — otherwise the surrogate itself introduces distortion. Enforcement mechanism (conditioning vs filtering) is unspecified.
  - **Compute: 200 DDPM steps/frame @ 512² is not feasible for 30 fps video on standard hardware.** Need DDIM/few-step distillation, reduced steps, or frame-subsampling — or state an A6000-class assumption. GANs (pix2pix/CIAGAN) are ~real-time; this asymmetry must be a down-select criterion.
  - **Benchmark CIAGAN as an equal GAN baseline** (matches GANonymization, better on Fear/Happy/Sadness). Shortlisting only GANonymization undersells the GAN family. G2Face = related work, not a candidate (reversible path unneeded, complexity).
  - **Down-select criteria should be pre-registered:** privacy (ASR/Rank-N-T), utility (emotion/pose/gaze), fairness (stratified age/ethnicity), compute (fps on target hardware). Sweep $d \in \{1.0,1.2,1.4,1.6\}$ for a privacy–utility curve figure.
  - Annotated `paper/main.tex` (Methodology, Chronogram Weeks 8–9/10–13/14, Expected Results) with `[NODE3-AGENT]` comments.
- [x] Node 4 — Inpainting/Compositing — **reviewed 2026-07-28 (NODE4-AGENT).** Verdict: **the separate-vs-end-to-end decision is currently assumed, not made — and the evidence favors end-to-end / learned fusion over classical blending.** Boundary handling, overlapping faces, and a "seamless" metric are all unspecified. Findings:

  **Separate compositing vs end-to-end (the key question)**
  - The verified candidates split on this axis: **end-to-end** (CIAGAN inpaints *in* the generator, no paste-back; LDFA inpaints in place with SD2) vs **separate** (ReferenceNet generates only the face region, blending is downstream; GANonymization re-synthesizes then pastes back over a U-Net-removed background).
  - **Recommendation: prefer end-to-end / learned fusion.** Our utility promise is expression/gaze/affect — exactly what classical alpha/Poisson blending does *not* model. A feathered paste can preserve pixels yet still break perceived affect via skin-tone/lighting discontinuity at the boundary. Classical blending = baseline only.
  - **The choice is not independent of Node 3.** Diffusion candidates (LDFA, ReferenceNet) naturally inpaint in place and tolerate context padding; GAN candidates differ among themselves (CIAGAN end-to-end vs GANonymization paste-back). → Make compositing an explicit scored criterion in the Week-14 down-select, not an afterthought.

  **Boundary handling (currently unspecified — must specify)**
  - **Context padding:** adopt LDFA's verified trick — pad each face box **32 px** before inpainting, paste back **only the unpadded region**. Cheap, hides the seam, gives the model boundary context.
  - **Learned adaptive mask:** G2Face's IFF blocks fuse generated ID features with original ID-irrelevant features via a **learned adaptive mask** — a stronger, learned alternative to fixed feathering that preserves hair/glasses/background edges. Consider IFF-style fusion if a separate compositor is kept.
  - **Mask shape:** use a **face-parsing / soft mask**, not the raw rectangular detector box — glasses and hair straddle the box and are the hardest to blend.

  **Boundary artifacts (pre-empt the verified failure modes)**
  - **Skin-tone / lighting mismatch:** LDFA shows skin-tone mismatch when glasses are removed + lighting inconsistency at the boundary — a direct threat in classrooms (glasses, hair over forehead, side window light).
  - **Glasses/hair edges:** fixed rectangular masks cut them.
  - **Metric gap:** "seamless" is asserted, never measured. Add a boundary-coherence metric to the utility axis (gradient/color discontinuity along the mask contour, or face-parsing consistency), reported per-scenario (glasses, hair occlusion, side lighting).

  **Overlapping faces / occlusion (unaddressed — classroom-critical)**
  - Students sit close → face boxes **overlap**. LDFA is verified to artifact on overlapping boxes (one face's inpaint overwrites another).
  - Specify an occlusion policy: process faces in depth/order, mask already-composited faces out of subsequent inpaint contexts, or composite back-to-front. Add an overlap scenario to the eval, or the "seamless" claim fails on crowded frames.

  Annotated `paper/main.tex` with `% [NODE4-AGENT]` comments at Methodology (separate-vs-end-to-end, boundary handling, artifacts, overlap, generator-dependence), Chronogram Weeks 10–13 (compositing sub-task), and Expected Results (mechanism, "seamless" success criterion, overlap policy). No text deleted.

- [x] Node 5 — Temporal Consistency — **REVIEWED 2026-07-28.** Verdict: plan is directionally sound but underspecified; "reuse previous face + gradient injection" has a drift failure mode and no defined identity-lock or flicker metric. Findings:
  - **Drift / error accumulation (main gap):** conditioning frame *t* on frame *t−1*'s synthetic output is a feedback loop — artifacts compound over long takes. Fix: anchor to a **canonical per-track latent** (Kung seed-lock per track ID) and use gradient injection to pull toward *that*, not toward the previous frame. Add periodic keyframe re-anchoring (every N frames / on confidence recovery) or EMA over latents instead of hard copy.
  - **"Target" is undefined:** DiffAIM's target = real identity (impersonation). Ours must be a **synthetic per-track pseudonym** — the seed-locked latent from Node 3. Reframe "reuse previous face" → "anchor to fixed per-track identity."
  - **Identity locking across occlusion/re-entry:** needs a persistent `track_id → synthetic identity (seed+latent)` table; re-match on re-entry via the **original** (pre-anonymization) face embedding, not the synthetic one. Hard dependency on Node 1 tracker (ByteTrack/DeepSORT still unspecified) and Node 3 seed-lock.
  - **Flicker unmeasured:** "no flicker" is asserted, never tested. Add a temporal utility metric: consecutive-frame LPIPS/embedding distance within a track, or optical-flow warping error (t vs t+1).
  - **Video-native vs per-frame+smoothing:** per-frame + locking is the right call for the 20-week timeline / compute (200 DDPM steps/frame); video-native diffusion stays future work. But the thesis currently defers *identity locking itself* to future work — that contradicts the differentiator. Commit to seed-locked per-track identity + smoothing as in-scope.
  - Annotated `paper/main.tex` with commented suggestions at Methodology (temporal sentence), Chronogram (frame-to-frame controls row), and Expected Results (both temporal paragraphs). No text deleted.
- [x] Cross-cutting — Architecture (merge vs modular) — **reviewed 2026-07-28 (ARCH-AGENT).** Verdict: **5 stages is the right design view, wrong deployment view. Target = 3 trainable networks + 2 thin stages.** Full recommendation now in §2 ("RESOLVED"). Findings:

  **The two merges (both training-free)**
  - **2+3 (expression estimation → generator): MERGE.** Confirms NODE2-AGENT. ReferenceNet conditions implicitly on a driving image — no estimation network to fail — and still gets best-in-table pose/gaze/expression. This deletes the single most damaging *silent* boundary: landmark/3DMM failure → GANonymization-style average-face fallback that destroys affect exactly in hard classroom conditions. Estimators (MediaPipe/3DMM) are kept as **evaluation probes only** — they compute the utility metrics, so they stay in the repo but leave the inference path.
  - **3+4 (compositing → generator): MERGE.** Verified precedent: CIAGAN and LDFA both generate the face in place (end-to-end inpainting, no compositor); ReferenceNet synthesizes in context via the driving image. Classical blending (Poisson/alpha-feather, ~0 params) demoted to a fallback utility for the GAN path. Deletes the seam/skin-tone-mismatch boundary (LDFA's known glasses/blur artifacts).

  **What stays separate — and why modularity survives**
  - **Node 1 (detection+tracking):** the highest-stakes boundary (missed face = privacy leak, not quality loss). Must remain independently benchmarkable (recall-first, per NODE1-AGENT).
  - **Node 5 (temporal):** a stage, not a network — reuses the generator U-Net via gradient injection. Distinct failure modes (drift, identity lock) justify a distinct benchmark, but no new weights.
  - Modularity is preserved where it has value: detection, generation, temporal remain independently swappable/benchmarkable; the merged stages were the ones whose modularity bought nothing (no one swaps a compositor) while adding failure surface.

  **Rejected merges**
  - 1+2 (RetinaFace landmarks → expression): only helps the explicit-carrier path we're abandoning; couples recall-critical detection to a conditioning choice.
  - 3+5 (video-native diffusion): correct end-state, out of scope (20 weeks, 200 DDPM steps/frame on standard hardware). Per-frame + locking closes the gap the baselines leave open.

  **Compute honesty:** merging saves failure modes, not meaningful FLOPs — Node 2 (~10–50M) and classical compositing (~0) are noise vs the ~900M generator at 200 steps/frame. The compute fix is generator-side (DDIM/few-step distillation, NODE3-AGENT). The thesis must argue the merges on robustness grounds or reviewers will expect a compute win that isn't there.

  **Scope guard:** both merges select pretrained models whose published behavior already subsumes the stage — no unified model is trained, so "don't invent a new generator, make existing models video-capable" holds.

  Annotated `paper/main.tex` Methodology with `% [ARCH-AGENT]` comments (verdict, error-compounding map, compute honesty, scope guard). No text deleted.

---

## 6. Conclusions (drawn from the 6 node reviews + literature validation)

### 6.1 The architecture makes sense — validated

Independent validation against 2024–2026 literature confirmed our core instincts are **correct and current**:

- ✅ **Merge conditioning into the generator** — confirmed. No recent explicit-conditioning method beats implicit driving-image conditioning on expression preservation. Kung's own follow-up (Reverse Personalization, WACV 2026) *abandoned* explicit conditioning entirely and still wins pose/gaze/expression. The field is moving our way.
- ✅ **Merge compositing into the generator** — confirmed. CIAGAN/LDFA/ReferenceNet all inpaint in place.
- ✅ **Per-frame + temporal smoothing** — confirmed. Video-native diffusion anonymization **does not exist yet** as a usable method; every 2024–2026 "video" solution (BLANKET, AnonNET, Muştu & Ekenel) is per-frame generation + a temporal transfer stage — exactly our plan.
- ✅ **Gaze as emergent** — confirmed (Kung achieves best gaze with no gaze module).
- ✅ **Diffusion over GAN** — confirmed, *especially for child faces* (GANs trained on adult FFHQ fail on infants; large-scale SD trained on LAION generalizes).

### 6.2 The one correction: our generator candidate set was outdated

The validation found we're missing a cluster of 2025–2026 methods — including the **direct successor to our primary candidate** and a **purpose-built solution to our gating risk**:

| Method | Venue | Why it matters |
|--------|-------|----------------|
| **BLANKET** (Hadera et al.) | ICDL 2025 | **Solves our exact gating risk**: infant/child face video anonymization. Beats DeepPrivacy2, temporally consistent, public code. Key insight matches ours: large-scale SD generalizes to infants, GANs don't. |
| **Reverse Personalization** (Kung et al.) | WACV 2026 | **Kung's own successor to ReferenceNet** — training-free, attribute-controllable (age/sex/race), SOTA privacy–utility. Gives demographic-similarity control for free. Code public. |
| **AnonNET** (Egin et al.) | ICCV-W 2025 | Current best "unified" video anonymization (diffusion inpaint + reenactment). Must-cite, must-compare. |
| **NullFace** (Kung et al.) | FG 2026 | Training-free localized anonymization; lowest re-ID in Reverse Personalization's table. |
| **Muştu & Ekenel** | DSP 2025 | The temporal-consistency evidence base: face-swap (SimSwap/FaceDancer) beats anonymization-specific methods on temporal consistency. |
| **FDeID-Toolbox** (Wei et al.) | arXiv 2603 | Standardized privacy/utility/quality eval — could save building eval infra + gain comparability. |

### 6.3 Confirmed risks (not hypothetical)

1. **Child-face fairness** — confirmed in the literature, not just our concern. BLANKET is the purpose-built mitigation.
2. **Temporal mechanism is our least-validated component** — gradient injection (DiffAIM) was designed for impersonation, not anonymization. Our reframe (anchor to synthetic per-track latent) is sound but unproven. BLANKET's "generate identity once, swap per frame" is a simpler, *validated-on-children* alternative — A/B test it.
3. **Compute honesty** — 200 DDPM steps/frame is not "standard hardware." Reverse Personalization: ~13 s/frame @ 1024² on A100. DDIM/few-step distillation or frame subsampling is **mandatory**, not optional.

---

## 7. FINAL PIPELINE (recommended)

### 7.1 How many steps?

**3 trainable networks + 2 thin stages** (down from the original 5). The 5-node view stays as the *design/benchmark* view; this is the *deployment* view.

### 7.2 The final architecture

```
Input Video (classroom lecture)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 1 — DETECTION + TRACKING  [trainable network #1]        │
│ YOLOv9-M (or YOLOv8n) + ByteTrack                             │
│ High recall priority. Output: [frame, track_id, box, conf]   │
│ • ByteTrack keeps low-conf detections → recall               │
│ • Tracker interpolation + RetinaFace 2nd pass on track drops │
└──────────────────────────────────────────────────────────────┘
    │ per-frame face crops + boxes + track IDs
    ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 2 — GENERATION + COMPOSITING  [trainable network #2]    │
│ (conditioning + compositing MERGED in — no separate nets)    │
│                                                               │
│ Candidates (down-select at Week 14, pre-registered criteria):│
│   • BLANKET (child-face specialist — NEW primary lean)       │
│   • Reverse Personalization (Kung WACV'26, attribute control)│
│   • ReferenceNet (Kung WACV'25, continuity baseline)         │
│   • CIAGAN (GAN baseline, ~real-time)                        │
│                                                               │
│ • Implicit conditioning via driving image (no expression net)│
│ • End-to-end inpainting (no separate compositor)             │
│ • Seed-lock per track_id → consistent synthetic identity     │
│ • 32px context padding + paste-back of unpadded region       │
│ • Demographic-similarity control (age/sex/race-matched)      │
└──────────────────────────────────────────────────────────────┘
    │ composited frames (synthetic face in place)
    ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 3 — TEMPORAL CONSISTENCY  [thin stage, not a network]  │
│ Anchor to fixed per-track canonical latent (NOT prev frame)  │
│ • Primary: gradient injection (DiffAIM) toward per-track     │
│   seed-locked latent — reuses generator U-Net                │
│ • Fallback/A-B test: BLANKET's "generate once, swap per      │
│   frame" (simpler, validated on children)                    │
│ • Persistent track_id → synthetic identity (seed+latent) map │
│ • Re-match on re-entry via ORIGINAL face embedding           │
│ • Keyframe re-anchoring / EMA to prevent drift               │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Output Video (de-identified, expression-preserved) + audio
```

### 7.3 Decision rationale — each choice explained

**Why 3 networks (not 5)?**
The two merges (conditioning→generator, compositing→generator) delete the two *silent quality-loss* boundaries: landmark/3DMM failure → average-face fallback (destroys affect exactly in hard classroom conditions), and generation→compositing (seam/skin-tone mismatch). Both merges are **training-free** — we select pretrained generators whose published behavior already subsumes the stage, so "don't invent a new generator" holds. The two boundaries that *must* stay visible — detection recall (a miss = privacy leak) and temporal identity — remain separate, benchmarkable stages.

**Why diffusion (not GAN)?**
- Best-in-table pose/gaze/expression preservation (our core utility axes)
- The anonymization knob $d$ = a free inference-time privacy–utility curve (thesis figure)
- Seed diversity → consistent per-student pseudonym (helps temporal)
- **Decisive for us:** GANs trained on adult FFHQ fail on infants; large-scale SD (LAION) generalizes to children. Our subjects are children.
- *Trade-off accepted:* diffusion is heavy (200 DDPM steps/frame). Mitigation = DDIM/few-step distillation or frame subsampling (mandatory). GANs (CIAGAN) kept as ~real-time baseline.

**Why BLANKET as the new primary lean (over ReferenceNet)?**
ReferenceNet was our lean, but (a) its own author superseded it (Reverse Personalization), and (b) it verifiably fails on infants — our exact subjects. BLANKET is purpose-built for child faces, temporally consistent, and public. **This is now the gating empirical question:** run the Week 8–9 fairness smoke test on BLANKET vs ReferenceNet vs Reverse Personalization vs CIAGAN.

**Why implicit conditioning (no expression/pose/gaze net)?**
ReferenceNet/Reverse Personalization achieve best-in-table pose/gaze/expression with *no* explicit estimator — the driving image carries it implicitly. This removes an entire failure boundary (MediaPipe/D3DFR fail on occlusion/children). Estimators survive only as **evaluation probes** (they compute our utility metrics), not pipeline stages. Gaze = emergent, measured not modeled.

**Why end-to-end compositing (no separate inpainter)?**
Our utility promise is expression/gaze/affect — exactly what classical alpha/Poisson blending does *not* model. A feathered paste can preserve pixels yet break perceived affect via skin-tone/lighting discontinuity. CIAGAN/LDFA/ReferenceNet all inpaint in place. Classical blending = fallback only.

**Why per-frame + temporal anchor (not video-native diffusion)?**
Video-native diffusion anonymization doesn't exist as a usable method yet. Every current "video" solution is per-frame + temporal transfer. Our plan is *more* aligned with SOTA than chasing video diffusion. Video-native = future work.

**Why anchor to a canonical latent (not the previous frame)?**
"Reuse previous frame's face" is a feedback loop — artifacts compound over a long lecture take (drift). Anchoring to a fixed per-track seed-locked latent (and pulling toward it via gradient injection) prevents drift. This is the same "fixed identity per track" principle BLANKET and Muştu & Ekenel validate.

### 7.4 Objectives & trade-offs summary

| Objective | How the pipeline achieves it | Trade-off accepted |
|-----------|------------------------------|--------------------|
| **Privacy** (suppress identity) | High-recall detection (no face missed) + diffusion anonymization knob $d$ + seed-locked per-track pseudonym | Diffusion compute (heavy) |
| **Utility** (preserve expression/gaze/affect) | Implicit driving-image conditioning (best-in-table) + end-to-end compositing | Not inspectable/editable signal |
| **Fairness** (children) | BLANKET child-face specialist + stratified eval | Extra candidate to benchmark |
| **Temporal stability** | Canonical-latent anchor + gradient injection | Least-validated component (has fallback) |
| **Compute** (standard hardware) | DDIM/distillation/subsampling | Must be stated, not assumed |
| **Modularity** | Detection / generation / temporal stay separately benchmarkable | Conditioning + compositing no longer swappable |

### 7.5 Pre-registered down-select criteria (Week 14)

Decide the primary generator empirically on:
1. **Privacy** — ASR, Rank-N-T, API confidence, PG/PIC, NoA (held-out verifier + parrot attack)
2. **Utility** — emotion agreement, 3DMM L2, pose/gaze, detection mAP, boundary coherence
3. **Fairness** — stratified by age/ethnicity (child faces are gating)
4. **Compute** — fps on target hardware
5. **Temporal** — flicker (consecutive-frame LPIPS/embedding distance), identity-lock stability

Sweep $d \in \{1.0, 1.2, 1.4, 1.6\}$ → privacy–utility curve figure.

---

## Links

- Problem: [[10-problem]] | Method: [[30-method]] | Evaluation: [[40-evaluation]]
- Stages: [[identification]], [[generation]]
- Decisions: [[70-decisions]] | Open questions: [[60-open-questions]]

#pipeline #architecture #phase1/detection #phase2/generation #phase2/inpainting
