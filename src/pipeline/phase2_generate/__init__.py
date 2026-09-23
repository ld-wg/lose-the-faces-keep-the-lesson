"""Phase 2 — synthetic face generation + inpainting (merged).

Candidates (research/pipeline.md): BLANKET / Reverse Personalization /
ReferenceNet / CIAGAN. Implicit conditioning + end-to-end inpainting.

`ciagan` is implemented (see `models/ciagan/`) — the GAN baseline, picked
first for its simplicity and settled public code/weights. BLANKET/
Reverse Personalization/ReferenceNet remain future candidates, pending the
fairness down-select described in research/pipeline.md.
"""
