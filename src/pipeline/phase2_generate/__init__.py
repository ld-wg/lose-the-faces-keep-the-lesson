"""Stage 2 — face replacement, plus the Stage 3 composition (compose_video.py).

Backends (models/): censor (blur, mosaic), ciagan, ganonymization, blanket (SDXL
identity once per track + inswapper swap per frame, through the sibling bridge repo).
run.py applies the per-face policies (policies.py: frame sampling, the changed-pixels
guard; the privacy gate and hide-low) and writes generation.jsonl; compose_video.py
builds the fail-closed output video from it.
"""
