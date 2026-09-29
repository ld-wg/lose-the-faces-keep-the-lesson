"""Evaluation tooling for the anonymization pipeline.

    identity_report — validates the identity pre-pass on real footage (plan Step 1)
    evaluate        — privacy/utility metrics for Phase 2 runs (plan Step 2)
    heldout         — the held-out FaceNet evaluator (never used for guidance)

Nothing here writes face embeddings to disk (LGPD); outputs are scalars.
"""
