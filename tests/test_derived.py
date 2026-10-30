"""Derived analyses over exported keys (experiments/derived.py), 2026-10-30 critical review."""

from __future__ import annotations

import pytest

import derived


def test_p2_geometry():
    c = 0.327
    assert derived.geometry(c, 0.0) == pytest.approx((1.0, c))
    synth, real = derived.geometry(c, c)          # beta = c removes the real component
    assert real == pytest.approx(0.0, abs=1e-12)
    synth, real = derived.geometry(c, 1.2)        # the operating point leans to the antipode
    assert synth == pytest.approx(0.47, abs=0.01)
    assert real == pytest.approx(-0.68, abs=0.01)


def test_frontier_uses_shown_faces_for_the_composed_video():
    keys = {"e5/pooled/ref/hidden-share": "0.2", "e5/pooled/ref/facenet_video.rank1": "0.3",
            "e2/pooled/blanket/anonymized": "0.28", "e2/pooled/blanket/generated/facenet.rank1": "0.475"}
    out: dict[str, str] = {}
    points = {p["key"]: p for p in derived.frontier(keys, out)}
    assert points["e5-ref"]["share"] == pytest.approx(0.8)
    assert points["e5-ref"]["rank1"] == pytest.approx(0.375)   # a filled box is never rank-1
    assert points["e2-blanket"]["rank1"] == pytest.approx(0.475)
    assert out["derived/frontier/e5-ref/rank1"] == "0.375"


@pytest.mark.parametrize("v,h,verdict", [(0.048, 0.040, "undecided"), (0.02, 0.01, "met"), (0.09, 0.02, "missed")])
def test_verification_target(v, h, verdict):
    out: dict[str, str] = {}
    derived.verification_targets({"e3e/pooled/gate/facenet_final.verified": str(v),
                                  "e3e/pooled/gate/facenet_final.verified.ci": str(h)}, out)
    assert out["derived/target/e3e-gate/verdict"] == verdict
