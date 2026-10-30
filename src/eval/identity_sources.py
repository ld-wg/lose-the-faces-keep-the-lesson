"""Which face of each cached identity image the swap used (2026-10-30 audit).

The identity image is the candidate's context crop (margin 0.8) with only the face
inpainted, so it can hold real neighbours. Before bridge `eabfbf3` the swap server took
FaceFusion's highest-scoring face (`source_faces[0]`); this asks the new server which face
overlaps the inpainted box and reports, per cache, how many identities had a different face
at index 0, i.e. swapped from the wrong face in the exported runs.

    python -m src.eval.identity_sources --phase1-dir <E3 detect dir> --video <video> \\
        --cache cache/e3_contribution/demo2/blanket --bridge-repo ~/projects/blanket-anonymizer-bridge \\
        --out identity_sources.json

Scalars only (counts and indices); no image or embedding leaves the process (LGPD).
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections import Counter
from pathlib import Path

from ..pipeline.identity.embedder import ArcFaceEmbedder, PoseEstimator
from ..pipeline.identity.prepass import load_identities, run_prepass
from ..pipeline.phase2_generate.models.blanket.backend import _RpcClient, candidate_box


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase1-dir", type=Path, required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--cache", type=Path, required=True, help="identity cache directory (index.json)")
    p.add_argument("--bridge-repo", type=Path, required=True)
    p.add_argument("--context-ratio", type=float, default=0.8)
    p.add_argument("--seed-candidates", type=int, default=5, help="as the run that built the cache")
    p.add_argument("--aggregation", default="quality_mean")
    p.add_argument("--ctx-id", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _, identities = load_identities(args.phase1_dir)
    prepass = run_prepass(args.phase1_dir / "detections.jsonl", args.video, identities, args.context_ratio,
                          embedder=ArcFaceEmbedder(ctx_id=args.ctx_id),
                          pose_estimator=PoseEstimator(ctx_id=args.ctx_id),
                          mode=args.aggregation, k_candidates=args.seed_candidates)
    by_seed = {identities[t].seed: tr for t, tr in prepass.tracks.items()}

    socket_dir = Path(tempfile.mkdtemp(prefix="identity_sources_"))
    swap = _RpcClient(socket_path=socket_dir / "swap.sock", python=args.bridge_repo / ".venv-swap" / "bin" / "python",
                      server_script=args.bridge_repo / "swap_server.py", cwd=args.bridge_repo,
                      extra_args=["--lenient-identity"], timeout=600, label="identity_sources/swap")
    index = json.loads((args.cache / "index.json").read_text())
    choices, no_box = Counter(), 0
    for seed, entry in index["entries"].items():
        if entry.get("status") != "ok":
            continue
        track = by_seed.get(int(seed))
        box = entry.get("box") or (candidate_box(track.seed_candidates, entry) if track is not None else None)
        if box is None:
            no_box += 1
            continue
        reply = swap.call("check_identity", identity_path=entry["path"], identity_box=box)
        choices["none" if reply["source_index"] < 0 else str(reply["source_index"])] += 1
    n = sum(choices.values())
    report = {"cache": str(args.cache), "identities_checked": n, "without_box": no_box,
              "source_index": dict(choices),
              "not_index_0": n - choices.get("0", 0),
              "not_index_0_share": round((n - choices.get("0", 0)) / n, 4) if n else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
