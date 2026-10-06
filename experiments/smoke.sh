#!/usr/bin/env bash
# Smoke test of every pipeline and evaluation component on a few frames, before
# a long experiment run. Writes to a scratch folder (default /tmp/smoke), keeps
# nothing in results/. Stops at the first failure.
#
#   experiments/smoke.sh [video] [frames]          # default: demo1, 20 frames
#   env: BLANKET_REPO, IDENTITY_GPU, SWAP_GPU, CTX_ID (as for the experiments), SMOKE_DIR
#
# Checks, in order:
#   - ArcFace torch vs ONNX parity (P1/P3 gradients; must be >= 0.999)
#   - P1 spike, if the bridge has tools/p1_spike.py
#   - Phase 1 with the recall post-pass
#   - censor blur and mosaic, CIAGAN with the P3 push
#   - BLANKET with P2, the privacy gate and hide-low
#   - final videos (fail-closed)
#   - the evaluator with every probe, the noise floor and the per-face export
#   - WIDER FACE on 50 images
set -euo pipefail
export PYTHONUTF8=1   # non-interactive shells on serra1 have an ASCII locale
cd "$(dirname "$0")/.."
VIDEO=${1:-demo1} N=${2:-20}
S=${SMOKE_DIR:-/tmp/smoke}
BR=${BLANKET_REPO:-$HOME/projects/blanket-anonymizer-bridge}
CTX=${CTX_ID:-0}
V=$(ls data/videos/$VIDEO.* | head -1)
step() { echo; echo "$(date '+%T') == $*"; }
py() { uv run python "$@"; }
rm -rf "$S"; mkdir -p "$S"

step "phase1 on $VIDEO"
py -m src.pipeline.phase1_detect.run --input "$V" --out "$S/det" > "$S/det.log" 2>&1
cat "$S/det/run_stats.json"

step "ArcFace torch vs ONNX parity"
py -m src.pipeline.identity.fr_torch --phase1-dir "$S/det" --video "$V" --faces 50

if [[ -f $BR/tools/p1_spike.py ]]; then
  step "P1 spike (identity venv)"
  img=$(ls "$BR"/output/identities/*_c0.jpg 2>/dev/null | head -1 || true)
  if [[ -n $img ]]; then
    (cd "$BR" && CUDA_VISIBLE_DEVICES=${IDENTITY_GPU:-0} .venv-identity/bin/python tools/p1_spike.py --image "$img")
  else
    echo "skipped: no identity image in $BR/output/identities"
  fi
fi

anon() {  # name, flags...
  local name=$1; shift
  step "anonymize $name ($N frames)"
  py -m src.pipeline.phase2_generate.run --phase1-dir "$S/det" --video "$V" --out "$S/$name" --limit "$N" "$@" \
    > "$S/$name.log" 2>&1 || { tail -20 "$S/$name.log"; exit 1; }
  py -m src.pipeline.phase2_generate.compose_video --phase1-dir "$S/det" --phase2-dir "$S/$name" --video "$V" \
    --limit "$N" | tail -1
  python3 -c "import json,collections; r=[json.loads(l) for l in open('$S/$name/generation.jsonl')]; \
print(dict(collections.Counter((x['status'], x.get('reason')) for x in r)), [x['push'] for x in r if x.get('push')][:2])"
}
anon censor-blur --model censor --censor-mode blur --ctx-id "$CTX"
anon censor-mosaic --model censor --censor-mode mosaic --ctx-id "$CTX"
anon ciagan-p3 --model ciagan --identity-prepass --ciagan-push track --ctx-id "$CTX"
anon blanket --model blanket --blanket-repo "$BR" --identity-prepass --blanket-swap-mode track --blanket-push-beta 1.2 \
  --blanket-identity-cache "$S/ids" --privacy-gate 0.5 --generate-min-conf 0.5 --ctx-id "$CTX" \
  ${IDENTITY_GPU:+--blanket-identity-gpu $IDENTITY_GPU} ${SWAP_GPU:+--blanket-swap-gpu $SWAP_GPU}

step "evaluate every arm with every probe"
py -m src.eval.evaluate --phase1-dir "$S/det" --video "$V" --phase2-dir "$S/censor-blur" --phase2-dir "$S/censor-mosaic" \
  --phase2-dir "$S/ciagan-p3" --phase2-dir "$S/blanket" --utility all --noise-floor --obs-out "$S/obs.csv" \
  --out "$S/eval" --limit "$N" --ctx-id "${EVAL_CTX_ID:-0}" > "$S/eval.log" 2>&1 || { tail -20 "$S/eval.log"; exit 1; }
cat "$S/eval/eval.md"
head -2 "$S/obs.csv"

step "WIDER FACE, 50 images"
py -m src.eval.widerface --model scrfd-10gf --out "$S/wf" --limit 50 --ctx-id "${EVAL_CTX_ID:-0}" > "$S/wf.log" 2>&1 \
  || { tail -20 "$S/wf.log"; exit 1; }
cat "$S/wf/widerface.json"

step "smoke OK ($S)"
