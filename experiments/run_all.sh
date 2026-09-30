#!/usr/bin/env bash
# Regenerate every paper experiment, in order. Each one deletes and rebuilds
# its own results/. Stops at the first failure; experiments whose components
# do not exist yet are reported as pending and skipped.
#
#   experiments/run_all.sh                 # e0 → e3
#   experiments/run_all.sh --only e0,e2    # a subset, still in order
#   experiments/run_all.sh --fresh         # also clear cache/ (SDXL identities)
set -euo pipefail

cd "$(dirname "$0")/.."
only="" extra=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --only) only=$2; shift 2 ;;
    --fresh) extra+=(--fresh); shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

declare -a finished=() pending=()
for dir in experiments/e*_*/; do
  name=$(basename "$dir") id=${name%%_*}
  [[ -n $only && ",$only," != *",$id,"* ]] && continue
  echo "==> $name"
  set +e
  uv run "$dir/run.py" ${extra[@]+"${extra[@]}"}
  rc=$?
  set -e
  case $rc in
    0) finished+=("$name") ;;
    2) pending+=("$name") ;;
    *) echo "✗ $name failed (exit $rc) — see $dir/results/run.log" >&2; exit "$rc" ;;
  esac
done

echo
echo "done:    ${finished[*]:-none}"
echo "pending: ${pending[*]:-none}"
