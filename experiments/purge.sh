#!/usr/bin/env bash
# Delete what the experiments keep on disk that still shows real people (LGPD), once the
# paper artifacts are exported (experiments/export_paper.sh). Lists by default; deletes
# only with --yes. Run on the machine that holds the results (serra1).
#
#   experiments/purge.sh            # list what would be deleted, with sizes
#   experiments/purge.sh --yes      # delete it
#   experiments/purge.sh --yes --obs   # also the per-face CSVs (obs.csv: real emotion labels)
#
# Deleted:
#   - experiments/*/results/<video>/<arm>/generated/   generated crops: synthetic face on the
#     real context crop (background, neighbours)
#   - experiments/*/results/<video>/<arm>/output.mp4   composed videos (real bodies, scene)
#   - cache/*/<video>/blanket*/seed_*.jpg              SDXL identity images (real background)
#   - $TMPDIR/blanket_ipc_*                            scratch left by a killed run
# Kept: results/latex, metrics, ledgers (scalars), provenance. --resume then regenerates.
set -euo pipefail
cd "$(dirname "$0")/.."

yes=0 obs=0
for a in "$@"; do
  case $a in
    --yes) yes=1 ;;
    --obs) obs=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 1 ;;
  esac
done

targets=()
while IFS= read -r -d '' p; do targets+=("$p"); done < <(
  find experiments/*/results -mindepth 3 -maxdepth 3 \( -type d -name generated -o -type f -name output.mp4 \) -print0 2>/dev/null
  find cache -type f -name 'seed_*.jpg' -print0 2>/dev/null
  find "${TMPDIR:-/tmp}" -maxdepth 1 -user "$(id -un)" -name 'blanket_ipc_*' -print0 2>/dev/null
  if [[ $obs == 1 ]]; then find experiments/*/results -name obs.csv -print0 2>/dev/null; fi
)

if [[ ${#targets[@]} -eq 0 ]]; then echo "nothing to purge"; exit 0; fi
du -sch "${targets[@]}" 2>/dev/null | tail -1
printf '%s\n' "${targets[@]}" | sed 's|/[^/]*$||' | sort | uniq -c | sort -rn | head -20
if [[ $yes == 1 ]]; then
  rm -rf "${targets[@]}"
  echo "purged ${#targets[@]} paths"
else
  echo "(dry run: ${#targets[@]} paths; pass --yes to delete)"
fi
