# Shared helpers for bin/ scripts. Sourced, not executed.
set -euo pipefail
export PYTHONUTF8=1   # non-interactive shells on serra1 have an ASCII locale

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS=${RESULTS_DIR:-$ROOT/results}
VIDEOS=${VIDEOS_DIR:-$ROOT/data/videos}

die() { echo "error: $*" >&2; exit 1; }

# A path to an existing file, or a short name found as data/videos/<name>.*
video_path() {
  if [[ -f $1 ]]; then (cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); return; fi
  local hit
  hit=$(ls "$VIDEOS/$1".* 2>/dev/null | head -1 || true)
  [[ -n $hit ]] || die "no video '$1': not a file, and not in $VIDEOS/"
  echo "$hit"
}

stem() { local b; b=$(basename "$1"); echo "${b%.*}"; }

# results/<YYYY-MM-DD_HHMM>_<step>-<model>-<video>, never reusing an existing folder
new_dir() {
  local d="$RESULTS/$(date +%Y-%m-%d_%H%M)_$1-$2-$3"
  [[ -e $d ]] && d="$d-$(date +%S)"
  mkdir -p "$d"
  echo "$d"
}

# run <dir> <label> <command...>: record the command in <dir>/<label>.sh, run it
# from the repo root with output in <dir>/<label>.log, and print <dir> last.
run() {
  local dir=$1 label=$2; shift 2
  { echo '#!/usr/bin/env bash'; echo "cd $(printf %q "$ROOT")"; printf '%q ' "$@"; echo; } > "$dir/$label.sh"
  (cd "$ROOT" && "$@") 2>&1 | tee "$dir/$label.log"
  echo "$dir"
}

# json_get <file> <key> [<key>...]: print one value of a JSON file
json_get() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))
for k in sys.argv[2:]: d = d[k]
print(d)' "$@"; }
