#!/bin/bash
# A/B two Ollama models on the consolidated regression set.
#
#   ./scripts/ab_model.sh <model-A> <model-B>
#   ./scripts/ab_model.sh mom-phase1 phi4
#   ./scripts/ab_model.sh mom-phase1 qwen3:14b
#
# Runs examples/combined_call{1,2,3}.* + combined_empty through each model
# (via MOM_MODEL), scores both with scripts/score.py, and prints a side-by-side
# summary + wall-clock time. Pull the challenger first:  ollama pull phi4
set -u
cd "$(dirname "$0")/.."
A="${1:-mom-phase1}"
B="${2:?usage: ab_model.sh <model-A> <model-B>}"
MP=".venv/bin/mom-phase1"; [ -x "$MP" ] || MP="mom-phase1"
CALLS=(examples/combined_call1.txt examples/combined_call2.vtt
       examples/combined_call3.txt examples/combined_empty.txt)

run_one () {
  local model="$1" dir="ab_$(echo "$model" | tr '/:.' '___')"
  rm -rf "$dir"; export MOM_DOCS_DIR="$dir" MOM_MODEL="$model"
  local t0 t1; t0=$(date +%s)
  for c in "${CALLS[@]}"; do "$MP" "$c" >/dev/null 2>&1; done
  t1=$(date +%s)
  echo "MODEL: $model     (${dir}/, $((t1 - t0))s)"
  python3 scripts/score.py "$dir" "${CALLS[@]}" 2>&1 | \
    grep -E "docs|established|grounded|flags|RESULT"
  echo "  doc list: $(ls "$dir" 2>/dev/null | sed 's/\.md//' | paste -sd, -)"
  echo
}

echo "════════ A/B: $A  vs  $B ════════"
run_one "$A"
run_one "$B"
echo "Compare: fact recall (established facts), supersede correctness (grep the"
echo "Change Logs), [NEEDS REVIEW]/[UNVERIFIED] counts, and doc cohesion by eye."
