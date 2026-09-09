#!/bin/bash
# Walk one example set through Phase 1 and print the resulting docs.
# Output goes to ./<set>_knowledge/ (gitignored), NOT your real knowledge/.
#
#   ./examples/run_example.sh delivery
#   ./examples/run_example.sh lms
#   ./examples/run_example.sh support
#
# For the consolidated regression set use examples/combined/run_combined.sh
# instead -- it adds the W2/W3/W4 checks.
set -e
cd "$(dirname "$0")/.."

SET="${1:?usage: run_example.sh <delivery|lms|expense|support|empty>}"
DIR="examples/$SET"
[ -d "$DIR" ] || { echo "no such set: $DIR"; exit 1; }

export MOM_DOCS_DIR="${MOM_DOCS_DIR:-${SET}_knowledge}"
rm -rf "$MOM_DOCS_DIR"

MP=".venv/bin/mom-phase1"
[ -x "$MP" ] || MP="mom-phase1"

shopt -s nullglob
CALLS=("$DIR"/*.txt "$DIR"/*.vtt)
IFS=$'\n' CALLS=($(sort <<<"${CALLS[*]}")); unset IFS

for t in "${CALLS[@]}"; do
    echo "════════════════════ $t ════════════════════"
    "$MP" "$t"
    echo
done

echo "════════════════════ score ════════════════════"
python3 scripts/score.py "$MOM_DOCS_DIR" "${CALLS[@]}" || true

echo
echo "════════════════════ RESULTING DOCS ($MOM_DOCS_DIR/) ════════════════════"
for f in "$MOM_DOCS_DIR"/*.md; do
    echo "───────── $f ─────────"
    cat "$f"
    echo
done
