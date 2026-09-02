#!/bin/bash
# One consolidated regression set. ~15-20 min vs ~90 for the 5-set sweep.
# Output -> ./combined_knowledge/  (gitignored).
#
#   ./examples/run_combined.sh
#
set -u
cd "$(dirname "$0")/.."
MP=".venv/bin/mom-phase1"; [ -x "$MP" ] || MP="mom-phase1"
export MOM_DOCS_DIR="${MOM_DOCS_DIR:-combined_knowledge}"
rm -rf "$MOM_DOCS_DIR"

for t in examples/combined_call1.txt examples/combined_call2.vtt \
         examples/combined_call3.txt examples/combined_empty.txt; do
  echo "════════════════════ $t ════════════════════"
  "$MP" "$t"
  echo
done

echo "════════════════════ score ════════════════════"
python3 scripts/score.py "$MOM_DOCS_DIR" \
  examples/combined_call1.txt examples/combined_call2.vtt \
  examples/combined_call3.txt examples/combined_empty.txt

echo
echo "════════════════════ docs ════════════════════"
for f in "$MOM_DOCS_DIR"/*.md; do echo "───────── $f"; cat "$f"; echo; done
