#!/bin/bash
# Consolidated regression set: 3 substantive calls + 1 empty. Structural
# parity with the full 5-set sweep -- every code path the 5 sets exercise has
# a trigger here, including 7B-stress (call1 is a dense ~70-line, 4-chunk
# scoping workshop). ~60-75 min vs ~2.5 h.  Output -> ./combined_knowledge/
#
#   ./examples/run_combined.sh
#
set -u
cd "$(dirname "$0")/.."
MP=".venv/bin/mom-phase1"; [ -x "$MP" ] || MP="mom-phase1"
export MOM_DOCS_DIR="${MOM_DOCS_DIR:-combined_knowledge}"
rm -rf "$MOM_DOCS_DIR"

CALLS=(examples/combined_call1.txt examples/combined_call2.vtt
       examples/combined_call3.txt examples/combined_empty.txt)

for t in "${CALLS[@]}"; do
  echo "════════════════════ $t ════════════════════"
  "$MP" "$t"
  echo
done

echo "════════════════════ score ════════════════════"
python3 scripts/score.py "$MOM_DOCS_DIR" "${CALLS[@]}"

echo
echo "════════════════════ W2/W3/W4 checks ════════════════════"
echo "docs                : $(ls "$MOM_DOCS_DIR"/*.md 2>/dev/null | wc -l | tr -d ' ')"
echo "GAP total           : $(grep -rc '\[GAP\]' "$MOM_DOCS_DIR"/*.md 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')"
grep -rn "\[RESOLVED\]"                          "$MOM_DOCS_DIR" | sed "s#$MOM_DOCS_DIR/##" || echo "(no RESOLVED)"
grep -rn "may contradict"                        "$MOM_DOCS_DIR" | sed "s#$MOM_DOCS_DIR/##" || echo "(no contradiction flags)"
grep -rn "updates PART of EF\|partially updates" "$MOM_DOCS_DIR" | sed "s#$MOM_DOCS_DIR/##" || echo "(no partial-supersede)"
echo "garbage names       : $(ls "$MOM_DOCS_DIR" 2>/dev/null | grep -Eic 'there|unnamed|^\[\]|^-') (want 0)"

echo
echo "════════════════════ docs ════════════════════"
for f in "$MOM_DOCS_DIR"/*.md; do echo "───────── $f"; cat "$f"; echo; done
