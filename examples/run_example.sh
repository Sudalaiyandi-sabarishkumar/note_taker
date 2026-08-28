#!/bin/bash
# Walk the three example calls through Phase 1 and print the resulting docs.
# Output goes to ./example_knowledge/ (gitignored), NOT your real knowledge/.
#
#   ./examples/run_example.sh
#
set -e
cd "$(dirname "$0")/.."

export MOM_DOCS_DIR="${MOM_DOCS_DIR:-example_knowledge}"
rm -rf "$MOM_DOCS_DIR"

MP=".venv/bin/mom-phase1"
[ -x "$MP" ] || MP="mom-phase1"   # fall back to whatever's on PATH

for t in examples/delivery_call1.txt examples/delivery_call2.txt examples/delivery_call3.txt; do
    echo "════════════════════ $t ════════════════════"
    "$MP" "$t"
    echo
done

echo "════════════════════ RESULTING DOCS ($MOM_DOCS_DIR/) ════════════════════"
for f in "$MOM_DOCS_DIR"/*.md; do
    echo "───────── $f ─────────"
    cat "$f"
    echo
done
