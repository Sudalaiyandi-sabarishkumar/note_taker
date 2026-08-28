# Manually verifying Phase 1

Everything below uses the example transcripts in [`examples/`](./examples/) and
writes to `example_knowledge/` (gitignored) so your real `knowledge/` is left
alone. Shortcut for the whole thing:

```bash
./examples/run_example.sh
```

The rest of this file is the same sequence step by step, with what to check.

---

## 0. Preconditions

```bash
ollama list | grep mom-phase1        # model is built  (else: ./build_model.sh)
ollama ps                            # after a run, PROCESSOR should be mostly GPU
.venv/bin/mom-phase1 --version       # CLI installed   (else: pip install -e ".[cli]")
export MOM_DOCS_DIR=example_knowledge # keep test output out of knowledge/
rm -rf "$MOM_DOCS_DIR"
```

---

## 1. First call — everything is new

```bash
.venv/bin/mom-phase1 examples/checkout_call1.txt
```

Expected summary line: **3 feature doc(s)**, each `1 new, 0 changed, 0 restated,
0 to review, 0 unverified`.

Check the docs exist and each fact carries a **verbatim** quote:

```bash
ls "$MOM_DOCS_DIR"
#   discount-codes.md? no -- not mentioned yet
#   guest-checkout.md   order-confirmation.md   payments.md   (names may vary slightly)

grep -A2 'Established Facts' "$MOM_DOCS_DIR/payments.md"
```

You should see something like:

```
- **EF-1**: ... — *"we need to support credit card and PayPal at launch."* — Maya Iyer, not available (source: checkout_call1, 2026-...)
```

**Prove the quote is real, not paraphrased** — pull the quote out of the doc and
find it in the transcript:

```bash
q=$(grep -oE '\*"[^"]+"\*' "$MOM_DOCS_DIR/payments.md" | head -1 | tr -d '*"')
grep -iF "$q" examples/checkout_call1.txt && echo "VERBATIM OK"
```

**Timestamps**: `checkout_call1.txt` is plain text with no timings, so every
attribution must say `not available` — never an invented `HH:MM:SS`:

```bash
grep -o 'source: checkout_call1[^)]*)' "$MOM_DOCS_DIR"/*.md
grep -E '[0-9]{2}:[0-9]{2}:[0-9]{2}' "$MOM_DOCS_DIR"/*.md && echo "BAD: fabricated timestamp" || echo "no fake timestamps OK"
```

**Small talk** ("talk then", "How was your trip") must not have produced any
fact — there is no doc for greetings/scheduling, and no EF quotes them.

---

## 2. Second call — restate / change / add / new feature

```bash
.venv/bin/mom-phase1 examples/checkout_call2.txt
```

`checkout_call2.txt` is built to hit every merge path at once. Expected:

| Statement in call 2 | Feature | What Phase 1 should do |
|---|---|---|
| "drop PayPal … add Apple Pay instead" | Payments | **CHANGE** — EF-1 kept but marked `[superseded by EF-2]`, EF-2 added |
| "guest checkout stays, no forced signup" | Guest Checkout / Checkout | **DUPLICATE** or **NEW** — EF-1 stays either way, nothing overwritten |
| "discount code field on the checkout page" | *new* Discounts | **new doc**, EF-1 |
| "downloadable PDF receipt to the order confirmation" | Order Confirmation | **NEW** fact (EF-2), EF-1 untouched |
| "How was your trip" / "wrap up" | — | ignored |

Now inspect Payments — a CHANGE **keeps** the old fact and adds the new one;
the Established Facts section stays complete and readable top to bottom:

```bash
cat "$MOM_DOCS_DIR/payments.md"
```

Expect:
- `## Established Facts`:
  - `**EF-1** [superseded by EF-2]:` … *"…credit card and PayPal at launch."*
  - `**EF-2**:` … *"Let's drop PayPal for now and add Apple Pay instead."*
- `## Open Questions / Ambiguities` → still `- None.` (a confident CHANGE is applied, not queued)
- `## Change Log` → `EF-1 for "Payments" superseded by EF-2 (from checkout_call2)` with `- was:` / `- now:` quote lines

The **current picture** is the non-superseded facts:
```bash
python3 -c "import mom_phase1.knowledge_docs as k; s=k._split_sections(open('$MOM_DOCS_DIR/payments.md').read()); f=k._parse_established_facts(s['established_facts']); print('active:', [x['id'] for x in k._active(f)])"
# -> active: [2, 3]   (after call 3 adds Google Pay as EF-3)
```

Guest Checkout / Checkout — EF-1 is never lost:

```bash
cat "$MOM_DOCS_DIR"/checkout*.md "$MOM_DOCS_DIR"/guest-checkout.md 2>/dev/null
# EF-1 still carries the call-1 quote; a re-confirmation appears as EF-2 or a
# Change-Log restate note, but EF-1 is not overwritten.
```

Discount Codes — brand new file:

```bash
cat "$MOM_DOCS_DIR/discount-codes.md"   # H1 "Discount Codes", one EF-1
```

---

## 3. A .vtt with real timestamps

```bash
.venv/bin/mom-phase1 examples/checkout_call3.vtt
cat "$MOM_DOCS_DIR/refunds.md"
```

Expect a new **Refunds** doc whose EF-1 attribution carries the cue start time,
e.g. `Dana Ford, 00:00:04.000 (source: checkout_call3, ...)` — the timestamp is
read from the `.vtt`, not from the model. The Google Pay line should land on
**Payments** as a CHANGE or NEEDS REVIEW (both acceptable — a human decides
whether Google Pay replaces or joins Apple Pay).

---

## 4. Negative check — an ungrounded claim cannot become a fact

Append a vague line and confirm it is not recorded as an Established Fact:

```bash
cp examples/checkout_call1.txt /tmp/vague.txt
printf '\nTom Becker: It should generally feel fast and modern.\n' >> /tmp/vague.txt
MOM_DOCS_DIR=/tmp/vague_kn .venv/bin/mom-phase1 /tmp/vague.txt
grep -ri "fast and modern" /tmp/vague_kn/ || echo "OK: vague statement not stored as a fact"
```

If the model does try to quote it, it can only be filed as an **Established
Fact** when that exact text is in the transcript; a near-miss shows up under
`## Open Questions / Ambiguities` tagged `[UNVERIFIED CITATION]`.

---

## 5. Re-running the same call is safe

```bash
.venv/bin/mom-phase1 examples/checkout_call1.txt   # again
```

Established Facts are append-only and the reconciler will mark the repeats as
DUPLICATE / NEEDS REVIEW rather than duplicating EF lines. The Change Log grows
by one line per run — that is the audit trail, by design.

---

## What "passing" looks like

- Every `EF-` line has a quote that `grep -F` finds in its source transcript.
- No `HH:MM:SS` in any attribution sourced from a `.txt` file.
- call 2's Payments fact is **rewritten**, with the call-1 quote preserved in the Change Log.
- Guest Checkout is untouched by call 2 (restate only).
- Discount Codes / Refunds appear as their own docs.
- Nothing from small talk becomes a fact.
