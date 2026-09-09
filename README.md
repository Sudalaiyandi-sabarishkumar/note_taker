# mom-phase1

**Phase 1** of a client-call → requirements pipeline: turn call transcripts into
per-feature requirements knowledge docs **without inventing anything**.

It extracts *cited* statements (verbatim quote + speaker + timestamp) from a
transcript into auto-discovered per-feature docs under `knowledge/`, and merges a
new call's statements into existing docs for the same feature instead of
overwriting — flagging anything ambiguous for human review rather than guessing.

> **The one rule:** *no citation, no statement.* The model only pulls quoted
> statements out of one transcript; the doc-writing is plain Python. A statement
> with no real quote to point at cannot reach a file. A quote that can't be found
> verbatim in the transcript is filed as a question, not a fact.

Each feature doc opens with a **`## User Story`** — one plain-language synthesis
of the feature's *current* (non-superseded) facts, regenerated every run. It only
combines what the cited facts say; the facts below it are the evidence.

### How a second call is merged

Before routing, a call's freshly-extracted area names are folded to canonical
names by one LLM call, so "Certificate Format" / "Certificate Content" /
"Certificate Issuance" all land in one **Certificates** doc rather than three.



When a statement is about a feature that already has recorded facts, it is
reconciled against them (one small LLM call per statement):

| Verdict | What happens |
|---|---|
| **NEW** | added as a new `EF-n` |
| **CHANGE** | the new statement is added as a new `EF-n`; the fact it reverses is **kept** but marked `**EF-k** [superseded by EF-n]:` — the section stays complete, and the supersession is also written to the Change Log with `was:` / `now:` quotes |
| **DUPLICATE** | not added; a "restated EF-n, no change" line goes in the Change Log |
| **UNCLEAR** | filed under Open Questions as `[NEEDS REVIEW]` for a human |

The current picture of a feature is its **non-superseded** facts. A call that
flips *"email only"* → *"SMS instead of email"* leaves the email fact visible
(struck) and adds the SMS fact — nothing is deleted or overwritten. The
reconciler is told: *if the existing fact is still true after this statement,
it is NEW or DUPLICATE, never CHANGE.*

## The model

Phase 1 runs a **custom Ollama model** built from [`Modelfile`](./Modelfile),
based on `qwen2.5:7b-instruct-q4_K_M` (a plain instruction model — fast, strict
with output formats and verbatim copying, which is what Phase 1 needs).

```bash
brew install ollama && brew services start ollama   # if you don't have it
./build_model.sh                                     # pulls qwen2.5:7b-instruct-q4_K_M, creates 'mom-phase1'
```

To skip the custom model and use the base directly: `export MOM_MODEL=qwen2.5:7b-instruct-q4_K_M`.

## Install

```bash
pip install -e ".[cli]"      # [cli] adds prompt_toolkit; plain `pip install -e .` also works
```

## Use — slash commands

```
$ mom-phase1
mom> /requirements test_transcripts/call1.txt
mom> /requirements test_transcripts/call2.txt
mom> /features
mom> /show notifications
mom> /exit
```

| Command | Does |
|---|---|
| `/requirements <file>` | Run Phase 1 on a `.txt` or `.vtt` transcript |
| `/extract <file>` | Alias for `/requirements` |
| `/features` | List feature docs under `knowledge/` |
| `/show <feature>` | Print a feature doc |
| `/model` | Show the active Ollama model |
| `/help`, `/exit` | — |

One-shot (no REPL): `mom-phase1 test_transcripts/call1.txt`

## Output shape (`knowledge/<feature>.md`)

Three calls about notifications — call 1 "to all persons via email", call 2
"to registered persons only, via email and SMS", call 3 "also via WhatsApp":

```markdown
# Notifications

## User Story
Notifications are sent only to registered persons, via email, SMS, and WhatsApp.

## Established Facts
- **EF-1** [superseded by EF-2]: Notifications go to all persons by email. — *"notification is sent to all persons via email"* — PM, not available (source: call1, ...)
- **EF-2**: Notifications go to registered persons only, by email and SMS. — *"notification is sent to only registered persons via email and sms"* — PM, not available (source: call2, ...)
- **EF-3**: Notifications are also sent by WhatsApp. — *"send notification via whatsapp also"* — PM, not available (source: call3, ...)

## Open Questions / Ambiguities
- None.

## Change Log
- ... : processed call1 — 1 new, 0 changed, ...
- ... : processed call2 — 0 new, 1 changed, ...
- ... : EF-1 superseded by EF-2 (from call2). ...  was: *"...all persons via email"*  now: *"...only registered persons via email and sms"*
- ... : processed call3 — 1 new, 0 changed, ...
```

**`## User Story`** is regenerated every run from the *non-superseded* facts —
it combines what they say and nothing more (EF-1 is struck, so "all persons"
does not appear). **`## Established Facts`** is the evidence, each with its
citation. Run [`./examples/run_example.sh`](./examples/run_example.sh) to see it
on the checkout example.

Tags a reviewer resolves later (Phase 3, not built here):
`[NEEDS REVIEW]` — reconciliation was unclear;
`[UNVERIFIED CITATION]` — quote not found verbatim in the transcript.
