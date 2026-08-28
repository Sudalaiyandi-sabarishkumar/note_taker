"""Turn a transcript into a list of grounded statement dicts.

The model does the pulling-out; this module does the checking. The gates
that enforce "no citation, no statement" structurally:

  1. A ``## STATEMENT`` block missing any field, or with an empty quote, is
     dropped outright.
  2. Quote snapping: a small local model paraphrases. So the model's quote
     is only ever used to *locate* the real sentence in the transcript --
     when it isn't already verbatim, it is matched against the transcript's
     own spans and, on a close-enough hit, replaced with the exact source
     text. Only a quote that still can't be placed stays ``verified=False``
     for the doc layer to file as a question rather than a fact.
  3. A timestamp is kept only if it literally appears in the source.
"""

import re
from difflib import SequenceMatcher

from .ollama_client import DEFAULT_MODEL, chat, strip_think
from .transcript import chunk_transcript

_SYSTEM = (
    "You extract product requirements from a real client-call transcript. "
    "The one rule that matters most: NO CITATION, NO STATEMENT. Every fact "
    "must be backed by an exact verbatim quote from the transcript part you "
    "are given. Never paraphrase, never combine sentences, never infer. If "
    "you cannot point to the exact words, do not write the fact."
)

_FIELDS = ("feature", "summary", "quote", "speaker", "timestamp")
_LABELS = {
    "feature": "Feature:",
    "summary": "Summary:",
    "quote": "Quote:",
    "speaker": "Speaker:",
    "timestamp": "Timestamp:",
}


def _instruction(part_no: int, total: int, known_features) -> str:
    if known_features:
        feat = (
            "Features already known from earlier calls: "
            + ", ".join(sorted(known_features))
            + ".\nIf a statement is about one of these, reuse that EXACT name "
            "(same spelling and case) -- even when the statement CHANGES or "
            "contradicts what was said before, it is still the same feature. "
            "Only invent a new feature name when a statement fits none of them."
        )
    else:
        feat = "No features are known yet -- this is the first call processed."
    return f"""This is PART {part_no} of {total} of one call transcript (not the whole call).
{feat}

For every distinct requirement, decision, constraint, or fact actually stated
by a person in THIS PART, output one block in exactly this format:

## STATEMENT
Feature: <the standing feature area in 1-3 plain words, e.g. "Notifications">
Summary: <one plain sentence for a human skimming later>
Quote: "<exact sentence(s) from THIS part, copied verbatim>"
Speaker: <name exactly as written, or "Unidentified speaker">
Timestamp: <the [HH:MM:SS...] prefix on that line, or "not available">

Name the lasting feature, NOT the change to it: a switch from email to SMS
is still Feature "Notifications", never "SMS Notifications" or "Switch to
SMS". No parentheses, no "from X to Y", no verbs.

Repeat for every distinct statement, however minor. If this part has only
small talk or scheduling, output the single line: NO STATEMENTS
Output nothing before the first block or after the last."""


def _normalise(s: str) -> str:
    """Fold quotes/whitespace/case so a verbatim quote still matches when the
    model tidied a curly apostrophe or collapsed a double space."""
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\n\"'.,")


_SPEAKER_PREFIX_RE = re.compile(r"^(?:\[[^\]]+\]\s*)?[^:]{1,40}:\s*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SNAP_THRESHOLD = 0.66


def _candidate_spans(transcript_text: str):
    """Every quotable span of the transcript: each speaker turn with its
    ``[ts] Name:`` prefix stripped, plus each sentence inside it. Longest
    first, so a snap prefers the fuller sentence over a fragment of it."""
    spans = []
    for line in transcript_text.splitlines():
        body = _SPEAKER_PREFIX_RE.sub("", line.strip()).strip()
        if len(body) < 8:
            continue
        spans.append(body)
        for sent in _SENT_SPLIT_RE.split(body):
            sent = sent.strip()
            if 8 <= len(sent) < len(body):
                spans.append(sent)
    seen, out = set(), []
    for s in sorted(spans, key=len, reverse=True):
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _similarity(a: str, b: str) -> float:
    """Max of char-level and word-level sequence ratio -- word-level rescues
    a one-word swap (want/need) that tanks the char ratio."""
    char_r = SequenceMatcher(None, a, b).ratio()
    word_r = SequenceMatcher(None, a.split(), b.split()).ratio()
    return max(char_r, word_r)


def _snap_quote(model_quote: str, spans, norm_spans, norm_transcript: str):
    """Returns ``(quote, verified, score)``. A verbatim quote is upgraded to
    the smallest span that contains it; a near quote is replaced with the
    best-matching span when the match clears ``_SNAP_THRESHOLD``; otherwise
    the model's quote is returned unchanged and unverified."""
    nq = _normalise(model_quote)
    if not nq:
        return model_quote, False, 0.0
    if nq in norm_transcript:
        for span, nspan in zip(spans, norm_spans):
            if nq in nspan:
                return span, True, 1.0
        return model_quote, True, 1.0
    best_span, best_score = None, 0.0
    for span, nspan in zip(spans, norm_spans):
        score = _similarity(nq, nspan)
        if score > best_score:
            best_span, best_score = span, score
    if best_span is not None and best_score >= _SNAP_THRESHOLD:
        return best_span, True, best_score
    return model_quote, False, best_score


def parse_statement_blocks(text: str):
    statements = []
    for block in text.split("## STATEMENT")[1:]:
        fields = {}
        for line in block.splitlines():
            line = line.strip()
            for key, label in _LABELS.items():
                if line.startswith(label) and key not in fields:
                    value = line[len(label):].strip()
                    if key == "quote":
                        value = value.strip().strip('"').strip()
                    fields[key] = value
        if all(k in fields and fields[k] for k in _FIELDS):
            statements.append(fields)
    return statements


def extract_statements(transcript_text: str, known_features, model=None,
                       progress=print):
    """Run the map step over every chunk and return grounded statement
    dicts, each with a ``verified`` bool set by checking the quote against
    the transcript."""
    model = model or DEFAULT_MODEL
    chunks = chunk_transcript(transcript_text)
    norm_transcript = _normalise(transcript_text)

    progress(f"Extracting cited statements from {len(chunks)} part(s)...")
    found = []
    for i, chunk in enumerate(chunks, start=1):
        progress(f"  part {i}/{len(chunks)}...")
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content":
                f"TRANSCRIPT PART {i} OF {len(chunks)}:\n{chunk}\n\n"
                + _instruction(i, len(chunks), known_features)},
        ]
        raw = chat(messages, model=model, show_progress=False)
        answer = strip_think(raw)
        if answer.strip().upper().startswith("NO STATEMENTS"):
            continue
        found.extend(parse_statement_blocks(answer))

    spans = _candidate_spans(transcript_text)
    norm_spans = [_normalise(sp) for sp in spans]
    for s in found:
        snapped, verified, score = _snap_quote(
            s["quote"], spans, norm_spans, norm_transcript)
        if snapped != s["quote"]:
            s["quote_as_extracted"] = s["quote"]
            s["quote"] = snapped
        s["verified"] = verified
        s["match_score"] = round(score, 2)
        # The model fabricates timestamps on plain-text transcripts that have
        # none. Only keep a timestamp that literally appears in the source;
        # anything else is an ungrounded citation detail -> drop it.
        ts = s.get("timestamp", "").strip()
        if ts and ts.lower() != "not available" and ts not in transcript_text:
            s["timestamp"] = "not available"
    return found


# --------------------------------------------------------------------------
# reconciliation: how does one new statement relate to a feature's facts?
# --------------------------------------------------------------------------
_RECONCILE_SYSTEM = (
    "You compare ONE new statement from a later client call against the facts "
    "already recorded for a single feature, and classify how it relates. Be "
    "strict and literal. The test for CHANGE: an existing fact would become "
    "WRONG if left as-is. If every existing fact is still true after this "
    "statement, the answer is NEW or DUPLICATE -- never CHANGE."
)

_VERDICT_RE = re.compile(r"^VERDICT:\s*(NEW|DUPLICATE|CHANGE|UNCLEAR)", re.I | re.M)
_TARGET_RE = re.compile(r"^TARGET:\s*EF-(\d+)", re.I | re.M)
_REASON_RE = re.compile(r"^REASON:\s*(.+)$", re.I | re.M)


def reconcile_statement(feature, existing_facts, statement, model=None):
    """Ask the model how ``statement`` relates to ``existing_facts`` for
    ``feature``. Returns ``(verdict, target_id, reason)``:

      * verdict  -- "NEW" | "DUPLICATE" | "CHANGE" | "UNCLEAR"
      * target_id -- int EF id for DUPLICATE/CHANGE, else None
      * reason   -- one-line rationale (may be "")

    A CHANGE whose TARGET is missing or not a real EF id is downgraded to
    UNCLEAR, so the caller can fall back to flagging it for review rather
    than editing the wrong fact.
    """
    model = model or DEFAULT_MODEL
    facts_block = "\n".join(
        f'EF-{f["id"]}: {f["summary"]} (quoted: "{f["quote"]}")'
        for f in existing_facts
    )
    prompt = f"""Feature: {feature}

Facts already recorded for this feature:
{facts_block}

New statement from a later call:
Summary: {statement['summary']}
Quoted: "{statement['quote']}"

Reply in EXACTLY this format and nothing else:

VERDICT: <NEW | DUPLICATE | CHANGE | UNCLEAR>
TARGET: <the one EF id this duplicates or changes, e.g. EF-2 -- or NONE>
REASON: <one sentence>

Definitions:
- DUPLICATE: it repeats or re-confirms one existing fact and adds nothing
  (e.g. "just to confirm, X still stands"). Put that id in TARGET.
- NEW: it adds a detail, capability, or constraint that no existing fact
  states -- even about the same feature. "Also send an SMS" when a fact
  says "send an email" is NEW. All existing facts stay true.
- CHANGE: it reverses or overwrites ONE existing fact so that fact would now
  be WRONG to keep (e.g. "drop PayPal" when a fact says "support PayPal").
  Put that id in TARGET.
- UNCLEAR: you cannot tell, or it could affect more than one fact.

Decision rule: if the existing fact is still true after this statement, it
is NEW or DUPLICATE, never CHANGE."""

    raw = chat(
        [{"role": "system", "content": _RECONCILE_SYSTEM},
         {"role": "user", "content": prompt}],
        model=model, show_progress=False, num_predict=1500,
    )
    answer = strip_think(raw)

    vm = _VERDICT_RE.search(answer)
    verdict = vm.group(1).upper() if vm else "UNCLEAR"
    tm = _TARGET_RE.search(answer)
    target_id = int(tm.group(1)) if tm else None
    rm = _REASON_RE.search(answer)
    reason = rm.group(1).strip() if rm else ""

    valid_ids = {f["id"] for f in existing_facts}
    if verdict in ("CHANGE", "DUPLICATE") and target_id not in valid_ids:
        # a verdict we can't act on safely is no better than "unclear"
        if verdict == "CHANGE":
            return "UNCLEAR", None, reason or "CHANGE verdict had no valid TARGET"
        target_id = None
    return verdict, target_id, reason
