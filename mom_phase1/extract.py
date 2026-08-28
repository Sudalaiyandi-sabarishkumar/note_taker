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
            "Feature areas already in use (reuse the EXACT name when a "
            "statement belongs to one of these -- same spelling and case -- "
            "even if the statement changes or contradicts what was said "
            "before; it is still the same feature area):\n  "
            + ", ".join(sorted(known_features))
            + "\nOnly invent a new area name when a statement fits none of them."
        )
    else:
        feat = "No feature areas exist yet -- this is the first part processed."
    return f"""This is PART {part_no} of {total} of one call transcript (not the whole call).
{feat}

For every distinct requirement, decision, constraint, or fact actually stated
by a person in THIS PART, output one block in exactly this format:

## STATEMENT
Feature: <a BROAD feature area, 1-2 plain words>
Summary: <one plain sentence for a human skimming later>
Quote: "<exact sentence(s) from THIS part, copied verbatim>"
Speaker: <name exactly as written, or "Unidentified speaker">
Timestamp: <the [HH:MM:SS...] prefix on that line, or "not available">

Rules for Feature -- read carefully, this is where extractions usually go wrong:
- A feature area is BROAD and collects many facts over many calls. Think
  "Certificates", "Notifications", "Enrollment", "Video Playback",
  "Course Publishing", "Reviews", "Payments" -- roughly 10-20 areas for a
  whole product.
- Do NOT create a separate area per sentence. "Certificate wording",
  "Certificate PDF", "Certificate issue trigger" are ALL just "Certificates".
  "Playback speed", "Resume position", "Offline download" are ALL just
  "Video Playback".
- When several statements in THIS part concern the same area, give them the
  IDENTICAL Feature name.
- Name the lasting area, never the change to it: switching email to SMS is
  still "Notifications", never "SMS Notifications" or "Switch to SMS". No
  parentheses, no "from X to Y", no verbs.

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
    # Feature names grow as chunks are processed so a later chunk of the SAME
    # call is told to reuse the areas its earlier chunks established, instead
    # of coining a near-duplicate.
    seen_features = list(dict.fromkeys(known_features))
    for i, chunk in enumerate(chunks, start=1):
        progress(f"  part {i}/{len(chunks)}...")
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content":
                f"TRANSCRIPT PART {i} OF {len(chunks)}:\n{chunk}\n\n"
                + _instruction(i, len(chunks), seen_features)},
        ]
        raw = chat(messages, model=model, show_progress=False)
        answer = strip_think(raw)
        if answer.strip().upper().startswith("NO STATEMENTS"):
            continue
        blocks = parse_statement_blocks(answer)
        found.extend(blocks)
        for b in blocks:
            if b["feature"] not in seen_features:
                seen_features.append(b["feature"])

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


# --------------------------------------------------------------------------
# feature-name canonicalisation: fold a call's fragmented area names
# --------------------------------------------------------------------------
_CANON_SYSTEM = (
    "You tidy a list of feature-area names extracted from one requirements "
    "call. Some are near-duplicates naming the same broad area. You map each "
    "name to a canonical area name. You never merge names that are genuinely "
    "different areas."
)

_CANON_LINE_RE = re.compile(r"^\s*(\d+)[.):]\s*(.+?)\s*$")


def canonicalize_feature_names(names, model=None):
    """One LLM call that folds a call's fragmented area names. Given
    ``["Certificate Format", "Certificate Content", "Video Playback",
    "Playback Speed"]`` it returns ``{"Certificate Format": "Certificates",
    "Certificate Content": "Certificates", "Video Playback": "Video
    Playback", "Playback Speed": "Video Playback"}``. On any failure it
    returns an identity map, so the caller always gets something usable."""
    names = list(dict.fromkeys(names))
    identity = {n: n for n in names}
    if len(names) < 2:
        return identity
    model = model or DEFAULT_MODEL
    numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
    prompt = f"""These feature-area names came from ONE requirements call. Some are
near-duplicates that name the same broad area -- for example "Certificate
Format" / "Certificate Content" / "Certificate Issuance" are all
"Certificates"; "Playback Speed" / "Resume Position" are both "Video
Playback".

Names:
{numbered}

Output exactly one line per name above, in the same order and numbering:

<n>. <canonical area name>

Rules:
- Names for the same broad area MUST get the identical canonical name.
- Canonical names are short (1-2 words); reuse an input name where sensible.
- A name with no duplicate maps to itself (cleaned up).
- Never merge names that are genuinely different areas.
- No text before line 1 or after the last line."""
    try:
        raw = chat([{"role": "system", "content": _CANON_SYSTEM},
                    {"role": "user", "content": prompt}],
                   model=model, show_progress=False, num_predict=800, temperature=0)
    except Exception:
        return identity
    mapping = {}
    for line in strip_think(raw).splitlines():
        m = _CANON_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        canon = m.group(2).strip().strip('"').strip()
        if 0 <= idx < len(names) and canon:
            mapping[names[idx]] = canon
    # any name the model skipped keeps its original name
    for n in names:
        mapping.setdefault(n, n)
    return mapping


# --------------------------------------------------------------------------
# user story: one plain-language synthesis of a feature's CURRENT facts
# --------------------------------------------------------------------------
_STORY_SYSTEM = (
    "You restate a feature's already-confirmed requirements as one short "
    "plain-language description of how the feature currently works. Use ONLY "
    "the words and facts you are given. Never add a detail, a channel, a "
    "number, a role, or a capability that is not written in the facts. If two "
    "facts overlap, merge them; if they do not, just state both. Every fact "
    "you are given is currently true."
)

_STORY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "be", "will", "must", "can", "should", "via", "from", "as",
    "that", "this", "when", "after", "before", "only", "also", "each", "all",
    "not", "no", "it", "its", "they", "their", "them", "user", "users",
    "customer", "customers", "feature", "system", "currently", "works", "work",
    "lets", "let", "pickup", "provide", "ensure", "allow", "include", "make",
    "using", "used", "use", "able", "then", "into", "over", "up",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _content_words(text):
    return {_stem(w) for w in _WORD_RE.findall(text.lower())
            if len(w) >= 3 and w not in _STORY_STOPWORDS}


def synthesize_user_story(feature, active_facts, model=None):
    """Return a short plain-language description of the CURRENT agreed
    behaviour of ``feature``, derived strictly from ``active_facts`` (the
    non-superseded Established Facts). Guardrails:

      * 0 facts   -> a fixed placeholder.
      * 1 fact    -> that fact's summary verbatim (no model call -- there is
                     nothing to synthesise and nothing to hallucinate).
      * 2+ facts  -> one model call to merge them; the result is rejected
                     (fall back to a mechanical join) if the model call
                     fails, or if the story introduces content words that
                     appear in none of the facts.
    """
    if not active_facts:
        return "No confirmed requirements yet."
    fallback = "; ".join(f["summary"].rstrip(".") for f in active_facts) + "."
    if len(active_facts) == 1:
        s = active_facts[0]["summary"].strip()
        return s if s.endswith((".", "!", "?")) else s + "."

    facts_block = "\n".join(f'- {f["summary"]} (exact words: "{f["quote"]}")'
                            for f in active_facts)
    prompt = f"""Feature: {feature}

Confirmed facts, all currently true:
{facts_block}

Write 1 to 3 sentences describing how "{feature}" currently works. Use ONLY the
information in the facts above. Do not introduce any channel, number, name,
role, or capability that is not written there. Where facts overlap, combine
them into one statement. Output only the sentences, nothing else."""
    try:
        raw = chat(
            [{"role": "system", "content": _STORY_SYSTEM},
             {"role": "user", "content": prompt}],
            model=model or DEFAULT_MODEL, show_progress=False,
            num_predict=400, temperature=0,
        )
    except Exception:
        return fallback
    story = strip_think(raw).strip().strip('"').strip()
    for lead in ("Sure,", "Here is", "Here's", "The feature", "Summary:", "User story:"):
        if story.startswith(lead):
            story = story.split(":", 1)[-1].strip() if ":" in story[:40] else fallback
            break
    if not story:
        return fallback

    # Structural guard: the story may not bring in content words that appear
    # in none of the facts (this is what caught an invented "email, SMS,
    # WhatsApp" leaking in from a prompt example).
    grounded = set()
    for f in active_facts:
        grounded |= _content_words(f["summary"]) | _content_words(f["quote"])
    grounded |= _content_words(feature)
    novel = _content_words(story) - grounded
    if len(novel) >= 3:  # a little phrasing slack; 3+ novel words = invented content
        return fallback
    return story
