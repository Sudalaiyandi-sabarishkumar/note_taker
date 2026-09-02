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

import os
import re
from difflib import SequenceMatcher

from .ollama_client import DEFAULT_MODEL, chat, strip_think
from .transcript import chunk_transcript

_SYSTEM = (
    "You extract product requirements from a real client-call transcript. "
    "The one rule that matters most: NO CITATION, NO STATEMENT. Every fact "
    "must be backed by an exact verbatim quote from the transcript part you "
    "are given. Never paraphrase, never combine sentences, never infer. If "
    "you cannot point to the exact words, do not write the fact.\n"
    "A statement is a requirement, decision, constraint, or fact ABOUT THE "
    "PRODUCT being built. It is NOT: someone's reaction or opinion, a remark "
    "about workload or scheduling, a greeting, or a meta-comment about the "
    "call itself. \"That changes my week, in a good way\" is not a statement."
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

If one sentence bundles two separable requirements (joined by "but",
"however", "and", or ";"), emit a SEPARATE ## STATEMENT for each, and in
each Quote copy only the clause that supports that statement -- e.g. "must
work on mobile browsers, but a native app is out of scope" becomes two
statements, one quoting "must work on mobile browsers" and one quoting "a
native app is out of scope".

Separately, whenever someone raises something that is explicitly UNDECIDED,
deferred, or still being discussed -- phrases like "we haven't decided",
"still under discussion", "we might want", "not sure yet", "TBD", "don't
build it yet", "keep it in mind", "open question" -- output an open-question
block instead of a STATEMENT:

## OPEN_QUESTION
Feature: <the feature area it concerns>
Question: <one sentence naming what is undecided>
Quote: "<exact sentence(s) from THIS part, copied verbatim>"
Speaker: <name exactly as written, or "Unidentified speaker">
Timestamp: <the [HH:MM:SS...] prefix on that line, or "not available">

Only extract requirements/decisions/constraints/facts about the product.
Skip reactions, opinions, workload remarks, scheduling, and small talk.
Repeat for every distinct statement or open question, however minor. If this
part has neither, output the single line: NO STATEMENTS
Output nothing before the first block or after the last."""


def _coverage_instruction(already_summaries) -> str:
    listed = "\n".join(f"- {a}" for a in already_summaries) or "(none)"
    return f"""A first pass already extracted these statements from THIS part:
{listed}

Re-read the part. Output a ## STATEMENT block ONLY for a concrete requirement,
decision, constraint, rule, number, limit, threshold, or policy about the
product that a person actually stated here and that is NOT already covered by
the list above. Same format as before:

## STATEMENT
Feature: <a BROAD feature area, 1-2 plain words>
Summary: <one plain sentence>
Quote: "<exact sentence(s) from THIS part, copied verbatim>"
Speaker: <name exactly as written, or "Unidentified speaker">
Timestamp: <the [HH:MM:SS...] prefix on that line, or "not available">

Do not repeat, rephrase, or split anything already in the list. Do not infer
or paraphrase -- the Quote must be word-for-word from THIS part. Skip
reactions, opinions, scheduling and small talk. If nothing is missing, output
the single line: NO STATEMENTS
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
    """Returns ``(quote, verified, score)``. A quote that is already a
    verbatim substring is kept as-is (so a deliberately-quoted single clause
    stays a single clause); a near quote is replaced with the best-matching
    transcript span when the match clears ``_SNAP_THRESHOLD``; otherwise the
    model's quote is returned unchanged and unverified."""
    nq = _normalise(model_quote)
    if not nq:
        return model_quote, False, 0.0
    if nq in norm_transcript:
        return model_quote.strip().strip('"').strip(), True, 1.0
    best_span, best_score = None, 0.0
    for span, nspan in zip(spans, norm_spans):
        score = _similarity(nq, nspan)
        if score > best_score:
            best_span, best_score = span, score
    if best_span is not None and best_score >= _SNAP_THRESHOLD:
        return best_span, True, best_score
    return model_quote, False, best_score


def _parse_blocks(text, marker, labels, required):
    out = []
    for block in text.split(marker)[1:]:
        block = block.split("## ")[0]  # stop at the next block of any kind
        fields = {}
        for line in block.splitlines():
            line = line.strip()
            for key, label in labels.items():
                if line.startswith(label) and key not in fields:
                    value = line[len(label):].strip()
                    if key in ("quote",):
                        value = value.strip('"').strip()
                    elif key == "feature":
                        value = _decamel(value.strip('"').strip())
                    fields[key] = value
        if all(fields.get(k) for k in required):
            out.append(fields)
    return out


def parse_statement_blocks(text: str):
    stmts = _parse_blocks(text, "## STATEMENT", _LABELS, _FIELDS)
    for s in stmts:
        s["kind"] = "fact"
    q_labels = {"feature": "Feature:", "summary": "Question:",
                "quote": "Quote:", "speaker": "Speaker:", "timestamp": "Timestamp:"}
    for q in _parse_blocks(text, "## OPEN_QUESTION", q_labels, _FIELDS):
        q["kind"] = "question"
        stmts.append(q)
    return stmts


# BUG 5 -- the model still extracts personal reactions / meta-comments about
# the call ("that changes my week, in a good way") despite being told not to.
# As a *whole extracted quote* these phrasings are essentially never a product
# requirement, so drop the statement outright.
_BANTER_RE = re.compile(
    r"""\bmy\s+(week|day|days|schedule|workload|plate|life|team|year)\b
      | \bchanges?\s+my\b
      | \bin\s+a\s+good\s+way\b
      | \b(i'm|i\s+am|we're|we\s+are)\s+(so\s+)?(excited|glad|happy|thrilled|relieved|pleased)\b
      | \b(can't|cannot|can\s+not)\s+wait\b
      | \blooking\s+forward\s+to\b
      | \bgood\s+(news|to\s+hear)\b
      | \bthanks?,?\s+(everyone|all|so\s+much|for)\b
      | \bhow\s+(was|are|did|is)\s+(your|the|you)\b
      | \b(survived|barely)\s+it\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# A short line that opens with an agreement word is a speaker echoing back what
# was just said ("Yes, enforced order, same idea as modules.") -- an
# acknowledgement, not a distinct requirement. Only drop the short ones;
# a long sentence after "Yes," may carry real content.
_REPLY_ECHO_RE = re.compile(
    r"^\s*(yes|no|right|correct|understood|agreed|sure|okay|ok|exactly|"
    r"indeed|noted|got\s+it|sounds\s+good|makes\s+sense)\b[\s,.:;-]",
    re.IGNORECASE,
)


def _is_banter(quote: str) -> bool:
    return bool(_BANTER_RE.search(quote or ""))


def _is_reply_echo(quote: str) -> bool:
    q = (quote or "").strip()
    return bool(_REPLY_ECHO_RE.match(q)) and len(q.split()) <= 11


# A wish for a "feel" with no measurable object -- not a requirement.
_ASPIRATION_RE = re.compile(
    r"\b(should (?:feel|just feel)|feels? (?:really |truly )?"
    r"(?:effortless|seamless|premium|delightful|magical|intuitive|"
    r"modern|slick|polished|snappy|smooth|clean|elegant|frictionless)|"
    r"really love|love (?:using )?(?:it|the (?:app|product|tool))|"
    r"best[- ]in[- ]class|world[- ]class|not (?:just )?tolerate|"
    r"hate expense tools|people hate)\b",
    re.IGNORECASE,
)
_MEASURABLE_RE = re.compile(
    r"\d|\b(second|minute|hour|day|week|month|percent|%|kb|mb|gb|"
    r"page|click|step|field|character)s?\b", re.IGNORECASE)


def _is_aspiration(quote: str) -> bool:
    return bool(_ASPIRATION_RE.search(quote or "")) and not _MEASURABLE_RE.search(quote or "")


# Logistics / scheduling chatter about the PROJECT, not the product being built.
_LOGISTICS_RE = re.compile(
    r"\b(reschedul\w*|re-?schedule"
    r"|(?:move|push|shift|bump|change)\b[^.?!]{0,40}\b(?:session|meeting|call|sync|invite|slot|review|standup|stand-up|check-?in|checkpoint|catch-?up)\b"
    r"|(?:move|push|shift|bump|reschedul\w*)\b[^.?!]{0,40}\bto\b[^.?!]{0,20}\b(?:next week|monday|tuesday|wednesday|thursday|friday|\d{1,2}\s*(?:am|pm)|\d{1,2}[:.]\d{2})"
    r"|send\b[^.?!]{0,25}\b(?:an? )?invite\b|calendar (?:hold|invite)|same time next week"
    r"|(?:talk|see you|meet|catch up)\b[^.?!]{0,15}\b(?:next week|thursday|monday|tuesday|wednesday|friday|then|soon)\b"
    r"|review (?:the )?(?:mockups?|designs?|deck) (?:next|later)"
    r"|mockups?\b[^.?!]{0,20}\b(?:running )?(?:a day )?late"
    r"|pick (?:this |it )?up next (?:week|time)|regroup next week|circle back next"
    r"|i'?ll (?:re-?)?(?:send|share|resend)\b[^.?!]{0,20}\b(?:invite|calendar|note)s?)\b",
    re.IGNORECASE,
)


def _is_logistics(quote: str) -> bool:
    return bool(_LOGISTICS_RE.search(quote or ""))


# The speaker is describing something NOT yet decided -- record it as an open
# question, never as an Established Fact. Runs on an extracted statement's
# summary+quote (the model often files these as ## STATEMENT).
_UNDECIDED_RE = re.compile(
    r"\b("
    r"(?:haven'?t|have not|hasn'?t|has not|not|never|yet to be|still to be)\s+"
    r"(?:yet\s+|been\s+)*(?:decided|settled|agreed|determined|finali[sz]ed|confirmed|chosen|nailed down)"
    r"|no (?:final )?decision|(?:still|currently) (?:open|undecided|tbd|under discussion|being discussed|in discussion|up in the air)"
    r"|to be (?:decided|determined|confirmed)|\btbd\b|undetermined|open question|open item"
    r"|still (?:checking|deciding|discussing|arguing|debating|working (?:it|this) out|figuring (?:it|this) out|to be worked out)"
    r"|(?:we|they|finance|legal|the team|it)(?:'re| are| is|'s)? still (?:being |under )?(?:checked|discussed|decided|reviewed)"
    r"|we (?:might|may) (?:want to|do|add|build|introduce|consider|offer)\b[^.?!]{0,60}\b(?:but|,)?[^.?!]{0,30}\bnot decided\b"
    r"|not (?:yet )?decided|don'?t build (?:it|this|that) yet|not for (?:now|launch|v1|the mvp|version one)"
    r"|keep (?:it|this|that) in mind (?:for now)?|park(?:ed|ing)? (?:it|this)(?: for now)?"
    r"|revisit\b[^.?!]{0,30}\blater|defer(?:red)?\b[^.?!]{0,15}\b(?:to|until)|placeholder for now"
    r"|remains? (?:undecided|undetermined|open)(?!\s+question)"
    r")\b",
    re.IGNORECASE,
)
# Language that CLOSES a question -- overrides a stray "open question" /
# "undecided" mention in the same sentence ("that resolves last call's open
# question", "we decided X").
_RESOLVES_RE = re.compile(
    r"\b(?:resolv\w+|clos\w+|settl\w+|answer\w+)\b[^.?!]{0,25}"
    r"\b(?:open question|last (?:week|call)|question|it|that)\b"
    r"|\b(?:we|they)\s+(?:have\s+)?(?:decided|agreed|settled|confirmed)\b"
    r"(?![^.?!]{0,12}\bnot\b)",
    re.IGNORECASE)


def _is_undecided(text: str) -> bool:
    t = text or ""
    if _RESOLVES_RE.search(t):
        return False
    return bool(_UNDECIDED_RE.search(t))


# A speaker flagging that the topic is unsettled -- an open question, not a
# fact. Scanned straight off the transcript because the model does not
# reliably emit an ## OPEN_QUESTION block for these.
_HEDGE_RE = re.compile(
    r"\b(we (?:may be |might be |are )?misaligned|let me check (?:with|on)|"
    r"still (?:checking|discussing|arguing|deciding|not sure)|"
    r"we (?:haven'?t|have not) (?:decided|settled|agreed)|"
    r"not (?:yet )?(?:sure|decided|settled)|to be (?:decided|determined|confirmed)|"
    r"\btbd\b|open question|we disagree|we'?re not aligned|"
    r"under discussion|circle back on|revisit (?:this|that) later|"
    r"we might want to|don'?t build (?:it|that) yet|keep (?:it|that) in mind for now)\b",
    re.IGNORECASE,
)
_SPEAKER_LINE_RE = re.compile(r"^(?:\[([^\]]+)\]\s*)?([^:]{1,40}):\s*(.+)$")


def _scan_open_questions(transcript_text, already):
    """Pull lines that flag an unsettled topic straight from the transcript
    and turn them into open-question items. ``already`` is a set of
    normalised quotes to skip (so a line the model already captured is not
    duplicated)."""
    out = []
    for line in transcript_text.splitlines():
        line = line.strip()
        if not _HEDGE_RE.search(line):
            continue
        m = _SPEAKER_LINE_RE.match(line)
        ts, spk, text = (m.group(1) or "not available", m.group(2).strip(),
                         m.group(3).strip()) if m else ("not available",
                                                        "Unidentified speaker", line)
        if _normalise(text) in already or len(text.split()) < 4:
            continue
        if _is_logistics(text) or _is_banter(text) or _is_reply_echo(text):
            continue  # scheduling / small talk, even when phrased as a hedge
        out.append({"kind": "question", "feature": "", "summary": text.rstrip("."),
                    "quote": text, "speaker": spk, "timestamp": ts,
                    "verified": True, "match_score": 1.0})
        already.add(_normalise(text))
    return out


# Strong clause boundaries that separate two independent requirements inside
# one sentence: a semicolon, or a contrastive conjunction. NOT a bare "and"
# (would shatter "title, description, and priority").
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\s*;\s*|,?\s+but\s+|\.?\s+however,?\s+|,?\s+whereas\s+)", re.IGNORECASE)


def _bundle_parts(q: str):
    parts = []
    for sent in _SENT_END_RE.split(q):
        for clause in _CLAUSE_SPLIT_RE.split(sent):
            c = clause.strip().strip(" ,;")
            if c:
                parts.append(c)
    return parts


def _unbundle(statement: dict, norm_transcript: str):
    """Split a Quote that packs two independent requirements -- multiple
    sentences, or clauses joined by ';' / 'but' / 'however' -- into one
    statement each, provided every part still verifies verbatim against the
    transcript and reads like a requirement on its own. Otherwise the
    statement is returned unchanged and the normal verify/snap path handles
    it. Runs even when the whole Quote verifies as one span (a valid
    compound sentence still hides a second requirement)."""
    q = statement["quote"]
    parts = _bundle_parts(q)
    if len(parts) < 2:
        return [statement]
    verified_parts = [p for p in parts
                      if _normalise(p) in norm_transcript and _looks_like_requirement(p)]
    if len(verified_parts) < 2:
        return [statement]  # not a clean multi-requirement bundle; let snap try
    out = []
    for p in verified_parts:
        piece = dict(statement)
        piece["quote"] = p
        piece["summary"] = (p[0].upper() + p[1:]).rstrip(".!?") + "."
        out.append(piece)
    return out


_HAS_VERBISH_RE = re.compile(
    r"\b(is|are|was|were|be|will|shall|should|must|can|may|need|have|has|"
    r"want|allow|require|send|add|show|keep|make|use|get|give|set|pay|"
    r"support|include|enter|submit|approve|reject|delegate|calculat|"
    r"appear|retain|store|export|display|goes?|go|do|does)\b", re.IGNORECASE)


def _looks_like_requirement(text: str) -> bool:
    """A split-off fragment is only worth keeping as its own statement if it
    reads like one -- has some substance and a verb-ish word. Drops debris
    like 'With one shortcut.' that happens to be a verbatim substring."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return len(words) >= 5 and bool(_HAS_VERBISH_RE.search(text))


_TS_PREFIX_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)\]")


def _ts_from_transcript(quote: str, transcript_text: str):
    """Find the transcript line that contains ``quote`` and return its
    ``[HH:MM:SS]`` prefix, if any. Used to backfill a timestamp onto a
    statement the model left un-timestamped (unbundled piece, paraphrase)."""
    needle = _normalise(quote)
    if not needle:
        return ""
    for line in transcript_text.splitlines():
        if needle in _normalise(line):
            m = _TS_PREFIX_RE.match(line.strip())
            if m:
                return m.group(1)
    return ""


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
        blocks = [] if answer.strip().upper().startswith("NO STATEMENTS") \
            else parse_statement_blocks(answer)
        found.extend(blocks)
        for b in blocks:
            if b["feature"] not in seen_features:
                seen_features.append(b["feature"])

        # Coverage pass: a second look at the SAME part, asking only for
        # concrete requirements the first pass missed. A 7B under-extracts
        # from dense spans; exact repeats are dropped by quote de-dup below,
        # and reconcile handles near-repeats as DUPLICATE at merge time.
        if os.environ.get("MOM_COVERAGE", "1") != "0" and len(chunk) > 400:
            progress(f"  part {i}/{len(chunks)} (coverage check)...")
            cov_raw = strip_think(chat(
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content":
                     f"TRANSCRIPT PART {i} OF {len(chunks)}:\n{chunk}\n\n"
                     + _coverage_instruction(b["summary"] for b in blocks)}],
                model=model, show_progress=False))
            if not cov_raw.strip().upper().startswith("NO STATEMENTS"):
                cov_blocks = parse_statement_blocks(cov_raw)
                for b in cov_blocks:
                    b["from_coverage"] = True  # best-effort: drop if it can't ground
                found.extend(cov_blocks)
                for b in cov_blocks:
                    if b["feature"] not in seen_features:
                        seen_features.append(b["feature"])

    spans = _candidate_spans(transcript_text)
    norm_spans = [_normalise(sp) for sp in spans]

    kept, seen_quotes = [], set()
    for s in found:
        if (_is_banter(s["quote"]) or _is_reply_echo(s["quote"])
                or _is_logistics(s["quote"])):
            continue  # reaction, acknowledgement, or project-scheduling chatter
        # An open question is one item -- verify its quote but never unbundle
        # or reconcile it. An aspiration or an "undecided" statement is really
        # a question too, so mark it now and skip unbundling (splitting "we
        # haven't decided X. Y is still discussing it." into two questions is
        # noise).
        if s.get("kind") != "question":
            if _is_aspiration(s["quote"]):
                s = dict(s, kind="question", summary=(
                    "Turn into a concrete, testable requirement: "
                    + s["summary"].rstrip(".")))
            elif _is_undecided(s["summary"] + " " + s["quote"]):
                s = dict(s, kind="question", summary=(
                    "Decision needed: " + s["summary"].rstrip(".")))
        pieces = [s] if s.get("kind") == "question" else _unbundle(s, norm_transcript)
        for piece in pieces:
            snapped, verified, score = _snap_quote(
                piece["quote"], spans, norm_spans, norm_transcript)
            if snapped != piece["quote"]:
                piece["quote_as_extracted"] = piece["quote"]
                piece["quote"] = snapped
            piece["verified"] = verified
            piece["match_score"] = round(score, 2)
            ts = piece.get("timestamp", "").strip()
            if ts and ts.lower() != "not available" and ts not in transcript_text:
                piece["timestamp"] = "not available"
            # Recover a timestamp from the transcript line the (snapped) quote
            # sits on -- e.g. an unbundled piece or a paraphrase the model
            # left un-timestamped, but the .vtt line carries a [HH:MM:SS] prefix.
            if piece.get("timestamp", "").strip().lower() in ("", "not available"):
                recovered = _ts_from_transcript(piece["quote"], transcript_text)
                if recovered:
                    piece["timestamp"] = recovered
            if piece.get("kind") == "question" and not verified:
                continue  # an ungrounded "open question" is just noise
            if piece.get("from_coverage") and not verified:
                continue  # the coverage pass is best-effort -- an addition it
                          # can't ground verbatim is dropped, not flagged
            piece.setdefault("kind", "fact")
            key = _normalise(piece["quote"])
            if key in seen_quotes:
                continue  # same quote from an overlapping chunk
            seen_quotes.add(key)
            # A vague aspiration ("should feel effortless") is not a testable
            # fact -- reclassify it as an open question so it lands in the
            # right section instead of Established Facts.
            if piece["kind"] == "fact" and _is_aspiration(piece["quote"]):
                piece["kind"] = "question"
                piece["summary"] = ("Turn into a concrete, testable requirement: "
                                    + piece["summary"].rstrip("."))
            # A statement about something NOT yet decided ("we haven't decided
            # the refund window", "might do a discount, not decided") is an
            # open question, not a fact -- whatever block the model used.
            elif piece["kind"] == "fact" and _is_undecided(
                    piece["summary"] + " " + piece["quote"]):
                piece["kind"] = "question"
                piece["summary"] = ("Decision needed: "
                                    + piece["summary"].rstrip("."))
            kept.append(piece)

    # Deterministic sweep for "we haven't decided / let me check / misaligned"
    # lines the model skipped -- record them as open questions.
    kept.extend(_scan_open_questions(transcript_text, seen_quotes))
    return kept


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
# feature-name canonicalisation: fold a call's fragmented area names, and
# route them onto existing docs, using each name's actual content
# --------------------------------------------------------------------------
_CANON_SYSTEM = (
    "You file each requirement statement under a feature area. Reuse an "
    "existing area whenever a statement plausibly belongs to it. Only put "
    "two statements in the same area when they are clearly the same topic; "
    "keep separate concerns apart -- a delivery radius and payment methods "
    "are different areas even in the same call, as are login/SSO vs "
    "interface language, pricing vs enrollment, proctoring vs publishing."
)

_CANON_LINE_RE = re.compile(r"^\s*(\d+)\s*[=.):\-]+\s*(.+?)\s*$")
_CAMEL_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _decamel(text):
    """"DeliveryRadius" -> "Delivery Radius", "SSOConfig" -> "SSO Config".
    A collapsed name tokenises as one word and won't match an existing
    doc, so the model's PascalCase output is split back into words."""
    text = _CAMEL_2.sub(r"\1 \2", text)
    text = _CAMEL_1.sub(r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


def _sig_words(text):
    """4+ char, non-filler, stemmed words -- the tokens that could link two
    areas."""
    out = set()
    for w in _WORD_RE.findall(text.lower()):
        w = _stem(w)
        if len(w) >= 4 and w not in _STORY_STOPWORDS and w not in _GENERIC_WORDS:
            out.add(w)
    return out


# Words that must never START a feature name (a transcript filler the model
# grabbed instead of a topic) and words that carry no topic on their own.
_AREA_BAD_LEAD = {
    "there", "here", "this", "that", "these", "those", "it", "they", "we",
    "also", "and", "but", "so", "then", "the", "a", "an",
    "yes", "no", "ok", "okay", "well", "just",
}


def _bad_area(name: str) -> bool:
    """True if ``name`` is not a usable feature-doc title -- empty, led by a
    transcript filler word, or made only of stopword/generic tokens."""
    toks = _WORD_RE.findall((name or "").lower())
    if not toks:
        return True
    if toks[0] in _AREA_BAD_LEAD:
        return True
    return all(t in _STORY_STOPWORDS or t in _GENERIC_WORDS or t in _AREA_BAD_LEAD
               or len(t) < 3 for t in toks)


def _clean_area(text):
    text = text.strip().strip('"').strip("*").strip()
    text = text.split(" (")[0].split(" -- ")[0].strip()  # drop echoed hints
    # "NEW:" may be leading (a proposed new area) or mid-string (the model
    # echoed the template). If there's a real name before it, that's the
    # answer; otherwise take what follows.
    m = re.search(r"\bNEW\s*:\s*", text, re.IGNORECASE)
    if m:
        before, after = text[:m.start()].strip(), text[m.end():].strip()
        text = before or after
    words = [w for w in _decamel(text).split() if w]
    # Drop a leading filler word ("There annual leave..." -> "annual leave...")
    while words and words[0].lower() in _AREA_BAD_LEAD:
        words = words[1:]
    name = " ".join(words[:4])
    return "" if _bad_area(name) else name


_LEADING_VERBS = {
    "send", "add", "adding", "show", "display", "provide", "change", "changed",
    "enable", "allow", "let", "make", "give", "set", "use", "support",
    "receive", "ensure", "create", "build", "keep", "bump", "drop", "remove",
    "increase", "decrease", "reduce", "update", "require", "include", "have",
    "want", "need", "define", "specify", "confirm", "confirmed",
}


def _area_from_summary(summary, fallback):
    toks = _WORD_RE.findall(summary.lower())
    while toks and toks[0] in (_LEADING_VERBS | _STORY_STOPWORDS | _AREA_BAD_LEAD):
        toks = toks[1:]
    sig = [w for w in toks
           if len(w) >= 4 and w not in _STORY_STOPWORDS and w not in _GENERIC_WORDS
           and w not in _LEADING_VERBS and w not in _AREA_BAD_LEAD]
    name = " ".join(w.title() for w in sig[:2])
    if name:
        return name
    return fallback if fallback and not _bad_area(fallback) else "Misc"


# Minimal stoplist for the cohesion-split link graph: only function words and
# a few universal UI nouns. Domain nouns (admin, role, course, catalog...) are
# KEPT -- the ambient-word filter below removes whichever ones are so common
# in THIS batch that they carry no signal.
_LINK_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "be", "will", "must", "can", "should", "would", "via", "from",
    "as", "that", "this", "these", "those", "when", "after", "before", "only",
    "also", "each", "all", "any", "not", "no", "it", "its", "they", "their",
    "them", "we", "our", "you", "your", "page", "screen", "system", "feature",
    "able", "into", "over", "then", "there", "here", "such", "per", "new",
    "first", "release", "other", "another",
}


def _topic_words(text):
    """Stemmed 4+ char content words with their first surface form --
    ``({stem, ...}, {stem: 'surfaceword'})``. Used both for the split link
    graph and for naming a cluster."""
    stems, surface = set(), {}
    for w in _WORD_RE.findall((text or "").lower()):
        if len(w) < 4 or w in _LINK_STOP:
            continue
        st = _stem(w)
        surface.setdefault(st, w)
        stems.add(st)
    return stems, surface


_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
    "hour", "hours", "day", "days", "week", "weeks", "month", "months",
    "minute", "minutes", "second", "seconds", "percent", "star", "stars",
}


def _name_cluster(members, fallback, ambient=frozenset()):
    """Name a cluster from the distinctive words that appear in the most of
    its summaries (skipping ambient words, generic filler, and bare
    numbers/units)."""
    from collections import Counter
    freq, surface = Counter(), {}
    for s in members:
        st, sf = _topic_words(s["summary"])
        surface.update(sf)
        for w in st:
            sw = sf.get(w, w)
            if (w in ambient or sw in _GENERIC_WORDS or sw in _NUMBER_WORDS
                    or sw.isdigit()):
                continue
            freq[w] += 1
    if not freq:
        return _area_from_summary(members[0]["summary"], fallback)
    # most common; tie-break toward the earlier-mentioned word for stability
    ordered = sorted(freq, key=lambda w: (-freq[w], w))
    picks = [surface[st] for st in ordered[:2]]
    name = " ".join(w.title() for w in picks)
    return name if not _bad_area(name) else _area_from_summary(
        members[0]["summary"], fallback)


# Broad/dumping-ground names a first call tends to produce -- a cohesion
# split of one of these is almost always right.
_DUMP_NAMES = {"misc", "general", "features", "feature", "requirements",
               "admin", "admin role", "roles", "system", "product",
               "tool", "platform", "overview", "scope"}


def _is_dump_name(feat):
    """A first-call catch-all label ('Admin Role', 'Ticketing Tool Features',
    'Misc') as opposed to a specific feature name."""
    low = feat.lower().strip()
    if low in _DUMP_NAMES or _bad_area(feat):
        return True
    toks = _WORD_RE.findall(low)
    # Only genuinely catch-all suffixes -- NOT "Settings"/"Module"/"Tool",
    # which are normal parts of a specific feature name.
    return bool(toks) and toks[-1] in {
        "features", "functionality", "capabilities", "requirements",
        "stuff", "things", "items", "misc",
    }


def _components(word_sets):
    """Connected components of the 'share >= 1 word' graph. Returns a list of
    index lists."""
    n = len(word_sets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(n):
        for b in range(a + 1, n):
            if word_sets[a] & word_sets[b]:
                parent[find(a)] = find(b)
    comps = {}
    for k in range(n):
        comps.setdefault(find(k), []).append(k)
    return list(comps.values())


def cohesion_split(statements, existing):
    """Split a NEW feature area that is really several unrelated topics (a
    whole first call dumped into one doc) into one area per topic cluster.
    Mutates ``s['feature']`` in place; returns ``[(old, [new, ...])]``.

    Deliberately conservative -- it fires only when ALL of these hold:
      * the area is not an existing doc and has >= 4 statements
      * the statements form >= 2 clusters on the "share a content word" graph
        (a genuinely single-topic area, where one recurring noun links
        everything, is never split) -- for a catch-all-named area the
        graph is retried with that recurring noun removed
      * either the area's name is a catch-all label, or the split yields
        >= 2 real clusters (2+ statements each)
    """
    if os.environ.get("MOM_NO_SPLIT"):
        return []
    by_feat = {}
    for i, s in enumerate(statements):
        by_feat.setdefault(s.get("feature", ""), []).append(i)

    from collections import Counter
    changes = []
    for feat, idxs in by_feat.items():
        if feat in existing or len(idxs) < 4:
            continue
        words = [_topic_words(statements[i]["summary"])[0] for i in idxs]
        df = Counter()
        for w in words:
            df.update(w)
        ambient = {k for k, c in df.items() if c > len(idxs) * 0.5}
        dump = _is_dump_name(feat)

        comps = _components(words)
        # A dump doc where one noun ("ticket") repeats in every line collapses
        # to a single component -- retry with the ambient nouns stripped so
        # the real sub-topics separate. Only for catch-all-named areas: a
        # specific name is trusted and never force-split this way.
        if len(comps) < 2 and dump and ambient:
            comps = _components([w - ambient for w in words])

        if len(comps) < 2:
            continue
        real_clusters = sum(1 for m in comps if len(m) >= 2)
        if not dump and real_clusters < 2:
            continue

        new_names = []
        for members in comps:
            local = [statements[idxs[m]] for m in members]
            name = _name_cluster(local, feat, ambient)
            for m in members:
                statements[idxs[m]]["feature"] = name
            new_names.append(name)
        changes.append((feat, new_names))

    if changes and os.environ.get("MOM_DEBUG"):
        with open(os.environ["MOM_DEBUG"], "a") as _f:
            for old, new in changes:
                _f.write(f"=== SPLIT ===\n{old!r} -> {new}\n\n")
    return changes


def canonicalize_statements(statements, existing=None, model=None):
    """Assign each statement to a feature area. Returns a list of area names
    parallel to ``statements``. ``existing`` is ``{area: one-line
    description}``; an assignment may reuse one of those.

    Guards: an assignment onto an existing area that shares no distinctive
    word with the statement is rejected (kept as its own area). Any failure
    falls back to the statement's own extracted feature name.
    """
    existing = existing or {}
    n = len(statements)
    fallback = [
        (f if (f := s.get("feature", "")) and not _bad_area(f)
         else _area_from_summary(s["summary"], "Misc"))
        for s in statements
    ]
    if n == 0 or (n < 2 and not existing):
        return fallback
    model = model or DEFAULT_MODEL

    exist_block = "\n".join(f"- {k}: {v}" for k, v in existing.items()) or "(none yet)"
    stmt_block = "\n".join(
        f"{i}. [{s.get('feature', '?')}] {s['summary']}"
        for i, s in enumerate(statements, 1)
    )
    prompt = f"""EXISTING areas (name: what it covers):
{exist_block}

STATEMENTS from this call:
{stmt_block}

For each statement 1..{n}, write one line:

<number> = <area>

<area> is an EXISTING area name copied EXACTLY, or  NEW: <short 2-3 word name>.
Write nothing else on the line -- no explanation, no parentheses.

Rules:
- If a statement clearly belongs to an EXISTING area, reuse that name.
- Give a statement its OWN area unless it is plainly the same topic as
  another statement here -- do not lump distinct concerns together (a
  delivery radius and payment methods are different areas; a quiz pass mark
  and a certificate are different areas).
- Only combine statements that are facets of one narrow topic (a catalog
  page + its search + its bookmark button; a pass mark + retry limit +
  lockout).
Output exactly {n} lines."""
    try:
        raw = chat([{"role": "system", "content": _CANON_SYSTEM},
                    {"role": "user", "content": prompt}],
                   model=model, show_progress=False, num_predict=1000, temperature=0)
    except Exception:
        return fallback

    assigned = list(fallback)
    exist_lower = {k.lower(): k for k in existing}
    for line in strip_think(raw).splitlines():
        m = _CANON_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        area = _clean_area(m.group(2))
        if not (0 <= idx < n) or not area:
            continue
        assigned[idx] = exist_lower.get(area.lower(), area)

    # anti-misroute guard, per statement
    for i, area in enumerate(assigned):
        if area not in existing:
            continue
        src = statements[i]["summary"].lower()
        tgt = (area + " " + existing.get(area, "")).lower()
        connected = (any(w in tgt for w in _sig_words(src))
                     or any(w in src for w in _sig_words(area)))
        if not connected:
            # The route is wrong and the extracted feature name was the thing
            # that got mis-grouped -- don't trust it either. Name the area from
            # the statement itself so the mechanical router can't re-merge it.
            own = statements[i].get("feature", "")
            keep_own = (own and own not in existing
                        and bool(_sig_words(own) & _sig_words(src)))
            assigned[i] = own if keep_own else _area_from_summary(src, own or "Misc")

    # Consolidate near-duplicate NEW area names coined within THIS batch
    # (e.g. "Final Exams" and "Exams" from two statements about the same
    # thing) so they land in one doc, not two.
    new_areas = list(dict.fromkeys(a for a in assigned if a not in existing))
    canon_new = {}
    for a in new_areas:
        wa = _sig_words(a)
        hit = next((c for c in canon_new.values()
                    if wa and (wa <= _sig_words(c) or _sig_words(c) <= wa
                               or len(wa & _sig_words(c)) / len(wa | _sig_words(c)) >= 0.5)),
                   None)
        canon_new[a] = hit or a
    assigned = [canon_new.get(a, a) for a in assigned]

    if os.environ.get("MOM_DEBUG"):
        import json
        with open(os.environ["MOM_DEBUG"], "a") as _f:
            _f.write("=== CANON ===\nexisting: " + json.dumps(list(existing)) + "\n")
            _f.write("statements:\n" + stmt_block + "\n")
            _f.write("raw:\n" + strip_think(raw) + "\n")
            _f.write("assigned: " + json.dumps(assigned) + "\n\n")
    return assigned


# --------------------------------------------------------------------------
# Layer 2: propose which existing docs name the same broad area
# --------------------------------------------------------------------------
_MERGE_SYSTEM = (
    "You review a knowledge base of feature-area docs and spot the ones that "
    "were fragmented -- several thin docs that are really facets of ONE broad "
    "feature. A doc describing a page, its search, and a bookmark button are "
    "one feature. Ratings, reviews, and review moderation are one feature. "
    "Draft/publish, post-publish editing, and instructor assignment are one "
    "'Course Publishing'. Be decisive: in a list of 20+ docs there are "
    "usually several such groups. But never merge genuinely different "
    "concerns (login vs interface, pricing vs enrollment, payments vs "
    "notifications, quizzes vs certificates)."
)

_MERGE_LINE_RE = re.compile(r"^\s*MERGE\s*:\s*([\d,\s+&]+?)\s*=>\s*(\d+)\s*$", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")


def propose_doc_merges(descs, model=None):
    """``descs`` is ``{doc_name: one-line description}``. Returns a list of
    groups; each group is ``[canonical, member, member, ...]`` naming docs
    that should be combined. Empty on 'nothing to merge' or any failure.

    Docs are numbered and the model answers in numbers -- it cannot
    hallucinate a doc that isn't in the list.
    """
    names = list(descs)
    if len(names) < 4:
        return []
    model = model or DEFAULT_MODEL
    block = "\n".join(f"{i}. {k} -- {v}" for i, (k, v) in enumerate(descs.items(), 1))
    prompt = f"""Feature-area docs in the knowledge base ({len(names)} of them):
{block}

Several are fragments of ONE broader feature. Output EVERY merge, one per
line, using the NUMBERS above and nothing else:

MERGE: <n> + <n> [+ <n> ...] => <n>

The number after "=>" is the doc to keep (it must be one of the numbers
before "=>"). Fragmentation patterns to look for:
- a thing + its search + its filter + its bookmark/save
- ratings + reviews + review moderation
- draft/publish rules + post-publish editing + instructor assignment
- a default limit + a cap + a waitlist
- two docs that are just two names for the same noun (e.g. an area and that area "... Cap")
NEVER merge distinct concerns: login/SSO vs interface language, pricing vs
enrollment, proctoring vs publishing, payments vs notifications, quizzes vs
certificates.
A list this size usually has 3-6 merges. If truly none, output: NONE"""
    try:
        raw = chat([{"role": "system", "content": _MERGE_SYSTEM},
                    {"role": "user", "content": prompt}],
                   model=model, show_progress=False, num_predict=800, temperature=0)
    except Exception:
        return []

    groups = []
    for line in strip_think(raw).splitlines():
        m = _MERGE_LINE_RE.match(line)
        if not m:
            continue
        idxs = [int(x) - 1 for x in _NUM_RE.findall(m.group(1))]
        canon_idx = int(m.group(2)) - 1
        members = [names[i] for i in dict.fromkeys(idxs) if 0 <= i < len(names)]
        members = list(dict.fromkeys(members))
        if len(members) < 2:
            continue
        if 0 <= canon_idx < len(names) and names[canon_idx] in members:
            members.remove(names[canon_idx])
            members.insert(0, names[canon_idx])
        groups.append(members)

    if os.environ.get("MOM_DEBUG"):
        with open(os.environ["MOM_DEBUG"], "a") as _f:
            _f.write("=== MERGE PROPOSAL ===\n" + strip_think(raw)
                     + "\nparsed: " + repr(groups) + "\n\n")
    return groups


# --------------------------------------------------------------------------
# gap analysis: what a set of facts leaves UNSPECIFIED
# --------------------------------------------------------------------------
_GAP_SYSTEM = (
    "You are a senior business analyst pressure-testing a feature's "
    "requirements before build. Real requirements ALWAYS have gaps -- an "
    "unstated limit, an undefined failure path, an unnamed owner, an "
    "uncovered edge case. Your job is to surface the 2-3 most important "
    "questions the client still has to answer. Every question must be "
    "concrete (answerable with a specific rule or value) and must follow "
    "from a listed requirement -- never restate one, never invent a feature."
)
_GAP_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?GAP:\s*(.+?)\s*(?:\[(?:from\s*)?((?:EF-\d+[,\s]*)+)\])\s*$",
    re.IGNORECASE | re.MULTILINE)


def analyze_gaps(feature, active_facts, model=None):
    """Up to 3 specific unanswered questions implied by ``active_facts`` --
    each tied to the EF number(s) it follows from. Returns a list of
    ``{"question": str, "refs": [int, ...]}``. Off when MOM_GAPS=0, or when
    there is too little to reason about."""
    if os.environ.get("MOM_GAPS", "1") == "0" or len(active_facts) < 2:
        return []
    numbered = "\n".join(f'EF-{f["id"]}: {f["summary"]}' for f in active_facts)
    prompt = f"""Feature: {feature}

Confirmed requirements:
{numbered}

Find the 2-3 most important decisions these requirements DO NOT yet answer.
Look for: a number/limit not given, what happens on failure or error, an
edge case (empty / duplicate / concurrent / expired), who is allowed to do
it, how it can be undone, what happens at a boundary.

Worked example --
requirement: "A customer gets an email when the booking is confirmed."
GAP: What happens if the confirmation email fails to deliver? [from EF-1]
GAP: Can the customer opt out of confirmation emails? [from EF-1]

Now do the same for the requirements above. One per line, EXACTLY this form:
GAP: <a concrete question> [from EF-<n>]

Never ask about a value, range, or rule a requirement already states (if a
fact says "one to five stars", do NOT ask the maximum rating). Give at most
TWO gaps -- the most important. If the requirements are complete, output NONE."""
    try:
        raw = strip_think(chat(
            [{"role": "system", "content": _GAP_SYSTEM},
             {"role": "user", "content": prompt}],
            model=model or DEFAULT_MODEL, show_progress=False,
            num_predict=400, temperature=0))
    except Exception:
        return []
    if raw.strip().upper().startswith("NONE"):
        return []
    by_id = {f["id"]: f for f in active_facts}
    fact_words = _content_words(" ".join(f["summary"] for f in active_facts))
    out = []
    for m in _GAP_LINE_RE.finditer(raw):
        q = m.group(1).strip().rstrip(".") + "?"
        q = re.sub(r"\?+$", "?", q)
        refs = [int(x) for x in re.findall(r"EF-(\d+)", m.group(2)) if int(x) in by_id]
        if not refs:
            continue
        qw = _content_words(q) - _STORY_GLUE
        # Off-piste: shares no vocabulary with any fact.
        if not (qw & fact_words):
            continue
        # Already answered by the cited fact(s):
        cited_text = " ".join(by_id[r]["summary"] + " " + by_id[r]["quote"]
                              for r in refs)
        cited = _content_words(cited_text)
        if qw and qw <= cited:
            continue
        # ...asks for a limit/count/range that the cited fact already gives.
        if (re.search(r"\b(maximum|minimum|max|min|how many|how much|"
                      r"what (?:is|are) the (?:limit|range|number|count))\b", q, re.I)
                and re.search(r"\b(?:one|two|three|four|five|ten|\d+)\b"
                              r"[^.?!]{0,20}\b(?:to|-|through|and)\b"
                              r"[^.?!]{0,10}\b(?:one|two|three|four|five|ten|\d+)\b"
                              r"|\bup to \d+|\bbetween \w+ and \w+", cited_text, re.I)):
            continue
        if q.lower() not in {o["question"].lower() for o in out}:
            out.append({"question": q, "refs": refs})
    return out[:2]


# --------------------------------------------------------------------------
# ask: answer a question strictly from the knowledge docs
# --------------------------------------------------------------------------
_ASK_SYSTEM = (
    "You answer questions about product requirements using ONLY the feature "
    "docs provided. Every doc lists Established Facts (each with an EF "
    "number, a verbatim client quote, the speaker, and the source call + "
    "date), a Change Log, and any superseded facts marked '[superseded by "
    "EF-N]'. Ground every claim in that material: cite the EF number and the "
    "source call/date, and quote the client where relevant. To say what was "
    "used BEFORE a change, look at the superseded fact and the Change Log's "
    "'was:' line. If the docs do not contain the answer, say exactly: "
    "\"The docs don't record that.\" Never guess or add outside knowledge.\n"
    "Established Facts is the authoritative record. The 'User Story' line is "
    "only a short, lossy summary -- never treat it as the full picture and "
    "never just echo it. When asked for the full / complete / overall "
    "requirements or story, account for EVERY Established Fact that is not "
    "marked superseded -- list each one you used, and do not drop a fact "
    "just because it is a constraint or a system behaviour rather than a "
    "user-facing capability."
)


def _doc_score(question_words, doc_text, doc_name):
    body = _content_words(doc_name) | _content_words(doc_text)
    return len(question_words & body)


def answer_question(question, doc_texts, model=None):
    """``doc_texts`` is ``{feature_name: full_markdown}``. Picks the docs most
    relevant to ``question`` and asks the model to answer from them only.
    Returns the answer string (or a 'name a feature' hint)."""
    if not doc_texts:
        return "No feature docs yet -- run /requirements on a transcript first."
    qwords = _content_words(question)
    ranked = sorted(doc_texts.items(),
                    key=lambda kv: _doc_score(qwords, kv[1], kv[0]),
                    reverse=True)
    top = [(n, t) for n, t in ranked if _doc_score(qwords, t, n) > 0][:3]
    if not top:
        names = ", ".join(sorted(doc_texts))
        return (f"Couldn't tell which feature you mean. Name one in the "
                f"question, or pick from: {names}")

    context = "\n\n".join(f"===== {n} =====\n{t}" for n, t in top)
    prompt = (f"Feature docs:\n\n{context}\n\n"
              f"Question: {question}\n\n"
              f"Answer using only the docs above, citing EF numbers and "
              f"source calls/dates.")
    try:
        raw = chat([{"role": "system", "content": _ASK_SYSTEM},
                    {"role": "user", "content": prompt}],
                   model=model or DEFAULT_MODEL, show_progress=False,
                   num_ctx=8192, num_predict=700, temperature=0)
    except Exception as exc:
        return f"(model error: {exc})"
    ans = strip_think(raw).strip()
    used = ", ".join(n for n, _ in top)
    return f"{ans}\n\n— from: {used}"


# --------------------------------------------------------------------------
# user story: one plain-language synthesis of a feature's CURRENT facts
# --------------------------------------------------------------------------
_STORY_SYSTEM = (
    "You turn a feature's confirmed requirements into ONE user story in the "
    "form: \"As a <role>, I want <capability>, so that <benefit>.\" Use ONLY "
    "what the facts state. Infer <role> from who the requirement serves "
    "(customer, learner, admin, instructor, manager...). Include the \"so "
    "that <benefit>\" clause ONLY if a benefit is actually stated in the "
    "facts; otherwise end after the capability. Never invent a channel, "
    "number, name, benefit, or capability that is not in the facts. If there "
    "are several capabilities, join them in one story with 'and'."
)

_STORY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "be", "will", "must", "can", "should", "via", "from", "as",
    "that", "this", "when", "after", "before", "only", "also", "each", "all",
    "not", "no", "it", "its", "they", "their", "them", "user", "users",
    "customer", "customers", "feature", "system", "currently", "works", "work",
    "lets", "let", "pickup", "provide", "ensure", "allow", "include", "make",
    "using", "used", "use", "able", "then", "into", "over", "up",
    # user-story scaffolding -- not "content" that could be hallucinated
    "want", "see", "know", "learner", "role", "benefit", "story", "need",
    "so", "that", "able", "view",
}
_WORD_RE = re.compile(r"[a-z0-9]+")

# Product/SaaS filler that is never a distinctive link between two areas.
_GENERIC_WORDS = {
    "course", "learner", "student", "user", "customer", "admin", "instructor",
    "manager", "page", "screen", "view", "button", "field", "form", "flow",
    "process", "management", "support", "service", "option", "setting", "data",
    "app", "product", "item", "list", "detail", "section", "area", "module",
    "feature", "system", "platform", "content", "info", "information", "tool",
}


def _stem(w):
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    if w.endswith("s") and len(w) >= 4:          # courses->course, buses->buse (fine)
        return w[:-1]
    return w


def _content_words(text):
    return {_stem(w) for w in _WORD_RE.findall(text.lower())
            if len(w) >= 3 and w not in _STORY_STOPWORDS}


# Build/dev verbs that make a fact summary read like a ticket title rather
# than a requirement ("Implement the export" -> "The export").
_STUB_VERBS = (r"implement|build|define|specify|create|set\s*up|"
               r"add(?:\s+support\s+for)?|introduce|develop|design|"
               r"configure|enable")
_STUB_LEAD_RE = re.compile(r"^(?:please\s+)?(?:" + _STUB_VERBS + r")\s+", re.IGNORECASE)
_STUB_WANT_RE = re.compile(r"(,\s*I want\s+)(?:to\s+)?(?:" + _STUB_VERBS + r")\s+",
                           re.IGNORECASE)


def _destub(text: str) -> str:
    if not _STUB_LEAD_RE.match(text or ""):
        return text
    out = _STUB_LEAD_RE.sub("", text, count=1).strip()
    return (out[0].upper() + out[1:]) if out else text


# Past-tense "we decided X" phrasing: the summary narrates the meeting rather
# than stating the requirement.
_NARRATION_LEAD_RE = re.compile(
    r"^(?:please\s+)?(?:adjust|adjusted|maintain|maintained|retain|retained|"
    r"keep|kept|clarif\w+|discuss\w+|note[d]?|decide[d]?|agree[d]?|review\w+|"
    r"expand\w+|introduc\w+|reaffirm\w+|reiterat\w+|address\w+|"
    r"defin\w+|establish\w+|determin\w+)\b",
    re.IGNORECASE,
)
_QUOTE_LEAD_FILLER_RE = re.compile(
    r"^(?:so|okay|ok|yeah|yes|no|well|right|sure|and|but|also|actually|"
    r"look|listen|hold on|i mean|you know)[,\s]+", re.IGNORECASE)


def _summary_from_quote(quote: str) -> str:
    """Render a plain one-line summary straight from the client's words --
    used when the model's own summary drifts from the quote."""
    q = (quote or "").strip().strip('"').strip()
    q = _QUOTE_LEAD_FILLER_RE.sub("", q).strip()
    if not q:
        return (quote or "").strip()
    words = q.split()
    if len(words) > 24:                       # keep it to the first clause-ish
        cut = re.split(r"(?<=[,;:])\s", q)
        q = cut[0] if cut and len(cut[0].split()) >= 6 else " ".join(words[:24])
    q = q.rstrip(" .!?;:,") + "."
    q = _destub(q[0].upper() + q[1:])
    return q


def fix_summary(summary: str, quote: str) -> str:
    """Keep the model's summary unless it barely overlaps the quote's wording
    or narrates the decision instead of stating it -- then rebuild it from
    the quote. Never invents; only ever falls back to the client's own text."""
    summary = (summary or "").strip()
    if not summary:
        return _summary_from_quote(quote)
    sw = _content_words(summary)
    if not sw:
        return _summary_from_quote(quote)
    qw = _content_words(quote)
    overlap = len(sw & qw) / len(sw)
    # Over-reach: the summary adds 2+ substantive words the quote never uses
    # ("...with specific response time" on a quote that says none of that).
    novel = {w for w in (sw - qw)
             if len(w) >= 4 and w not in _STORY_GLUE and w not in _NUMBER_WORDS}
    if overlap < 0.35 or len(novel) >= 2 or _NARRATION_LEAD_RE.match(summary):
        return _summary_from_quote(quote)
    return summary


def _join_fact_summaries(active_facts, cap=8):
    """Every current requirement as one plain sentence -- the deterministic
    fallback when a synthesised story can't be trusted. Each clause is a
    (de-stubbed) fact summary; nothing is added."""
    clauses = []
    for f in active_facts[:cap]:
        c = _destub(f["summary"].strip()).rstrip(" .;")
        c = re.sub(r"\s*[;\n]\s*", ", ", c)   # no inner ';' -- it's our separator
        if not c:
            continue
        first = c.split()[0]
        if clauses and c[0].isupper() and not (len(first) > 1 and first.isupper()):
            c = c[0].lower() + c[1:]          # mid-sentence clause, not an acronym
        clauses.append(c)
    if not clauses:
        return "No confirmed requirements yet."
    text = "; ".join(clauses)
    extra = len(active_facts) - len(clauses)
    if extra > 0:
        text += f"; plus {extra} more (see Established Facts)"
    return text + "."


def synthesize_user_story(feature, active_facts, model=None):
    """Return ONE user story ("As a <role>, I want <capability>[, so that
    <benefit>].") for the feature's CURRENT (non-superseded) facts.

    One model call, grounded strictly in ``active_facts``. Falls back to a
    mechanical "As a user, I want ..." built from the fact summaries if the
    call fails or the result introduces content not in the facts.
    """
    if not active_facts:
        return "No confirmed requirements yet."

    # Mechanical fallback -- used only if the model call fails or its output
    # can't be trusted. Not forced into "As a ... I want ..." shape (the fact
    # summaries are declarative); every current requirement, joined plainly.
    # 100% grounded: it IS the fact summaries.
    fallback = _join_fact_summaries(active_facts)

    facts_block = "\n".join(f'- {f["summary"]} (exact words: "{f["quote"]}")'
                            for f in active_facts)
    prompt = f"""Feature: {feature}

Confirmed facts, all currently true:
{facts_block}

Write ONE user story: "As a <role>, I want <capability>, so that <benefit>."
- <role>: whoever the requirement serves, inferred from the facts.
- <capability>: what the facts say the product does, in your own words.
  Combine multiple facts with "and".
- ", so that <benefit>": include ONLY if a benefit is stated in the facts;
  otherwise end after <capability>.
It must be one grammatical sentence starting "As a". Use only the facts --
no invented detail. Output only the story."""
    try:
        raw = chat(
            [{"role": "system", "content": _STORY_SYSTEM},
             {"role": "user", "content": prompt}],
            model=model or DEFAULT_MODEL, show_progress=False,
            num_predict=300, temperature=0,
        )
    except Exception:
        return fallback
    story = _clean_story_text(strip_think(raw))
    if story.lower().lstrip("*_ ").startswith("as a"):
        story = _STUB_WANT_RE.sub(r"\1", story, count=1)  # "I want to implement X" -> "I want X"
        story = _constrain_role(story, active_facts)
        if story and not _story_problem(story, active_facts, feature):
            return story

    # The one-sentence story failed (bad shape, invented a detail, or left
    # whole facts out). Try once more for a short prose summary that must
    # cover EVERY fact, then re-check it. Still fall back to the plain join.
    try:
        raw2 = chat(
            [{"role": "system", "content": _STORY_SYSTEM},
             {"role": "user", "content":
                 f"Feature: {feature}\n\nThese are ALL the confirmed "
                 f"requirements, every one still true:\n{facts_block}\n\n"
                 "Write 2-4 plain declarative sentences that cover EVERY "
                 "requirement above -- omit none. No \"As a...\" framing. Add "
                 "no number, name, channel, or capability that is not stated. "
                 "Output only the summary."}],
            model=model or DEFAULT_MODEL, show_progress=False,
            num_predict=350, temperature=0,
        )
        story2 = _clean_story_text(strip_think(raw2))
        if story2 and not _story_problem(story2, active_facts, feature):
            return story2
    except Exception:
        pass
    return fallback


def _clean_story_text(raw):
    s = raw.strip().strip('"').strip()
    for lead in ("Sure,", "Here is", "Here's", "User story:", "Story:",
                 "Summary:"):
        if s.lower().startswith(lead.lower()):
            s = s.split(":", 1)[-1].strip() if ":" in s else s[len(lead):].strip()
            break
    return s


def _constrain_role(story, active_facts):
    """The <role> must be a word that appears in this doc's facts (stops a
    stale 'learner' leaking in from another domain)."""
    fact_text = " ".join(f["summary"] + " " + f["quote"]
                         for f in active_facts).lower()
    m = re.match(r"\s*as an?\s+([a-z][a-z ]{1,20}?)[,\s]", story, re.IGNORECASE)
    if m:
        head = m.group(1).strip().lower().split()[-1].rstrip("s")
        if head and head != "user" and head not in fact_text:
            story = re.sub(r"^\s*as an?\s+[a-z][a-z ]{1,20}?([,\s])",
                           r"As a user\1", story, count=1, flags=re.IGNORECASE)
    return story


def _story_problem(story, active_facts, feature):
    """None if the story is usable; a short reason string if not.

    Two failure modes:
      * FABRICATION -- a number, or a proper noun / acronym, that no fact
        states (this is what an invented "email, SMS, WhatsApp" looked like).
        Lowercase rewording ("issue" for "ticket") is fine.
      * INCOMPLETE -- it leaves 2+ whole facts unmentioned, so it misreads as
        the full picture when it isn't.
    """
    fact_lc = " ".join(f["summary"] + " " + f["quote"]
                       for f in active_facts).lower()
    if _numbers_in(story) - _numbers_in(fact_lc):
        return "invented number"
    grounded = set(_WORD_RE.findall(fact_lc)) | _content_words(feature)
    toks = re.findall(r"\S+", story)
    for i, tok in enumerate(toks):
        core = re.sub(r"[^A-Za-z]", "", tok)
        if len(core) < 3:
            continue
        at_start = i == 0 or toks[i - 1][-1:] in ".!?:;"
        propery = core.isupper() or (core[0].isupper() and not at_start)
        if propery and core.lower() not in grounded and core.lower() not in _STORY_GLUE:
            return f"invented proper noun: {core}"
    # Coverage: a fact is "mentioned" only if the story shares one of its
    # DISTINCTIVE words -- content words minus glue minus words common to
    # half+ the facts ("email", "ticket"...), which would otherwise let one
    # fact stand in for an unrelated one.
    from collections import Counter
    fact_words, df = [], Counter()
    for f in active_facts:
        w = _content_words(f["summary"]) - _STORY_GLUE
        fact_words.append(w)
        df.update(w)
    ambient = ({k for k, c in df.items() if c * 2 > len(active_facts)}
               if len(active_facts) > 2 else set())
    sw = _content_words(story)
    missed = sum(1 for w in fact_words if (w - ambient) and not (w - ambient) & sw)
    if missed >= 2 or (missed and len(active_facts) <= 3):
        return f"leaves {missed} fact(s) out"
    return None


_STORY_GLUE = {
    "within", "appropriate", "relevant", "various", "certain", "particular",
    "additional", "additionally", "whether", "based", "according", "able",
    "using", "via", "regarding", "related", "necessary", "required",
    "available", "applicable", "specific", "same", "such", "both", "either",
    "each", "every", "their", "there", "when", "once", "after", "before",
    "while", "always", "never", "also", "then", "however", "instead",
    "system", "product", "platform", "tool", "feature", "user", "users",
    "the", "and", "for",
}

_NUM_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def _numbers_in(text):
    text = (text or "").lower()
    nums = {t.replace(",", "") for t in _NUM_TOKEN_RE.findall(text)}
    nums |= {w for w in _WORD_RE.findall(text) if w in _NUMBER_WORDS}
    return nums
