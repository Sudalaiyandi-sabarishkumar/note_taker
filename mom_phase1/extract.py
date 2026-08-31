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

Only extract requirements/decisions/constraints/facts about the product.
Skip reactions, opinions, workload remarks, scheduling, and small talk.
Repeat for every distinct statement, however minor. If this part has none,
output the single line: NO STATEMENTS
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
                    elif key == "feature":
                        value = _decamel(value.strip('"').strip())
                    fields[key] = value
        if all(k in fields and fields[k] for k in _FIELDS):
            statements.append(fields)
    return statements


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


def _unbundle(statement: dict, norm_transcript: str):
    """If the Quote already verifies as one contiguous span, return the
    statement unchanged. Otherwise, if it is 2+ sentences and each on its own
    verifies against the transcript, split it: one statement per sentence,
    the sentence text becoming its own Summary (the quote stays ground
    truth). Sentences that don't verify are dropped here and the leftover is
    handled by the normal verify/snap path."""
    q = statement["quote"]
    if _normalise(q) in norm_transcript:
        return [statement]
    parts = [p.strip() for p in _SENT_END_RE.split(q) if p.strip()]
    if len(parts) < 2:
        return [statement]
    verified_parts = [p for p in parts if _normalise(p) in norm_transcript]
    if len(verified_parts) < 2:
        return [statement]  # not a clean multi-quote bundle; let snap try
    out = []
    for p in verified_parts:
        piece = dict(statement)
        piece["quote"] = p
        piece["summary"] = p.rstrip(".!?") + "."
        out.append(piece)
    return out


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

    kept = []
    for s in found:
        if _is_banter(s["quote"]) or _is_reply_echo(s["quote"]):
            continue  # reaction, meta-comment, or bare acknowledgement
        # A model that jammed several transcript sentences into one Quote is
        # unbundled here so each real sentence becomes its own statement.
        for piece in _unbundle(s, norm_transcript):
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
            kept.append(piece)
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
    return " ".join(words[:4])


_LEADING_VERBS = {
    "send", "add", "adding", "show", "display", "provide", "change", "changed",
    "enable", "allow", "let", "make", "give", "set", "use", "support",
    "receive", "ensure", "create", "build", "keep", "bump", "drop", "remove",
    "increase", "decrease", "reduce", "update", "require", "include", "have",
    "want", "need", "define", "specify", "confirm", "confirmed",
}


def _area_from_summary(summary, fallback):
    toks = _WORD_RE.findall(summary.lower())
    while toks and toks[0] in (_LEADING_VERBS | _STORY_STOPWORDS):
        toks = toks[1:]
    sig = [w for w in toks
           if len(w) >= 4 and w not in _STORY_STOPWORDS and w not in _GENERIC_WORDS
           and w not in _LEADING_VERBS]
    return " ".join(w.title() for w in sig[:2]) or fallback


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
    fallback = [s.get("feature", "") or _area_from_summary(s["summary"], "Misc")
                for s in statements]
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
    "\"The docs don't record that.\" Never guess or add outside knowledge."
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


def synthesize_user_story(feature, active_facts, model=None):
    """Return ONE user story ("As a <role>, I want <capability>[, so that
    <benefit>].") for the feature's CURRENT (non-superseded) facts.

    One model call, grounded strictly in ``active_facts``. Falls back to a
    mechanical "As a user, I want ..." built from the fact summaries if the
    call fails or the result introduces content not in the facts.
    """
    if not active_facts:
        return "No confirmed requirements yet."

    def _cap_phrase(text):
        toks = text.strip().rstrip(".").split()
        if toks and toks[0].lower() in _LEADING_VERBS:
            toks = toks[1:]
        if not toks:
            return text.strip().rstrip(".")
        return toks[0][0].lower() + " ".join(toks)[1:]

    fallback = ("As a user, I want "
                + " and ".join(_cap_phrase(f["summary"]) for f in active_facts[-3:])
                + ".")

    facts_block = "\n".join(f'- {f["summary"]} (exact words: "{f["quote"]}")'
                            for f in active_facts)
    prompt = f"""Feature: {feature}

Confirmed facts, all currently true:
{facts_block}

Write ONE user story: "As a <role>, I want <capability>, so that <benefit>."
- <role>: whoever the requirement serves, inferred from the facts.
- <capability>: what the facts say the product does. Combine multiple facts
  with "and".
- ", so that <benefit>": include ONLY if a benefit is stated in the facts;
  otherwise end the sentence after <capability>.
Use only the facts above -- no invented detail. Output only the story."""
    try:
        raw = chat(
            [{"role": "system", "content": _STORY_SYSTEM},
             {"role": "user", "content": prompt}],
            model=model or DEFAULT_MODEL, show_progress=False,
            num_predict=300, temperature=0,
        )
    except Exception:
        return fallback
    story = strip_think(raw).strip().strip('"').strip()
    for lead in ("Sure,", "Here is", "Here's", "User story:", "Story:"):
        if story.lower().startswith(lead.lower()):
            story = story.split(":", 1)[-1].strip() if ":" in story[:40] else fallback
            break
    if not story or "as a" not in story.lower()[:12]:
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
