"""Merge extracted statements into per-feature knowledge docs.

The doc structure written here is pure Python -- every Established Fact
traces back to a quote the extract step already verified ("no citation, no
statement"). Two optional callbacks add model-written prose that is
DERIVED from those cited facts, never new information: ``canon_fn`` folds a
call's fragmented area names, and ``story_fn`` writes the "## User Story"
line by combining a feature's current (non-superseded) facts.

Each doc has four sections:
  ## User Story          one plain-language synthesis of the current facts
  ## Established Facts    the cited facts, superseded ones kept + marked
  ## Open Questions       [NEEDS REVIEW] / [UNVERIFIED CITATION] items
  ## Change Log           append-only audit trail

Merge policy:
  * A call's freshly-extracted area names are first folded to canonical
    names (``canon_fn``, or a mechanical word-overlap fallback), then
    routed onto an existing doc where the words overlap enough.
  * A statement for a feature with NO existing facts goes straight in as an
    Established Fact.
  * A statement for a feature that ALREADY has facts is reconciled against
    those facts (``reconcile`` callback, an LLM call wired in by the CLI):
      - NEW       -> added as a new Established Fact
      - DUPLICATE -> not added; noted in the Change Log
      - CHANGE    -> the new statement is added as a new Established Fact and
                     the one it replaces is kept but marked
                     "[superseded by EF-N]", so the section stays complete
                     and the supersession is also logged.
      - UNCLEAR (or no reconcile callback) -> filed under Open Questions as
                     [NEEDS REVIEW] with the new quote next to the existing
                     facts, for a human to resolve.
  * A statement whose quote could not be verified against the transcript is
    filed as [UNVERIFIED CITATION] regardless of feature state.
  * The Change Log is append-only and records every add, restate, and
    rewrite -- it is the audit trail.
"""

import os
import re
from datetime import date

DOCS_DIR = os.environ.get("MOM_DOCS_DIR", "knowledge")

_STOPWORDS = {"a", "an", "the", "for", "in", "on", "of", "to", "and", "or",
              "from", "as", "via", "with", "by", "at",
              "app", "flow", "feature", "screen", "page", "system", "module",
              "new", "change", "changing", "switch", "switching", "update",
              "updating", "method", "option", "support"}
_EF_LINE_RE = re.compile(
    r'^-\s*\*\*EF-(\d+)\*\*(?:\s*\[superseded by EF-(\d+)\])?'
    r':\s*(.+?)\s*—\s*\*"(.+?)"\*\s*—\s*(.+)$'
)


# --------------------------------------------------------------------------
# feature-name routing
# --------------------------------------------------------------------------
def _feature_words(name: str):
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in words if w not in _STOPWORDS}


def match_existing_feature(new_name, existing_names, threshold=0.5):
    """Route a freshly-extracted feature name onto an existing doc's name
    when they plainly refer to the same thing, else return ``new_name``.

    Word-overlap only (no LLM). Plain Jaccard is too weak when a local
    model emits a verbose name ("Switching Notification Method from Email
    to SMS") whose one shared head noun ("notification") is diluted by
    descriptive filler -- so the score is the max of Jaccard and, when the
    shorter name's words are wholly contained in the longer's, a
    containment score. A single shared word only counts when it is that
    subset relationship, to avoid merging "Leave Approval" into "Leave
    Balance" on the shared "leave".
    """
    new_words = _feature_words(new_name)
    if not new_words:
        return new_name
    best, best_score = new_name, threshold
    for existing in existing_names:
        ew = _feature_words(existing)
        if not ew:
            continue
        inter = len(new_words & ew)
        if not inter:
            continue
        jaccard = inter / len(new_words | ew)
        subset = new_words <= ew or ew <= new_words
        containment = inter / min(len(new_words), len(ew))
        if inter == 1 and not subset:
            score = jaccard
        else:
            score = max(jaccard, containment)
        if score > best_score or (score == best_score and score >= threshold
                                  and best is new_name):
            best, best_score = existing, score
    return best


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unnamed-feature"


def canonicalize_batch_features(statements):
    """Collapse near-duplicate feature names *within one call's* extractions
    before anything is written -- the model still coins "Certificate
    Content" / "Certificate Format" for one area within a single call, and
    those never get reconciled against each other later. Groups names that
    share a distinctive word (one not used by 3+ other names in the batch)
    or that clear the same overlap bar as cross-call routing, then rewrites
    every statement to the shortest name in its group."""
    names = list(dict.fromkeys(s["feature"] for s in statements))
    if len(names) < 2:
        return statements

    word_freq = {}
    for n in names:
        for w in _feature_words(n):
            word_freq[w] = word_freq.get(w, 0) + 1
    generic = {w for w, c in word_freq.items() if c >= 3}

    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb, key=len)] = min(ra, rb, key=len)

    for i, a in enumerate(names):
        wa = _feature_words(a)
        for b in names[i + 1:]:
            wb = _feature_words(b)
            shared = wa & wb
            distinctive = shared - generic
            if distinctive or match_existing_feature(a, [b]) == b:
                union(a, b)

    canon = {n: find(n) for n in names}
    for s in statements:
        s["feature"] = canon.get(s["feature"], s["feature"])
    return statements


# --------------------------------------------------------------------------
# reading an existing doc
# --------------------------------------------------------------------------
def discover_features(docs_dir=None):
    """Returns ``{feature_name_from_H1: path}``."""
    docs_dir = docs_dir or DOCS_DIR
    out = {}
    if not os.path.isdir(docs_dir):
        return out
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(docs_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                first = f.readline().strip()
        except OSError:
            continue
        m = re.match(r"^#\s+(.+)$", first)
        if m:
            out[m.group(1).strip()] = path
    return out


def _split_sections(doc_text: str):
    sections = {"user_story": "", "established_facts": "", "open_questions": "",
                "change_log": ""}
    for m in re.finditer(r"^## (.+?)\n(.*?)(?=\n## |\Z)", doc_text,
                         re.DOTALL | re.MULTILINE):
        heading = m.group(1).strip().lower()
        body = m.group(2).strip()
        if "user story" in heading:
            sections["user_story"] = body
        elif "established fact" in heading:
            sections["established_facts"] = body
        elif "open question" in heading:
            sections["open_questions"] = body
        elif "change log" in heading:
            sections["change_log"] = body
    return sections


def describe_features(docs_dir=None):
    """``{feature_name: one-line description}`` -- the User Story if the doc
    has one, else its first Established Fact summary. Feeds the canon step so
    it can route a new area onto the right existing doc by content."""
    out = {}
    for name, path in discover_features(docs_dir).items():
        try:
            with open(path, encoding="utf-8") as f:
                sec = _split_sections(f.read())
        except OSError:
            continue
        desc = sec["user_story"].strip().splitlines()[0] if sec["user_story"] else ""
        if not desc:
            facts = _parse_established_facts(sec["established_facts"])
            desc = facts[0]["summary"] if facts else name
        out[name] = desc
    return out


def _parse_established_facts(body: str):
    facts = []
    for line in body.splitlines():
        m = _EF_LINE_RE.match(line.strip())
        if m:
            facts.append({"id": int(m.group(1)),
                          "superseded_by": int(m.group(2)) if m.group(2) else None,
                          "summary": m.group(3).strip(),
                          "quote": m.group(4).strip(),
                          "attribution": m.group(5).strip()})
    return facts


def _active(facts):
    """Facts that still stand -- not replaced by a later CHANGE."""
    return [f for f in facts if not f.get("superseded_by")]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _ef_line(f):
    tag = (f' [superseded by EF-{f["superseded_by"]}]'
           if f.get("superseded_by") else "")
    return (f'- **EF-{f["id"]}**{tag}: {f["summary"]} — '
            f'*"{f["quote"]}"* — {f["attribution"]}')


def _render(feature, facts, new_questions, prior_questions_raw,
            prior_changelog_raw, changelog_entry, user_story):
    ef_lines = "\n".join(
        _ef_line(f) for f in sorted(facts, key=lambda f: f["id"])
    ) or "- None yet."

    if prior_questions_raw.strip().lower() in ("- none.", "none.", "none", ""):
        prior_questions_raw = ""
    q_parts = [p for p in (prior_questions_raw, "\n\n".join(new_questions)) if p]
    q_text = "\n\n".join(q_parts) if q_parts else "- None."

    cl_text = "\n".join(p for p in (prior_changelog_raw, changelog_entry) if p)

    return (
        f"# {feature}\n\n"
        f"## User Story\n{user_story}\n\n"
        f"## Established Facts\n{ef_lines}\n\n"
        f"## Open Questions / Ambiguities\n{q_text}\n\n"
        f"## Change Log\n{cl_text}\n"
    )


# --------------------------------------------------------------------------
# the merge
# --------------------------------------------------------------------------
def merge_statements(statements, source_name, docs_dir=None, reconcile=None,
                     story_fn=None, canon_fn=None):
    """Write/update one doc per feature. Returns a list of per-feature
    summary strings for the CLI to print.

    ``reconcile(feature, existing_facts, statement) -> (verdict, target_id,
    reason)`` decides how a statement relates to a feature that already has
    facts (see module docstring). If it is None, every such statement is
    filed as [NEEDS REVIEW] -- the old conservative behaviour.

    ``story_fn(feature, active_facts) -> str`` synthesises the "## User
    Story" line from the non-superseded facts. If None, a mechanical join of
    the fact summaries is used.

    ``canon_fn(batch, existing) -> {name: canonical}`` folds this call's
    fragmented area names and routes them onto existing docs, using each
    name's statement summaries as context. If None, a mechanical
    word-overlap fold is used.
    """
    docs_dir = docs_dir or DOCS_DIR
    os.makedirs(docs_dir, exist_ok=True)
    today = date.today().isoformat()
    existing = discover_features(docs_dir)
    known = list(existing.keys())

    # First fold near-duplicate area names coined this call (with their
    # statement summaries as context, and the existing docs as targets),
    # then run the mechanical routing as a safety net.
    if canon_fn:
        batch = {}
        for s in statements:
            batch.setdefault(s["feature"], []).append(s["summary"])
        mapping = canon_fn(batch, describe_features(docs_dir))
        for s in statements:
            s["feature"] = mapping.get(s["feature"], s["feature"])
    else:
        canonicalize_batch_features(statements)
    for s in statements:
        s["feature"] = match_existing_feature(s["feature"], known)

    by_feature = {}
    for s in statements:
        by_feature.setdefault(s["feature"], []).append(s)

    summary = []
    for feature, group in by_feature.items():
        path = existing.get(feature)
        if path:
            with open(path, encoding="utf-8") as f:
                sections = _split_sections(f.read())
            facts = _parse_established_facts(sections["established_facts"])
        else:
            path = os.path.join(docs_dir, f"{_slug(feature)}.md")
            sections = {"established_facts": "", "open_questions": "",
                        "change_log": ""}
            facts = []

        # all_facts keeps every fact ever recorded for this feature, including
        # ones a later CHANGE superseded (kept for the audit trail, rendered
        # struck-through). _active(all_facts) is the current picture.
        all_facts = list(facts)
        next_id = max([f["id"] for f in all_facts], default=0) + 1
        new_questions, extra_log = [], []
        n_new = n_dup = n_change = n_review = n_unverified = 0

        for i, s in enumerate(group, start=1):
            attribution = f"{s['speaker']}, {s['timestamp']} (source: {source_name}, {today})"
            q_id = f"Q-{today}-{_slug(feature)}-{i}"

            if not s["verified"]:
                new_questions.append(
                    f'- **{q_id}** [UNVERIFIED CITATION]: extracted for "{feature}" '
                    f'but the quote was not found verbatim in {source_name} -- '
                    f'confirm wording before this becomes a fact.\n'
                    f'  - Claimed: *"{s["quote"]}"* — {s["speaker"]}, {s["timestamp"]}\n'
                    f'  - Summary: {s["summary"]}'
                )
                n_unverified += 1
                continue

            active = _active(all_facts)

            if not active:
                all_facts.append({"id": next_id, "superseded_by": None,
                                  "summary": s["summary"], "quote": s["quote"],
                                  "attribution": attribution})
                next_id += 1
                n_new += 1
                continue

            verdict, target_id, reason = (
                reconcile(feature, active, s) if reconcile
                else ("UNCLEAR", None, "")
            )
            reason_txt = f" Reason: {reason}" if reason else ""
            target = next((f for f in active if f["id"] == target_id), None)

            if verdict == "NEW":
                all_facts.append({"id": next_id, "superseded_by": None,
                                  "summary": s["summary"], "quote": s["quote"],
                                  "attribution": attribution})
                next_id += 1
                n_new += 1

            elif verdict == "DUPLICATE" and target is not None:
                extra_log.append(
                    f"- {today}: {source_name} restated EF-{target_id} for "
                    f'"{feature}" — no change.{reason_txt}'
                )
                n_dup += 1

            elif verdict == "CHANGE" and target is not None:
                # Keep the old fact, mark it superseded, add the new one --
                # the Established Facts section stays complete and glanceable.
                target["superseded_by"] = next_id
                all_facts.append({"id": next_id, "superseded_by": None,
                                  "summary": s["summary"], "quote": s["quote"],
                                  "attribution": attribution})
                extra_log.append(
                    f'- {today}: EF-{target_id} for "{feature}" superseded by '
                    f'EF-{next_id} (from {source_name}).{reason_txt}\n'
                    f'  - was: *"{target["quote"]}"* — {target["attribution"]}\n'
                    f'  - now: *"{s["quote"]}"* — {attribution}'
                )
                next_id += 1
                n_change += 1

            else:  # UNCLEAR, no reconcile callback, or a stale target id
                existing_summary = "; ".join(
                    f'EF-{f["id"]}: {f["summary"]}' for f in active
                ) or "none"
                new_questions.append(
                    f'- **{q_id}** [NEEDS REVIEW]: new statement for "{feature}" -- '
                    f'could not tell if this is new information, a restatement, or '
                    f'a change to an existing fact.{reason_txt}\n'
                    f'  - New: *"{s["quote"]}"* — {attribution}\n'
                    f'  - Existing facts on this feature: {existing_summary}'
                )
                n_review += 1

        run_line = (
            f"- {today}: processed {source_name} — {n_new} new, {n_change} "
            f"changed, {n_dup} restated, {n_review} to review, "
            f"{n_unverified} unverified"
        )
        changelog_entry = "\n".join([run_line, *extra_log])

        active_now = _active(all_facts)
        if story_fn:
            user_story = story_fn(feature, active_now)
        else:
            user_story = ("; ".join(f["summary"].rstrip(".") for f in active_now)
                          + "." if active_now else "No confirmed requirements yet.")

        doc_text = _render(feature, all_facts, new_questions,
                           sections["open_questions"], sections["change_log"],
                           changelog_entry, user_story)
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc_text)

        summary.append(
            f"- {feature}: {n_new} new, {n_change} changed, {n_dup} restated, "
            f"{n_review} to review, {n_unverified} unverified  ->  {path}"
        )
    return summary
