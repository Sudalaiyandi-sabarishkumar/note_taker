#!/usr/bin/env python3
"""Score a knowledge/ directory against the transcripts it was built from.

    python3 scripts/score.py <knowledge_dir> <transcript.txt> [<transcript.txt> ...]

Checks (exit code is non-zero if any HARD check fails):
  HARD  every quote asserted as fact (Established Facts, OPEN QUESTION, and a
        NEEDS REVIEW block's "New:" line) is verbatim (normalised) in some
        transcript. Quotes inside an [UNVERIFIED CITATION] block are exempt --
        that block exists precisely to flag a quote that did NOT verify, so
        the pipeline is working, not failing.
  HARD  no fabricated HH:MM:SS timestamp on a fact sourced from a .txt
  soft  counts: docs, facts, superseded, [NEEDS REVIEW], [UNVERIFIED CITATION],
        [OPEN QUESTION], and any EF summary that looks like a bare fragment
"""
import glob
import os
import re
import sys

_QUOTE_RE = re.compile(r'\*"([^"]+)"\*')
_EF_RE = re.compile(r'^- \*\*EF-\d+\*\*(?: \[superseded by EF-\d+\])?:', re.M)
_TS_RE = re.compile(r'\b\d{2}:\d{2}:\d{2}\b')


def norm(s):
    s = s.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip(" \t\n\"'.,")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    kdir, tpaths = argv[0], argv[1:]
    tx = " || ".join(norm(open(p, encoding="utf-8").read()) for p in tpaths)
    txt_sources = {os.path.splitext(os.path.basename(p))[0] for p in tpaths
                   if p.lower().endswith(".txt")}

    docs = sorted(glob.glob(os.path.join(kdir, "*.md")))
    if not docs:
        print(f"no .md files in {kdir}")
        return 2

    n_fact = n_sup = n_review = n_unver = n_open = 0
    n_exempt_q = 0  # quotes inside [UNVERIFIED CITATION] blocks -- not hard-checked
    ungrounded, fake_ts, fragments = [], [], []
    for d in docs:
        body = open(d, encoding="utf-8").read()
        name = os.path.basename(d)
        n_fact += len(_EF_RE.findall(body))
        n_sup += body.count("[superseded by EF-")
        n_review += body.count("[NEEDS REVIEW]")
        n_unver += body.count("[UNVERIFIED CITATION]")
        n_open += body.count("[OPEN QUESTION]")
        # Hard-check quotes, but skip any inside an [UNVERIFIED CITATION]
        # block -- that block is the pipeline flagging a quote it could NOT
        # ground, which is correct behaviour, not a failure.
        in_unverified = False
        for line in body.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("## "):
                in_unverified = False
            elif stripped.startswith("- ") and line == stripped:  # top-level bullet
                in_unverified = "[UNVERIFIED CITATION]" in line
            for q in _QUOTE_RE.findall(line):
                if in_unverified:
                    n_exempt_q += 1
                elif norm(q) not in tx:
                    ungrounded.append((name, q))
        for m in re.finditer(r"source: ([a-zA-Z0-9_\-]+),[^)]*\)", body):
            src = m.group(1)
            line = body[body.rfind("\n", 0, m.start()) + 1: m.end()]
            if src in txt_sources and _TS_RE.search(line.split("source:")[0]):
                fake_ts.append((name, line.strip()[:90]))
        for line in body.splitlines():
            m = re.match(r"- \*\*EF-\d+\*\*[^:]*:\s*(.+?)\s*—", line)
            if m:
                summ = m.group(1)
                w = re.findall(r"[A-Za-z0-9']+", summ)
                if len(w) < 4:
                    fragments.append((name, summ))

    all_q = len(sum([_QUOTE_RE.findall(open(d, encoding="utf-8").read())
                     for d in docs], []))
    total_q = all_q - n_exempt_q          # quotes subject to the hard check
    grounded = total_q - len(ungrounded)

    print(f"docs                : {len(docs)}")
    print(f"established facts    : {n_fact}  ({n_sup} superseded)")
    print(f"quotes grounded     : {grounded}/{total_q}")
    print(f"flags               : {n_review} needs-review, {n_unver} unverified, "
          f"{n_open} open-question")
    if fragments:
        print(f"fragment summaries  : {len(fragments)}")
        for n, s in fragments[:8]:
            print(f"    [{n}] {s!r}")
    if ungrounded:
        print("UNGROUNDED QUOTES:")
        for n, q in ungrounded:
            print(f"    [{n}] {q[:80]}")
    if fake_ts:
        print("FABRICATED TIMESTAMPS:")
        for n, l in fake_ts:
            print(f"    [{n}] {l}")

    hard_fail = bool(ungrounded or fake_ts)
    print("\nRESULT:", "FAIL" if hard_fail else "PASS")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
