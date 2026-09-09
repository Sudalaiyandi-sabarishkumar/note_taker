# Gold set — what it covers

The **Gold Manager** universe: a single-operator gold **buy/sell ledger** with
weighted-average stock costing, deferred (part / later) payments, per-party
Receivable / Payable balances, and a cash-or-gold **lending** side. Reverse-
engineered from the `~/Documents/gold` codebase (Flutter · Node/Express ·
MongoDB), so this set doubles as a ground-truth ("gold") check.

Unlike the other `examples/` sets, these are written as **realistic Microsoft
Teams `.vtt` exports** — per-utterance cues, repeated speaker names, filler and
false starts, interruptions ("Sorry, go ahead"), a screen-share fumble, a dog,
audio break-up, a late joiner, and requirements that only emerge through
back-and-forth. **3 substantive calls + 1 empty:**

- **`gold_call1.vtt`** — a ~136-line, **7-chunk** scoping call, 4 speakers
  (Mani the operator; Latha BA; Ravi engineering; Priya design cameo). ~25
  requirements across ~15 feature areas: single fixed login, one-pool grams +
  weighted-average cost, buy/sell maths, ledger-replay stock, over-sell block,
  transaction fields, server-computed total, rounding, history + search,
  deferred payments, derived paid/part/unpaid status, per-party page + one
  net total, one-payment-many-bills settle, case-insensitive party names,
  editable opening balances, cash/gold flowing through everything, yellow-on-
  black, the Flutter/Node/Mongo stack. This is the **7B-stress** call.
- **`gold_call2.vtt`** — changes + confirmations, and the new **Loans** area
  (cash = per-day interest, gold = per-month; `principal ÷ ref × rate ×
  periods`; month = 30 days; count-start-day toggle; single repayment;
  can't lend past cash/stock).
- **`gold_call3.vtt`** — refinements; the call-1 open item is closed; a buried
  change sandwiched between two restates; three new features.
- **`gold_empty.vtt`** — reschedule + slipped mockups + expo banter → **zero docs**.

Run: `MOM_DOCS_DIR=gold_knowledge ./examples/run_example.sh gold`
→ `gold_knowledge/`. `score.py` must print **RESULT: PASS**, then check the table.

| # | Scenario | Where planted | Pass looks like |
|---|---|---|---|
| 1 | **Multi-chunk** + overlap de-dup | call1 = 7 chunks | no duplicated EF at a chunk boundary |
| 2 | **Feature separation**, no dumping ground | call1's ~15 topics | cohesive per-feature docs (stock / payments / party balances / history / login / opening balances / theme / stack …), NOT one mega-doc |
| 3 | **Confusable numbers** not conflated | password `1977`; cash `2` dp vs gold `4` dp; `15 lakh` cash vs `100` g gold; month `30` → `31` days | each value lands on the right fact; the password `1977` is never read as a quantity |
| 4 | **Doc-name guard / de-camel** | call3 "**There's a** settle action on the party page…" | doc `Settle` / `Party Settlement`, never `There's Settle` / `# []` |
| 5 | **Bundled sentence → unbundle** | call2 cue 20 "show every one of that person's bills, **and** the one net total, **and** the name matching ignores case" | → separate facts (all-bills view / single net total / case-insensitive names) |
| 6 | **Timestamp on unbundled pieces** | call2 cue 20 (split) | each piece carries `[00:00:23.900]` |
| 7 | **Same-call complementary** | call2 cues 28–32 "a cash loan… interest is per day" then "a gold loan… per month" | both kept, no `[NEEDS REVIEW]`, no supersede |
| 8 | **Coverage-pass recall** of a buried detail | call1 cue 92 "if I go to add a payment onto a bill, it can't let me put in more than what's still owing" | captured as a fact despite being a throwaway aside |
| 9 | **Undecided → open question** | call1 cue 127 "haven't decided… whether the opening balances should lock"; call2 cue 47 "I don't know yet… partial repayment" | `[OPEN QUESTION]` "Decision needed: …", **no fact-doc** |
| 10 | **Aspiration → open question** | call1 cue 129 "feel fast and calm"; call3 cue 43 "feel trustworthy… the numbers always have to reconcile" | `[OPEN QUESTION]` "turn into a concrete, testable requirement", not an EF |
| 11 | **Logistics / banter dropped** | call1 dog + screen-share fumble + "association's new billing rules" tangent; call3 "festival rush"; the whole empty call | none become facts or questions |
| 12 | **.vtt parsing** — voice tags, timestamps, **accented name**, no-name `<v>` | call2 | speakers incl. **`José Álvarez`**; cue 45 `<v>` with no name → `Unidentified speaker`; `[HH:MM:SS]` prefix on facts |
| 13 | **Cross-call reversal + supersede (value)** | call1 "no default — prompt for the amount paid every time" → call2 cues 8–11 "default the amount paid to the full total" | amount-paid EF `[superseded by EF-N]`, was/now logged; end state = **defaults to full total** |
| 14 | **Drop-then-restore** | call2 cue 13 "remove the note field" → call3 cue 10 "put it back… I do use it" | the call3 fact supersedes the call2 removal; the note field ends up **active** (optional) |
| 15 | **Cross-call value change, non-reversal** | call2 cue 34 "a month is thirty days" → call3 cue 13 "a month should be thirty one days" | call2 fact `[superseded by EF-N]`, `was: …thirty days` / `now: …thirty one days` |
| 16 | **W4 partial supersede** (multi-value fact) | call1 cues 63–65 "round cash to two decimal places, and… gold weight to four decimal places" (stays one fact) → call3 cues 7–9 "gold rounding, four down to three" | `[NEEDS REVIEW]` "EF-N updates PART of EF-M"; EF-M **not** fully superseded; the cash 2-dp part still visible |
| 17 | **`_NOCHANGE_RE` restate** | call2 cues 17–19 "weighted average… unchanged" / "can't sell past stock… unchanged"; call3 cues 18–20 login `mani`/`1977` / yellow-on-black / server-computed total | each → "restated … no change", not a new EF, not a flag |
| 18 | **Buried value change between restates** | call3 cues 21–27: status-derivation (restate) → **"quietly cap it at the outstanding instead of blocking"** (change) → "payments never touching stock or profit" (restate) | the middle line supersedes call1's "a new payment can't exceed the outstanding" fact; the two restates log "no change" |
| 19 | **W3 open-question resolution** | call1 cue 127 "haven't decided if opening balances lock" → call3 cues 4–6 "they stay editable… always. That closes that one" | the call1 question flips to `[RESOLVED]` with a `resolved by … EF-N` pointer; the call3 line is filed as a **fact**, not re-flagged as undecided |
| 20 | **New undecided item late** | call3 cues 40–41 "a second login, for an assistant… don't build anything for it… Not now" | `[OPEN QUESTION]`, no EF |
| 21 | **Dialogue-driven extraction** | call1 — the buy-maths rule emerges only from Ravi's "walk me through the maths on a buy"; rationale attached ("that's how mistakes happen", "it drifts") | the requirement is captured cleanly; the rationale line is **not** a separate fact |
| 22 | **Disfluency / false-start tolerance** | call1 "so it's, okay it's a small thing"; "the, the dog"; "we, we should"; cut-offs | facts extract cleanly through the noise; filler never becomes a statement |
| 23 | **ASR-style artefacts** | call2 cue 30 "per, uh, per day per one lakh"; cue 31 "sorry, you broke up" | the interest-formula fact still lands; "you broke up" is dropped |
| 24 | **W2 gap analysis** | multi-fact docs (payments, loans, stock) | grounded `[GAP]` questions, ≤2/doc, tied to EF numbers; no vague "what are the failure conditions?" |
| 25 | **Empty call → zero docs** | `gold_empty.vtt` | "No concrete, citable statements…"; no new `.md` |
| 26 | **Idempotency** | re-run `gold_call1.vtt` a 2nd time | "re-stated an already-recorded fact — no change"; fact count unchanged |

## Not exercised here
Cross-doc contradiction sweep (no two docs are given conflicting *unlinked*
numbers), `.txt` ingestion (this set is all `.vtt`), and a genuine 4th call.
Use the `combined` set for those.

## Quick asserts

```
grep -rn "UNVERIFIED CITATION" gold_knowledge/     # expect: none
grep -rn "\[RESOLVED\]"        gold_knowledge/      # expect: the opening-balances lock question
grep -rn "updates PART of EF"  gold_knowledge/      # expect: the gold-rounding 4→3 change
grep -c  "superseded by EF-"   gold_knowledge/*.md | grep -v ':0'   # amount-paid, note field, month 30→31, payment-cap rule
ls gold_knowledge/*.md | wc -l
```
