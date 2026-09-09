# Combined regression set — what it covers

The **HomeFix** universe (a home-repair booking platform + its provider
training portal + partner billing). **3 substantive calls + 1 empty call:**

- **`combined_call1.txt`** — a ~75-line, **4-chunk** scoping workshop with 6
  speakers. Requirements emerge through genuine back-and-forth — an
  engineer's follow-up draws out the rule, rationale gets attached ("the
  24-hour lockout came from our safety team"), a clarification turns into an
  undecided ("same everywhere or by region?" → "revisit later"), there's a
  mid-call tangent. ~35 requirements across ~30 feature areas, LMS-flavoured
  (training portal) and finance-flavoured (billing) vocabulary, confusable
  numbers throughout. This is the **7B-stress** call (under-extraction,
  misroute, over-club, rationale-as-fact).
- **`combined_call2.vtt`** — Teams export: a batch of changes + confirmations.
- **`combined_call3.txt`** — refinements, and open items get closed.
- **`combined_empty.txt`** — logistics only → must produce zero docs.

Run: `./examples/run_combined.sh`  → `combined_knowledge/` (~60-75 min).
`score.py` must print **RESULT: PASS**. Then check the table.

| # | Scenario | Where planted | Pass looks like |
|---|---|---|---|
| 1 | **Multi-chunk** + overlap de-dup | call1 = 4 chunks | no duplicated EF at a chunk boundary |
| 2 | **Feature separation**, no dumping ground | call1's ~30 topics | ~28-34 cohesive docs, NOT one mega-doc |
| 3 | **7B-stress / gap 3** — dense terse turns, confusable numbers (3.5★ / 20 jobs / 5 photos; 3 attempts / 3 business days / 3 months; 12 mo / 24 h / 48-8-1 h; net 30 / $10,000; 600 rpm / 429) | call1 ratings turn, training turns, billing turns | facts land in the right docs; numbers not conflated; recall is reasonable (coverage pass helps) |
| 4 | **Doc-name guard / de-camel** | call1 "There is a re-book button…" | doc `Re-book` / `Rebooking`, never `There Rebook` / `# []` |
| 5 | **Bundled sentence → unbundle** | call1 "pay by card or cash… but an itemised invoice is emailed within 24 hours"; call2 cue 5 "upload a valid ID and insurance certificate… , and both are re-checked every 12 months" | each → two separate facts |
| 6 | **Timestamp on unbundled pieces** | call2 cue 5 (split) | both pieces carry `[00:00:20.000]` |
| 7 | **Same-call complementary** | call1 email notif, then "also send an SMS" | both kept, no `[NEEDS REVIEW]`, no supersede |
| 8 | **Coverage-pass recall** of a buried detail | call1 "45 dollar call-out fee… applies even if the customer cancels after dispatch" | both the fee and the "after dispatch" clause captured |
| 9 | **Undecided → open question** | call1 "haven't decided the refund window"; "advanced electrical module … mandatory or opt-in"; "whether we offer volume discounts is still open" | `[OPEN QUESTION]` "Decision needed: …", **no fact-doc** |
| 10 | **Aspiration → open question** | call1 "the whole booking flow should just feel effortless" / "ops dashboard should feel calm, not busy" | `[OPEN QUESTION]` "turn into a concrete, testable requirement", not an EF |
| 11 | **Logistics / banter dropped** | call1 "push next week's review to Thursday" / "I'll send an invite"; call3 "how was the conference"; empty call | none become facts or questions |
| 12 | **.vtt parsing** — voice tags, timestamps, **accented name**, no-name `<v>` | call2 | speakers `Priya Nair` / `Jose Ruiz` / **`José Álvarez`**; cue 7 `<v>` → `Unidentified speaker`; `[HH:MM:SS]` on facts |
| 13 | **Cross-call CHANGE + supersede (value)** | call1 "card or cash" → call2 "drop the cash payment option. Card only" | payment EF `[superseded by EF-N]`, was/now logged |
| 14 | **Batch 4 wrong-target retarget** | call2 "change the booking-confirmation **reminder** from 2 hours to 1 hour" vs call1's "email on confirmation" **and** "reminder email … within 2 hours" | supersedes the **reminder** fact, not "email on confirmation"; log may say "re-aimed … by wording" |
| 15 | **W4-2 partial supersede** (multi-value fact) | call1 "Response-time targets: Basic 48h, Pro 8h, **and** Enterprise 1h" (stays one fact) → call3 "change the Pro target from 8 hours to 4 hours" | `[NEEDS REVIEW]` "EF-N updates PART of EF-M"; EF-M **not** superseded; Basic/Enterprise still visible |
| 16 | **`_NOCHANGE_RE` restate** | call2 "three-strike cancellation rule stays as is" / "24-hour invoice rule is unchanged"; call3 "45 dollar call-out fee is unchanged" / "net 30 payment terms are unchanged" / "3 business day dispute review window is unchanged" | each → "restated … no change", not a new EF, not a flag |
| 17 | **Reversal / drop-then-restore** (#34) | call2 "remove the phone support channel" → call3 "bring back phone support" | call3 fact supersedes the call2 "remove"; phone support ends up **active** |
| 18 | **Cross-call change, non-reversal** | call2 "calls go through a masked number" → call3 "drop the masked number, show the real phone number" | call2 fact `[superseded by EF-N]` |
| 19 | **W3-1 open-question resolution** | call1 "still checking whether weekend slots are allowed" → call3 "weekend slots are allowed … 9am-5pm. That resolves last call's open question" | the call1 question flips to `[RESOLVED]` with a `resolved by … EF-N` pointer; the call3 line itself is filed as a **fact**, NOT re-flagged as undecided |
| 20 | **W3 reworded-answer resolution** (LLM near-miss) | call1 "Spanish support is not decided for launch" → call3 "Spanish is confirmed for the initial launch" | the call1 question flips to `[RESOLVED]` although "confirmed"/"decided" don't lexically match ≥3 words |
| 21 | **W3-2 cross-doc contradiction** (#31) | call1 "a provider can attach up to 5 photos" vs call2 "the completed-job report can include up to 10 photos" | an `X-…` `[NEEDS REVIEW]` cross-reference in **both** docs, "5 vs 10" |
| 22 | **New undecided item late** | call3 "loyalty discount … not decided … don't build it yet" | `[OPEN QUESTION]`, no EF |
| 23 | **W2 gap analysis** — grounded `[GAP]` questions, ≤2/doc, tied to EF numbers | multi-fact docs | e.g. `[GAP] Who is assigned when no provider accepts within 60 minutes? [from EF-N]` |
| 24 | **W2 vague-gap filter** | any doc | no `[GAP]` reading "what are the failure conditions?" / "how is this handled?" / "any other requirements?" |
| 25 | **Word-list generalisation** — regexes on LMS + finance phrasing | call1 training + billing sections | undecided/logistics/aspiration classify correctly on non-booking vocabulary |
| 26 | **LMS multi-fact feature stays whole** | call1 "quiz … 80 percent to pass"; "three failed attempts … locked out for 24 hours" | pass mark + lockout as separate EFs, not one bundle, not over-split |
| 27 | **Certificate lifecycle, no false CHANGE** | call1 "certificate valid for 12 months" + "must re-certify before it expires, otherwise inactive" | two complementary EFs, no `[NEEDS REVIEW]`, no supersede |
| 28 | **User Story quality** | every multi-fact doc | prose or a complete `; `-joined list; never "(Plus N more requirements below.)"; covers all active facts on small docs |
| 29 | **Empty call → zero docs** | `combined_empty.txt` | "No concrete, citable statements…"; no new `.md` |
| 30 | **Idempotency** | re-run `combined_call1.txt` a 2nd time | "re-stated an already-recorded fact — no change"; fact count unchanged |
| 31 | **Dialogue-driven extraction** — requirements stated as answers to follow-ups, with rationale attached, and clarification loops that become undecideds | call1 throughout (e.g. Kofi: "and if nobody nearby takes it?" → the 60-min rule; "same everywhere or by region?" → "revisit per-region later" undecided; "the 24-hour lockout came from our safety team") | the requirement is captured cleanly; the rationale line ("came from the safety team") is NOT a separate fact; the region clarification → `[OPEN QUESTION]` |
| 32 | **Junk-fragment guard** | call2 cue 5 "There's a wrinkle here." (a lead-in fragment, like expense's "With one shortcut.") | never becomes an EF or a question |
| 33 | **Buried cross-call value change** (recall test) | call3 "the review-hold threshold stays at 3.5 stars, the weekly Friday payout is unchanged, and shorten the manual-dispatch window from 60 minutes to 45 minutes" — one real change sandwiched between two restates | the 60→45 change **is** applied (supersedes the call1 "60 minutes → manual dispatch" fact); the two restates log as "no change" |

## What this still can't reproduce
Genuine real-transcript disfluency (crosstalk, half-sentences, mid-call
reversals) and *unanticipated* failure modes — I author these knowing which
scenarios I'm testing. Run a real 5-set occasionally as the release gate.

## Quick asserts

```
grep -rn "UNVERIFIED CITATION" combined_knowledge/       # expect: none
grep -rn "\[RESOLVED\]"        combined_knowledge/        # expect: weekend slots + Spanish
grep -rn "may contradict"      combined_knowledge/        # expect: the 5-vs-10 photos pair
grep -rn "updates PART of EF"  combined_knowledge/        # expect: the Pro SLA change
grep -c  "superseded by EF-"   combined_knowledge/*.md | grep -v ':0'
ls combined_knowledge/*.md | wc -l                        # expect: ~28-34
```
