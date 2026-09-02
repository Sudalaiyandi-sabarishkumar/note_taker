# Combined regression set — what it covers

One domain (**HomeFix**, a home-repair booking platform), 3 substantive calls
+ 1 empty call. Replaces a full 5-set sweep for a fast pass (~15-20 min).

Run: `./examples/run_combined.sh`  → `combined_knowledge/`

`score.py` must print **RESULT: PASS** (every fact quote grounded, no
fabricated timestamps). Then eyeball the checks below.

| # | Scenario (which old set it came from) | Where planted | Pass looks like |
|---|---|---|---|
| 1 | **Multi-chunk** transcript + overlap de-dup (lms) | call1 is 2 chunks | no duplicated EF from the chunk boundary |
| 2 | **Clean feature separation**, no dumping ground (lms Batch 2) | call1 ~9 topics | ~8-12 docs (Booking, Provider Assignment, Notifications, Payments, Call-out Fee, Cancellation, Ratings, Provider Onboarding, Accessibility, Re-book…), NOT one mega-doc |
| 3 | **Doc-name guard / de-camel** (Batch 1) | "There is a re-book button…", "booking flow" | doc named `Re-book` / `Rebooking`, never `There Rebook` / `BookingFlow` / `# []` |
| 4 | **Bundled sentence → unbundle** (support) | call1 "pay by card or cash… but an itemised invoice is emailed within 24 hours" | two separate facts: payment methods, and 24-hour invoice |
| 5 | **Same-call complementary** notification, not a self-reversal (Batch 1 #5) | call1 email notif, then "also send an SMS" | both kept, **no** `[NEEDS REVIEW]`, no supersede |
| 6 | **Coverage pass recall** of a buried detail (expense $50) | call1 "45 dollar call-out fee… applies even if the customer cancels after dispatch" | both the fee **and** the "applies after dispatch" clause are captured |
| 7 | **Undecided → open question** (expense hedge scan) | call1 "haven't decided the refund window yet" | `[OPEN QUESTION]` under a Refunds area, no fact invented |
| 8 | **Aspiration → open question** (support) | call1 "the whole booking flow should just feel effortless" | `[OPEN QUESTION]` "turn into a concrete, testable requirement", NOT an EF |
| 9 | **Logistics / banter dropped** (empty, delivery) | call1 "push next week's review to Thursday" / "I'll send an invite"; call3 "how was the conference" | none of these become facts or questions |
| 10 | **.vtt parsing** — voice tags, timestamps, accented name, no-name `<v>` (support) | call2 | speakers `Priya Nair` / `Jose Ruiz`; line 7 `<v>` → `Unidentified speaker`; `[HH:MM:SS]` on facts |
| 11 | **Timestamp on unbundled pieces** (Batch 1 #7) | call2 line 5 "upload a valid ID and insurance certificate… re-checked every 12 months" | the split pieces carry `[00:00:20.000]`, not "not available" |
| 12 | **Cross-call CHANGE + supersede (value)** (delivery 5→8 km) | call1 "cash on completion" → call2 "drop the cash payment option. Card only" | payment-methods EF `[superseded by EF-N]`, was/now in Change Log |
| 13 | **`_NOCHANGE_RE` restate** (delivery, support) | call2 "three-strike cancellation rule stays as is, no change"; call2 "24-hour invoice rule is unchanged"; call3 "45 dollar call-out fee is unchanged" | each logged as "restated … no change", **not** a new EF, not a flag |
| 14 | **Batch 4 wrong-target retarget** (support 4h→2h) | call3 "accept within 30 minutes, not 60" vs call1's assign rule **and** its 60-min fallback | supersedes the **60-minute** fact, not the "auto-assign nearest ≥4★" one; log may say "re-aimed … by wording" |
| 15 | **Reversal / drop-then-restore** (support #34) | call2 "remove the phone support channel" → call3 "bring back phone support" | call3 fact supersedes the call2 "remove" fact; phone support ends up **active** |
| 16 | **Open question raised then answered later** (support) | call1 "still checking whether weekend slots are allowed" → call3 "weekend slots are allowed … 9am-5pm" | call3 records the weekend-slots **fact**. (Known gap: the call1 open question is not auto-closed — note whether it sits stale or is mis-filed.) |
| 17 | **New undecided item late** (expense) | call3 "loyalty discount … not decided … don't build it yet" | `[OPEN QUESTION]`, no EF |
| 18 | **Empty call → zero docs** (empty) | `combined_empty.txt` | "No concrete, citable statements…"; no new `.md`, no junk facts |
| 19 | **User Story quality** (this session's fixes) | every multi-fact doc | reads as prose or a complete `; `-joined list; never "(Plus N more requirements below.)"; covers all active facts on small docs |
| 20 | **Idempotency** (Batch 1 #22) | re-run `combined_call1.txt` a 2nd time | "re-stated an already-recorded fact — no change"; fact count unchanged |

## Quick asserts after a run

```
grep -rl "NEEDS REVIEW"        combined_knowledge/   # expect: few/none, each defensible
grep -rl "UNVERIFIED CITATION" combined_knowledge/   # expect: none
grep -c  "superseded by EF-"   combined_knowledge/*.md | grep -v ':0'   # expect: payments, provider-assignment(60→30), phone/support
ls combined_knowledge/ | wc -l                       # expect: ~8-12
test ! -e combined_knowledge/*there* && echo "no garbage names"
```
