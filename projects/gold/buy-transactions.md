# Buy Transactions

## User Story
As a user, I want to record buy transactions that update the average stock based on old grams, new grams, and rate, and show the updated stock history, so that the current stock status is accurately reflected.

## Established Facts
- **EF-1**: On a buy the new average is old grams times old average, plus the grams bought times the rate, over the total grams after. — *"On a buy the new average is old grams times old average, plus the grams bought times the rate, over the total grams after."* — Mani Selvam, [00:00:32.100] (source: compressed_gold_call1, 2026-09-22)
- **EF-2**: Sale, the average doesn't move. Grams come off. — *"Sale, the average doesn't move. Grams come off."* — Mani Selvam, [00:00:43.100] (source: compressed_gold_call1, 2026-09-22)
- **EF-3**: Includes date, party name, weight in grams, rate per gram, and optional note. — *"Buy or sell, the date, the party's name, weight in grams, rate per gram, and, uh, an optional note."* — Mani Selvam, [00:01:11.900] (source: compressed_gold_call1, 2026-09-22)
- **EF-4**: History's a list, newest first, each line shows the stock after. — *"History's a list, newest first, each line shows the stock after it."* — Mani Selvam, [00:01:32.100] (source: compressed_gold_call1, 2026-09-22)
- **EF-5**: Each party has their own page with all bills and net number. — *"Every party gets their own page, all their bills, and one net number, receivable if they owe me, payable if I owe them."* — Mani Selvam, [00:02:14.100] (source: compressed_gold_call1, 2026-09-22)

## Open Questions / Ambiguities
- **Q-2026-09-22-compressed_gold_call1-buy-transactions-6** [OPEN QUESTION]: One thing I haven't decided, whether the opening balances lock once there's transactions in. Leave that open. — raised by Mani Selvam, compressed_gold_call1 (2026-09-22)
  - *"One thing I haven't decided, whether the opening balances lock once there's transactions in. Leave that open."*

- **G-2026-09-22-compressed_gold_call1-buy-transactions-2** [GAP]: What happens if the grams bought exceed the available stock? — not yet decided; follows from EF-1. (raised by gap analysis, compressed_gold_call1 2026-09-22)

- **G-2026-09-22-compressed_gold_call1-buy-transactions-3** [GAP]: Who is responsible for updating the stock after a transaction? — not yet decided; follows from EF-4. (raised by gap analysis, compressed_gold_call1 2026-09-22)

## Change Log
- 2026-09-22: processed compressed_gold_call1 — 5 new, 0 changed, 0 restated, 0 to review, 0 unverified, 3 open question(s)
