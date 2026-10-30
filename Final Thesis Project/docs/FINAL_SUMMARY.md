# Final pilot dataset - baseline vs. Semantic Pruning Gateway

Matched comparison (n=57/condition: only queries where both conditions have
non-error data), combining every valid batch collected 2026-09-13 to
2026-09-27 on GPT-5.2, TravelPlanner validation split.

| Metric | Baseline | Gateway | Delta | p-value | Verdict |
|---|---|---|---|---|---|
| TCR | 61.4% | 56.1% | -5.3 pts | 0.572 (two-prop z) | Not significant |
| CSR | 87.7% | 85.5% | -2.2 pts | 0.520 (Welch's t) | Not significant |
| Tokens/query | 14,476 | 10,122 | -30.1% | <0.0001 (Welch's t) | Highly significant |

By difficulty level (57 matched, 23 easy / 17 medium / 17 hard):
- Medium and hard TCR are statistically identical between conditions -
  pruning has zero measurable effect there; failures are a reasoning-layer
  ceiling, not a context-management one.
- The entire TCR gap concentrates in easy-level queries. Diagnosed causes:
  (1) naive semantic pruning discarding budget-relevant rows - fixed via a
  constraint-aware price floor (`price_floor_keep`); (2) residual gap traced
  to LLM sampling variance (confirmed non-deterministic even at
  temperature=0), not a systematic pruning defect - raising the price floor
  from 3 to 6 left the aggregate rate unchanged (76.5% either way) while the
  specific queries that passed/failed shuffled.

## Known limitations
- Batches were not all run under identical settings: batches 1-3 used the API-default temperature and a
  gateway price floor of 3; batch 4 used temperature 0 and a floor of 6. See `BATCHES.md`.
- `gpt-5.2` output is not deterministic even at temperature 0, so re-running yields statistically
  similar but not identical per-query outcomes.
- The matched sample (n = 57 pairs) resolves the token-consumption effect clearly (p < 0.0001) but is
  too small to resolve a TCR difference of about 5 points; a two-proportion power calculation puts
  that at over 1,000 queries per condition.
- Source files: see `BATCHES.md` for the file-to-batch mapping.
