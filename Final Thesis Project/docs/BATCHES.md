# Data-collection batches and exact run configuration

The thesis results (Chapter 5) combine four collection batches. They were **not** all run with
identical settings; this table is the authoritative record. Query selection is deterministic
(`stratified_sample` takes queries in file order per difficulty level, skipping `--offset`), and
`results/final/` stores one JSON record per query.

| Batch | Files (`results/final/`) | Queries drawn (validation split) | Gateway price floor | Temperature | Date | Matched valid pairs |
|---|---|---|---|---|---|---|
| 1 | `*_pilot_n9_v2` (baseline) / `*_pilot_n9_v3` (gateway) | first 3 per level | 3 | API default | 2026-09-14 | 9 |
| 2 | `*_beyond_n18` | levels' queries 4-9 | 3 | API default | 2026-09-14 | 18 |
| 3 | `*_beyond2_n24` | levels' queries 10-17 | 3 | API default | 2026-09-26 | 24 |
| 4 | `*_easy43_new` | easy queries 18-60 only | 6 | 0 | 2026-09-27 | 6 (of 43; see below) |
| | | | | | **Total** | **57** |

Other settings, constant across all batches: reasoning model `gpt-5.2` (OpenAI), max 12 Reason-Act
steps, tool results truncated to 50 rows per call, `fetch_more` returns 10 cached rows, embedding
model `all-MiniLM-L6-v2` (local), adaptive threshold = mean + 0.5 sd of the response's own cosine
scores clipped to 5-15 kept rows.

Batch 4 contributes only 6 matched pairs because only 7 of its 43 Gateway queries returned a valid record
(the rest carry an API error in the `error` field). Errored records are kept in the files (field
`error`) and excluded by `analyze.py`, which pairs the two conditions on query index.

## Commands that generated each batch

```bash
# Batch 1 (price floor 3, temperature default)
THESIS_PRICE_FLOOR=3 THESIS_TEMPERATURE=default python -m src.run_experiment --agent baseline --n-per-level 3 --out final/baseline_pilot_n9_v2.jsonl
THESIS_PRICE_FLOOR=3 THESIS_TEMPERATURE=default python -m src.run_experiment --agent gateway  --n-per-level 3 --out final/gateway_pilot_n9_v3.jsonl
# Batch 2
THESIS_PRICE_FLOOR=3 THESIS_TEMPERATURE=default python -m src.run_experiment --agent baseline --n-per-level 6 --offset 3 --out final/baseline_beyond_n18.jsonl
THESIS_PRICE_FLOOR=3 THESIS_TEMPERATURE=default python -m src.run_experiment --agent gateway  --n-per-level 6 --offset 3 --out final/gateway_beyond_n18.jsonl
# Batch 3
THESIS_PRICE_FLOOR=3 THESIS_TEMPERATURE=default python -m src.run_experiment --agent baseline --n-per-level 8 --offset 9 --out final/baseline_beyond2_n24.jsonl
THESIS_PRICE_FLOOR=3 THESIS_TEMPERATURE=default python -m src.run_experiment --agent gateway  --n-per-level 8 --offset 9 --out final/gateway_beyond2_n24.jsonl
# Batch 4 (repo defaults: price floor 6, temperature 0)
python -m src.run_experiment --agent baseline --levels easy --offset 17 --n-per-level 43 --out final/baseline_easy43_new.jsonl
python -m src.run_experiment --agent gateway  --levels easy --offset 17 --n-per-level 43 --out final/gateway_easy43_new.jsonl
```

(`--out` is resolved relative to `results/`.) Note that the original batches were launched before
the `THESIS_*` environment variables existed; the commands above are the equivalent invocations of
the current code, and the selected query indices were verified identical to the stored files.

Re-running will **not** reproduce the stored per-query outcomes bit-for-bit: `gpt-5.2` output is
non-deterministic even at temperature 0 (thesis Section 5.3.2). The stored records are the
evidence; `scripts/reproduce_results.sh` regenerates the reported statistics from them.

## Other result folders

* `results/diagnostics/gateway_easy17_floor6.jsonl` - the 17 easy queries re-run with price floor 6
  and temperature 0 (thesis Section 5.3.1, compared with the floor-3 outcomes for the same queries).
* `results/superseded/` - early pilot runs made before two later fixes (city-name normalisation in
  `tools/registry.py`; the constraint-aware price floor). Kept for transparency; not used in any
  reported statistic.
