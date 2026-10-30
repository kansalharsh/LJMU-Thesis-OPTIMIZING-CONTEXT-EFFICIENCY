"""Times the Gateway's pruning step on the tool responses of the matched sample.

Measures constraint_aware_prune() only (embedding + scoring + selection), warm model, one call
per tool type per query (outbound flight, accommodation, restaurant, attraction). No API calls.
Run from the repository root:  python scripts/benchmark_pruning.py
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, data_loader, semantic_gateway as sg  # noqa: E402
from src.tools import registry as R  # noqa: E402


def valid(files):
    ids = set()
    for f in files:
        for line in open(config.RESULTS_DIR / "final" / f):
            r = json.loads(line)
            if not r["error"]:
                ids.add(r["idx"])
    return ids


base = ["baseline_pilot_n9_v2.jsonl", "baseline_beyond_n18.jsonl", "baseline_beyond2_n24.jsonl", "baseline_easy43_new.jsonl"]
gate = ["gateway_pilot_n9_v3.jsonl", "gateway_beyond_n18.jsonl", "gateway_beyond2_n24.jsonl", "gateway_easy43_new.jsonl"]
matched = sorted(valid(base) & valid(gate))
queries = data_loader.load_queries("validation")

t0 = time.perf_counter()
model = sg._embedder()
print(f"model load: {time.perf_counter() - t0:.2f} s on device {model.device}")
sg.rank_by_relevance(R.accommodation_search("Atlanta"), "warm-up")

per_call, per_query = [], []
for i in matched:
    q = queries[i]
    intent = f"{q.query} Destination: {q.dest}. Budget: ${q.budget}. Constraints: {q.local_constraint}"
    responses = [R.flight_search(q.org, q.dest, q.date[0]), R.accommodation_search(q.dest),
                 R.restaurant_search(q.dest), R.attraction_search(q.dest)]
    total = 0.0
    for rows in responses:
        if rows:
            t = time.perf_counter()
            sg.constraint_aware_prune(rows, intent, price_floor_keep=config.PRICE_FLOOR_KEEP)
            dt = time.perf_counter() - t
            per_call.append(dt)
            total += dt
    per_query.append(total)

per_call.sort()
print(f"{len(per_call)} calls: median {st.median(per_call)*1e3:.1f} ms, mean {st.mean(per_call)*1e3:.1f} ms, "
      f"p95 {per_call[int(0.95 * len(per_call))]*1e3:.1f} ms, max {per_call[-1]*1e3:.1f} ms")
print(f"per query ({len(per_query)} queries): mean {st.mean(per_query):.3f} s, max {max(per_query):.3f} s")
