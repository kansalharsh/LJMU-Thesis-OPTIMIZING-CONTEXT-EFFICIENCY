"""Statistical comparison of the Baseline and Gateway conditions.

`python -m src.analyze --final` reproduces Tables 5.1 and 5.2 of the thesis from
the raw per-query records in results/final/.

Matching: a query contributes only if BOTH conditions produced a valid
(non-error) record for it, so an uneven number of errored runs cannot bias the
comparison (thesis Section 3.6). Records are paired on the dataset row index.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from scipy import stats

from . import config

FINAL_BASELINE = [
    "final/baseline_pilot_n9_v2.jsonl",
    "final/baseline_beyond_n18.jsonl",
    "final/baseline_beyond2_n24.jsonl",
    "final/baseline_easy43_new.jsonl",
]
FINAL_GATEWAY = [
    "final/gateway_pilot_n9_v3.jsonl",
    "final/gateway_beyond_n18.jsonl",
    "final/gateway_beyond2_n24.jsonl",
    "final/gateway_easy43_new.jsonl",
]


def load_records(paths: list[str]) -> list[dict]:
    records = []
    for p in paths:
        with open(config.RESULTS_DIR / p, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    return records


def tokens(r: dict) -> int:
    return r["input_tokens"] + r["output_tokens"]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def two_prop_z(b: list[int], g: list[int]) -> tuple[float, float]:
    n1, n2 = len(b), len(g)
    p1, p2 = sum(b) / n1, sum(g) / n2
    pooled = (sum(b) + sum(g)) / (n1 + n2)
    se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2)) ** 0.5
    z = (p2 - p1) / se if se > 0 else 0.0
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def verdict(p: float) -> str:
    return "significant (p<0.05)" if p < 0.05 else "not significant"


def report(base: list[dict], gate: list[dict]) -> None:
    b_tcr = [int(r["tcr"]) for r in base]
    g_tcr = [int(r["tcr"]) for r in gate]
    rows = [
        ("CSR", [r["csr"] for r in base], [r["csr"] for r in gate]),
        ("Tokens/query", [tokens(r) for r in base], [tokens(r) for r in gate]),
        ("Latency (s)", [r["latency_s"] for r in base], [r["latency_s"] for r in gate]),
    ]
    z, pz = two_prop_z(b_tcr, g_tcr)
    print(f"n = {len(base)} matched pairs")
    print(f"TCR           baseline={mean(b_tcr):.3f} gateway={mean(g_tcr):.3f} "
          f"delta={mean(g_tcr) - mean(b_tcr):+.3f}  two-proportion z={z:.3f} p={pz:.4f}  {verdict(pz)}")
    for name, bv, gv in rows:
        t, p = stats.ttest_ind(bv, gv, equal_var=False)
        print(f"{name:<13} baseline={mean(bv):.3f} gateway={mean(gv):.3f} "
              f"delta={mean(gv) - mean(bv):+.3f}  Welch t={t:.3f} p={p:.4f}  {verdict(p)}")
    tb, tg = mean([tokens(r) for r in base]), mean([tokens(r) for r in gate])
    print(f"Token reduction: {(tb - tg) / tb:.1%}")
    seen = [r for r in gate if r.get("rows_seen")]
    if seen:
        print(f"Gateway pruned {mean([r['rows_pruned'] / r['rows_seen'] for r in seen]):.1%} of tool-result rows on average "
              f"(mean {mean([r['rows_pruned'] for r in gate]):.1f} of {mean([r['rows_seen'] for r in gate]):.1f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="use the thesis result files in results/final/")
    ap.add_argument("--baseline", nargs="+", help="baseline JSONL files, relative to results/")
    ap.add_argument("--gateway", nargs="+", help="gateway JSONL files, relative to results/")
    ap.add_argument("--unmatched", action="store_true", help="skip pairing (not what the thesis reports)")
    args = ap.parse_args()

    base_files = FINAL_BASELINE if args.final else args.baseline
    gate_files = FINAL_GATEWAY if args.final else args.gateway
    if not base_files or not gate_files:
        ap.error("pass --final, or both --baseline and --gateway")

    base_all, gate_all = load_records(base_files), load_records(gate_files)
    base_ok = {r["idx"]: r for r in base_all if not r["error"]}
    gate_ok = {r["idx"]: r for r in gate_all if not r["error"]}
    print(f"baseline: {len(base_all)} records, {len(base_ok)} valid | "
          f"gateway: {len(gate_all)} records, {len(gate_ok)} valid")

    if args.unmatched:
        base, gate = list(base_ok.values()), list(gate_ok.values())
    else:
        common = sorted(set(base_ok) & set(gate_ok))
        base, gate = [base_ok[i] for i in common], [gate_ok[i] for i in common]

    print("\n=== Overall (thesis Table 5.1) ===")
    report(base, gate)

    print("\n=== By difficulty level (thesis Table 5.2) ===")
    by_level = defaultdict(list)
    for b, g in zip(base, gate):
        by_level[b["level"]].append((b, g))
    for level in ("easy", "medium", "hard"):
        pairs = by_level.get(level, [])
        if not pairs:
            continue
        bt = mean([int(b["tcr"]) for b, _ in pairs])
        gt = mean([int(g["tcr"]) for _, g in pairs])
        bk = mean([tokens(b) for b, _ in pairs])
        gk = mean([tokens(g) for _, g in pairs])
        print(f"{level:<7} n={len(pairs):<3} TCR baseline={bt:.1%} gateway={gt:.1%} delta={100 * (gt - bt):+.1f} pts | "
              f"tokens baseline={bk:,.0f} gateway={gk:,.0f}")


if __name__ == "__main__":
    main()
