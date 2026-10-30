"""CLI entrypoint: run an agent condition over a TravelPlanner query sample.

Usage:
    python -m src.run_experiment --agent baseline --n-per-level 5 --out results/baseline_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config, constraint_checker, data_loader, metrics
from .agents.baseline_agent import BaselineAgent
from .agents.gateway_agent import GatewayAgent
from .llm_provider import get_provider

AGENTS = {
    "baseline": BaselineAgent,
    "gateway": GatewayAgent,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=AGENTS.keys(), default="baseline")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-per-level", type=int, default=5, help="Queries per difficulty level")
    parser.add_argument("--levels", nargs="+", choices=["easy", "medium", "hard"], default=None,
                        help="Restrict to these difficulty levels (default: all three)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N queries per level (draw a fresh batch)")
    parser.add_argument("--out", default=None, help="Output JSONL path (default: results/<agent>_<split>.jsonl)")
    args = parser.parse_args()

    queries = data_loader.load_queries(args.split)
    sample = data_loader.stratified_sample(queries, args.n_per_level, args.offset, args.levels)

    out_path = config.RESULTS_DIR / (args.out or f"{args.agent}_{args.split}_pilot.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = get_provider()
    agent = AGENTS[args.agent](llm)

    records: list[metrics.QueryRecord] = []
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(sample, start=1):
            print(f"[{i}/{len(sample)}] {q.level:8s} {q.org} -> {q.dest} ({q.days}d)", file=sys.stderr)
            result = agent.run(q)

            if result.error:
                csr_val, tcr_val = 0.0, False
            else:
                try:
                    checks = constraint_checker.evaluate_plan(result.plan_text, q)
                    csr_val = constraint_checker.csr(checks)
                    tcr_val = constraint_checker.tcr(checks)
                except Exception as e:  # noqa: BLE001 - a bad plan shouldn't crash a paid run
                    print(f"    [eval error] {e}", file=sys.stderr)
                    csr_val, tcr_val = 0.0, False
                    result.error = f"eval_error: {e}"

            record = metrics.QueryRecord(
                idx=q.idx,
                level=q.level,
                org=q.org,
                dest=q.dest,
                tcr=tcr_val,
                csr=csr_val,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_s=result.latency_s,
                rate_limit_wait_s=result.rate_limit_wait_s,
                react_steps=result.steps,
                error=result.error,
                rows_seen=getattr(agent, "total_rows_seen", 0),
                rows_pruned=getattr(agent, "total_rows_pruned", 0),
                plan_text=result.plan_text,
            )
            records.append(record)
            out_f.write(json.dumps(record.__dict__) + "\n")
            out_f.flush()
            print(
                f"    -> tcr={record.tcr} csr={record.csr:.2f} "
                f"tokens={record.input_tokens + record.output_tokens} "
                f"latency={record.latency_s:.1f}s wait={record.rate_limit_wait_s:.1f}s "
                f"error={record.error}",
                file=sys.stderr,
            )

    summary = metrics.summarize(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
