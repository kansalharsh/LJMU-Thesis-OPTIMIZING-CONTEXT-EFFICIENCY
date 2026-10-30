"""Per-query metric records and run-level aggregation.

Definitions follow the research proposal:
- TCR: fraction of queries whose plan violates zero applicable constraints.
- CSR: fraction of individual applicable constraints satisfied, averaged
  across all queries (more granular than TCR).
- Token-to-Signal Ratio: total tokens consumed per successful (TCR=1) plan.
- System Latency: end-to-end wall time per query, excluding rate-limit
  backoff waits (tracked separately so free-tier throttling doesn't
  masquerade as gateway/reasoning latency).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class QueryRecord:
    idx: int
    level: str
    org: str
    dest: str
    tcr: bool
    csr: float
    input_tokens: int
    output_tokens: int
    latency_s: float
    rate_limit_wait_s: float
    react_steps: int
    error: str | None = None
    rows_seen: int = 0
    rows_pruned: int = 0
    plan_text: str = ""  # added after the thesis batches; empty in results/final


def write_jsonl(records: list[QueryRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def summarize(records: list[QueryRecord]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}

    successful = [r for r in records if r.tcr]
    total_tokens = sum(r.input_tokens + r.output_tokens for r in records)
    successful_tokens = sum(r.input_tokens + r.output_tokens for r in successful)

    return {
        "n": n,
        "tcr": len(successful) / n,
        "csr_mean": sum(r.csr for r in records) / n,
        "token_to_signal_ratio": (
            successful_tokens / len(successful) if successful else float("inf")
        ),
        "total_tokens": total_tokens,
        "avg_latency_s": sum(r.latency_s for r in records) / n,
        "avg_rate_limit_wait_s": sum(r.rate_limit_wait_s for r in records) / n,
        "errors": sum(1 for r in records if r.error),
    }
