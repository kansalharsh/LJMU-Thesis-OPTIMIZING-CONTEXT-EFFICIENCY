"""Loads TravelPlanner queries (validation split) into typed records.

The validation split is used (not test) because it carries `local_constraint`
and `budget` fields needed to compute CSR; the test split deliberately hides
these to prevent leaderboard contamination, so it can't be self-scored locally.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config


@dataclass
class TravelQuery:
    idx: int
    org: str
    dest: str
    days: int
    visiting_city_number: int
    date: list[str]
    people_number: int
    local_constraint: dict[str, Any]
    budget: float
    query: str
    level: str


def _parse_pyliteral(value: str) -> Any:
    return ast.literal_eval(value)


def load_queries(split: str = "validation") -> list[TravelQuery]:
    path = config.QUERY_DB / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    queries: list[TravelQuery] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            queries.append(
                TravelQuery(
                    idx=i,
                    org=row["org"],
                    dest=row["dest"],
                    days=int(row["days"]),
                    visiting_city_number=int(row["visiting_city_number"]),
                    date=_parse_pyliteral(row["date"]),
                    people_number=int(row["people_number"]),
                    local_constraint=_parse_pyliteral(row["local_constraint"]),
                    budget=float(row["budget"]),
                    query=row["query"],
                    level=row["level"],
                )
            )
    return queries


def stratified_sample(
    queries: list[TravelQuery], n_per_level: int, offset: int = 0, levels: list[str] | None = None
) -> list[TravelQuery]:
    """Take up to n_per_level queries from each difficulty level, in file order.

    `offset` skips the first N queries per level - use it to draw a fresh,
    non-overlapping batch after an earlier pilot already spent queries
    [0:offset) per level.
    """
    by_level: dict[str, list[TravelQuery]] = {}
    for q in queries:
        by_level.setdefault(q.level, []).append(q)

    sample: list[TravelQuery] = []
    for level, qs in by_level.items():
        if levels and level not in levels:
            continue
        sample.extend(qs[offset : offset + n_per_level])
    return sample
