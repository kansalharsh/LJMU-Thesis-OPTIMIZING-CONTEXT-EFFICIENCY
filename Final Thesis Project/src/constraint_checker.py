"""Hard-constraint validators for a produced itinerary.

TravelPlanner's official repo ships its own commonsense/hard-constraint
checker; it is not included in this dataset drop, so this is a
from-scratch reimplementation covering the constraint fields present in
`local_constraint` and `budget`. Each check returns True/False; CSR is the
fraction of applicable checks that pass, TCR is True only if every
applicable check passes.

Chapter 3 (Methodology) should document this as the evaluation protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data_loader import TravelQuery


@dataclass
class ConstraintResult:
    name: str
    applicable: bool
    satisfied: bool


def _extract_total_cost(plan_text: str) -> float | None:
    import re

    # Heuristic: look for an explicit "Total Cost" / "Total:" line.
    for line in plan_text.splitlines():
        stripped = line.lower().strip().strip("*# ")
        if "total cost" in stripped or stripped.startswith("total:") or stripped.startswith("total "):
            nums = re.findall(r"\d[\d,]*\.?\d*", line)
            if nums:
                return float(nums[-1].replace(",", ""))
    return None


def check_budget(plan_text: str, query: TravelQuery) -> ConstraintResult:
    total = _extract_total_cost(plan_text)
    if total is None:
        return ConstraintResult("budget", True, False)
    return ConstraintResult("budget", True, total <= query.budget)


def check_house_rule(plan_text: str, query: TravelQuery) -> ConstraintResult:
    rule = query.local_constraint.get("house rule")
    if not rule:
        return ConstraintResult("house_rule", False, True)
    return ConstraintResult("house_rule", True, rule.lower() in plan_text.lower())


def check_cuisine(plan_text: str, query: TravelQuery) -> ConstraintResult:
    cuisine = query.local_constraint.get("cuisine")
    if not cuisine:
        return ConstraintResult("cuisine", False, True)
    cuisines = cuisine if isinstance(cuisine, list) else [cuisine]
    text_lower = plan_text.lower()
    satisfied = all(c.lower() in text_lower for c in cuisines)
    return ConstraintResult("cuisine", True, satisfied)


def check_room_type(plan_text: str, query: TravelQuery) -> ConstraintResult:
    room_type = query.local_constraint.get("room type")
    if not room_type:
        return ConstraintResult("room_type", False, True)
    return ConstraintResult("room_type", True, room_type.lower() in plan_text.lower())


def check_transportation(plan_text: str, query: TravelQuery) -> ConstraintResult:
    transportation = query.local_constraint.get("transportation")
    if not transportation:
        return ConstraintResult("transportation", False, True)
    return ConstraintResult("transportation", True, transportation.lower() in plan_text.lower())


def check_days_present(plan_text: str, query: TravelQuery) -> ConstraintResult:
    days_mentioned = sum(1 for d in range(1, query.days + 1) if f"day {d}" in plan_text.lower())
    return ConstraintResult("days_coverage", True, days_mentioned == query.days)


ALL_CHECKS = [
    check_budget,
    check_house_rule,
    check_cuisine,
    check_room_type,
    check_transportation,
    check_days_present,
]


def evaluate_plan(plan_text: str, query: TravelQuery) -> list[ConstraintResult]:
    return [check(plan_text, query) for check in ALL_CHECKS]


def csr(results: list[ConstraintResult]) -> float:
    applicable = [r for r in results if r.applicable]
    if not applicable:
        return 1.0
    return sum(1 for r in applicable if r.satisfied) / len(applicable)


def tcr(results: list[ConstraintResult]) -> bool:
    return all(r.satisfied for r in results if r.applicable)
