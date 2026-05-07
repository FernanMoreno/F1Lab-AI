"""Campaign ranking helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from reglabsim.failures.taxonomy import failure_priority_score


def rank_failures(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate and rank failures across runs."""
    counter: Counter[str] = Counter()
    safety_counter: Counter[str] = Counter()
    priority_counter: Counter[str] = Counter()
    for run in runs:
        for failure in run.get("failure_log", []):
            counter[failure["failure_type"]] += 1
            safety_counter[failure["failure_type"]] += {
                "low": 1,
                "medium": 2,
                "high": 3,
                "critical": 4,
            }.get(failure["severity"], 1)
            priority_counter[failure["failure_type"]] += round(
                failure_priority_score(failure) * 1000
            )
    ranking = []
    for failure_type, count in counter.most_common():
        ranking.append(
            {
                "failure_type": failure_type,
                "count": count,
                "weighted_severity": safety_counter[failure_type],
                "priority_score": round(priority_counter[failure_type] / 1000, 4),
            }
        )
    return ranking
