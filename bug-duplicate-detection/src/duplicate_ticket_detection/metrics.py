from __future__ import annotations

from collections.abc import Iterable, Mapping


def average_precision(ranked_ids: Iterable[str], relevant_ids: Iterable[str]) -> float:
    """Compute AP for one ranked duplicate search result."""

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen_relevant: set[str] = set()
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant and ticket_id not in seen_relevant:
            seen_relevant.add(ticket_id)
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)


def mean_average_precision(results: Mapping[str, tuple[Iterable[str], Iterable[str]]]) -> float:
    """Compute MAP from query id to (ranked ids, relevant ids)."""

    scores = [average_precision(ranked, relevant) for ranked, relevant in results.values() if set(relevant)]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
