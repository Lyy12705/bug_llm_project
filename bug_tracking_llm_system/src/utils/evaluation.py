from __future__ import annotations

from collections import Counter
from typing import Iterable


def accuracy(gold: Iterable[str], predicted: Iterable[str]) -> float:
    pairs = list(zip(gold, predicted))
    if not pairs:
        return 0.0
    return sum(1 for left, right in pairs if left == right) / len(pairs)


def classification_counts(gold: Iterable[str], predicted: Iterable[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(gold) | set(predicted))
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    for expected, actual in zip(gold, predicted):
        if expected == actual:
            counts[expected]["tp"] += 1
        else:
            counts.setdefault(actual, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
            counts.setdefault(expected, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
    return counts


def field_level_accuracy(gold_rows: list[dict], predicted_rows: list[dict], fields: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in fields:
        result[field] = accuracy(
            [str(row.get(field, "")) for row in gold_rows],
            [str(row.get(field, "")) for row in predicted_rows],
        )
    return result


def label_distribution(rows: Iterable[dict], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "")) for row in rows))
