from __future__ import annotations

import argparse
from typing import Any

from common import accuracy, add_common_args, nested, read_records, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate assignee triage outputs.")
    add_common_args(parser)
    args = parser.parse_args()
    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    expected = [normalize(row.get("assignee")) for row in gold_rows]
    predicted = [predicted_assignee(row) for row in pred_rows]
    candidates = [ranked_candidates(row) for row in pred_rows]
    metrics = {
        "rows": min(len(gold_rows), len(pred_rows)),
        "top1_accuracy": accuracy(expected, predicted),
        "top3_accuracy": hit_at_k(expected, candidates, 3),
        "top5_accuracy": hit_at_k(expected, candidates, 5),
        "mrr": mrr(expected, candidates),
        "macro_f1": macro_f1(expected, predicted),
        "unable_to_decide_rate": ratio(
            sum(1 for row, pred in zip(pred_rows, predicted, strict=False) if pred == "manual_triage" or nested(row, "assignee", "needs_manual_triage", default=False)),
            min(len(gold_rows), len(pred_rows)),
        ),
        "fallback_trigger_rate": ratio(
            sum(1 for row in pred_rows if nested(row, "assignee", "fallback_used", default=row.get("fallback_used", False))),
            min(len(gold_rows), len(pred_rows)),
        ),
    }
    write_metrics(metrics, args.output)


def predicted_assignee(row: dict[str, Any]) -> str:
    return normalize(row.get("predicted_assignee") or nested(row, "assignee", "assignee", default=row.get("assignee", "")))


def ranked_candidates(row: dict[str, Any]) -> list[str]:
    raw = row.get("ranked_candidates") or nested(row, "assignee", "ranked_candidates", default=[])
    if not isinstance(raw, list):
        return []
    return [normalize(candidate) for candidate in raw if normalize(candidate)]


def hit_at_k(expected: list[str], candidates_by_row: list[list[str]], k: int) -> float:
    pairs = list(zip(expected, candidates_by_row, strict=False))
    return ratio(sum(1 for gold, candidates in pairs if gold and gold in candidates[:k]), len(pairs))


def mrr(expected: list[str], candidates_by_row: list[list[str]]) -> float:
    pairs = list(zip(expected, candidates_by_row, strict=False))
    reciprocal_sum = 0.0
    for gold, candidates in pairs:
        for index, candidate in enumerate(candidates, start=1):
            if gold and candidate == gold:
                reciprocal_sum += 1.0 / index
                break
    return ratio(reciprocal_sum, len(pairs))


def macro_f1(expected: list[str], predicted: list[str]) -> float:
    pairs = list(zip(expected, predicted, strict=False))
    labels = sorted({label for pair in pairs for label in pair if label})
    if not labels:
        return 0.0
    values = []
    for label in labels:
        true_positive = sum(1 for gold, pred in pairs if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in pairs if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(values) / len(values), 6)


def ratio(numerator: float, denominator: int) -> float:
    return round(float(numerator) / denominator, 6) if denominator else 0.0


def normalize(value: Any) -> str:
    return str(value or "").strip().lower()


if __name__ == "__main__":
    main()
