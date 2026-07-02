#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SCENARIOS = (
    "best_f1:-:-",
    "precision90:0.90:-",
    "precision80_recall30:0.80:0.30",
    "precision70_recall50:0.70:0.50",
    "recall50:-:0.50",
)


@dataclass(frozen=True)
class DecisionScore:
    query_id: str
    score: float
    actual_is_duplicate: bool
    best_is_correct_duplicate: bool


@dataclass(frozen=True)
class Scenario:
    name: str
    min_precision: float | None
    min_recall: float | None


@dataclass(frozen=True)
class ThresholdResult:
    scenario: str
    satisfied: bool
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    correct_duplicate_links: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep duplicate decision thresholds from a decisions CSV without rerunning SBERT.")
    parser.add_argument("--decisions-csv", required=True, help="CSV produced by tune-threshold --output-decisions-csv")
    parser.add_argument("--scenario", action="append", help="Scenario as name:min_precision:min_recall. Use '-' for no constraint.")
    parser.add_argument("--output-csv", default="reports/decision_threshold_sweep.csv")
    args = parser.parse_args()

    scores = read_decision_scores(args.decisions_csv)
    scenarios = [parse_scenario(value) for value in (args.scenario or DEFAULT_SCENARIOS)]
    results = [best_threshold_for_scenario(scores, scenario) for scenario in scenarios]
    write_results(results, args.output_csv)
    print(render_results(results))
    print(f"output_csv={args.output_csv}")
    return 0


def read_decision_scores(path: str | Path) -> list[DecisionScore]:
    rows: list[DecisionScore] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("score"):
                continue
            rows.append(
                DecisionScore(
                    query_id=row.get("query_id", ""),
                    score=float(row["score"]),
                    actual_is_duplicate=row.get("actual_is_duplicate") == "true",
                    best_is_correct_duplicate=row.get("best_is_correct_duplicate") == "true",
                )
            )
    if not rows:
        raise SystemExit(f"No usable decision scores found: {path}")
    return rows


def parse_scenario(value: str) -> Scenario:
    parts = value.split(":")
    if len(parts) != 3:
        raise SystemExit(f"Invalid --scenario {value!r}; expected name:min_precision:min_recall")
    name, precision_text, recall_text = parts
    if not name:
        raise SystemExit("Scenario name cannot be empty")
    return Scenario(
        name=name,
        min_precision=parse_optional_float(precision_text),
        min_recall=parse_optional_float(recall_text),
    )


def parse_optional_float(value: str) -> float | None:
    value = value.strip()
    if value in {"", "-"}:
        return None
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise SystemExit("Scenario precision/recall constraints must be between 0 and 1")
    return parsed


def best_threshold_for_scenario(scores: list[DecisionScore], scenario: Scenario) -> ThresholdResult:
    candidates = threshold_candidates(scores)
    constrained = [
        candidate for candidate in candidates
        if passes_constraints(candidate, min_precision=scenario.min_precision, min_recall=scenario.min_recall)
    ]
    if constrained:
        best = max(constrained, key=result_key)
        return replace_scenario(best, scenario.name, satisfied=True)
    best = max(candidates, key=result_key)
    return replace_scenario(best, scenario.name, satisfied=False)


def threshold_candidates(scores: list[DecisionScore]) -> list[ThresholdResult]:
    records = sorted(scores, key=lambda score: score.score, reverse=True)
    total = len(records)
    positives = sum(1 for record in records if record.actual_is_duplicate)
    negatives = total - positives
    candidates: list[ThresholdResult] = []

    true_positives = 0
    false_positives = 0
    correct_duplicate_links = 0
    index = 0
    while index < total:
        threshold = records[index].score
        while index < total and records[index].score == threshold:
            record = records[index]
            if record.actual_is_duplicate:
                true_positives += 1
                if record.best_is_correct_duplicate:
                    correct_duplicate_links += 1
            else:
                false_positives += 1
            index += 1

        false_negatives = positives - true_positives
        true_negatives = negatives - false_positives
        precision = ratio(true_positives, true_positives + false_positives)
        recall = ratio(true_positives, true_positives + false_negatives)
        f1 = ratio(2 * precision * recall, precision + recall)
        accuracy = ratio(true_positives + true_negatives, total)
        candidates.append(
            ThresholdResult(
                scenario="",
                satisfied=True,
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=f1,
                accuracy=accuracy,
                true_positives=true_positives,
                false_positives=false_positives,
                true_negatives=true_negatives,
                false_negatives=false_negatives,
                correct_duplicate_links=correct_duplicate_links,
            )
        )
    return candidates


def passes_constraints(result: ThresholdResult, *, min_precision: float | None, min_recall: float | None) -> bool:
    if min_precision is not None and result.precision < min_precision:
        return False
    if min_recall is not None and result.recall < min_recall:
        return False
    return True


def result_key(result: ThresholdResult) -> tuple[float, float, float, float, int, float]:
    return (
        result.f1,
        result.precision,
        result.recall,
        result.accuracy,
        result.correct_duplicate_links,
        result.threshold,
    )


def replace_scenario(result: ThresholdResult, scenario: str, *, satisfied: bool) -> ThresholdResult:
    return ThresholdResult(
        scenario=scenario,
        satisfied=satisfied,
        threshold=result.threshold,
        precision=result.precision,
        recall=result.recall,
        f1=result.f1,
        accuracy=result.accuracy,
        true_positives=result.true_positives,
        false_positives=result.false_positives,
        true_negatives=result.true_negatives,
        false_negatives=result.false_negatives,
        correct_duplicate_links=result.correct_duplicate_links,
    )


def write_results(results: list[ThresholdResult], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ThresholdResult.__dataclass_fields__.keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fieldnames})


def render_results(results: list[ThresholdResult]) -> str:
    lines = [
        "Decision Threshold Sweep",
        f"{'Scenario':<24} {'OK':>3} {'Threshold':>9} {'Precision':>9} {'Recall':>8} {'F1':>8} {'FP':>4} {'FN':>4}",
        "-" * 78,
    ]
    for result in results:
        lines.append(
            f"{result.scenario:<24} {str(result.satisfied):>3} {result.threshold:>9.6f} "
            f"{result.precision:>9.4f} {result.recall:>8.4f} {result.f1:>8.4f} "
            f"{result.false_positives:>4} {result.false_negatives:>4}"
        )
    return "\n".join(lines)


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
