#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (str(ROOT), str(SRC)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from scripts.evaluate_fault_localization import (
    _file_matches,
    _gold_files,
    _repository_name,
    _stage1_candidate_files,
    _ticket_id,
    read_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Top-20-only Stage-1 metrics and paired method differences."
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="LABEL=predictions.jsonl; repeat for each method on the same ticket split.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally compare only the first N gold tickets, useful for a paired smoke test.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive when supplied.")
    predictions = parse_prediction_args(args.prediction)
    gold_rows = read_records(args.gold)
    if args.limit is not None:
        gold_rows = gold_rows[: args.limit]
    analyses = {
        label: analyze_method(
            gold_rows,
            read_records(path),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + position * 100,
        )
        for position, (label, path) in enumerate(predictions.items())
    }
    baseline_label = next(iter(predictions))
    baseline_rows = ticket_metric_rows(gold_rows, read_records(predictions[baseline_label]))
    paired: dict[str, Any] = {}
    for position, (label, path) in enumerate(list(predictions.items())[1:], start=1):
        candidate_rows = ticket_metric_rows(gold_rows, read_records(path))
        paired[label] = paired_differences(
            baseline_rows,
            candidate_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 10_000 + position * 100,
        )
    output = {
        "metric_scope": "Stage-1 Top-20 candidate files only",
        "gold_path": str(Path(args.gold)),
        "rows": len(gold_rows),
        "methods": analyses,
        "paired_baseline": baseline_label,
        "paired_differences": paired,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "definitions": {
            "hit_at_20": "Fraction of tickets with at least one gold file in the first 20 candidates.",
            "recall_at_20": "Mean fraction of all gold files recovered in the first 20 candidates.",
            "top_1_accuracy": "Fraction of tickets whose first candidate matches a gold file.",
            "mrr_at_20": "Mean reciprocal rank of the first gold file within candidates 1-20; zero on misses.",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def parse_prediction_args(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Prediction must use LABEL=PATH syntax: {value}")
        label, raw_path = value.split("=", maxsplit=1)
        if not label or label in parsed:
            raise ValueError(f"Prediction label is empty or duplicated: {label}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed[label] = path
    return parsed


def analyze_method(
    gold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = ticket_metric_rows(gold_rows, predictions)
    metric_names = ("hit_at_20", "recall_at_20", "top_1_accuracy", "mrr_at_20")
    summary: dict[str, Any] = {
        "rows": len(rows),
        **{metric: mean([float(row[metric]) for row in rows]) for metric in metric_names},
        "full_recall_rows": sum(row["outcome"] == "full_recall" for row in rows),
        "partial_recall_rows": sum(row["outcome"] == "partial_recall" for row in rows),
        "miss_rows": sum(row["outcome"] == "miss" for row in rows),
        "bootstrap_95_ci": {
            metric: bootstrap_mean_ci(
                [float(row[metric]) for row in rows],
                samples=bootstrap_samples,
                seed=seed + offset,
            )
            for offset, metric in enumerate(metric_names)
        },
    }
    repositories = sorted({str(row["repo"]) for row in rows})
    summary["per_repository"] = {}
    for repository in repositories:
        subset = [row for row in rows if row["repo"] == repository]
        summary["per_repository"][repository] = {
            "rows": len(subset),
            **{metric: mean([float(row[metric]) for row in subset]) for metric in metric_names},
            "miss_rows": sum(row["outcome"] == "miss" for row in subset),
        }
    return summary


def ticket_metric_rows(
    gold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_by_id = {_ticket_id(row): row for row in predictions if _ticket_id(row)}
    rows: list[dict[str, Any]] = []
    for gold in gold_rows:
        current_id = _ticket_id(gold)
        gold_files = _gold_files(gold)
        if not gold_files:
            continue
        prediction = pred_by_id.get(current_id, {})
        candidate_files = _stage1_candidate_files(prediction)[:20]
        recovered = sum(
            any(_file_matches(candidate, gold_file) for candidate in candidate_files)
            for gold_file in gold_files
        )
        rank = first_relevant_rank(candidate_files, gold_files)
        recall = recovered / len(gold_files)
        rows.append(
            {
                "ticket_id": current_id,
                "repo": _repository_name(gold) or _repository_name(prediction) or "unknown",
                "hit_at_20": float(recovered > 0),
                "recall_at_20": recall,
                "top_1_accuracy": float(rank == 1),
                "mrr_at_20": 1.0 / rank if rank is not None else 0.0,
                "outcome": "full_recall" if recovered == len(gold_files) else "partial_recall" if recovered else "miss",
            }
        )
    return rows


def first_relevant_rank(candidate_files: list[str], gold_files: list[str]) -> int | None:
    for rank, candidate in enumerate(candidate_files, start=1):
        if any(_file_matches(candidate, gold_file) for gold_file in gold_files):
            return rank
    return None


def paired_differences(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = {str(row["ticket_id"]): row for row in baseline_rows}
    candidate = {str(row["ticket_id"]): row for row in candidate_rows}
    if baseline.keys() != candidate.keys():
        raise ValueError("Paired methods do not cover the same ticket IDs.")
    metric_names = ("hit_at_20", "recall_at_20", "top_1_accuracy", "mrr_at_20")
    output: dict[str, Any] = {"rows": len(baseline)}
    for offset, metric in enumerate(metric_names):
        differences = [
            float(candidate[current_id][metric]) - float(baseline[current_id][metric])
            for current_id in baseline
        ]
        output[metric] = {
            "mean_difference": mean(differences),
            "bootstrap_95_ci": bootstrap_mean_ci(
                differences,
                samples=bootstrap_samples,
                seed=seed + offset,
            ),
        }
    return output


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    if not values:
        return {"lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    lower_index = max(0, int(samples * 0.025))
    upper_index = min(samples - 1, int(samples * 0.975))
    return {
        "lower": round(estimates[lower_index], 6),
        "upper": round(estimates[upper_index], 6),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
