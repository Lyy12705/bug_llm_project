#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import PipelineConfig  # noqa: E402
from modules.duplicate_detector import DuplicateDetector  # noqa: E402
from modules.priority_classifier import PRIORITY_TO_SCORE, PriorityClassifier  # noqa: E402


DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "integrated_baseline_eval.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the integrated deterministic baselines on a frozen smoke fixture.")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--section", choices=("all", "duplicate", "priority"), default="all")
    parser.add_argument("--duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    report = evaluate_fixture(payload, duplicate_threshold=args.duplicate_threshold)
    selected = report if args.section == "all" else {args.section: report[args.section]}
    text = json.dumps(selected, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def evaluate_fixture(payload: dict[str, Any], *, duplicate_threshold: float) -> dict[str, Any]:
    historical = list(payload.get("historical_tickets") or [])
    queries = list(payload.get("queries") or [])
    detector = DuplicateDetector(
        config=PipelineConfig(project_root=ROOT, save_checkpoints=False),
        threshold=duplicate_threshold,
    )
    classifier = PriorityClassifier()
    duplicate_rows: list[dict[str, Any]] = []
    priority_rows: list[dict[str, Any]] = []
    for row in queries:
        ticket = dict(row.get("ticket") or {})
        duplicate = detector.detect(ticket, historical)
        duplicate_rows.append(
            {
                "ticket_id": ticket.get("ticket_id"),
                "gold": bool(row.get("duplicate")),
                "predicted": bool(duplicate.get("is_duplicate")),
                "confidence": float(duplicate.get("similarity_score") or 0.0),
            }
        )
        priority = classifier.predict(ticket, list(duplicate.get("top_k_candidates") or []))
        priority_rows.append(
            {
                "ticket_id": ticket.get("ticket_id"),
                "gold": str(row.get("priority") or ""),
                "predicted": str(priority.get("predicted_priority") or ""),
                "confidence": float(priority.get("confidence") or 0.0),
            }
        )

    return {
        "schema_version": 1,
        "fixture_description": payload.get("description", ""),
        "rows": len(queries),
        "duplicate": _binary_metrics(duplicate_rows),
        "priority": _multiclass_metrics(priority_rows),
    }


def _binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["gold"] and row["predicted"] for row in rows)
    fp = sum(not row["gold"] and row["predicted"] for row in rows)
    fn = sum(row["gold"] and not row["predicted"] for row in rows)
    tn = len(rows) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": len(rows),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round((tp + tn) / len(rows), 6) if rows else 0.0,
        "ece": round(_ece(rows), 6),
    }


def _multiclass_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted(PRIORITY_TO_SCORE)
    per_label: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(row["gold"] == label and row["predicted"] == label for row in rows)
        fp = sum(row["gold"] != label and row["predicted"] == label for row in rows)
        fn = sum(row["gold"] == label and row["predicted"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if any(row["gold"] == label for row in rows):
            f1_values.append(f1)
        per_label[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    correct = sum(row["gold"] == row["predicted"] for row in rows)
    return {
        "rows": len(rows),
        "accuracy": round(correct / len(rows), 6) if rows else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else 0.0,
        "gold_distribution": dict(Counter(row["gold"] for row in rows)),
        "predicted_distribution": dict(Counter(row["predicted"] for row in rows)),
        "per_label": per_label,
        "ece": round(_ece([{**row, "gold": row["gold"] == row["predicted"]} for row in rows]), 6),
    }


def _ece(rows: list[dict[str, Any]], bins: int = 5) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [row for row in rows if lower <= float(row["confidence"]) <= upper if index == bins - 1 or float(row["confidence"]) < upper]
        if not bucket:
            continue
        accuracy = sum(bool(row["gold"]) for row in bucket) / len(bucket)
        confidence = sum(float(row["confidence"]) for row in bucket) / len(bucket)
        total += len(bucket) / len(rows) * abs(accuracy - confidence)
    return total


if __name__ == "__main__":
    raise SystemExit(main())
