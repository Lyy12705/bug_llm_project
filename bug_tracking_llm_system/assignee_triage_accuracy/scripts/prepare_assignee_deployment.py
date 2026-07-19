from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import PipelineConfig  # noqa: E402
from modules.assignee_deployment import (  # noqa: E402
    build_roster,
    derive_ownership,
    load_assignee_set,
    normalize_history_record,
    temporal_split,
)
from modules.assignee_triager import AssigneeTriager  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a review-gated assignee deployment bundle from project history.")
    parser.add_argument("--history", type=Path, required=True, help="Project issue history JSONL with ground-truth assignees.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="project_assignee")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-assignee-history", type=int, default=5)
    parser.add_argument(
        "--inactive-assignees",
        type=Path,
        default=None,
        help="Reviewed JSON list of inactive/departed assignees to exclude before candidate generation.",
    )
    parser.add_argument("--ownership-minimum-rows", type=int, default=5)
    parser.add_argument("--ownership-minimum-share", type=float, default=0.70)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.80)
    parser.add_argument("--minimum-auto-coverage", type=float, default=0.10)
    parser.add_argument("--minimum-calibration-rows", type=int, default=30)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve artifacts only when all validation gates pass; otherwise outputs remain research-only.",
    )
    parser.add_argument(
        "--confirm-roster-reviewed",
        action="store_true",
        help="Confirm that the generated roster and inactive list were reviewed by a project maintainer.",
    )
    parser.add_argument(
        "--confirm-upstream-duplicate-filtered",
        action="store_true",
        help="Confirm that input history has already passed the separate upstream duplicate-detection feature.",
    )
    args = parser.parse_args()

    for name, value in (
        ("ownership minimum share", args.ownership_minimum_share),
        ("target auto accuracy", args.target_auto_accuracy),
        ("minimum auto coverage", args.minimum_auto_coverage),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be between 0 and 1")

    raw_rows = read_jsonl(args.history)
    normalized = [row for row in (normalize_history_record(value) for value in raw_rows) if row]
    train, validation, test, leakage = temporal_split(normalized, args.train_ratio, args.validation_ratio)
    inactive = load_assignee_set(args.inactive_assignees)
    roster_payload, roster = build_roster(train, max(1, args.minimum_assignee_history), inactive)
    if not roster:
        raise SystemExit("No assignee meets --minimum-assignee-history; lower it or provide more history.")
    prefilter_train_rows = len(train)
    train = [row for row in train if row["assignee"] in roster]
    roster_payload["review_confirmed"] = bool(args.confirm_roster_reviewed)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "history": output / "history_train.jsonl",
        "validation": output / "validation_set.jsonl",
        "test": output / "test_set.jsonl",
        "roster": output / "active_assignees.json",
        "component_ownership": output / "component_ownership.candidate.json",
        "file_ownership": output / "file_ownership.candidate.json",
        "calibrator": output / "assignee_confidence_calibrator.json",
        "policy": output / "routing_policy.json",
        "open_set": output / "open_set_detector.candidate.json",
        "feedback": output / "assignee_feedback.jsonl",
        "bundle": output / "deployment_bundle.json",
        "report": output / "deployment_report.json",
    }
    write_jsonl(paths["history"], train)
    write_jsonl(paths["validation"], validation)
    write_jsonl(paths["test"], test)
    write_json(paths["roster"], roster_payload)

    component_ownership, file_ownership = derive_ownership(
        train, roster, max(1, args.ownership_minimum_rows), args.ownership_minimum_share
    )
    write_json(paths["component_ownership"], component_ownership)
    write_json(paths["file_ownership"], file_ownership)

    validation_predictions = predict(paths["history"], paths["roster"], validation)
    test_predictions = predict(paths["history"], paths["roster"], test)
    calibration = fit_calibration(validation_predictions, args.minimum_calibration_rows)
    routing = select_routing_threshold(
        validation_predictions,
        calibration,
        target_accuracy=args.target_auto_accuracy,
        minimum_coverage=args.minimum_auto_coverage,
    )
    gates = {
        "enough_calibration_rows": len(validation_predictions) >= args.minimum_calibration_rows,
        "both_correct_and_incorrect_examples": len({row["correct"] for row in validation_predictions}) == 2,
        "target_auto_accuracy_met": bool(routing.get("target_met")),
        "active_roster_review_confirmed": bool(args.confirm_roster_reviewed),
        "upstream_duplicate_filter_confirmed": bool(args.confirm_upstream_duplicate_filtered),
        "no_cross_split_ticket_id_overlap": leakage["cross_split_ticket_id_overlap"] == 0,
        "no_cross_split_content_overlap": leakage["cross_split_content_overlap"] == 0,
        "chronological_split_valid": bool(leakage["chronological_order_valid"]),
    }
    approved = bool(args.approve and all(gates.values()) and calibration)
    deployment_status = "approved" if approved else "research_only"

    calibrator_artifact = {
        "schema_version": 1,
        "artifact_type": "assignee_confidence_calibrator",
        "name": "isotonic_regression",
        "deployment_status": deployment_status,
        "runtime_supported": bool(calibration),
        "ranker_confidence_version": "hybrid_ranker_v1",
        "dataset": args.dataset,
        "mapping": calibration,
        "validation_metrics": prediction_metrics(validation_predictions, calibration, routing),
        "frozen_test_metrics": prediction_metrics(test_predictions, calibration, routing),
        "approval_gates": gates,
    }
    write_json(paths["calibrator"], calibrator_artifact)
    policy = {
        "schema_version": 1,
        "name": "project_selective_routing",
        "deployment_status": deployment_status,
        "t_high": routing.get("threshold", 1.0),
        "t_low": min(float(routing.get("threshold", 1.0)), 0.50),
        "target_auto_accuracy": args.target_auto_accuracy,
        "minimum_auto_coverage": args.minimum_auto_coverage,
        "selection_split": "validation",
    }
    write_json(paths["policy"], policy)
    write_json(paths["open_set"], candidate_open_set_artifact(args.dataset))

    pipeline_config = {
        "assignee_dataset_path": "history_train.jsonl",
        "assignee_active_roster_path": "active_assignees.json",
        "assignee_feedback_path": "assignee_feedback.jsonl",
        "assignee_routing_policy_path": "routing_policy.json" if approved else None,
        "assignee_routing_policy_name": "project_selective_routing" if approved else "",
        "assignee_calibration_artifact_path": "assignee_confidence_calibrator.json" if approved else None,
        "assignee_open_set_enabled": True,
        "assignee_open_set_risk_threshold": 0.75,
        "assignee_allow_uncalibrated_auto_assignment": False,
    }
    bundle = {
        "schema_version": 1,
        "artifact_type": "assignee_deployment_bundle",
        "dataset": args.dataset,
        "deployment_status": deployment_status,
        "pipeline_config": pipeline_config,
        "candidate_ownership_files": {
            "component": paths["component_ownership"].name,
            "file": paths["file_ownership"].name,
            "note": "Review with maintainers before adding these paths to pipeline_config.",
        },
        "approval_gates": gates,
    }
    report = {
        "dataset": args.dataset,
        "deployment_status": deployment_status,
        "explicit_approval_requested": args.approve,
        "roster_review_confirmed": args.confirm_roster_reviewed,
        "raw_rows": len(raw_rows),
        "normalized_rows": len(normalized),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "roster_size": len(roster),
        "long_tail_or_inactive_train_rows_excluded": prefilter_train_rows - len(train),
        "leakage_audit": leakage,
        "approval_gates": gates,
        "routing_selection": routing,
        "validation_metrics": calibrator_artifact["validation_metrics"],
        "frozen_test_metrics": calibrator_artifact["frozen_test_metrics"],
        "class_imbalance": class_imbalance(train),
    }
    write_json(paths["bundle"], bundle)
    write_json(paths["report"], report)
    print(json.dumps({"bundle": str(paths["bundle"]), **report}, ensure_ascii=False, indent=2))


def predict(history_path: Path, roster_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config = PipelineConfig(
        project_root=SYSTEM_ROOT,
        assignee_dataset_path=history_path,
        assignee_active_roster_path=roster_path,
        assignee_allow_uncalibrated_auto_assignment=True,
        assignee_top_k=5,
        save_checkpoints=False,
    )
    triager = AssigneeTriager(config=config)
    predictions = []
    for row in rows:
        result = triager.assign(row, {"predicted_priority": row.get("priority", "P3")})
        candidates = [owner for owner in result.get("ranked_candidates", []) if owner != "manual_triage"]
        predictions.append(
            {
                "ticket_id": row["ticket_id"],
                "expected": row["assignee"],
                "candidates": candidates,
                "raw_confidence": float(result.get("raw_confidence") or 0.0),
                "correct": bool(candidates and candidates[0] == row["assignee"]),
            }
        )
    return predictions


def fit_calibration(predictions: list[dict[str, Any]], minimum_rows: int) -> dict[str, list[float]]:
    if len(predictions) < minimum_rows or len({row["correct"] for row in predictions}) < 2:
        return {}
    x = np.asarray([row["raw_confidence"] for row in predictions], dtype=float)
    y = np.asarray([int(row["correct"]) for row in predictions], dtype=float)
    model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    x_values = [round(float(value), 12) for value in model.X_thresholds_]
    y_values = [round(float(value), 12) for value in model.y_thresholds_]
    if len(x_values) < 2:
        return {}
    return {"x_thresholds": x_values, "y_thresholds": y_values}


def calibrated_probability(raw: float, mapping: dict[str, list[float]]) -> float:
    if not mapping:
        return raw
    x_values = mapping["x_thresholds"]
    y_values = mapping["y_thresholds"]
    if raw <= x_values[0]:
        return y_values[0]
    if raw >= x_values[-1]:
        return y_values[-1]
    for index in range(1, len(x_values)):
        if raw <= x_values[index]:
            width = x_values[index] - x_values[index - 1]
            if width <= 0:
                return y_values[index]
            ratio = (raw - x_values[index - 1]) / width
            return y_values[index - 1] + ratio * (y_values[index] - y_values[index - 1])
    return y_values[-1]


def select_routing_threshold(
    predictions: list[dict[str, Any]],
    mapping: dict[str, list[float]],
    *,
    target_accuracy: float,
    minimum_coverage: float,
) -> dict[str, Any]:
    if not predictions or not mapping:
        return {"target_met": False, "threshold": 1.0, "coverage": 0.0, "accuracy": 0.0}
    scored = [(calibrated_probability(row["raw_confidence"], mapping), row["correct"]) for row in predictions]
    best: dict[str, Any] | None = None
    for threshold in sorted({score for score, _ in scored}):
        selected = [correct for score, correct in scored if score >= threshold]
        coverage = len(selected) / len(scored)
        accuracy = sum(selected) / len(selected) if selected else 0.0
        if accuracy >= target_accuracy and coverage >= minimum_coverage:
            candidate = {
                "target_met": True,
                "threshold": round(float(threshold), 6),
                "coverage": round(coverage, 6),
                "accuracy": round(accuracy, 6),
                "selected_rows": len(selected),
            }
            if best is None or candidate["coverage"] > best["coverage"]:
                best = candidate
    return best or {"target_met": False, "threshold": 1.0, "coverage": 0.0, "accuracy": 0.0}


def prediction_metrics(
    predictions: list[dict[str, Any]], mapping: dict[str, list[float]], routing: dict[str, Any]
) -> dict[str, Any]:
    total = len(predictions)
    threshold = float(routing.get("threshold", 1.0))
    routed = [
        row
        for row in predictions
        if mapping and calibrated_probability(row["raw_confidence"], mapping) >= threshold
    ]
    return {
        "rows": total,
        "top1_accuracy": ratio(sum(row["correct"] for row in predictions), total),
        "top3_accuracy": ratio(sum(row["expected"] in row["candidates"][:3] for row in predictions), total),
        "top5_accuracy": ratio(sum(row["expected"] in row["candidates"][:5] for row in predictions), total),
        "mrr": ratio(sum(reciprocal_rank(row["expected"], row["candidates"]) for row in predictions), total),
        "macro_f1": macro_f1(predictions),
        "auto_assignment_coverage": ratio(len(routed), total),
        "auto_assignment_accuracy": ratio(sum(row["correct"] for row in routed), len(routed)),
        "unable_to_decide_rate": ratio(total - len(routed), total),
        "fallback_trigger_rate": ratio(total - len(routed), total),
    }


def macro_f1(predictions: list[dict[str, Any]]) -> float:
    labels = {str(row["expected"]) for row in predictions}
    labels.update(str(row["candidates"][0]) if row["candidates"] else "manual_triage" for row in predictions)
    scores = []
    for label in labels:
        true_positive = false_positive = false_negative = 0
        for row in predictions:
            expected = str(row["expected"])
            predicted = str(row["candidates"][0]) if row["candidates"] else "manual_triage"
            true_positive += int(expected == label and predicted == label)
            false_positive += int(expected != label and predicted == label)
            false_negative += int(expected == label and predicted != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 6) if scores else 0.0


def class_imbalance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        owner = str(row["assignee"])
        counts[owner] = counts.get(owner, 0) + 1
    values = sorted(counts.values(), reverse=True)
    return {
        "classes": len(values),
        "largest_class_rows": values[0] if values else 0,
        "smallest_class_rows": values[-1] if values else 0,
        "largest_class_share": ratio(values[0] if values else 0, sum(values)),
        "distribution": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def candidate_open_set_artifact(dataset: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "assignee_open_set_detector",
        "name": "rule_based_novelty",
        "deployment_status": "research_only",
        "ranker_confidence_version": "hybrid_ranker_v1",
        "dataset": dataset,
        "threshold": 0.75,
        "parameters": {
            "weights": {
                "unseen_component": 0.25,
                "unseen_product_component_pair": 0.25,
                "rare_component": 0.15,
                "rare_product_component_pair": 0.15,
                "no_component_history": 0.10,
                "no_text_similarity": 0.05,
                "disagreement": 0.05,
            },
            "rare_component_max_rows": 5,
            "rare_product_component_max_rows": 5,
            "ownership_risk_reduction": 0.20,
        },
        "approval_note": "Candidate only. Validate unseen-owner cases before approval.",
    }


def reciprocal_rank(expected: str, candidates: list[str]) -> float:
    try:
        return 1.0 / (candidates.index(expected) + 1)
    except ValueError:
        return 0.0


def ratio(numerator: float, denominator: int) -> float:
    return round(float(numerator) / denominator, 6) if denominator else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
