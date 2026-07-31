from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
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
    BUNDLE_SCHEMA_VERSION,
    apply_assignee_aliases,
    artifact_manifest_entry,
    build_roster,
    canonicalize_assignee_set,
    derive_ownership,
    load_assignee_set,
    load_assignee_alias_map,
    normalize_history_record,
    split_temporal_partition,
    temporal_split,
    temporal_protocol_manifest,
    wilson_interval,
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
        "--assignee-alias-map",
        type=Path,
        default=None,
        help="Reviewed JSON mapping from historical aliases to canonical assignee IDs.",
    )
    parser.add_argument(
        "--inactive-assignees",
        type=Path,
        default=None,
        help="Reviewed JSON list of inactive/departed assignees to exclude before candidate generation.",
    )
    parser.add_argument(
        "--active-assignees",
        type=Path,
        default=None,
        help="Optional reviewed active roster; may include cold-start owners with no training history.",
    )
    parser.add_argument("--ownership-minimum-rows", type=int, default=5)
    parser.add_argument("--ownership-minimum-share", type=float, default=0.70)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.85)
    parser.add_argument("--minimum-auto-coverage", type=float, default=0.10)
    parser.add_argument("--maximum-unseen-auto-rate", type=float, default=0.05)
    parser.add_argument("--maximum-unseen-rate-upper-bound", type=float, default=0.05)
    parser.add_argument("--minimum-calibration-rows", type=int, default=30)
    parser.add_argument(
        "--calibration-fit-ratio",
        type=float,
        default=0.50,
        help="Earlier share of validation used to fit calibration; the later share selects routing thresholds.",
    )
    parser.add_argument("--minimum-test-rows", type=int, default=2500)
    parser.add_argument("--minimum-auto-rows", type=int, default=250)
    parser.add_argument("--minimum-auto-accuracy-lower-bound", type=float, default=0.80)
    parser.add_argument("--minimum-component-auto-rows", type=int, default=30)
    parser.add_argument("--minimum-component-auto-accuracy", type=float, default=0.75)
    parser.add_argument("--minimum-component-auto-accuracy-lower-bound", type=float, default=0.60)
    parser.add_argument("--target-top3-confirmation-accuracy", type=float, default=0.90)
    parser.add_argument("--bundle-valid-days", type=int, default=30)
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
        ("maximum unseen auto rate", args.maximum_unseen_auto_rate),
        ("maximum unseen rate upper bound", args.maximum_unseen_rate_upper_bound),
        ("calibration fit ratio", args.calibration_fit_ratio),
        ("minimum auto accuracy lower bound", args.minimum_auto_accuracy_lower_bound),
        ("minimum component auto accuracy", args.minimum_component_auto_accuracy),
        (
            "minimum component auto accuracy lower bound",
            args.minimum_component_auto_accuracy_lower_bound,
        ),
        ("target top3 confirmation accuracy", args.target_top3_confirmation_accuracy),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be between 0 and 1")
    if not 0.0 < args.calibration_fit_ratio < 1.0:
        raise SystemExit("calibration fit ratio must be strictly between 0 and 1")
    if (
        args.minimum_calibration_rows < 2
        or args.minimum_test_rows < 1
        or args.minimum_auto_rows < 1
        or args.minimum_component_auto_rows < 1
    ):
        raise SystemExit("minimum row requirements must be positive")
    if args.bundle_valid_days < 1:
        raise SystemExit("bundle valid days must be positive")

    raw_rows = read_jsonl(args.history)
    normalized = [row for row in (normalize_history_record(value) for value in raw_rows) if row]
    alias_map = load_assignee_alias_map(args.assignee_alias_map)
    normalized, alias_rows_remapped = apply_assignee_aliases(normalized, alias_map)
    train, validation, test, leakage = temporal_split(normalized, args.train_ratio, args.validation_ratio)
    calibration_fit, policy_selection, validation_split_audit = split_temporal_partition(
        validation, args.calibration_fit_ratio
    )
    inactive = load_assignee_set(args.inactive_assignees)
    reviewed_active = load_assignee_set(args.active_assignees)
    inactive = canonicalize_assignee_set(inactive, alias_map)
    reviewed_active = canonicalize_assignee_set(reviewed_active, alias_map)
    all_train_assignees = {str(row["assignee"]) for row in train}
    roster_payload, roster = build_roster(
        train,
        max(1, args.minimum_assignee_history),
        inactive,
        reviewed_active,
    )
    if not roster:
        raise SystemExit("No assignee meets --minimum-assignee-history; lower it or provide more history.")
    prefilter_train_rows = len(train)
    train = [row for row in train if row["assignee"] in roster]
    roster_payload["review_confirmed"] = bool(args.confirm_roster_reviewed)
    protocol = temporal_protocol_manifest(
        {
            "train": train,
            "calibration_fit": calibration_fit,
            "policy_selection": policy_selection,
            "sealed_test": test,
        },
        source=str(args.history.resolve()),
    )
    protocol["training_filter"] = {
        "rows_before_roster_filter": prefilter_train_rows,
        "rows_after_roster_filter": len(train),
        "excluded_rows": prefilter_train_rows - len(train),
        "minimum_assignee_history": max(1, args.minimum_assignee_history),
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "history": output / "history_train.jsonl",
        "validation": output / "validation_set.jsonl",
        "calibration_fit": output / "calibration_fit_set.jsonl",
        "policy_selection": output / "policy_selection_set.jsonl",
        "test": output / "test_set.jsonl",
        "protocol": output / "temporal_protocol_manifest.json",
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
    write_jsonl(paths["calibration_fit"], calibration_fit)
    write_jsonl(paths["policy_selection"], policy_selection)
    write_jsonl(paths["test"], test)
    write_json(paths["protocol"], protocol)
    write_json(paths["roster"], roster_payload)

    component_ownership, file_ownership = derive_ownership(
        train, roster, max(1, args.ownership_minimum_rows), args.ownership_minimum_share
    )
    write_json(paths["component_ownership"], component_ownership)
    write_json(paths["file_ownership"], file_ownership)

    calibration_predictions = predict(
        paths["history"], paths["roster"], calibration_fit, roster, all_train_assignees, inactive
    )
    policy_predictions = predict(
        paths["history"], paths["roster"], policy_selection, roster, all_train_assignees, inactive
    )
    test_predictions = predict(
        paths["history"], paths["roster"], test, roster, all_train_assignees, inactive
    )
    calibration = fit_calibration(calibration_predictions, args.minimum_calibration_rows)
    routing = select_routing_threshold(
        policy_predictions,
        calibration,
        target_accuracy=args.target_auto_accuracy,
        minimum_coverage=args.minimum_auto_coverage,
        maximum_unseen_auto_rate=args.maximum_unseen_auto_rate,
        target_review_accuracy=args.target_top3_confirmation_accuracy,
        minimum_candidate_source_count=2,
    )
    calibration_metrics = prediction_metrics(calibration_predictions, calibration, routing)
    policy_metrics = prediction_metrics(policy_predictions, calibration, routing)
    test_metrics = prediction_metrics(test_predictions, calibration, routing)
    test_gate = selective_routing_gate(
        test_metrics,
        target_accuracy=args.target_auto_accuracy,
        minimum_coverage=args.minimum_auto_coverage,
        maximum_unseen_auto_rate=args.maximum_unseen_auto_rate,
        minimum_rows=args.minimum_test_rows,
        minimum_auto_rows=args.minimum_auto_rows,
        minimum_accuracy_lower_bound=args.minimum_auto_accuracy_lower_bound,
        minimum_component_auto_rows=args.minimum_component_auto_rows,
        minimum_component_accuracy=args.minimum_component_auto_accuracy,
        minimum_component_accuracy_lower_bound=args.minimum_component_auto_accuracy_lower_bound,
        maximum_unseen_rate_upper_bound=args.maximum_unseen_rate_upper_bound,
    )
    gates = {
        "enough_calibration_rows": len(calibration_predictions) >= args.minimum_calibration_rows,
        "both_correct_and_incorrect_calibration_examples": len(
            {row["correct"] for row in calibration_predictions}
        )
        == 2,
        "policy_selection_constraints_met": bool(routing.get("target_met")),
        "frozen_test_gate_passed": bool(test_gate["passed"]),
        "active_roster_review_confirmed": bool(args.confirm_roster_reviewed),
        "upstream_duplicate_filter_confirmed": bool(args.confirm_upstream_duplicate_filtered),
        "unique_ticket_ids": leakage["non_unique_ticket_ids"] == 0,
        "no_cross_split_ticket_id_overlap": leakage["cross_split_ticket_id_overlap"] == 0,
        "no_cross_split_content_overlap": leakage["cross_split_content_overlap"] == 0,
        "chronological_split_valid": bool(leakage["chronological_order_valid"]),
        "validation_subsplit_chronological": bool(validation_split_audit["chronological_order_valid"]),
        "temporal_protocol_leakage_free": bool(protocol["audit"]["leakage_free"]),
    }
    approval_gate_passed = bool(all(gates.values()) and calibration)
    approved = bool(args.approve and approval_gate_passed)
    deployment_status = "approved" if approved else "research_only"
    created_at = datetime.now(UTC)
    expires_at = created_at + timedelta(days=args.bundle_valid_days)

    calibrator_artifact = {
        "schema_version": 1,
        "artifact_type": "assignee_confidence_calibrator",
        "name": "isotonic_regression",
        "deployment_status": deployment_status,
        "runtime_supported": bool(calibration),
        "approval_gate_passed": approval_gate_passed,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "ranker_confidence_version": "hybrid_ranker_v1",
        "dataset": args.dataset,
        "mapping": calibration,
        "calibration_fit_metrics": calibration_metrics,
        "policy_selection_metrics": policy_metrics,
        "frozen_test_metrics": test_metrics,
        "approval_gates": gates,
    }
    write_json(paths["calibrator"], calibrator_artifact)
    policy = {
        "schema_version": 1,
        "name": "project_selective_routing",
        "deployment_status": deployment_status,
        "approval_gate_passed": approval_gate_passed,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "t_high": routing.get("t_high", routing.get("threshold", 1.0)),
        "t_low": routing.get("t_low", routing.get("threshold", 1.0)),
        "open_set_threshold": routing.get("open_set_threshold", 0.0),
        "require_active_owner": True,
        "minimum_candidate_source_count": 2,
        "target_auto_accuracy": args.target_auto_accuracy,
        "minimum_auto_coverage": args.minimum_auto_coverage,
        "maximum_unseen_auto_rate": args.maximum_unseen_auto_rate,
        "selection_split": "policy_selection",
        "frozen_test_gate": test_gate,
    }
    write_json(paths["policy"], policy)
    write_json(
        paths["open_set"],
        candidate_open_set_artifact(
            args.dataset,
            deployment_status="research_only",
            approval_gate_passed=False,
            threshold=float(routing.get("open_set_threshold", 0.0)),
            created_at=created_at,
            expires_at=expires_at,
        ),
    )

    pipeline_config = {
        "assignee_dataset_path": "history_train.jsonl",
        "assignee_active_roster_path": "active_assignees.json",
        "assignee_feedback_path": "assignee_feedback.jsonl",
        "assignee_routing_policy_path": "routing_policy.json" if approved else None,
        "assignee_routing_policy_name": "project_selective_routing" if approved else "",
        "assignee_calibration_artifact_path": "assignee_confidence_calibrator.json" if approved else None,
        "assignee_open_set_enabled": True,
        "assignee_open_set_risk_threshold": float(routing.get("open_set_threshold", 0.0)),
        "assignee_open_set_artifact_path": None,
        "assignee_allow_uncalibrated_auto_assignment": False,
    }
    artifact_manifest = {}
    if approved:
        artifact_manifest = {
            "assignee_dataset_path": artifact_manifest_entry(paths["history"], output),
            "assignee_active_roster_path": artifact_manifest_entry(paths["roster"], output),
            "assignee_routing_policy_path": artifact_manifest_entry(paths["policy"], output),
            "assignee_calibration_artifact_path": artifact_manifest_entry(paths["calibrator"], output),
        }
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_type": "assignee_deployment_bundle",
        "dataset": args.dataset,
        "deployment_status": deployment_status,
        "approval_gate_passed": approval_gate_passed,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "pipeline_config": pipeline_config,
        "artifact_manifest": artifact_manifest,
        "runtime_open_set_detector": "fallback_rule_based_v1",
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
        "approval_gate_passed": approval_gate_passed,
        "raw_rows": len(raw_rows),
        "normalized_rows": len(normalized),
        "assignee_aliases": len(alias_map),
        "assignee_alias_rows_remapped": alias_rows_remapped,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "calibration_fit_rows": len(calibration_fit),
        "policy_selection_rows": len(policy_selection),
        "test_rows": len(test),
        "roster_size": len(roster),
        "reviewed_active_roster_rows": len(reviewed_active),
        "long_tail_or_inactive_train_rows_excluded": prefilter_train_rows - len(train),
        "leakage_audit": leakage,
        "validation_split_audit": validation_split_audit,
        "temporal_protocol": protocol,
        "approval_gates": gates,
        "frozen_test_gate": test_gate,
        "routing_selection": routing,
        "calibration_fit_metrics": calibration_metrics,
        "policy_selection_metrics": policy_metrics,
        "frozen_test_metrics": test_metrics,
        "class_imbalance": class_imbalance(train),
    }
    write_json(paths["bundle"], bundle)
    write_json(paths["report"], report)
    print(json.dumps({"bundle": str(paths["bundle"]), **report}, ensure_ascii=False, indent=2))


def predict(
    history_path: Path,
    roster_path: Path,
    rows: list[dict[str, Any]],
    active_roster: set[str],
    historical_assignees: set[str],
    inactive_assignees: set[str],
) -> list[dict[str, Any]]:
    config = PipelineConfig(
        project_root=SYSTEM_ROOT,
        assignee_dataset_path=history_path,
        assignee_active_roster_path=roster_path,
        assignee_allow_uncalibrated_auto_assignment=True,
        assignee_open_set_enabled=True,
        assignee_open_set_risk_threshold=1.0,
        assignee_top_k=5,
        save_checkpoints=False,
    )
    triager = AssigneeTriager(config=config)
    predictions = []
    for row in rows:
        result = triager.assign(row, {"predicted_priority": row.get("priority", "P3")})
        candidates = [owner for owner in result.get("ranked_candidates", []) if owner != "manual_triage"]
        top_candidate = candidates[0] if candidates else ""
        expected = str(row["assignee"])
        if expected in active_roster:
            owner_status = "known_active" if expected in historical_assignees else "cold_start_known"
        elif expected in inactive_assignees:
            owner_status = "inactive_or_transferred"
        elif expected in historical_assignees:
            owner_status = "long_tail_not_deployable"
        else:
            owner_status = "unresolvable_owner"
        details = result.get("candidate_details") or []
        top_signals = details[0].get("signals", []) if details and isinstance(details[0], dict) else []
        predictions.append(
            {
                "ticket_id": row["ticket_id"],
                "component": row.get("component") or "unknown",
                "expected": expected,
                "candidates": candidates,
                "raw_confidence": float(result.get("raw_confidence") or 0.0),
                "correct": bool(candidates and candidates[0] == expected),
                "top3_correct": expected in candidates[:3],
                "known_owner": expected in active_roster,
                "owner_status": owner_status,
                "predicted_owner_active": bool(top_candidate and top_candidate in active_roster),
                "candidate_source_count": len(set(str(value) for value in top_signals)),
                "open_set_risk": float(result.get("open_set_risk") or 0.0),
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
    maximum_unseen_auto_rate: float,
    target_review_accuracy: float,
    minimum_candidate_source_count: int,
) -> dict[str, Any]:
    if not predictions or not mapping:
        return {
            "target_met": False,
            "threshold": 1.0,
            "t_high": 1.0,
            "t_low": 1.0,
            "open_set_threshold": 0.0,
            "coverage": 0.0,
            "accuracy": 0.0,
        }
    scored = [
        {
            **row,
            "calibrated_probability": calibrated_probability(row["raw_confidence"], mapping),
        }
        for row in predictions
    ]
    risk_thresholds = _threshold_candidates(
        [float(row["open_set_risk"]) for row in scored], include=(0.0, 1.0)
    )
    confidence_thresholds = _threshold_candidates(
        [float(row["calibrated_probability"]) for row in scored], include=(1.0,)
    )
    unknown_total = sum(not row["known_owner"] for row in scored)
    best: dict[str, Any] | None = None
    for risk_threshold in risk_thresholds:
        for threshold in confidence_thresholds:
            selected = [
                row
                for row in scored
                if row["predicted_owner_active"]
                and int(row["candidate_source_count"]) >= minimum_candidate_source_count
                and float(row["open_set_risk"]) < risk_threshold
                and float(row["calibrated_probability"]) >= threshold
            ]
            if not selected:
                continue
            correct = sum(bool(row["correct"]) for row in selected)
            coverage = len(selected) / len(scored)
            accuracy = correct / len(selected)
            unseen_auto = sum(not row["known_owner"] for row in selected)
            unseen_rate = unseen_auto / unknown_total if unknown_total else 0.0
            if not (
                accuracy >= target_accuracy
                and coverage >= minimum_coverage
                and unseen_rate < maximum_unseen_auto_rate
            ):
                continue
            candidate = {
                "target_met": True,
                "threshold": round(float(threshold), 6),
                "t_high": round(float(threshold), 6),
                "open_set_threshold": round(float(risk_threshold), 6),
                "coverage": round(coverage, 6),
                "accuracy": round(accuracy, 6),
                "selected_rows": len(selected),
                "correct_rows": correct,
                "accuracy_ci95": wilson_interval(correct, len(selected)),
                "unseen_auto_rows": unseen_auto,
                "unseen_auto_rate": round(unseen_rate, 6),
                "minimum_candidate_source_count": minimum_candidate_source_count,
            }
            if best is None or (
                candidate["coverage"],
                candidate["accuracy"],
                -candidate["unseen_auto_rate"],
            ) > (
                best["coverage"],
                best["accuracy"],
                -best["unseen_auto_rate"],
            ):
                best = candidate
    if best is None:
        return {
            "target_met": False,
            "threshold": 1.0,
            "t_high": 1.0,
            "t_low": 1.0,
            "open_set_threshold": 0.0,
            "coverage": 0.0,
            "accuracy": 0.0,
            "blocker": "policy_selection_constraints_not_met",
        }

    review_pool = [
        row
        for row in scored
        if row["predicted_owner_active"]
        and int(row["candidate_source_count"]) >= minimum_candidate_source_count
        and float(row["open_set_risk"]) < best["open_set_threshold"]
        and float(row["calibrated_probability"]) < best["t_high"]
    ]
    best_review: dict[str, Any] | None = None
    for threshold in _threshold_candidates(
        [float(row["calibrated_probability"]) for row in review_pool],
        include=(best["t_high"],),
    ):
        selected = [row for row in review_pool if row["calibrated_probability"] >= threshold]
        if not selected:
            continue
        accuracy = sum(bool(row["top3_correct"]) for row in selected) / len(selected)
        if accuracy < target_review_accuracy:
            continue
        candidate = {
            "t_low": round(float(threshold), 6),
            "top3_confirmation_rows": len(selected),
            "top3_confirmation_accuracy": round(accuracy, 6),
        }
        if best_review is None or candidate["top3_confirmation_rows"] > best_review["top3_confirmation_rows"]:
            best_review = candidate
    best.update(
        best_review
        or {
            "t_low": best["t_high"],
            "top3_confirmation_rows": 0,
            "top3_confirmation_accuracy": 0.0,
        }
    )
    return best


def prediction_metrics(
    predictions: list[dict[str, Any]], mapping: dict[str, list[float]], routing: dict[str, Any]
) -> dict[str, Any]:
    total = len(predictions)
    high = float(routing.get("t_high", routing.get("threshold", 1.0)))
    low = float(routing.get("t_low", high))
    risk_threshold = float(routing.get("open_set_threshold", 0.0))
    minimum_sources = int(routing.get("minimum_candidate_source_count", 2))
    scored = [
        {
            **row,
            "calibrated_probability": calibrated_probability(row["raw_confidence"], mapping),
        }
        for row in predictions
    ]
    routed = [
        row
        for row in scored
        if mapping
        and row["predicted_owner_active"]
        and int(row["candidate_source_count"]) >= minimum_sources
        and float(row["open_set_risk"]) < risk_threshold
        and float(row["calibrated_probability"]) >= high
    ]
    reviewed = [
        row
        for row in scored
        if row not in routed
        and mapping
        and row["predicted_owner_active"]
        and int(row["candidate_source_count"]) >= minimum_sources
        and float(row["open_set_risk"]) < risk_threshold
        and float(row["calibrated_probability"]) >= low
    ]
    correct_auto = sum(bool(row["correct"]) for row in routed)
    unseen = [row for row in scored if not row["known_owner"]]
    unseen_auto = [row for row in routed if not row["known_owner"]]
    return {
        "rows": total,
        "known_rows": total - len(unseen),
        "unseen_rows": len(unseen),
        "owner_status_breakdown": dict(sorted(Counter(row["owner_status"] for row in scored).items())),
        "candidate_source_count_breakdown": dict(
            sorted(Counter(str(row["candidate_source_count"]) for row in scored).items())
        ),
        "top1_accuracy": ratio(sum(row["correct"] for row in scored), total),
        "top3_accuracy": ratio(sum(row["expected"] in row["candidates"][:3] for row in scored), total),
        "top5_accuracy": ratio(sum(row["expected"] in row["candidates"][:5] for row in scored), total),
        "mrr": ratio(sum(reciprocal_rank(row["expected"], row["candidates"]) for row in scored), total),
        "macro_f1": macro_f1(scored),
        "auto_assignment_rows": len(routed),
        "auto_assignment_coverage": ratio(len(routed), total),
        "auto_assignment_accuracy": ratio(correct_auto, len(routed)),
        "auto_assignment_accuracy_ci95": wilson_interval(correct_auto, len(routed)),
        "unseen_auto_assignment_rows": len(unseen_auto),
        "unseen_auto_assignment_rate": ratio(len(unseen_auto), len(unseen)),
        "unseen_auto_assignment_rate_ci95": wilson_interval(len(unseen_auto), len(unseen)),
        "top3_confirmation_rows": len(reviewed),
        "top3_confirmation_coverage": ratio(len(reviewed), total),
        "top3_confirmation_accuracy": ratio(
            sum(bool(row["top3_correct"]) for row in reviewed), len(reviewed)
        ),
        "component_metrics": component_routing_metrics(scored, routed),
        "unable_to_decide_rate": ratio(total - len(routed), total),
        "fallback_trigger_rate": ratio(total - len(routed), total),
    }


def component_routing_metrics(
    scored: list[dict[str, Any]], routed: list[dict[str, Any]]
) -> dict[str, Any]:
    row_counts = Counter(str(row.get("component") or "unknown") for row in scored)
    auto_counts = Counter(str(row.get("component") or "unknown") for row in routed)
    correct_counts = Counter(
        str(row.get("component") or "unknown") for row in routed if bool(row.get("correct"))
    )
    return {
        component: {
            "rows": rows,
            "auto_assignment_rows": auto_counts[component],
            "auto_assignment_coverage": ratio(auto_counts[component], rows),
            "auto_assignment_accuracy": ratio(
                correct_counts[component], auto_counts[component]
            ),
            "auto_assignment_accuracy_ci95": wilson_interval(
                correct_counts[component], auto_counts[component]
            ),
        }
        for component, rows in sorted(row_counts.items())
    }


def selective_routing_gate(
    metrics: dict[str, Any],
    *,
    target_accuracy: float,
    minimum_coverage: float,
    maximum_unseen_auto_rate: float,
    minimum_rows: int,
    minimum_auto_rows: int,
    minimum_accuracy_lower_bound: float,
    minimum_component_auto_rows: int = 30,
    minimum_component_accuracy: float = 0.75,
    minimum_component_accuracy_lower_bound: float = 0.60,
    maximum_unseen_rate_upper_bound: float = 0.05,
) -> dict[str, Any]:
    component_metrics = metrics.get("component_metrics")
    component_metrics_available = isinstance(component_metrics, dict) and bool(component_metrics)
    component_failures = {
        name: values
        for name, values in (component_metrics.items() if component_metrics_available else ())
        if values.get("auto_assignment_rows", 0) >= minimum_component_auto_rows
        and values.get("auto_assignment_accuracy", 0.0) < minimum_component_accuracy
    }
    component_confidence_failures = {
        name: values
        for name, values in (component_metrics.items() if component_metrics_available else ())
        if values.get("auto_assignment_rows", 0) >= minimum_component_auto_rows
        and values.get("auto_assignment_accuracy_ci95", {"lower": 0.0})["lower"]
        < minimum_component_accuracy_lower_bound
    }
    checks = {
        "auto_assignment_accuracy": metrics["auto_assignment_accuracy"] >= target_accuracy,
        "auto_assignment_coverage": metrics["auto_assignment_coverage"] >= minimum_coverage,
        "unseen_auto_assignment_rate": metrics["unseen_auto_assignment_rate"] < maximum_unseen_auto_rate,
        "unseen_auto_assignment_rate_confidence_upper_bound": metrics.get(
            "unseen_auto_assignment_rate_ci95", {"upper": 1.0}
        )["upper"]
        < maximum_unseen_rate_upper_bound,
        "minimum_test_rows": metrics["rows"] >= minimum_rows,
        "minimum_auto_rows": metrics["auto_assignment_rows"] >= minimum_auto_rows,
        "auto_accuracy_confidence_lower_bound": metrics["auto_assignment_accuracy_ci95"]["lower"]
        >= minimum_accuracy_lower_bound,
        "component_metrics_available": component_metrics_available,
        "component_accuracy_floor": not component_failures,
        "component_accuracy_confidence_floor": not component_confidence_failures,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "component_failures": component_failures,
        "component_confidence_failures": component_confidence_failures,
        "targets": {
            "target_auto_accuracy": target_accuracy,
            "minimum_auto_coverage": minimum_coverage,
            "maximum_unseen_auto_rate": maximum_unseen_auto_rate,
            "minimum_test_rows": minimum_rows,
            "minimum_auto_rows": minimum_auto_rows,
            "minimum_accuracy_lower_bound": minimum_accuracy_lower_bound,
            "minimum_component_auto_rows": minimum_component_auto_rows,
            "minimum_component_accuracy": minimum_component_accuracy,
            "minimum_component_accuracy_lower_bound": minimum_component_accuracy_lower_bound,
            "maximum_unseen_rate_upper_bound": maximum_unseen_rate_upper_bound,
        },
    }


def _threshold_candidates(values: list[float], *, include: tuple[float, ...]) -> list[float]:
    if not values:
        return sorted(set(include))
    array = np.asarray(values, dtype=float)
    quantiles = np.linspace(0.0, 1.0, min(101, len(array)))
    candidates = {float(value) for value in np.quantile(array, quantiles)} | set(include)
    candidates.update(min(1.0, float(value) + 0.000001) for value in array)
    return sorted(candidates)


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


def candidate_open_set_artifact(
    dataset: str,
    *,
    deployment_status: str,
    approval_gate_passed: bool,
    threshold: float,
    created_at: datetime,
    expires_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "assignee_open_set_detector",
        "name": "rule_based_novelty",
        "deployment_status": deployment_status,
        "approval_gate_passed": approval_gate_passed,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "ranker_confidence_version": "hybrid_ranker_v1",
        "dataset": dataset,
        "threshold": max(0.0, min(1.0, threshold)),
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
        "approval_note": (
            "Candidate only and not loaded by the approved bundle. Train and validate a detector "
            "against the same score definition before replacing fallback_rule_based_v1."
        ),
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
