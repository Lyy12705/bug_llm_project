from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_eligibility import (  # noqa: E402
    annotate_eligible_owner_labels_from_snapshots,
    read_assignee_eligibility_index,
)

from assignee_open_set_common import (
    OPEN_SET_FEATURES,
    apply_top5_assist_policy,
    attach_calibrated_probability,
    attach_open_set_probability,
    build_drift_reference,
    build_leave_assignee_out_folds,
    calibration_metrics,
    fit_portable_logistic,
    open_set_metrics,
    route_predictions,
    routing_metrics,
    routing_score_diagnostics,
    select_multi_window_routing_policy,
    select_multi_window_top5_assist_policy,
    top5_assist_metrics,
    write_json,
    write_jsonl,
)
from calibrate_assignee_ltr import scored_predictions
from evaluate_assignee_open_set_holdout import recommendation_metrics
from train_assignee_ltr import (
    DEFAULT_DATA_DIR,
    CandidateIndex,
    apply_label_map,
    build_semantic_backend,
    canonicalize_source_rows,
    read_jsonl,
    read_label_map,
    exclude_cross_split_overlap,
    parse_timestamp,
    row_sort_key,
)
from train_assignee_open_set import EXPECTED_RANKER_NAME, load_ranker


RAW_DIR = PROJECT_ROOT / "assignee_triage_accuracy" / "paper_grade" / "data" / "raw"
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "phase6_candidate_ltr"
    / "reports"
    / "bmo_public_q4_holdout_source_quota_sbert"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "phase7_rolling_open_set"
    / "reports"
    / "bmo_rolling_through_2021q3"
)
ROLLING_CALIBRATION_FEATURES = (
    "top_probability",
    "probability_margin",
    "one_minus_entropy",
    "component_owner_share",
    "product_component_owner_share",
    "owner_recency",
    "strong_source_agreement",
    "candidate_source_count",
    "component_token_owner_share",
    "reporter_owner_share",
    "file_owner_share",
)
DRIFT_FEATURES = (
    "calibrated_probability",
    "open_set_probability",
    "top_probability",
    "probability_margin",
)


@dataclass
class RollingWindow:
    name: str
    source_path: Path
    history_before: list[dict[str, Any]]
    query_rows: list[dict[str, Any]]
    audit: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train rolling-history calibration and open-set routing across multiple temporal windows."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--expected-ranker-name", default=EXPECTED_RANKER_NAME)
    parser.add_argument(
        "--train", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_history_train.jsonl"
    )
    parser.add_argument("--base-raw", type=Path, default=RAW_DIR / "bmo_public_10k_raw.jsonl")
    parser.add_argument(
        "--validation", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_validation_set.jsonl"
    )
    parser.add_argument("--q3", type=Path, default=RAW_DIR / "bmo_public_future_2020q3_raw.jsonl")
    parser.add_argument("--q4", type=Path, default=RAW_DIR / "bmo_public_future_2020q4_raw.jsonl")
    parser.add_argument("--q1", type=Path, default=RAW_DIR / "bmo_public_future_2021q1_raw.jsonl")
    parser.add_argument("--q2", type=Path, default=RAW_DIR / "bmo_public_future_2021q2_raw.jsonl")
    parser.add_argument("--q3-2021", type=Path, default=RAW_DIR / "bmo_public_future_2021q3_raw.jsonl")
    parser.add_argument(
        "--rolling-window",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override the legacy rolling windows with a repeated ordered NAME=PATH list.",
    )
    parser.add_argument(
        "--fit-window-count",
        type=int,
        default=3,
        help="Leading rolling windows used only to fit calibration/open-set models.",
    )
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-query-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-owner-history", type=int, default=3)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.85)
    parser.add_argument("--target-review-accuracy", type=float, default=0.90)
    parser.add_argument("--target-top5-assist-accuracy", type=float, default=0.85)
    parser.add_argument(
        "--top5-correctness-field",
        choices=("is_top5_correct", "is_top5_eligible_correct"),
        default="is_top5_correct",
        help=(
            "Freeze Top-5 quality against the historical exact owner or a reviewed "
            "eligible-owner set. Eligible mode requires complete time-versioned snapshots."
        ),
    )
    parser.add_argument(
        "--eligibility-roster",
        type=Path,
        action="append",
        default=[],
        help="Repeat for non-overlapping reviewed eligibility snapshots.",
    )
    parser.add_argument("--minimum-top5-assist-coverage", type=float, default=0.10)
    parser.add_argument("--minimum-top5-assist-rows-per-window", type=int, default=100)
    parser.add_argument(
        "--minimum-top5-assist-accuracy-lower-bound", type=float, default=0.80
    )
    parser.add_argument("--minimum-top5-assist-confidence", type=float, default=0.0)
    parser.add_argument("--minimum-auto-coverage", type=float, default=0.10)
    parser.add_argument("--maximum-unseen-auto-rate", type=float, default=0.05)
    parser.add_argument("--minimum-high-confidence", type=float, default=0.50)
    parser.add_argument("--minimum-review-confidence", type=float, default=0.20)
    parser.add_argument("--minimum-review-rows-per-window", type=int, default=1)
    parser.add_argument("--minimum-review-coverage", type=float, default=0.0)
    parser.add_argument("--minimum-candidate-source-count", type=int, default=2)
    parser.add_argument("--minimum-auto-rows-per-window", type=int, default=250)
    parser.add_argument(
        "--minimum-auto-accuracy-lower-bound", type=float, default=0.80
    )
    parser.add_argument(
        "--maximum-unseen-rate-upper-bound", type=float, default=0.05
    )
    parser.add_argument("--minimum-component-auto-rows", type=int, default=30)
    parser.add_argument("--minimum-component-auto-accuracy", type=float, default=0.75)
    parser.add_argument(
        "--minimum-component-auto-accuracy-lower-bound", type=float, default=0.60
    )
    parser.add_argument(
        "--require-review-policy",
        action="store_true",
        help="Make a cross-window Top-3 confirmation policy mandatory for the development gate.",
    )
    parser.add_argument("--maximum-drift-psi", type=float, default=0.25)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--artifact-name", default="assignee_rolling_open_set_logistic_v3")
    args = parser.parse_args()

    probability_arguments = {
        "target auto accuracy": args.target_auto_accuracy,
        "target review accuracy": args.target_review_accuracy,
        "target Top-5 assist accuracy": args.target_top5_assist_accuracy,
        "minimum Top-5 assist coverage": args.minimum_top5_assist_coverage,
        "minimum Top-5 assist accuracy lower bound": (
            args.minimum_top5_assist_accuracy_lower_bound
        ),
        "minimum Top-5 assist confidence": args.minimum_top5_assist_confidence,
        "minimum auto coverage": args.minimum_auto_coverage,
        "maximum unseen auto rate": args.maximum_unseen_auto_rate,
        "minimum auto accuracy lower bound": args.minimum_auto_accuracy_lower_bound,
        "maximum unseen rate upper bound": args.maximum_unseen_rate_upper_bound,
        "minimum component auto accuracy": args.minimum_component_auto_accuracy,
        "minimum component auto accuracy lower bound": (
            args.minimum_component_auto_accuracy_lower_bound
        ),
    }
    if any(not 0.0 <= value <= 1.0 for value in probability_arguments.values()):
        raise SystemExit(
            "routing probability thresholds must be between zero and one"
        )
    if (
        args.minimum_candidate_source_count < 2
        or args.minimum_auto_rows_per_window < 1
        or args.minimum_component_auto_rows < 1
        or args.minimum_top5_assist_rows_per_window < 1
    ):
        raise SystemExit(
            "v3 requires at least two candidate sources and positive sample floors"
        )

    ranker, ranker_artifact = load_ranker(args.model_dir, args.expected_ranker_name)
    requirements = ranker_artifact.get("parameters", {})
    pool_size = int(requirements.get("candidate_pool_size", 30))
    half_life_days = float(requirements.get("half_life_days", 90.0))
    smoothing_alpha = float(requirements.get("smoothing_alpha", 5.0))
    embedding_backend = str(requirements.get("embedding_backend") or "none")
    candidate_generator_version = str(
        requirements.get("candidate_generator_version") or "v2"
    )
    semantic = build_semantic_backend(embedding_backend, args.sbert_model)
    if embedding_backend == "sbert" and not semantic.status.startswith("sbert_local:"):
        raise SystemExit(f"The fixed ranker requires a local SBERT model: {semantic.status}")

    label_map = read_label_map(args.assignee_label_map)
    availability = read_label_availability(args.base_raw)
    base_history = apply_label_availability(load_rows(args.train, label_map), availability)
    default_specs = [
        ("validation", args.validation),
        ("2020_q3", args.q3),
        ("2020_q4", args.q4),
        ("2021_q1", args.q1),
        ("2021_q2", args.q2),
        ("2021_q3", args.q3_2021),
    ]
    window_specs = (
        [parse_named_path(value) for value in args.rolling_window]
        if args.rolling_window
        else default_specs
    )
    if len(window_specs) < 3 or len({name for name, _ in window_specs}) != len(window_specs):
        raise SystemExit("rolling windows require at least three unique ordered names")
    if not 1 <= args.fit_window_count < len(window_specs):
        raise SystemExit("--fit-window-count must leave at least one policy-selection window")
    windows = prepare_rolling_windows(
        base_history,
        window_specs,
        label_map,
        availability,
        strict_duplicate_families=candidate_generator_version == "v3",
    )
    fit_windows = windows[: args.fit_window_count]
    selection_windows = windows[args.fit_window_count :]

    calibration_predictions: list[dict[str, Any]] = []
    normal_fit_by_window: dict[str, list[dict[str, Any]]] = {}
    for window in fit_windows:
        print(f"[rolling] scoring calibration window: {window.name}", flush=True)
        predictions = score_window(
            window,
            ranker,
            pool_size,
            half_life_days,
            smoothing_alpha,
            semantic,
            candidate_generator_version,
        )
        normal_fit_by_window[window.name] = predictions
        calibration_predictions.extend(predictions)
    calibrator = fit_portable_logistic(
        calibration_predictions,
        [int(row["is_top1_correct"]) for row in calibration_predictions],
        ROLLING_CALIBRATION_FEATURES,
        seed=args.seed,
        balanced=False,
    )
    for predictions in normal_fit_by_window.values():
        attach_calibrated_probability(predictions, calibrator)

    detector_training_predictions: list[dict[str, Any]] = []
    leave_owner_audits = []
    for window in fit_windows:
        print(f"[rolling] scoring leave-assignee-out window: {window.name}", flush=True)
        fold = build_leave_assignee_out_folds(
            window.history_before,
            window.query_rows,
            folds=1,
            hidden_query_fraction=args.hidden_query_fraction,
            minimum_history=args.minimum_owner_history,
            seed=args.seed + len(leave_owner_audits),
        )[0]
        reduced_window = RollingWindow(
            name=window.name,
            source_path=window.source_path,
            history_before=fold.history_rows,
            query_rows=fold.query_rows,
            audit=window.audit,
        )
        predictions = score_window(
            reduced_window,
            ranker,
            pool_size,
            half_life_days,
            smoothing_alpha,
            semantic,
            candidate_generator_version,
        )
        attach_calibrated_probability(predictions, calibrator)
        for row in predictions:
            row["window"] = window.name
            row["synthetic_unknown"] = not row["known_owner"]
        detector_training_predictions.extend(predictions)
        leave_owner_audits.append({"window": window.name, **fold.audit})

    detector = fit_portable_logistic(
        detector_training_predictions,
        [int(not row["known_owner"]) for row in detector_training_predictions],
        OPEN_SET_FEATURES,
        seed=args.seed,
        balanced=True,
    )
    attach_open_set_probability(detector_training_predictions, detector)

    selection_predictions: dict[str, list[dict[str, Any]]] = {}
    for window in selection_windows:
        print(f"[rolling] scoring policy-selection window: {window.name}", flush=True)
        predictions = score_window(
            window,
            ranker,
            pool_size,
            half_life_days,
            smoothing_alpha,
            semantic,
            candidate_generator_version,
        )
        attach_calibrated_probability(predictions, calibrator)
        attach_open_set_probability(predictions, detector)
        selection_predictions[window.name] = predictions
    if args.top5_correctness_field == "is_top5_eligible_correct":
        if not args.eligibility_roster:
            raise SystemExit(
                "eligible-owner Top-5 policy requires --eligibility-roster"
            )
        eligibility_indexes = [
            read_assignee_eligibility_index(path)
            for path in args.eligibility_roster
        ]
        for predictions in selection_predictions.values():
            annotate_eligible_owner_labels_from_snapshots(
                predictions, eligibility_indexes
            )
        unlabeled = sum(
            row.get("eligibility_label_available") is not True
            for rows in selection_predictions.values()
            for row in rows
        )
        if unlabeled:
            raise SystemExit(
                "eligible-owner Top-5 policy cannot be frozen: "
                f"{unlabeled} selection rows lack a time-valid reviewed snapshot"
            )
    policy = select_multi_window_routing_policy(
        selection_predictions,
        target_auto_accuracy=args.target_auto_accuracy,
        minimum_auto_coverage=args.minimum_auto_coverage,
        maximum_unseen_auto_rate=args.maximum_unseen_auto_rate,
        target_review_accuracy=args.target_review_accuracy,
        minimum_high_confidence=args.minimum_high_confidence,
        minimum_review_confidence=args.minimum_review_confidence,
        minimum_review_rows_per_window=max(1, args.minimum_review_rows_per_window),
        minimum_review_coverage=max(0.0, args.minimum_review_coverage),
        minimum_candidate_source_count=max(0, args.minimum_candidate_source_count),
        minimum_auto_rows_per_window=max(1, args.minimum_auto_rows_per_window),
        minimum_accuracy_lower_bound=args.minimum_auto_accuracy_lower_bound,
        maximum_unseen_rate_upper_bound=args.maximum_unseen_rate_upper_bound,
        minimum_component_auto_rows=max(1, args.minimum_component_auto_rows),
        minimum_component_accuracy=args.minimum_component_auto_accuracy,
        minimum_component_accuracy_lower_bound=(
            args.minimum_component_auto_accuracy_lower_bound
        ),
    )
    top5_policy = select_multi_window_top5_assist_policy(
        selection_predictions,
        target_accuracy=args.target_top5_assist_accuracy,
        minimum_coverage=args.minimum_top5_assist_coverage,
        minimum_rows_per_window=max(1, args.minimum_top5_assist_rows_per_window),
        minimum_accuracy_lower_bound=(
            args.minimum_top5_assist_accuracy_lower_bound
        ),
        minimum_candidate_source_count=max(0, args.minimum_candidate_source_count),
        minimum_confidence=args.minimum_top5_assist_confidence,
        correctness_field=args.top5_correctness_field,
    )
    for predictions in selection_predictions.values():
        route_predictions(predictions, policy)
        apply_top5_assist_policy(predictions, top5_policy)

    pooled_selection = [row for rows in selection_predictions.values() for row in rows]
    drift_reference = build_drift_reference(
        pooled_selection,
        DRIFT_FEATURES,
        maximum_psi=args.maximum_drift_psi,
    )
    selection_routing = {
        name: routing_metrics(predictions) for name, predictions in selection_predictions.items()
    }
    selection_top5_assist = {
        name: top5_assist_metrics(
            predictions, correctness_field=args.top5_correctness_field
        )
        for name, predictions in selection_predictions.items()
    }
    label_availability = label_availability_audit(
        [
            row
            for window in windows
            for row in [*window.history_before, *window.query_rows]
        ]
    )
    global_development_checks = {
        "candidate_generator_version_v3": candidate_generator_version == "v3",
        "assignment_label_timestamps_explicit": label_availability[
            "all_rows_explicit"
        ],
    }
    selection_window_checks = {
        name: {
            "auto_accuracy": metrics["auto_assignment_accuracy"]
            >= args.target_auto_accuracy,
            "auto_coverage": metrics["auto_assignment_coverage"]
            >= args.minimum_auto_coverage,
            "unseen_point_rate": metrics["unseen_auto_assignment_rate"]
            < args.maximum_unseen_auto_rate,
            "unseen_confidence_upper_bound": metrics[
                "unseen_auto_assignment_rate_ci95"
            ]["upper"]
            < args.maximum_unseen_rate_upper_bound,
            "minimum_auto_rows": metrics["auto_assignment_rows"]
            >= args.minimum_auto_rows_per_window,
            "auto_accuracy_confidence_lower_bound": metrics[
                "auto_assignment_accuracy_ci95"
            ]["lower"]
            >= args.minimum_auto_accuracy_lower_bound,
            "candidate_source_support_frozen": int(
                policy.get("minimum_candidate_source_count", 0)
            )
            >= args.minimum_candidate_source_count,
            "component_safety": bool(
                policy.get("selection_window_metrics", {})
                .get(name, {})
                .get("component_safety", {})
                .get("passed", False)
            ),
        }
        for name, metrics in selection_routing.items()
    }
    selection_passed = bool(
        policy.get("found")
        and all(global_development_checks.values())
        and all(
            all(checks.values()) for checks in selection_window_checks.values()
        )
        and (not args.require_review_policy or policy.get("review_policy_found") is True)
    )
    top5_selection_window_checks = {
        name: {
            "top5_accuracy": metrics["top5_assist_accuracy"]
            >= args.target_top5_assist_accuracy,
            "top5_coverage": metrics["top5_assist_coverage"]
            >= args.minimum_top5_assist_coverage,
            "minimum_top5_rows": metrics["top5_assist_rows"]
            >= args.minimum_top5_assist_rows_per_window,
            "top5_accuracy_confidence_lower_bound": metrics[
                "top5_assist_accuracy_ci95"
            ]["lower"]
            >= args.minimum_top5_assist_accuracy_lower_bound,
            "never_auto_assigns": metrics["auto_assignment_rows"] == 0,
        }
        for name, metrics in selection_top5_assist.items()
    }
    top5_selection_passed = bool(
        top5_policy.get("found")
        and all(global_development_checks.values())
        and all(
            all(checks.values())
            for checks in top5_selection_window_checks.values()
        )
    )

    report = {
        "schema_version": 1,
        "method": "rolling_history_multi_window_open_set_v3",
        "deployment_status": "research_only",
        "ranker": ranker_artifact.get("name"),
        "protocol": {
            "fit_windows": [window.name for window in fit_windows],
            "policy_selection_windows": [window.name for window in selection_windows],
            "window_audits": [window.audit for window in windows],
            "label_availability_policy": (
                "explicit_assignment_event_time_at_or_before_each_history_cutoff"
            ),
            "label_availability_audit": label_availability,
            "assignment_time_history_available": label_availability[
                "all_rows_explicit"
            ],
            "leave_assignee_out_audits": leave_owner_audits,
            "all_temporal_checks_passed": all(window.audit["temporal_order_valid"] for window in windows),
            "all_leave_owner_checks_passed": all(row["leakage_free"] for row in leave_owner_audits),
        },
        "training_rows": {
            "calibration": len(calibration_predictions),
            "open_set_detector": len(detector_training_predictions),
            "open_set_unknown": sum(not row["known_owner"] for row in detector_training_predictions),
        },
        "calibration_features": list(ROLLING_CALIBRATION_FEATURES),
        "open_set_features": list(OPEN_SET_FEATURES),
        "calibration_by_fit_window": {
            name: calibration_metrics(rows) for name, rows in normal_fit_by_window.items()
        },
        "calibration_by_selection_window": {
            name: calibration_metrics(rows) for name, rows in selection_predictions.items()
        },
        "open_set_detector_fit": open_set_metrics(detector_training_predictions),
        "open_set_by_selection_window": {
            name: open_set_metrics(rows, threshold=policy["open_set_threshold"])
            for name, rows in selection_predictions.items()
        },
        "recommendation_by_selection_window": {
            name: recommendation_metrics(rows) for name, rows in selection_predictions.items()
        },
        "recommendation_known_owner_by_selection_window": {
            name: recommendation_metrics([row for row in rows if row["known_owner"]])
            for name, rows in selection_predictions.items()
        },
        "routing_policy": policy,
        "routing_by_selection_window": selection_routing,
        "top5_assist_policy": top5_policy,
        "top5_assist_by_selection_window": selection_top5_assist,
        "top5_assist_development_gate": {
            "passed": top5_selection_passed,
            "requires_every_selection_window": True,
            "global_checks": global_development_checks,
            "selection_window_checks": top5_selection_window_checks,
            "requires_user_selection": True,
            "auto_assignment_authorized": False,
            "blocker": (
                "new_untouched_holdout_required"
                if top5_selection_passed
                else "protocol_provenance_gate_failed"
                if top5_policy.get("found")
                and all(
                    all(checks.values())
                    for checks in top5_selection_window_checks.values()
                )
                else "multi_window_top5_quality_gate_failed"
            ),
        },
        "routing_diagnostics_by_selection_window": {
            name: routing_score_diagnostics(rows, policy) for name, rows in selection_predictions.items()
        },
        "development_gate": {
            "passed": selection_passed,
            "requires_every_selection_window": True,
            "global_checks": global_development_checks,
            "selection_window_checks": selection_window_checks,
            "requires_review_policy": args.require_review_policy,
            "review_policy_found": policy.get("review_policy_found") is True,
            "blocker": (
                "new_untouched_holdout_required" if selection_passed
                else "multi_window_routing_safety_gate_failed"
            ),
        },
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": "assignee_rolling_open_set_routing_bundle",
        "name": args.artifact_name,
        "deployment_status": "research_only",
        "ranker_name": ranker_artifact.get("name"),
        "ranker_requirements": {
            "candidate_pool_size": pool_size,
            "half_life_days": half_life_days,
            "smoothing_alpha": smoothing_alpha,
            "embedding_backend": embedding_backend,
            "candidate_generator_version": candidate_generator_version,
            "sbert_model": args.sbert_model if embedding_backend == "sbert" else "",
        },
        "history_windows": [window.name for window in windows],
        "label_availability_policy": (
            "explicit_assignment_event_time_at_or_before_each_history_cutoff"
        ),
        "label_availability_audit": label_availability,
        "calibrator": calibrator.to_artifact(),
        "open_set_detector": detector.to_artifact(),
        "routing_policy": policy,
        "top5_assist_policy": top5_policy,
        "top5_assist_development_gate": report[
            "top5_assist_development_gate"
        ],
        "drift_reference": drift_reference,
        "development_gate": report["development_gate"],
        "targets": {
            "target_auto_accuracy": args.target_auto_accuracy,
            "minimum_auto_coverage": args.minimum_auto_coverage,
            "maximum_unseen_auto_rate": args.maximum_unseen_auto_rate,
            "target_review_accuracy": args.target_review_accuracy,
            "target_top5_assist_accuracy": args.target_top5_assist_accuracy,
            "minimum_top5_assist_coverage": args.minimum_top5_assist_coverage,
            "minimum_top5_assist_rows_per_window": max(
                1, args.minimum_top5_assist_rows_per_window
            ),
            "minimum_top5_assist_accuracy_lower_bound": (
                args.minimum_top5_assist_accuracy_lower_bound
            ),
            "minimum_review_rows_per_window": max(1, args.minimum_review_rows_per_window),
            "minimum_review_coverage": max(0.0, args.minimum_review_coverage),
            "minimum_candidate_source_count": max(
                0, args.minimum_candidate_source_count
            ),
            "minimum_auto_rows_per_window": max(
                1, args.minimum_auto_rows_per_window
            ),
            "minimum_auto_accuracy_lower_bound": (
                args.minimum_auto_accuracy_lower_bound
            ),
            "maximum_unseen_rate_upper_bound": (
                args.maximum_unseen_rate_upper_bound
            ),
            "minimum_component_auto_rows": max(
                1, args.minimum_component_auto_rows
            ),
            "minimum_component_auto_accuracy": (
                args.minimum_component_auto_accuracy
            ),
            "minimum_component_auto_accuracy_lower_bound": (
                args.minimum_component_auto_accuracy_lower_bound
            ),
        },
        "approval_note": (
            f"Frozen after {windows[-1].name} development. Requires one new untouched temporal holdout, "
            "reviewed active roster, and explicit operator approval."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "rolling_open_set_report.json", report)
    write_json(args.output_dir / "rolling_open_set_artifact.json", artifact)
    write_jsonl(args.output_dir / "detector_training_predictions.jsonl", detector_training_predictions)
    for name, predictions in selection_predictions.items():
        write_jsonl(args.output_dir / f"{name}_routing_predictions.jsonl", predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def prepare_rolling_windows(
    base_history: list[dict[str, Any]],
    window_specs: list[tuple[str, Path]],
    label_map: dict[str, str],
    availability: dict[str, str] | None = None,
    *,
    strict_duplicate_families: bool = False,
) -> list[RollingWindow]:
    history = sorted((dict(row) for row in base_history), key=row_sort_key)
    availability = availability or {}
    output = []
    for name, path in window_specs:
        loaded = apply_label_availability(load_rows(path, label_map), availability)
        rows, overlap_rows_excluded = exclude_cross_split_overlap(
            loaded,
            history,
            strict_duplicate_families=strict_duplicate_families,
        )
        if not rows:
            raise ValueError(f"rolling window {name} is empty after cross-split overlap exclusion")
        query_start_epoch = parse_timestamp(str(rows[0].get("created_at") or ""))
        available_history = []
        unavailable_future_change = 0
        unavailable_missing_time = 0
        for row in history:
            available_epoch = parse_timestamp(str(row.get("label_available_at") or ""))
            if available_epoch is None:
                unavailable_missing_time += 1
            elif query_start_epoch is not None and available_epoch <= query_start_epoch:
                available_history.append(row)
            else:
                unavailable_future_change += 1
        temporal_order_valid = bool(
            available_history and row_sort_key(available_history[-1]) <= row_sort_key(rows[0])
        )
        if not temporal_order_valid:
            raise ValueError(f"rolling window {name} is not later than its history")
        known = {str(row.get("assignee") or "") for row in available_history}
        unseen_rows = sum(str(row.get("assignee") or "") not in known for row in rows)
        audit = {
            "window": name,
            "source": str(path),
            "prior_snapshot_rows": len(history),
            "history_rows": len(available_history),
            "history_rows_withheld_future_last_change": unavailable_future_change,
            "history_rows_withheld_missing_last_change": unavailable_missing_time,
            "query_rows_loaded": len(loaded),
            "query_rows": len(rows),
            "overlap_rows_excluded": overlap_rows_excluded,
            "history_end": str(available_history[-1].get("created_at") or ""),
            "label_availability_cutoff": str(rows[0].get("created_at") or ""),
            "query_start": str(rows[0].get("created_at") or ""),
            "query_end": str(rows[-1].get("created_at") or ""),
            "known_owner_rows": len(rows) - unseen_rows,
            "unseen_owner_rows": unseen_rows,
            "unseen_owner_rate": round(unseen_rows / len(rows), 6),
            "temporal_order_valid": temporal_order_valid,
        }
        output.append(RollingWindow(name, path, list(available_history), rows, audit))
        history.extend(rows)
        history.sort(key=row_sort_key)
    return output


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise SystemExit(f"Expected NAME=PATH rolling window, got: {value}")
    return name.strip(), Path(raw_path.strip())


def score_window(
    window: RollingWindow,
    ranker: Any,
    pool_size: int,
    half_life_days: float,
    smoothing_alpha: float,
    semantic: Any,
    candidate_generator_version: str = "v2",
) -> list[dict[str, Any]]:
    index = CandidateIndex(
        window.history_before,
        half_life_days=half_life_days,
        smoothing_alpha=smoothing_alpha,
        semantic=semantic,
        candidate_generator_version=candidate_generator_version,
    )
    predictions = scored_predictions(window.query_rows, index, ranker, pool_size)
    for row in predictions:
        row["window"] = window.name
        row["history_rows"] = len(window.history_before)
    return predictions


def load_rows(path: Path, label_map: dict[str, str]) -> list[dict[str, Any]]:
    return sorted(
        apply_label_map(canonicalize_source_rows(read_jsonl(path)), label_map),
        key=row_sort_key,
    )


def read_label_availability(path: Path) -> dict[str, str]:
    output = {}
    for row in read_jsonl(path):
        ticket_id = str(row.get("ticket_id") or "").strip()
        if not ticket_id and row.get("id") is not None:
            ticket_id = f"bmo_{row['id']}"
        value = str(row.get("label_available_at") or row.get("last_change_time") or "").strip()
        if ticket_id and value:
            output[ticket_id] = value
    return output


def apply_label_availability(
    rows: list[dict[str, Any]], availability: dict[str, str]
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        explicit = str(
            row.get("assignment_label_available_at")
            or row.get("assignment_changed_at")
            or row.get("assigned_at")
            or ""
        ).strip()
        existing_source = str(row.get("label_availability_source") or "").strip()
        value = str(
            explicit
            or row.get("label_available_at")
            or row.get("last_change_time")
            or availability.get(str(row.get("ticket_id") or ""), "")
        ).strip()
        row["label_available_at"] = value
        if explicit:
            row["label_availability_source"] = "explicit_assignment_event_time"
        elif existing_source:
            row["label_availability_source"] = existing_source
        elif row.get("last_change_time") or availability.get(
            str(row.get("ticket_id") or ""), ""
        ):
            row["label_availability_source"] = "last_change_time_proxy"
        elif value:
            row["label_availability_source"] = "explicit_assignment_event_time"
        else:
            row["label_availability_source"] = "missing"
        output.append(row)
    return output


def label_availability_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize label-time provenance once per ticket for fail-closed v3 gates."""

    priority = {
        "explicit_assignment_event_time": 0,
        "last_change_time_proxy": 1,
        "missing": 2,
    }
    provenance_by_row: dict[str, str] = {}
    for index, row in enumerate(rows):
        ticket_id = str(row.get("ticket_id") or "").strip()
        key = ticket_id or f"__anonymous_{index}"
        source = str(
            row.get("label_availability_source") or "missing"
        ).strip()
        if source not in priority:
            source = "missing"
        current = provenance_by_row.get(key)
        if current is None or priority[source] > priority[current]:
            provenance_by_row[key] = source
    sources = Counter(provenance_by_row.values())
    audited_rows = len(provenance_by_row)
    explicit = sources["explicit_assignment_event_time"]
    proxy = sources["last_change_time_proxy"]
    missing = audited_rows - explicit - proxy
    return {
        "rows": audited_rows,
        "explicit_assignment_event_rows": explicit,
        "last_change_time_proxy_rows": proxy,
        "missing_or_unknown_provenance_rows": missing,
        "explicit_assignment_event_coverage": round(
            explicit / audited_rows, 6
        )
        if audited_rows
        else 0.0,
        "all_rows_explicit": bool(audited_rows and explicit == audited_rows),
    }


if __name__ == "__main__":
    main()
