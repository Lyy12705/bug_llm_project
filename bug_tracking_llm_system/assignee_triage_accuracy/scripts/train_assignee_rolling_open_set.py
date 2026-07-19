from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assignee_open_set_common import (
    OPEN_SET_FEATURES,
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-query-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-owner-history", type=int, default=3)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.85)
    parser.add_argument("--target-review-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-auto-coverage", type=float, default=0.10)
    parser.add_argument("--maximum-unseen-auto-rate", type=float, default=0.05)
    parser.add_argument("--minimum-high-confidence", type=float, default=0.50)
    parser.add_argument("--minimum-review-confidence", type=float, default=0.20)
    parser.add_argument("--maximum-drift-psi", type=float, default=0.25)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    ranker, ranker_artifact = load_ranker(args.model_dir, args.expected_ranker_name)
    requirements = ranker_artifact.get("parameters", {})
    pool_size = int(requirements.get("candidate_pool_size", 30))
    half_life_days = float(requirements.get("half_life_days", 90.0))
    smoothing_alpha = float(requirements.get("smoothing_alpha", 5.0))
    embedding_backend = str(requirements.get("embedding_backend") or "none")
    semantic = build_semantic_backend(embedding_backend, args.sbert_model)
    if embedding_backend == "sbert" and not semantic.status.startswith("sbert_local:"):
        raise SystemExit(f"The fixed ranker requires a local SBERT model: {semantic.status}")

    label_map = read_label_map(args.assignee_label_map)
    availability = read_label_availability(args.base_raw)
    base_history = apply_label_availability(load_rows(args.train, label_map), availability)
    windows = prepare_rolling_windows(
        base_history,
        [
            ("validation", args.validation),
            ("2020_q3", args.q3),
            ("2020_q4", args.q4),
            ("2021_q1", args.q1),
            ("2021_q2", args.q2),
            ("2021_q3", args.q3_2021),
        ],
        label_map,
        availability,
    )
    fit_windows = windows[:3]
    selection_windows = windows[3:]

    calibration_predictions: list[dict[str, Any]] = []
    normal_fit_by_window: dict[str, list[dict[str, Any]]] = {}
    for window in fit_windows:
        predictions = score_window(
            window, ranker, pool_size, half_life_days, smoothing_alpha, semantic
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
            reduced_window, ranker, pool_size, half_life_days, smoothing_alpha, semantic
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
        predictions = score_window(
            window, ranker, pool_size, half_life_days, smoothing_alpha, semantic
        )
        attach_calibrated_probability(predictions, calibrator)
        attach_open_set_probability(predictions, detector)
        selection_predictions[window.name] = predictions
    policy = select_multi_window_routing_policy(
        selection_predictions,
        target_auto_accuracy=args.target_auto_accuracy,
        minimum_auto_coverage=args.minimum_auto_coverage,
        maximum_unseen_auto_rate=args.maximum_unseen_auto_rate,
        target_review_accuracy=args.target_review_accuracy,
        minimum_high_confidence=args.minimum_high_confidence,
        minimum_review_confidence=args.minimum_review_confidence,
    )
    for predictions in selection_predictions.values():
        route_predictions(predictions, policy)

    pooled_selection = [row for rows in selection_predictions.values() for row in rows]
    drift_reference = build_drift_reference(
        pooled_selection,
        DRIFT_FEATURES,
        maximum_psi=args.maximum_drift_psi,
    )
    selection_routing = {
        name: routing_metrics(predictions) for name, predictions in selection_predictions.items()
    }
    selection_passed = bool(
        policy.get("found")
        and all(
            metrics["auto_assignment_accuracy"] >= args.target_auto_accuracy
            and metrics["auto_assignment_coverage"] >= args.minimum_auto_coverage
            and metrics["unseen_auto_assignment_rate"] <= args.maximum_unseen_auto_rate
            for metrics in selection_routing.values()
        )
    )

    report = {
        "schema_version": 1,
        "method": "rolling_history_multi_window_open_set_v1",
        "deployment_status": "research_only",
        "ranker": ranker_artifact.get("name"),
        "protocol": {
            "fit_windows": [window.name for window in fit_windows],
            "policy_selection_windows": [window.name for window in selection_windows],
            "window_audits": [window.audit for window in windows],
            "label_availability_policy": "last_change_time_at_or_before_query_start",
            "assignment_time_history_available": False,
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
        "routing_diagnostics_by_selection_window": {
            name: routing_score_diagnostics(rows, policy) for name, rows in selection_predictions.items()
        },
        "development_gate": {
            "passed": selection_passed,
            "requires_every_selection_window": True,
            "blocker": (
                "new_untouched_holdout_required" if selection_passed
                else "multi_window_routing_safety_gate_failed"
            ),
        },
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": "assignee_rolling_open_set_routing_bundle",
        "name": "assignee_rolling_open_set_logistic_v1",
        "deployment_status": "research_only",
        "ranker_name": ranker_artifact.get("name"),
        "ranker_requirements": {
            "candidate_pool_size": pool_size,
            "half_life_days": half_life_days,
            "smoothing_alpha": smoothing_alpha,
            "embedding_backend": embedding_backend,
            "sbert_model": args.sbert_model if embedding_backend == "sbert" else "",
        },
        "history_windows": [window.name for window in windows],
        "label_availability_policy": "last_change_time_at_or_before_query_start",
        "calibrator": calibrator.to_artifact(),
        "open_set_detector": detector.to_artifact(),
        "routing_policy": policy,
        "drift_reference": drift_reference,
        "development_gate": report["development_gate"],
        "targets": {
            "target_auto_accuracy": args.target_auto_accuracy,
            "minimum_auto_coverage": args.minimum_auto_coverage,
            "maximum_unseen_auto_rate": args.maximum_unseen_auto_rate,
        },
        "approval_note": (
            "Frozen after 2021 Q3 development. Requires one new untouched temporal holdout, "
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
) -> list[RollingWindow]:
    history = sorted((dict(row) for row in base_history), key=row_sort_key)
    availability = availability or {}
    output = []
    for name, path in window_specs:
        loaded = apply_label_availability(load_rows(path, label_map), availability)
        rows, overlap_rows_excluded = exclude_cross_split_overlap(loaded, history)
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


def score_window(
    window: RollingWindow,
    ranker: Any,
    pool_size: int,
    half_life_days: float,
    smoothing_alpha: float,
    semantic: Any,
) -> list[dict[str, Any]]:
    index = CandidateIndex(
        window.history_before,
        half_life_days=half_life_days,
        smoothing_alpha=smoothing_alpha,
        semantic=semantic,
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
        value = str(
            row.get("label_available_at")
            or row.get("last_change_time")
            or availability.get(str(row.get("ticket_id") or ""), "")
        ).strip()
        row["label_available_at"] = value
        output.append(row)
    return output


if __name__ == "__main__":
    main()
