from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from assignee_open_set_common import (
    calibration_metrics,
    deployment_gate,
    evaluate_score_drift,
    open_set_metrics,
    predict_portable_logistic,
    route_predictions,
    routing_metrics,
    routing_score_diagnostics,
    write_json,
    write_jsonl,
)
from evaluate_assignee_open_set_holdout import recommendation_metrics
from train_assignee_ltr import (
    DEFAULT_DATA_DIR,
    build_semantic_backend,
    read_label_map,
)
from train_assignee_open_set import EXPECTED_RANKER_NAME
from train_assignee_rolling_open_set import (
    DEFAULT_MODEL_DIR,
    RAW_DIR,
    RollingWindow,
    apply_label_availability,
    load_rows,
    prepare_rolling_windows,
    read_label_availability,
    score_window,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "phase7_rolling_open_set"
    / "reports"
    / "bmo_rolling_through_2021q3"
    / "rolling_open_set_artifact.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen rolling-history assignee routing bundle on a new holdout."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--routing-artifact", type=Path, default=DEFAULT_ARTIFACT)
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
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--holdout-name", default="new_holdout")
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-untouched-holdout", action="store_true")
    args = parser.parse_args()

    if not args.confirm_untouched_holdout:
        raise SystemExit("--confirm-untouched-holdout is required")
    artifact = json.loads(args.routing_artifact.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "assignee_rolling_open_set_routing_bundle":
        raise SystemExit("Unsupported rolling open-set artifact")
    if artifact.get("deployment_status") != "research_only":
        raise SystemExit("Holdout evaluation expects a research-only artifact")
    if not artifact.get("development_gate", {}).get("passed"):
        raise SystemExit("The rolling development gate did not pass")
    if artifact.get("ranker_name") != EXPECTED_RANKER_NAME:
        raise SystemExit("Frozen bundle does not reference the expected shallow ranker")

    ranker_artifact = json.loads(
        (args.model_dir / "candidate_ltr_artifact.json").read_text(encoding="utf-8")
    )
    if ranker_artifact.get("name") != artifact.get("ranker_name"):
        raise SystemExit("Ranker model directory does not match the rolling bundle")
    ranker = joblib.load(args.model_dir / "candidate_ltr_model.joblib")
    requirements = artifact["ranker_requirements"]
    semantic = build_semantic_backend(
        str(requirements.get("embedding_backend") or "none"),
        str(requirements.get("sbert_model") or "sentence-transformers/all-MiniLM-L6-v2"),
    )
    if requirements.get("embedding_backend") == "sbert" and not semantic.status.startswith("sbert_local:"):
        raise SystemExit(f"The frozen ranker requires a local SBERT model: {semantic.status}")

    label_map = read_label_map(args.assignee_label_map)
    availability = read_label_availability(args.base_raw)
    base_history = apply_label_availability(load_rows(args.train, label_map), availability)
    all_windows = prepare_rolling_windows(
        base_history,
        [
            ("validation", args.validation),
            ("2020_q3", args.q3),
            ("2020_q4", args.q4),
            ("2021_q1", args.q1),
            ("2021_q2", args.q2),
            ("2021_q3", args.q3_2021),
            (args.holdout_name, args.holdout),
        ],
        label_map,
        availability,
    )
    windows = all_windows[:-1]
    if [window.name for window in windows] != artifact.get("history_windows"):
        raise SystemExit("Rolling history windows do not match the frozen artifact")
    window = all_windows[-1]
    predictions = score_window(
        window,
        ranker,
        int(requirements["candidate_pool_size"]),
        float(requirements["half_life_days"]),
        float(requirements["smoothing_alpha"]),
        semantic,
    )
    for row, probability in zip(
        predictions,
        predict_portable_logistic(predictions, artifact["calibrator"]),
        strict=True,
    ):
        row["calibrated_probability"] = round(float(probability), 8)
    for row, probability in zip(
        predictions,
        predict_portable_logistic(predictions, artifact["open_set_detector"]),
        strict=True,
    ):
        row["open_set_probability"] = round(float(probability), 8)

    policy = artifact["routing_policy"]
    route_predictions(predictions, policy)
    routing_before_drift_gate = routing_metrics(predictions)
    drift = evaluate_score_drift(predictions, artifact["drift_reference"])
    if drift["severe_drift"]:
        for row in predictions:
            if row["routing_status"] == "auto_assign":
                row["routing_status"] = "top3_confirmation"
                row["fallback_reason"] = "score_distribution_drift"
    routing = routing_metrics(predictions)
    targets = artifact["targets"]
    gate = deployment_gate(
        routing,
        target_auto_accuracy=float(targets["target_auto_accuracy"]),
        minimum_auto_coverage=float(targets["minimum_auto_coverage"]),
        maximum_unseen_auto_rate=float(targets["maximum_unseen_auto_rate"]),
    )
    gate["score_drift_check"] = not drift["severe_drift"]
    gate["passed"] = bool(gate["passed"] and not drift["severe_drift"])
    gate["production_integration_allowed"] = False
    gate["production_blocker"] = (
        "reviewed_active_roster_and_explicit_operator_approval_required"
        if gate["passed"]
        else "untouched_holdout_or_drift_safety_gate_failed"
    )

    report = {
        "schema_version": 1,
        "method": "frozen_rolling_history_holdout_evaluation_v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "deployment_status": "research_only",
        "protocol": {
            **window.audit,
            "holdout_used_for_fitting": False,
            "holdout_used_for_threshold_selection": False,
            "rolling_history_updated_only_from_prior_windows": True,
            "label_availability_policy": artifact.get("label_availability_policy"),
            "assignment_time_history_available": False,
        },
        "recommendation_metrics": recommendation_metrics(predictions),
        "recommendation_metrics_known_owner": recommendation_metrics(
            [row for row in predictions if row["known_owner"]]
        ),
        "calibration_metrics": calibration_metrics(predictions),
        "open_set_metrics": open_set_metrics(
            predictions, threshold=float(policy["open_set_threshold"])
        ),
        "routing_before_drift_gate": routing_before_drift_gate,
        "score_drift": drift,
        "routing_metrics": routing,
        "routing_score_diagnostics": routing_score_diagnostics(predictions, policy),
        "deployment_gate": gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "rolling_holdout_report.json", report)
    write_jsonl(args.output_dir / "rolling_holdout_predictions.jsonl", predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
