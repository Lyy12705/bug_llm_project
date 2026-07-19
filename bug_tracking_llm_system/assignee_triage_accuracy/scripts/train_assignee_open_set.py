from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from assignee_open_set_common import (
    OPEN_SET_FEATURES,
    attach_calibrated_probability,
    attach_open_set_probability,
    build_leave_assignee_out_folds,
    deployment_gate,
    fit_portable_logistic,
    open_set_metrics,
    route_predictions,
    routing_score_diagnostics,
    routing_metrics,
    select_routing_policy,
    write_json,
    write_jsonl,
)
from calibrate_assignee_ltr import CALIBRATION_FEATURES, scored_predictions
from train_assignee_ltr import (
    DEFAULT_DATA_DIR,
    CandidateIndex,
    apply_label_map,
    build_semantic_backend,
    canonicalize_source_rows,
    read_jsonl,
    read_label_map,
    exclude_cross_split_overlap,
    row_sort_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "phase6_candidate_ltr"
    / "reports"
    / "bmo_public_q4_holdout_source_quota_sbert"
)
DEFAULT_DEVELOPMENT = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "paper_grade"
    / "data"
    / "raw"
    / "bmo_public_future_2020q3_raw.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "phase6_candidate_ltr"
    / "reports"
    / "bmo_public_q3_open_set_shallow_v3"
)
EXPECTED_RANKER_NAME = "candidate_hist_gradient_boosting_shallow_ltr_v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a leakage-audited open-set detector and fix routing thresholds on development data."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--expected-ranker-name", default=EXPECTED_RANKER_NAME)
    parser.add_argument(
        "--train", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_history_train.jsonl"
    )
    parser.add_argument(
        "--validation", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_validation_set.jsonl"
    )
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    parser.add_argument("--leave-owner-folds", type=int, default=3)
    parser.add_argument("--hidden-query-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-owner-history", type=int, default=3)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.85)
    parser.add_argument("--target-review-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-auto-coverage", type=float, default=0.10)
    parser.add_argument("--maximum-unseen-auto-rate", type=float, default=0.05)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if not 0.3 <= args.calibration_fraction <= 0.7:
        raise SystemExit("--calibration-fraction must be between 0.3 and 0.7")
    ranker, ranker_artifact = load_ranker(args.model_dir, args.expected_ranker_name)
    parameters = ranker_artifact.get("parameters", {})
    pool_size = int(parameters.get("candidate_pool_size", 30))
    half_life_days = float(parameters.get("half_life_days", 90.0))
    smoothing_alpha = float(parameters.get("smoothing_alpha", 5.0))
    embedding_backend = str(parameters.get("embedding_backend") or "none")

    label_map = read_label_map(args.assignee_label_map)
    train_rows = load_rows(args.train, label_map)
    validation_rows = load_rows(args.validation, label_map)
    development_rows = load_rows(args.development, label_map)
    development_rows, overlap_rows_excluded = exclude_cross_split_overlap(
        development_rows, [*train_rows, *validation_rows]
    )
    split_at = max(20, min(len(validation_rows) - 20, int(len(validation_rows) * args.calibration_fraction)))
    calibration_rows = validation_rows[:split_at]
    detector_rows = validation_rows[split_at:]

    semantic = build_semantic_backend(embedding_backend, args.sbert_model)
    if embedding_backend == "sbert" and not semantic.status.startswith("sbert_local:"):
        raise SystemExit(f"The fixed ranker requires a local SBERT model: {semantic.status}")
    full_index = CandidateIndex(
        train_rows,
        half_life_days=half_life_days,
        smoothing_alpha=smoothing_alpha,
        semantic=semantic,
    )

    calibration_predictions = scored_predictions(calibration_rows, full_index, ranker, pool_size)
    calibrator = fit_portable_logistic(
        calibration_predictions,
        [int(row["is_top1_correct"]) for row in calibration_predictions],
        CALIBRATION_FEATURES,
        seed=args.seed,
        balanced=False,
    )

    folds = build_leave_assignee_out_folds(
        train_rows,
        detector_rows,
        folds=args.leave_owner_folds,
        hidden_query_fraction=args.hidden_query_fraction,
        minimum_history=args.minimum_owner_history,
        seed=args.seed,
    )
    detector_training_predictions: list[dict[str, Any]] = []
    for fold in folds:
        fold_index = CandidateIndex(
            fold.history_rows,
            half_life_days=half_life_days,
            smoothing_alpha=smoothing_alpha,
            semantic=semantic,
        )
        predictions = scored_predictions(fold.query_rows, fold_index, ranker, pool_size)
        attach_calibrated_probability(predictions, calibrator)
        for row in predictions:
            row["leave_assignee_out_fold"] = fold.fold
            row["synthetic_unknown"] = not row["known_owner"]
        detector_training_predictions.extend(predictions)

    detector = fit_portable_logistic(
        detector_training_predictions,
        [int(not row["known_owner"]) for row in detector_training_predictions],
        OPEN_SET_FEATURES,
        seed=args.seed,
        balanced=True,
    )
    attach_open_set_probability(detector_training_predictions, detector)

    development_predictions = scored_predictions(development_rows, full_index, ranker, pool_size)
    attach_calibrated_probability(development_predictions, calibrator)
    attach_open_set_probability(development_predictions, detector)
    policy = select_routing_policy(
        development_predictions,
        target_auto_accuracy=args.target_auto_accuracy,
        minimum_auto_coverage=args.minimum_auto_coverage,
        maximum_unseen_auto_rate=args.maximum_unseen_auto_rate,
        target_review_accuracy=args.target_review_accuracy,
    )
    route_predictions(development_predictions, policy)
    development_routing = routing_metrics(development_predictions)
    development_gate = deployment_gate(
        development_routing,
        target_auto_accuracy=args.target_auto_accuracy,
        minimum_auto_coverage=args.minimum_auto_coverage,
        maximum_unseen_auto_rate=args.maximum_unseen_auto_rate,
    )

    rule_predictions = [dict(row, open_set_probability=float(row["open_set_risk"])) for row in development_predictions]
    report = {
        "schema_version": 1,
        "method": "temporal_leave_assignee_out_logistic_v2",
        "deployment_status": "research_only",
        "ranker": ranker_artifact.get("name"),
        "protocol": {
            "calibration_fit_rows": len(calibration_predictions),
            "detector_fit_rows": len(detector_training_predictions),
            "development_selection_rows": len(development_predictions),
            "development_overlap_rows_excluded": overlap_rows_excluded,
            "q4_used_for_fitting": False,
            "q4_used_for_threshold_selection": False,
            "development_used_for_threshold_selection": True,
            "leave_assignee_out_folds": [fold.audit for fold in folds],
            "all_folds_leakage_free": all(fold.audit["leakage_free"] for fold in folds),
        },
        "features": {
            "calibration": list(CALIBRATION_FEATURES),
            "open_set": list(OPEN_SET_FEATURES),
            "identity_features_forbidden": True,
            "future_statistics_forbidden": True,
        },
        "detector_fit": open_set_metrics(detector_training_predictions),
        "development_open_set": open_set_metrics(
            development_predictions, threshold=policy["open_set_threshold"]
        ),
        "development_rule_based_reference": open_set_metrics(rule_predictions, threshold=0.75),
        "routing_policy": policy,
        "development_routing": development_routing,
        "development_score_diagnostics": routing_score_diagnostics(development_predictions, policy),
        "development_gate": development_gate,
        "holdout_status": "not_evaluated_use_evaluate_assignee_open_set_holdout.py",
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": "assignee_open_set_routing_bundle",
        "name": "assignee_open_set_logistic_v2",
        "deployment_status": "research_only",
        "ranker_name": ranker_artifact.get("name"),
        "ranker_model_dir": str(args.model_dir),
        "ranker_requirements": {
            "candidate_pool_size": pool_size,
            "half_life_days": half_life_days,
            "smoothing_alpha": smoothing_alpha,
            "embedding_backend": embedding_backend,
            "sbert_model": args.sbert_model if embedding_backend == "sbert" else "",
        },
        "calibrator": calibrator.to_artifact(),
        "open_set_detector": detector.to_artifact(),
        "routing_policy": policy,
        "development_gate": development_gate,
        "approval_note": (
            "Frozen on development data. A new untouched temporal holdout, reviewed roster, and explicit "
            "operator approval are required before runtime integration."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "open_set_training_report.json", report)
    write_json(args.output_dir / "open_set_routing_artifact.json", artifact)
    write_jsonl(args.output_dir / "detector_training_predictions.jsonl", detector_training_predictions)
    write_jsonl(args.output_dir / "development_routing_predictions.jsonl", development_predictions)
    write_summary(args.output_dir / "open_set_training_summary.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_ranker(model_dir: Path, expected_name: str) -> tuple[Any, dict[str, Any]]:
    model_path = model_dir / "candidate_ltr_model.joblib"
    artifact_path = model_dir / "candidate_ltr_artifact.json"
    if not model_path.exists() or not artifact_path.exists():
        raise SystemExit(f"Missing fixed ranker files in {model_dir}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("deployment_status") != "research_only":
        raise SystemExit("Open-set development expects an explicitly research-only ranker artifact")
    if artifact.get("name") != expected_name:
        raise SystemExit(
            f"Ranker mismatch: expected {expected_name}, got {artifact.get('name') or 'missing'}"
        )
    return joblib.load(model_path), artifact


def load_rows(path: Path, label_map: dict[str, str]) -> list[dict[str, Any]]:
    return sorted(
        apply_label_map(canonicalize_source_rows(read_jsonl(path)), label_map),
        key=row_sort_key,
    )


def write_summary(path: Path, report: dict[str, Any]) -> None:
    routing = report["development_routing"]
    detector = report["development_open_set"]
    lines = [
        "# Open-set routing development summary",
        "",
        "This artifact was fixed using temporal validation and Q3 development data. Q4 was not used.",
        "",
        f"- Open-set AUROC: {detector['auroc']}",
        f"- Open-set AUPRC: {detector['auprc']}",
        f"- Auto-assignment accuracy: {routing['auto_assignment_accuracy']}",
        f"- Auto-assignment coverage: {routing['auto_assignment_coverage']}",
        f"- Unseen-owner auto-assignment rate: {routing['unseen_auto_assignment_rate']}",
        f"- Development safety gate: {report['development_gate']['passed']}",
        "",
        "The next valid step is a single evaluation on a new untouched temporal holdout.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
