from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from train_assignee_ltr import (
    DEFAULT_DATA_DIR,
    CandidateIndex,
    apply_label_map,
    build_semantic_backend,
    candidate_source_families_from_features,
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
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "calibration"
CALIBRATION_FEATURES = (
    "top_probability",
    "probability_margin",
    "one_minus_entropy",
    "component_support_log",
    "product_component_support_log",
    "owner_recency",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate an already-fixed candidate LTR ranker.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_history_train.jsonl")
    parser.add_argument("--validation", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_validation_set.jsonl")
    parser.add_argument(
        "--holdout",
        type=Path,
        default=DEFAULT_DATA_DIR.parent / "raw" / "bmo_public_future_2020q4_raw.jsonl",
    )
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-pool-size", type=int, default=30)
    parser.add_argument("--half-life-days", type=float, default=90.0)
    parser.add_argument("--smoothing-alpha", type=float, default=5.0)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--calibration-fit-fraction", type=float, default=0.60)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.85)
    parser.add_argument("--target-review-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-auto-coverage", type=float, default=0.10)
    parser.add_argument("--maximum-unseen-auto-rate", type=float, default=0.05)
    parser.add_argument("--open-set-risk-threshold", type=float, default=0.75)
    args = parser.parse_args()

    if not 0.4 <= args.calibration_fit_fraction <= 0.8:
        raise SystemExit("--calibration-fit-fraction must be between 0.4 and 0.8")
    model_path = args.model_dir / "candidate_ltr_model.joblib"
    artifact_path = args.model_dir / "candidate_ltr_artifact.json"
    if not model_path.exists() or not artifact_path.exists():
        raise SystemExit(f"Missing fixed ranker files in {args.model_dir}")

    ranker_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if ranker_artifact.get("deployment_status") != "research_only":
        raise SystemExit("Calibration expects an explicitly research-only ranker artifact.")
    ranker = joblib.load(model_path)
    candidate_generator_version = str(
        ranker_artifact.get("parameters", {}).get(
            "candidate_generator_version", "v2"
        )
    )
    label_map = read_label_map(args.assignee_label_map)
    train_rows = sorted(
        apply_label_map(canonicalize_source_rows(read_jsonl(args.train)), label_map), key=row_sort_key
    )
    validation_rows = sorted(
        apply_label_map(canonicalize_source_rows(read_jsonl(args.validation)), label_map), key=row_sort_key
    )
    holdout_rows = sorted(
        apply_label_map(canonicalize_source_rows(read_jsonl(args.holdout)), label_map), key=row_sort_key
    )
    holdout_rows, overlap_rows_excluded = exclude_cross_split_overlap(
        holdout_rows,
        [*train_rows, *validation_rows],
        strict_duplicate_families=candidate_generator_version == "v3",
    )

    semantic = build_semantic_backend("sbert", args.sbert_model)
    if not semantic.status.startswith("sbert_local:"):
        raise SystemExit(f"The fixed ranker requires a local SBERT model: {semantic.status}")
    index = CandidateIndex(
        train_rows,
        half_life_days=args.half_life_days,
        smoothing_alpha=args.smoothing_alpha,
        semantic=semantic,
        candidate_generator_version=candidate_generator_version,
    )
    validation_predictions = scored_predictions(validation_rows, index, ranker, args.candidate_pool_size)
    split_at = max(20, min(len(validation_predictions) - 20, int(len(validation_predictions) * args.calibration_fit_fraction)))
    calibration_fit = validation_predictions[:split_at]
    threshold_selection = validation_predictions[split_at:]

    calibrators = fit_calibrators(calibration_fit)
    calibrator_metrics = {
        name: calibration_metrics(rows=threshold_selection, calibrator=calibrator)
        for name, calibrator in calibrators.items()
    }
    best_name = min(
        calibrator_metrics,
        key=lambda name: (
            calibrator_metrics[name]["brier"],
            calibrator_metrics[name]["ece"],
            name,
        ),
    )
    calibrator = calibrators[best_name]
    apply_calibrator(threshold_selection, calibrator)
    high_policy = select_threshold(
        threshold_selection,
        target_accuracy=args.target_auto_accuracy,
        open_set_risk_threshold=args.open_set_risk_threshold,
    )
    low_policy = select_threshold(
        threshold_selection,
        target_accuracy=args.target_review_accuracy,
        open_set_risk_threshold=args.open_set_risk_threshold,
    )

    holdout_predictions = scored_predictions(holdout_rows, index, ranker, args.candidate_pool_size)
    apply_calibrator(holdout_predictions, calibrator)
    route_predictions(
        holdout_predictions,
        high_threshold=high_policy["threshold"],
        low_threshold=low_policy["threshold"],
        open_set_risk_threshold=args.open_set_risk_threshold,
    )
    routing = routing_metrics(holdout_predictions)
    passed = bool(
        high_policy["found"]
        and routing["auto_assignment_accuracy"] >= args.target_auto_accuracy
        and routing["auto_assignment_coverage"] >= args.minimum_auto_coverage
        and routing["unseen_auto_assignment_rate"] <= args.maximum_unseen_auto_rate
    )

    report = {
        "method": "candidate_ltr_posthoc_calibration_v1",
        "deployment_status": "research_only",
        "ranker": ranker_artifact.get("name"),
        "protocol": {
            "calibration_fit_rows": len(calibration_fit),
            "threshold_selection_rows": len(threshold_selection),
            "holdout_rows": len(holdout_rows),
            "holdout_overlap_rows_excluded": overlap_rows_excluded,
            "holdout_used_for_fitting": False,
            "holdout_used_for_threshold_selection": False,
        },
        "calibrator_selection": {
            "selected": best_name,
            "metrics": calibrator_metrics,
        },
        "routing_policy": {
            "t_high": high_policy,
            "t_low": low_policy,
            "open_set_risk_threshold": args.open_set_risk_threshold,
        },
        "holdout_routing": routing,
        "deployment_gate": {
            "passed": passed,
            "target_auto_accuracy": args.target_auto_accuracy,
            "minimum_auto_coverage": args.minimum_auto_coverage,
            "maximum_unseen_auto_rate": args.maximum_unseen_auto_rate,
            "blocker": "explicit_operator_approval_required" if passed else "routing_safety_gate_failed",
        },
    }
    calibration_artifact = build_calibration_artifact(
        best_name, calibrator, report, ranker_artifact.get("name", "")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "candidate_ltr_calibration_report.json", report)
    write_json(args.output_dir / "candidate_ltr_calibrator.json", calibration_artifact)
    write_jsonl(args.output_dir / "holdout_routing_predictions.jsonl", holdout_predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


class ProbabilityCalibrator:
    def __init__(self, name: str, model: Any) -> None:
        self.name = name
        self.model = model

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if self.name == "isotonic":
            values = np.asarray([row["top_probability"] for row in rows])
            return np.asarray(self.model.predict(values), dtype=float)
        matrix = np.asarray([[row[name] for name in CALIBRATION_FEATURES] for row in rows])
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=float)


def fit_calibrators(rows: list[dict[str, Any]]) -> dict[str, ProbabilityCalibrator]:
    labels = np.asarray([int(row["is_top1_correct"]) for row in rows])
    probabilities = np.asarray([row["top_probability"] for row in rows])
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(probabilities, labels)
    matrix = np.asarray([[row[name] for name in CALIBRATION_FEATURES] for row in rows])
    logistic = LogisticRegression(C=0.5, max_iter=1000, random_state=3407).fit(matrix, labels)
    return {
        "isotonic": ProbabilityCalibrator("isotonic", isotonic),
        "multi_feature_logistic": ProbabilityCalibrator("multi_feature_logistic", logistic),
    }


def scored_predictions(
    rows: list[dict[str, Any]], index: CandidateIndex, ranker: Any, pool_size: int
) -> list[dict[str, Any]]:
    output = []
    feature_names = list(ranker.feature_names_)
    for row in rows:
        candidates = index.candidates(row, pool_size)
        if not candidates:
            continue
        matrix = np.asarray(
            [[candidate.features.get(name, 0.0) for name in feature_names] for candidate in candidates]
        )
        probabilities = np.asarray(ranker.predict_proba(matrix)[:, 1], dtype=float)
        ranked = sorted(
            zip(candidates, probabilities, strict=True),
            key=lambda item: (-float(item[1]), item[0].assignee),
        )
        top_candidate, top_probability = ranked[0]
        second_probability = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        normalized = probabilities / probabilities.sum() if probabilities.sum() else np.zeros_like(probabilities)
        entropy = -sum(float(value) * math.log(max(float(value), 1e-12)) for value in normalized)
        entropy /= math.log(len(normalized)) if len(normalized) > 1 else 1.0
        expected = str(row.get("assignee") or "").strip().lower()
        top_features = top_candidate.features
        if index.candidate_generator_version == "v3":
            source_families = candidate_source_families_from_features(top_features)
            candidate_source_count = len(source_families)
            strong_source_agreement = len(
                source_families
                & {
                    "component_history",
                    "reporter_history",
                    "file_ownership",
                    "lexical_retrieval",
                    "semantic_retrieval",
                }
            )
        else:
            legacy_sources = (
                "product_component",
                "component",
                "bm25",
                "sbert",
                "recent_component",
                "product",
                "global_prior",
            )
            source_families = {
                source
                for source in legacy_sources
                if bool(top_features.get(f"source_{source}", 0.0))
            }
            candidate_source_count = len(source_families)
            strong_source_agreement = sum(
                bool(top_features.get(f"source_{source}", 0.0))
                for source in (
                    "product_component",
                    "component",
                    "bm25",
                    "sbert",
                )
            )
        prediction = {
            "ticket_id": row.get("ticket_id"),
            "product": row.get("product") or "unknown",
            "component": row.get("component") or "unknown",
            "expected_assignee": expected,
            "predicted_assignee": top_candidate.assignee,
            "known_owner": expected in index.global_counts,
            "is_top1_correct": top_candidate.assignee == expected,
            "is_top3_correct": expected in [candidate.assignee for candidate, _ in ranked[:3]],
            "is_top5_correct": expected in [candidate.assignee for candidate, _ in ranked[:5]],
            "top_probability": float(top_probability),
            "probability_margin": float(top_probability) - second_probability,
            "one_minus_entropy": 1.0 - entropy,
            "component_support_log": float(top_features.get("component_support_log", 0.0)),
            "product_component_support_log": float(top_features.get("product_component_support_log", 0.0)),
            "owner_recency": float(top_features.get("owner_recency", 0.0)),
            "predicted_owner_global_log_count": float(top_features.get("global_log_count", 0.0)),
            "predicted_owner_component_log_count": float(top_features.get("component_log_count", 0.0)),
            "predicted_owner_pair_log_count": float(top_features.get("product_component_log_count", 0.0)),
            "component_owner_share": float(top_features.get("component_share_smoothed", 0.0)),
            "product_component_owner_share": float(
                top_features.get("product_component_share_smoothed", 0.0)
            ),
            "recent_component_owner_share": float(top_features.get("recent_90d_component_share", 0.0)),
            "component_token_owner_share": float(
                top_features.get("component_token_share", 0.0)
            ),
            "reporter_owner_share": float(
                top_features.get("reporter_owner_share", 0.0)
            ),
            "file_owner_share": float(top_features.get("file_owner_share", 0.0)),
            "bm25_owner_score": float(top_features.get("bm25_owner_score", 0.0)),
            "bm25_best_issue_score": float(top_features.get("bm25_best_issue_score", 0.0)),
            "sbert_owner_score": float(top_features.get("sbert_owner_score", 0.0)),
            "candidate_source_count": candidate_source_count,
            "candidate_source_families": sorted(source_families),
            "strong_source_agreement": strong_source_agreement,
            "component_recency_drift": abs(
                float(top_features.get("recent_90d_component_share", 0.0))
                - float(top_features.get("component_share_smoothed", 0.0))
            ),
            "candidate_count": len(ranked),
            "open_set_risk": open_set_risk(top_features, float(top_probability), float(top_probability) - second_probability),
            "ranked_candidates": [candidate.assignee for candidate, _ in ranked[:10]],
            "candidate_pool": [candidate.assignee for candidate, _ in ranked],
            "title": row.get("title") or row.get("summary") or "",
        }
        output.append(prediction)
    return output


def open_set_risk(features: dict[str, float], top_probability: float, margin: float) -> float:
    risk = 0.0
    if features.get("component_support_log", 0.0) <= 0.0:
        risk += 0.25
    if features.get("product_component_support_log", 0.0) <= 0.0:
        risk += 0.20
    if features.get("component_log_count", 0.0) <= 0.0:
        risk += 0.20
    if not (features.get("source_component") or features.get("source_product_component")):
        risk += 0.15
    if top_probability < 0.50:
        risk += 0.10
    if margin < 0.10:
        risk += 0.10
    return round(min(1.0, risk), 6)


def calibration_metrics(rows: list[dict[str, Any]], calibrator: ProbabilityCalibrator) -> dict[str, float]:
    labels = [int(row["is_top1_correct"]) for row in rows]
    probabilities = calibrator.predict(rows)
    return {
        "brier": round(float(brier_score_loss(labels, probabilities)), 6),
        "ece": round(expected_calibration_error(labels, probabilities), 6),
    }


def expected_calibration_error(labels: list[int], probabilities: np.ndarray, bins: int = 10) -> float:
    total = len(labels)
    if not total:
        return 0.0
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, value in enumerate(probabilities) if low <= value < high or (index == bins - 1 and value == 1.0)]
        if not members:
            continue
        accuracy = sum(labels[i] for i in members) / len(members)
        confidence = sum(float(probabilities[i]) for i in members) / len(members)
        result += len(members) / total * abs(accuracy - confidence)
    return result


def apply_calibrator(rows: list[dict[str, Any]], calibrator: ProbabilityCalibrator) -> None:
    probabilities = calibrator.predict(rows)
    for row, probability in zip(rows, probabilities, strict=True):
        row["calibrated_probability"] = round(float(min(1.0, max(0.0, probability))), 6)


def select_threshold(
    rows: list[dict[str, Any]], *, target_accuracy: float, open_set_risk_threshold: float
) -> dict[str, Any]:
    candidates = sorted({float(row["calibrated_probability"]) for row in rows}, reverse=True)
    best = None
    for threshold in candidates:
        accepted = [
            row for row in rows
            if row["calibrated_probability"] >= threshold and row["open_set_risk"] < open_set_risk_threshold
        ]
        if not accepted:
            continue
        accuracy = sum(row["is_top1_correct"] for row in accepted) / len(accepted)
        if accuracy >= target_accuracy and (best is None or len(accepted) > best["accepted_rows"]):
            best = {
                "found": True,
                "threshold": round(threshold, 6),
                "accepted_rows": len(accepted),
                "coverage": round(len(accepted) / len(rows), 6),
                "accuracy": round(accuracy, 6),
            }
    return best or {"found": False, "threshold": 1.000001, "accepted_rows": 0, "coverage": 0.0, "accuracy": 0.0}


def route_predictions(
    rows: list[dict[str, Any]], *, high_threshold: float, low_threshold: float, open_set_risk_threshold: float
) -> None:
    for row in rows:
        if row["open_set_risk"] >= open_set_risk_threshold:
            row["routing_status"] = "manual_triage_open_set"
        elif row["calibrated_probability"] >= high_threshold:
            row["routing_status"] = "auto_assign"
        elif row["calibrated_probability"] >= low_threshold:
            row["routing_status"] = "top3_confirmation"
        else:
            row["routing_status"] = "manual_triage_low_confidence"


def routing_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    auto = [row for row in rows if row["routing_status"] == "auto_assign"]
    known = [row for row in rows if row["known_owner"]]
    unknown = [row for row in rows if not row["known_owner"]]
    unknown_auto = [row for row in unknown if row["routing_status"] == "auto_assign"]
    return {
        "rows": len(rows),
        "known_rows": len(known),
        "unseen_rows": len(unknown),
        "auto_assignment_rows": len(auto),
        "auto_assignment_coverage": round(len(auto) / len(rows), 6) if rows else 0.0,
        "auto_assignment_accuracy": round(
            sum(row["is_top1_correct"] for row in auto) / len(auto), 6
        ) if auto else 0.0,
        "unseen_auto_assignment_rows": len(unknown_auto),
        "unseen_auto_assignment_rate": round(len(unknown_auto) / len(unknown), 6) if unknown else 0.0,
        "top3_confirmation_rate": round(
            sum(row["routing_status"] == "top3_confirmation" for row in rows) / len(rows), 6
        ) if rows else 0.0,
        "manual_triage_rate": round(
            sum(row["routing_status"].startswith("manual_triage") for row in rows) / len(rows), 6
        ) if rows else 0.0,
        "routing_status_breakdown": dict(
            sorted(Counter(row["routing_status"] for row in rows).items())
        ),
    }


def build_calibration_artifact(
    name: str, calibrator: ProbabilityCalibrator, report: dict[str, Any], ranker_name: str
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "assignee_ltr_calibrator",
        "name": name,
        "deployment_status": "research_only",
        "ranker": ranker_name,
        "routing_policy": report["routing_policy"],
        "holdout_routing": report["holdout_routing"],
        "approval_note": "Runtime integration requires an approved ranker and explicit operator approval.",
    }
    if name == "isotonic":
        payload["mapping"] = {
            "x_thresholds": [float(value) for value in calibrator.model.X_thresholds_],
            "y_thresholds": [float(value) for value in calibrator.model.y_thresholds_],
        }
    else:
        payload["feature_names"] = list(CALIBRATION_FEATURES)
        payload["coefficients"] = [float(value) for value in calibrator.model.coef_[0]]
        payload["intercept"] = float(calibrator.model.intercept_[0])
    return payload


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
