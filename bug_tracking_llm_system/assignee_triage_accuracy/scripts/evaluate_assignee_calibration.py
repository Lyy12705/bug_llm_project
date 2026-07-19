from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from assignee_phase1_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_PHASE1_ROOT,
    SYSTEM_ROOT,
    build_history_counts,
    build_metrics,
    evaluate_current_triager,
    read_jsonl,
    rows_to_prediction_csv,
    write_csv,
    write_json,
    write_jsonl,
)
from assignee_phase2_common import (
    DEFAULT_PHASE2_ROOT,
    DEFAULT_RAW_PATH,
    build_window_view,
    default_windows,
    load_normalized_raw_records,
    summarize_top1,
)


DEFAULT_PHASE3_ROOT = (
    Path(__file__).resolve().parents[1] / "phase3_confidence_calibration"
)
DATASET = "bmo_paper_2024_3k"
PROBABILITY_EPSILON = 1e-6
CONFIDENCE_BINS = [round(index / 10, 1) for index in range(11)]
ACCURACY_TARGETS = [0.80, 0.85, 0.90, 0.95]
COVERAGE_TARGETS = [round(index / 10, 1) for index in range(1, 11)]
EXPECTED_CLOSED_SET_METRICS = {
    "top1_accuracy": 0.765625,
    "hit_at_3": 0.90625,
    "hit_at_5": 0.9375,
    "hit_at_10": 0.958333,
    "mrr": 0.838777,
}


@dataclass
class Calibrator:
    name: str
    feature_set: str
    model: Any = None

    def predict(self, predictions: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> list[float]:
        if self.name == "raw_confidence":
            return [clip_probability(raw_confidence(row)) for row in predictions]
        matrix = feature_matrix(predictions, train_rows, self.feature_set)
        if self.name == "isotonic_regression":
            values = self.model.predict(matrix[:, 0])
        else:
            values = self.model.predict_proba(matrix)[:, 1]
        return [clip_probability(float(value)) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate confidence calibration and selective prediction for current assignee triage."
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--approve-artifact",
        action="store_true",
        help="Mark the exported runtime artifact approved after an explicit domain review.",
    )
    args = parser.parse_args()

    dataset = args.dataset
    output_dir = args.output_dir or DEFAULT_PHASE3_ROOT / "reports" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = processed_paths(args.data_dir, dataset)
    train_rows = read_jsonl(paths["train"])
    validation_rows = read_jsonl(paths["validation"])
    test_rows = read_jsonl(paths["test"])
    roster = read_roster(paths["roster"])

    validation_predictions, validation_ranking_metrics = evaluate_current_triager(
        train_path=paths["train"],
        train_rows=train_rows,
        test_rows=validation_rows,
        roster=roster,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    test_predictions, test_ranking_metrics = evaluate_current_triager(
        train_path=paths["train"],
        train_rows=train_rows,
        test_rows=test_rows,
        roster=roster,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    validation_predictions = add_prediction_features(validation_predictions, train_rows)
    test_predictions = add_prediction_features(test_predictions, train_rows)
    ranking_check = verify_ranking_unchanged(test_ranking_metrics)

    raw_audit_rows = (
        confidence_bin_rows("validation", validation_predictions)
        + confidence_bin_rows("test", test_predictions)
    )
    write_csv(output_dir / "raw_confidence_audit.csv", raw_audit_rows, raw_audit_fields())
    confidence_distribution = {
        "validation": confidence_distribution_summary(validation_predictions),
        "test": confidence_distribution_summary(test_predictions),
        "confidence_source": confidence_source_summary(),
        "ranking_check": ranking_check,
    }
    write_json(output_dir / "confidence_distribution.json", confidence_distribution)

    calibrators = fit_calibrators(validation_predictions, train_rows, args.seed)
    method_rows, method_metrics = compare_calibrators(
        calibrators=calibrators,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        train_rows=train_rows,
    )
    write_csv(output_dir / "calibration_method_comparison.csv", method_rows, calibration_method_fields())

    best_method = select_best_method(method_metrics)
    best_calibrator = calibrators[best_method]
    validation_best_probs = best_calibrator.predict(validation_predictions, train_rows)
    test_best_probs = best_calibrator.predict(test_predictions, train_rows)
    calibrated_predictions = attach_probabilities(
        test_predictions,
        method_metrics["test_probabilities_by_method"],
        best_method=best_method,
    )
    write_csv(
        output_dir / "calibrated_predictions.csv",
        rows_to_prediction_csv(calibrated_predictions),
        calibrated_prediction_fields(method_metrics["method_names"]),
    )
    write_jsonl(output_dir / "calibrated_predictions.jsonl", calibrated_predictions)
    calibration_artifact = build_runtime_calibration_artifact(
        dataset=dataset,
        calibrator=best_calibrator,
        method=best_method,
        validation_metrics=method_metrics["validation"][best_method],
        test_metrics=method_metrics["test"][best_method],
        approved=args.approve_artifact,
    )
    write_json(output_dir / "assignee_confidence_calibrator.json", calibration_artifact)

    selective = evaluate_selective_prediction(
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        validation_probs=validation_best_probs,
        test_probs=test_best_probs,
        best_method=best_method,
    )
    write_csv(output_dir / "coverage_accuracy_curve.csv", selective["coverage_grid_rows"], coverage_fields())
    write_csv(output_dir / "coverage_risk_curve.csv", selective["coverage_grid_rows"], coverage_fields())
    write_json(output_dir / "selective_prediction_metrics.json", selective["metrics"])

    routing = evaluate_routing_policies(
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        validation_probs=validation_best_probs,
        test_probs=test_best_probs,
    )
    write_csv(output_dir / "routing_policy_comparison.csv", routing["rows"], routing_fields())

    multi_window = evaluate_multi_window_calibration(
        best_method=best_method,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_csv(output_dir / "multi_window_calibration_summary.csv", multi_window["rows"], multi_window_fields())
    write_calibration_drift_report(output_dir / "calibration_drift_report.md", multi_window)

    open_world = evaluate_open_world_groups(
        best_method=best_method,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_csv(output_dir / "open_world_group_calibration.csv", open_world["group_rows"], open_world_fields())
    write_csv(output_dir / "open_world_predictions_calibrated.csv", open_world["prediction_rows"], open_world_prediction_fields())
    write_json(output_dir / "open_world_calibration_metrics.json", open_world["metrics"])

    plot_reliability_diagram(
        output_dir / "raw_reliability_diagram.png",
        validation_predictions,
        test_predictions,
        [raw_confidence(row) for row in validation_predictions],
        [raw_confidence(row) for row in test_predictions],
        title="Raw Confidence Reliability",
    )
    plot_reliability_diagram(
        output_dir / "reliability_diagram_raw.png",
        validation_predictions,
        test_predictions,
        [raw_confidence(row) for row in validation_predictions],
        [raw_confidence(row) for row in test_predictions],
        title="Raw Confidence Reliability",
    )
    plot_reliability_diagram(
        output_dir / "reliability_diagram_calibrated.png",
        validation_predictions,
        test_predictions,
        validation_best_probs,
        test_best_probs,
        title=f"Calibrated Reliability ({best_method})",
    )
    plot_coverage_risk_curve(output_dir / "coverage_risk_curve.png", selective["coverage_grid_rows"])

    summary = build_phase3_summary(
        args=args,
        paths=paths,
        output_dir=output_dir,
        validation_ranking_metrics=validation_ranking_metrics,
        test_ranking_metrics=test_ranking_metrics,
        ranking_check=ranking_check,
        confidence_distribution=confidence_distribution,
        method_metrics=method_metrics,
        best_method=best_method,
        selective=selective,
        routing=routing,
        multi_window=multi_window,
        open_world=open_world,
    )
    write_json(output_dir / "phase3_calibration_summary.json", summary)
    write_phase3_markdown(output_dir / "phase3_calibration_summary.md", summary)
    print(json.dumps(summary["final_recommendation"], ensure_ascii=False, indent=2, sort_keys=True))


def processed_paths(data_dir: Path, dataset: str) -> dict[str, Path]:
    paths = {
        "train": data_dir / f"{dataset}_history_train.jsonl",
        "validation": data_dir / f"{dataset}_validation_set.jsonl",
        "test": data_dir / f"{dataset}_test_set.jsonl",
        "roster": data_dir / f"{dataset}_candidate_roster.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing processed dataset files:\n" + "\n".join(missing))
    return paths


def read_roster(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("candidates", []) if isinstance(payload, dict) else payload
    return {str(value).strip().lower() for value in values if str(value).strip()}


def add_prediction_features(predictions: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = build_history_counts(train_rows)
    output = []
    for row in predictions:
        feature_payload = prediction_time_features(row, counts)
        output.append({**row, **feature_payload})
    return output


def prediction_time_features(row: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("candidate_scores") if isinstance(row.get("candidate_scores"), dict) else {}
    ranked = row.get("ranked_candidates") if isinstance(row.get("ranked_candidates"), list) else []
    predicted = str(row.get("predicted_assignee") or "").strip().lower()
    product = str(row.get("product") or "unknown").strip().lower() or "unknown"
    component = str(row.get("component") or "unknown").strip().lower() or "unknown"
    product_component = f"{product}::{component}"
    sorted_scores = sorted((float(value) for value in scores.values()), reverse=True)
    top1_score = float(scores.get(predicted, sorted_scores[0] if sorted_scores else 0.0))
    top2_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = float(row.get("margin") or 0.0)
    signal_count = count_signals(row.get("reason"))
    global_count = int(counts["global_counts"].get(predicted, 0))
    component_support = int(counts["component_counts"].get(component, Counter()).get(predicted, 0))
    product_support = int(counts["product_counts"].get(product, Counter()).get(predicted, 0))
    product_component_support = int(
        counts["product_component_counts"].get(product_component, Counter()).get(predicted, 0)
    )
    return {
        "raw_confidence": raw_confidence(row),
        "top1_score": top1_score,
        "top2_score": top2_score,
        "score_margin": top1_score - top2_score,
        "normalized_margin": margin,
        "candidate_count": len(ranked),
        "signal_count": signal_count,
        "assignee_train_frequency": global_count,
        "component_history_support": component_support,
        "product_history_support": product_support,
        "product_component_history_support": product_component_support,
    }


def count_signals(reason: Any) -> int:
    text = str(reason or "")
    marker = "strongest signals:"
    if marker not in text:
        return 0
    signal_text = text.split(marker, 1)[1].split(".", 1)[0]
    signals = [value.strip() for value in signal_text.split(",") if value.strip()]
    return len(signals)


def raw_confidence(row: dict[str, Any]) -> float:
    return clip_probability(float(row.get("confidence") or row.get("raw_confidence") or 0.0))


def correctness_labels(predictions: list[dict[str, Any]]) -> list[int]:
    return [1 if row.get("is_top1_correct") else 0 for row in predictions]


def confidence_bin_rows(split: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    labels = correctness_labels(predictions)
    confidences = [raw_confidence(row) for row in predictions]
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        selected = [
            (confidence, label)
            for confidence, label in zip(confidences, labels, strict=True)
            if low <= confidence < high or (index == 9 and confidence == 1.0)
        ]
        count = len(selected)
        mean_confidence = sum(conf for conf, _ in selected) / count if count else 0.0
        accuracy = sum(label for _, label in selected) / count if count else 0.0
        rows.append(
            {
                "split": split,
                "bin_low": low,
                "bin_high": high,
                "sample_count": count,
                "mean_confidence": round(mean_confidence, 6),
                "actual_accuracy": round(accuracy, 6),
                "calibration_gap": round(mean_confidence - accuracy, 6),
            }
        )
    return rows


def confidence_distribution_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    values = [raw_confidence(row) for row in predictions]
    if not values:
        return {}
    arr = np.array(values)
    histogram = {}
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        histogram[f"[{low:.1f},{high:.1f}{']' if index == 9 else ')'}"] = int(
            sum(1 for value in values if low <= value < high or (index == 9 and value == 1.0))
        )
    duplicate_counts = Counter(values)
    return {
        "rows": len(values),
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "std": round(float(arr.std()), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "quantiles": {
            str(q): round(float(np.quantile(arr, q)), 6)
            for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        },
        "histogram": histogram,
        "invalid_low_count": sum(1 for value in values if value < 0),
        "invalid_high_count": sum(1 for value in values if value > 1),
        "unique_value_count": len(duplicate_counts),
        "most_common_values": [
            {"confidence": confidence, "count": count}
            for confidence, count in duplicate_counts.most_common(10)
        ],
        "saturation_0_95_or_above_count": sum(1 for value in values if value >= 0.95),
        "saturation_0_95_or_above_ratio": round(sum(1 for value in values if value >= 0.95) / len(values), 6),
    }


def confidence_source_summary() -> dict[str, Any]:
    return {
        "source": "AssigneeTriager._confidence",
        "formula": "min(0.95, max(0.35, 0.55 + 0.30 * top1_top2_margin + signal_bonus))",
        "top1_top2_margin": "(best_score - second_score) / best_score",
        "signal_bonus": "min(0.20, 0.05 * number_of_top_candidate_signals)",
        "fallback_confidences": {
            "mapped_owner_without_ranked_candidates": 0.65,
            "manual_triage_no_signal": 0.25,
        },
        "temperature_scaling_note": "Temperature scaling is not appropriate here because the deterministic ranker does not expose calibrated logits.",
    }


def fit_calibrators(predictions: list[dict[str, Any]], train_rows: list[dict[str, Any]], seed: int) -> dict[str, Calibrator]:
    y = np.array(correctness_labels(predictions))
    if len(set(y.tolist())) < 2:
        raise SystemExit("Calibration validation split needs both correct and incorrect predictions.")
    calibrators: dict[str, Calibrator] = {"raw_confidence": Calibrator("raw_confidence", "raw")}

    raw_matrix = feature_matrix(predictions, train_rows, "raw")
    platt = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(random_state=seed, max_iter=1000)),
        ]
    )
    platt.fit(raw_matrix, y)
    calibrators["platt_logistic"] = Calibrator("platt_logistic", "raw", platt)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_matrix[:, 0], y)
    calibrators["isotonic_regression"] = Calibrator("isotonic_regression", "raw", isotonic)

    margin_matrix = feature_matrix(predictions, train_rows, "margin")
    margin_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(random_state=seed, max_iter=1000)),
        ]
    )
    margin_model.fit(margin_matrix, y)
    calibrators["margin_logistic"] = Calibrator("margin_logistic", "margin", margin_model)

    multi_matrix = feature_matrix(predictions, train_rows, "multi")
    multi_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(random_state=seed, max_iter=1000)),
        ]
    )
    multi_model.fit(multi_matrix, y)
    calibrators["multi_feature_logistic"] = Calibrator("multi_feature_logistic", "multi", multi_model)
    return calibrators


def feature_matrix(predictions: list[dict[str, Any]], train_rows: list[dict[str, Any]], feature_set: str) -> np.ndarray:
    enriched = add_prediction_features(predictions, train_rows)
    if feature_set == "raw":
        names = ["raw_confidence"]
    elif feature_set == "margin":
        names = ["normalized_margin"]
    elif feature_set == "multi":
        names = [
            "raw_confidence",
            "normalized_margin",
            "top1_score",
            "top2_score",
            "score_margin",
            "candidate_count",
            "signal_count",
            "assignee_train_frequency",
            "component_history_support",
            "product_history_support",
            "product_component_history_support",
        ]
    else:
        raise SystemExit(f"Unknown feature set: {feature_set}")
    return np.array([[safe_float(row.get(name)) for name in names] for row in enriched], dtype=float)


def compare_calibrators(
    *,
    calibrators: dict[str, Calibrator],
    validation_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    val_labels = correctness_labels(validation_predictions)
    test_labels = correctness_labels(test_predictions)
    val_probabilities: dict[str, list[float]] = {}
    test_probabilities: dict[str, list[float]] = {}
    metrics: dict[str, Any] = {
        "method_names": list(calibrators.keys()),
        "validation": {},
        "test": {},
        "validation_probabilities_by_method": val_probabilities,
        "test_probabilities_by_method": test_probabilities,
    }
    for name, calibrator in calibrators.items():
        val_probs = calibrator.predict(validation_predictions, train_rows)
        test_probs = calibrator.predict(test_predictions, train_rows)
        val_probabilities[name] = val_probs
        test_probabilities[name] = test_probs
        val_metrics = probability_metrics(val_labels, val_probs)
        test_metrics = probability_metrics(test_labels, test_probs)
        metrics["validation"][name] = val_metrics
        metrics["test"][name] = test_metrics
        rows.append(method_row(name, "validation", calibrator.feature_set, val_metrics))
        rows.append(method_row(name, "test", calibrator.feature_set, test_metrics))
    return rows, metrics


def probability_metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    y = np.array(labels, dtype=int)
    p = np.array([clip_probability(value) for value in probabilities], dtype=float)
    ece, mce = ece_mce(labels, probabilities)
    metrics = {
        "rows": len(labels),
        "mean_probability": round(float(p.mean()), 6) if len(p) else 0.0,
        "actual_accuracy": round(float(y.mean()), 6) if len(y) else 0.0,
        "ece": ece,
        "mce": mce,
        "brier_score": round(float(brier_score_loss(y, p)), 6) if len(set(labels)) > 1 else None,
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 6) if len(y) else None,
        "roc_auc": safe_metric(lambda: roc_auc_score(y, p), labels),
        "pr_auc": safe_metric(lambda: average_precision_score(y, p), labels),
    }
    slope = calibration_slope_intercept(labels, probabilities)
    metrics.update(slope)
    return metrics


def ece_mce(labels: list[int], probabilities: list[float]) -> tuple[float, float]:
    total = len(labels)
    if total == 0:
        return 0.0, 0.0
    ece = 0.0
    mce = 0.0
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        selected = [
            (prob, label)
            for prob, label in zip(probabilities, labels, strict=True)
            if low <= prob < high or (index == 9 and prob == 1.0)
        ]
        if not selected:
            continue
        mean_prob = sum(prob for prob, _ in selected) / len(selected)
        accuracy = sum(label for _, label in selected) / len(selected)
        gap = abs(mean_prob - accuracy)
        ece += len(selected) / total * gap
        mce = max(mce, gap)
    return round(ece, 6), round(mce, 6)


def calibration_slope_intercept(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    if len(set(labels)) < 2:
        return {"calibration_slope": None, "calibration_intercept": None}
    logits = np.array([[logit(probability)] for probability in probabilities], dtype=float)
    model = LogisticRegression(max_iter=1000)
    try:
        model.fit(logits, np.array(labels, dtype=int))
    except Exception:
        return {"calibration_slope": None, "calibration_intercept": None}
    return {
        "calibration_slope": round(float(model.coef_[0][0]), 6),
        "calibration_intercept": round(float(model.intercept_[0]), 6),
    }


def safe_metric(fn: Any, labels: list[int]) -> float | None:
    if len(set(labels)) < 2:
        return None
    try:
        return round(float(fn()), 6)
    except Exception:
        return None


def method_row(name: str, split: str, feature_set: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": name,
        "split": split,
        "feature_set": feature_set,
        **metrics,
    }


def select_best_method(method_metrics: dict[str, Any]) -> str:
    candidates = []
    for name, metrics in method_metrics["validation"].items():
        candidates.append(
            (
                metrics["ece"],
                metrics["brier_score"] if metrics["brier_score"] is not None else 999.0,
                metrics["log_loss"] if metrics["log_loss"] is not None else 999.0,
                name,
            )
        )
    candidates.sort()
    return candidates[0][3]


def attach_probabilities(
    predictions: list[dict[str, Any]],
    probabilities_by_method: dict[str, list[float]],
    *,
    best_method: str,
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(predictions):
        copy = dict(row)
        for method, probabilities in probabilities_by_method.items():
            copy[f"{method}_probability"] = round(probabilities[index], 6)
        copy["best_calibrator"] = best_method
        copy["calibrated_probability"] = round(probabilities_by_method[best_method][index], 6)
        output.append(copy)
    return output


def build_runtime_calibration_artifact(
    *,
    dataset: str,
    calibrator: Calibrator,
    method: str,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    approved: bool,
) -> dict[str, Any]:
    """Export only a dependency-free, review-gated calibration representation."""
    mapping: dict[str, list[float]] = {}
    runtime_supported = method == "isotonic_regression" and calibrator.model is not None
    if runtime_supported:
        mapping = {
            "x_thresholds": [round(float(value), 12) for value in calibrator.model.X_thresholds_],
            "y_thresholds": [round(float(value), 12) for value in calibrator.model.y_thresholds_],
        }
    return {
        "schema_version": 1,
        "artifact_type": "assignee_confidence_calibrator",
        "name": method,
        "deployment_status": "approved" if approved and runtime_supported else "research_only",
        "runtime_supported": runtime_supported,
        "ranker_confidence_version": "hybrid_ranker_v1",
        "dataset": dataset,
        "mapping": mapping,
        "validation_metrics": validation_metrics,
        "frozen_test_metrics": test_metrics,
        "approval_note": (
            "Explicitly approved by the operator; only use after validating the target project's temporal split."
            if approved and runtime_supported
            else "Research artifact. Runtime will reject it until an operator explicitly approves a target-domain artifact."
        ),
    }


def evaluate_selective_prediction(
    *,
    validation_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
    validation_probs: list[float],
    test_probs: list[float],
    best_method: str,
) -> dict[str, Any]:
    val_labels = correctness_labels(validation_predictions)
    target_rows = []
    targets_payload = {}
    for target in ACCURACY_TARGETS:
        threshold_info = threshold_for_target_accuracy(validation_probs, val_labels, target)
        test_eval = selective_eval(test_probs, correctness_labels(test_predictions), threshold_info["threshold"])
        row = {
            "target_accuracy": target,
            **threshold_info,
            **prefix_keys(test_eval, "test_"),
        }
        target_rows.append(row)
        targets_payload[str(target)] = row

    coverage_grid_rows = []
    for target_coverage in COVERAGE_TARGETS:
        threshold = threshold_for_validation_coverage(validation_probs, target_coverage)
        test_eval = selective_eval(test_probs, correctness_labels(test_predictions), threshold)
        coverage_grid_rows.append(
            {
                "target_validation_coverage": target_coverage,
                "threshold": threshold,
                "test_coverage": test_eval["coverage"],
                "test_accuracy": test_eval["selective_accuracy"],
                "test_risk": test_eval["risk"],
                "test_error_count": test_eval["error_count"],
                "accepted_ticket_count": test_eval["accepted_count"],
                "rejected_ticket_count": test_eval["rejected_count"],
            }
        )
    return {
        "target_rows": target_rows,
        "coverage_grid_rows": coverage_grid_rows,
        "metrics": {
            "best_calibrator": best_method,
            "targets": targets_payload,
            "coverage_grid": coverage_grid_rows,
        },
    }


def threshold_for_target_accuracy(probabilities: list[float], labels: list[int], target: float) -> dict[str, Any]:
    candidates = sorted(set(probabilities), reverse=True)
    best: dict[str, Any] | None = None
    for threshold in candidates:
        evaluation = selective_eval(probabilities, labels, threshold)
        if evaluation["accepted_count"] == 0 or evaluation["selective_accuracy"] < target:
            continue
        candidate = {
            "threshold": round(float(threshold), 6),
            "validation_coverage": evaluation["coverage"],
            "validation_accuracy": evaluation["selective_accuracy"],
            "validation_error_rate": evaluation["risk"],
            "validation_accepted_count": evaluation["accepted_count"],
            "validation_rejected_count": evaluation["rejected_count"],
            "threshold_found": True,
        }
        if best is None or candidate["validation_coverage"] > best["validation_coverage"]:
            best = candidate
    if best is not None:
        return best
    return {
        "threshold": 1.000001,
        "validation_coverage": 0.0,
        "validation_accuracy": 0.0,
        "validation_error_rate": 0.0,
        "validation_accepted_count": 0,
        "validation_rejected_count": len(labels),
        "threshold_found": False,
    }


def threshold_for_validation_coverage(probabilities: list[float], target_coverage: float) -> float:
    if not probabilities:
        return 1.000001
    sorted_probs = sorted(probabilities, reverse=True)
    accepted = max(1, min(len(sorted_probs), math.ceil(target_coverage * len(sorted_probs))))
    return round(float(sorted_probs[accepted - 1]), 6)


def selective_eval(probabilities: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    accepted = [(prob, label) for prob, label in zip(probabilities, labels, strict=True) if prob >= threshold]
    accepted_count = len(accepted)
    correct = sum(label for _, label in accepted)
    error_count = accepted_count - correct
    accuracy = correct / accepted_count if accepted_count else 0.0
    return {
        "threshold": threshold,
        "coverage": round(accepted_count / len(labels), 6) if labels else 0.0,
        "selective_accuracy": round(accuracy, 6),
        "risk": round(1.0 - accuracy, 6) if accepted_count else 0.0,
        "error_rate": round(error_count / accepted_count, 6) if accepted_count else 0.0,
        "error_count": int(error_count),
        "accepted_count": accepted_count,
        "rejected_count": len(labels) - accepted_count,
        "abstention_rate": round((len(labels) - accepted_count) / len(labels), 6) if labels else 0.0,
    }


def evaluate_routing_policies(
    *,
    validation_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
    validation_probs: list[float],
    test_probs: list[float],
) -> dict[str, Any]:
    policy_specs = {
        "safety_first": {"auto_min_accuracy": 0.95, "top3_min_hit": 0.80, "mode": "safety"},
        "balanced": {"auto_min_accuracy": 0.85, "top3_min_hit": 0.80, "mode": "balanced"},
        "automation_first": {"auto_min_accuracy": 0.80, "top3_min_hit": 0.70, "mode": "automation"},
    }
    rows = []
    for policy_name, spec in policy_specs.items():
        thresholds = select_routing_thresholds(
            validation_predictions,
            validation_probs,
            spec["auto_min_accuracy"],
            spec["top3_min_hit"],
            spec["mode"],
        )
        val_eval = routing_eval(validation_predictions, validation_probs, thresholds["t_low"], thresholds["t_high"])
        test_eval = routing_eval(test_predictions, test_probs, thresholds["t_low"], thresholds["t_high"])
        rows.append(
            {
                "policy": policy_name,
                "t_low": thresholds["t_low"],
                "t_high": thresholds["t_high"],
                "validation_objective": thresholds["objective"],
                **prefix_keys(val_eval, "validation_"),
                **prefix_keys(test_eval, "test_"),
            }
        )
    return {"rows": rows}


def select_routing_thresholds(
    predictions: list[dict[str, Any]],
    probabilities: list[float],
    auto_min_accuracy: float,
    top3_min_hit: float,
    mode: str,
) -> dict[str, Any]:
    unique = sorted(set(round(float(prob), 6) for prob in probabilities))
    thresholds = [0.0, *unique, 1.000001]
    best: dict[str, Any] | None = None
    for t_high in thresholds:
        for t_low in thresholds:
            if t_low > t_high:
                continue
            evaluation = routing_eval(predictions, probabilities, t_low, t_high)
            if evaluation["auto_assign_count"] and evaluation["auto_assign_accuracy"] < auto_min_accuracy:
                continue
            if evaluation["top3_count"] and evaluation["top3_hit_at_3"] < top3_min_hit:
                continue
            objective = routing_objective(evaluation, mode)
            candidate = {"t_low": t_low, "t_high": t_high, "objective": objective}
            if best is None or objective > best["objective"]:
                best = candidate
    if best is None:
        fallback_high = threshold_for_target_accuracy(
            probabilities,
            correctness_labels(predictions),
            auto_min_accuracy,
        )["threshold"]
        best = {"t_low": fallback_high, "t_high": fallback_high, "objective": -999.0}
    return best


def routing_objective(evaluation: dict[str, Any], mode: str) -> float:
    if mode == "safety":
        return (
            evaluation["auto_assign_coverage"]
            + 0.25 * evaluation["top3_coverage"]
            + 0.10 * evaluation["top3_hit_at_3"]
            - 5.0 * evaluation["wrong_auto_assign_rate"]
        )
    if mode == "automation":
        return (
            evaluation["auto_assign_coverage"]
            + 0.20 * evaluation["top3_coverage"]
            - 2.0 * evaluation["wrong_auto_assign_rate"]
            - 0.10 * evaluation["manual_triage_rate"]
        )
    return (
        evaluation["auto_assign_coverage"] * evaluation["auto_assign_accuracy"]
        + evaluation["top3_coverage"] * evaluation["top3_hit_at_3"]
        - 0.50 * evaluation["wrong_auto_assign_rate"]
        - 0.10 * evaluation["manual_triage_rate"]
    )


def routing_eval(predictions: list[dict[str, Any]], probabilities: list[float], t_low: float, t_high: float) -> dict[str, Any]:
    total = len(predictions)
    auto = []
    top3 = []
    manual = []
    for prediction, probability in zip(predictions, probabilities, strict=True):
        if probability >= t_high:
            auto.append(prediction)
        elif probability >= t_low:
            top3.append(prediction)
        else:
            manual.append(prediction)
    auto_correct = sum(1 for row in auto if row.get("is_top1_correct"))
    top3_hit = sum(1 for row in top3 if row.get("is_hit_at_3"))
    wrong_auto = len(auto) - auto_correct
    return {
        "auto_assign_count": len(auto),
        "auto_assign_coverage": round(len(auto) / total, 6) if total else 0.0,
        "auto_assign_accuracy": round(auto_correct / len(auto), 6) if auto else 0.0,
        "top3_count": len(top3),
        "top3_coverage": round(len(top3) / total, 6) if total else 0.0,
        "top3_hit_at_3": round(top3_hit / len(top3), 6) if top3 else 0.0,
        "manual_triage_count": len(manual),
        "manual_triage_rate": round(len(manual) / total, 6) if total else 0.0,
        "wrong_auto_assign_count": wrong_auto,
        "wrong_auto_assign_rate": round(wrong_auto / len(auto), 6) if auto else 0.0,
    }


def evaluate_multi_window_calibration(
    *,
    best_method: str,
    top_k: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    records = load_normalized_raw_records(DEFAULT_RAW_PATH)
    windows = default_windows()
    rows = []
    global_calibrator: Calibrator | None = None
    global_train_rows: list[dict[str, Any]] | None = None
    for window in windows:
        view = build_window_view(records, window, min_train_assignee_count=10)
        train_rows = view["train"]
        validation_rows = view["validation"]
        test_rows = view["test"]
        roster = set(view["roster"])
        window_data_dir = DEFAULT_PHASE3_ROOT / "window_data" / window.name
        train_path = window_data_dir / "history_train.jsonl"
        write_jsonl(train_path, train_rows)
        validation_predictions, _ = evaluate_current_triager(
            train_path=train_path,
            train_rows=train_rows,
            test_rows=validation_rows,
            roster=roster,
            top_k=top_k,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        test_predictions, _ = evaluate_current_triager(
            train_path=train_path,
            train_rows=train_rows,
            test_rows=test_rows,
            roster=roster,
            top_k=top_k,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        validation_predictions = add_prediction_features(validation_predictions, train_rows)
        test_predictions = add_prediction_features(test_predictions, train_rows)

        calibrators = fit_calibrators(validation_predictions, train_rows, seed)
        per_window_calibrator = calibrators[best_method]
        if global_calibrator is None:
            global_calibrator = per_window_calibrator
            global_train_rows = train_rows
        raw_probs = [raw_confidence(row) for row in test_predictions]
        per_window_probs = per_window_calibrator.predict(test_predictions, train_rows)
        assert global_calibrator is not None and global_train_rows is not None
        global_probs = global_calibrator.predict(test_predictions, train_rows)
        labels = correctness_labels(test_predictions)
        threshold = threshold_for_target_accuracy(
            per_window_calibrator.predict(validation_predictions, train_rows),
            correctness_labels(validation_predictions),
            0.85,
        )["threshold"]
        selective = selective_eval(per_window_probs, labels, threshold)
        raw_metrics = probability_metrics(labels, raw_probs)
        per_metrics = probability_metrics(labels, per_window_probs)
        global_metrics = probability_metrics(labels, global_probs)
        rows.append(
            {
                "window_name": window.name,
                "window_kind": window.kind,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "test_count": len(test_rows),
                "roster_size": len(roster),
                "best_method": best_method,
                "raw_ece": raw_metrics["ece"],
                "per_window_calibrated_ece": per_metrics["ece"],
                "global_calibrated_ece": global_metrics["ece"],
                "raw_brier": raw_metrics["brier_score"],
                "per_window_brier": per_metrics["brier_score"],
                "global_brier": global_metrics["brier_score"],
                "threshold_target_0_85": threshold,
                "coverage": selective["coverage"],
                "selective_accuracy": selective["selective_accuracy"],
                "accepted_count": selective["accepted_count"],
                "rejected_count": selective["rejected_count"],
            }
        )
    aggregate = {
        "raw_ece": summarize_top1([float(row["raw_ece"]) for row in rows]),
        "per_window_calibrated_ece": summarize_top1([float(row["per_window_calibrated_ece"]) for row in rows]),
        "global_calibrated_ece": summarize_top1([float(row["global_calibrated_ece"]) for row in rows]),
        "coverage": summarize_top1([float(row["coverage"]) for row in rows]),
        "selective_accuracy": summarize_top1([float(row["selective_accuracy"]) for row in rows]),
    }
    return {"rows": rows, "aggregate": aggregate}


def evaluate_open_world_groups(
    *,
    best_method: str,
    top_k: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    data_dir = DEFAULT_PHASE1_ROOT / "reports" / DATASET / "prefilter_open_world_eval" / "data"
    train_path = data_dir / "prefilter_history_train.jsonl"
    validation_path = data_dir / "prefilter_validation_set.jsonl"
    test_path = data_dir / "prefilter_test_set.jsonl"
    roster_path = data_dir / "prefilter_candidate_roster.json"
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    test_rows = read_jsonl(test_path)
    train_counts = Counter(str(row.get("assignee") or "").strip().lower() for row in train_rows)
    train_assignees = read_roster(roster_path)
    validation_predictions, _ = evaluate_current_triager(
        train_path=train_path,
        train_rows=train_rows,
        test_rows=validation_rows,
        roster=train_assignees,
        top_k=top_k,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    test_predictions, _ = evaluate_current_triager(
        train_path=train_path,
        train_rows=train_rows,
        test_rows=test_rows,
        roster=train_assignees,
        top_k=top_k,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    validation_predictions = add_open_world_groups(add_prediction_features(validation_predictions, train_rows), train_counts)
    test_predictions = add_open_world_groups(add_prediction_features(test_predictions, train_rows), train_counts)
    calibrators = fit_calibrators(validation_predictions, train_rows, seed)
    calibrator = calibrators[best_method]
    validation_probs = calibrator.predict(validation_predictions, train_rows)
    test_probs = calibrator.predict(test_predictions, train_rows)
    raw_threshold_85 = threshold_for_target_accuracy(
        [raw_confidence(row) for row in validation_predictions],
        correctness_labels(validation_predictions),
        0.85,
    )["threshold"]
    calibrated_threshold_85 = threshold_for_target_accuracy(
        validation_probs,
        correctness_labels(validation_predictions),
        0.85,
    )["threshold"]

    prediction_rows = []
    for row, probability in zip(test_predictions, test_probs, strict=True):
        prediction_rows.append(
            {
                "ticket_id": row["ticket_id"],
                "open_set_group": row["open_set_group"],
                "train_frequency": row["train_frequency"],
                "expected_assignee": row["expected_assignee"],
                "predicted_assignee": row["predicted_assignee"],
                "is_top1_correct": row["is_top1_correct"],
                "is_hit_at_3": row["is_hit_at_3"],
                "raw_confidence": raw_confidence(row),
                "calibrated_probability": round(probability, 6),
                "raw_auto_assign_at_0_85_policy": raw_confidence(row) >= raw_threshold_85,
                "calibrated_auto_assign_at_0_85_policy": probability >= calibrated_threshold_85,
                "title": row.get("title"),
            }
        )
    group_rows = []
    for group, rows_for_group in sorted(group_by_predictions(test_predictions, "open_set_group").items()):
        indexes = [test_predictions.index(row) for row in rows_for_group]
        group_probs = [test_probs[index] for index in indexes]
        group_raw = [raw_confidence(row) for row in rows_for_group]
        calibrated_auto = [
            (probability, row)
            for probability, row in zip(group_probs, rows_for_group, strict=True)
            if probability >= calibrated_threshold_85
        ]
        raw_auto = [
            row for row in rows_for_group if raw_confidence(row) >= raw_threshold_85
        ]
        group_rows.append(
            {
                "open_set_group": group,
                "rows": len(rows_for_group),
                "mean_raw_confidence": round(float(np.mean(group_raw)), 6) if group_raw else 0.0,
                "median_raw_confidence": round(float(np.median(group_raw)), 6) if group_raw else 0.0,
                "mean_calibrated_probability": round(float(np.mean(group_probs)), 6) if group_probs else 0.0,
                "raw_abstention_rate_at_0_85_policy": round(1 - len(raw_auto) / len(rows_for_group), 6) if rows_for_group else 0.0,
                "calibrated_abstention_rate_at_0_85_policy": round(1 - len(calibrated_auto) / len(rows_for_group), 6) if rows_for_group else 0.0,
                "raw_false_auto_assign_rate": false_auto_assign_rate(raw_auto, group),
                "calibrated_false_auto_assign_rate": false_auto_assign_rate([row for _, row in calibrated_auto], group),
                "calibrated_manual_triage_recall": manual_triage_recall(rows_for_group, group_probs, calibrated_threshold_85, group),
            }
        )
    unseen_labels = [1 if row["open_set_group"] == "unseen_assignee" else 0 for row in test_predictions]
    raw_unseen_auc = safe_metric(lambda: roc_auc_score(unseen_labels, [-raw_confidence(row) for row in test_predictions]), unseen_labels)
    calibrated_unseen_auc = safe_metric(lambda: roc_auc_score(unseen_labels, [-prob for prob in test_probs]), unseen_labels)
    return {
        "group_rows": group_rows,
        "prediction_rows": prediction_rows,
        "metrics": {
            "best_method": best_method,
            "raw_threshold_for_0_85_validation_accuracy": raw_threshold_85,
            "calibrated_threshold_for_0_85_validation_accuracy": calibrated_threshold_85,
            "raw_unseen_detection_auc_using_negative_confidence": raw_unseen_auc,
            "calibrated_unseen_detection_auc_using_negative_probability": calibrated_unseen_auc,
            "confidence_alone_open_set_detector_judgment": "insufficient_if_unseen_false_auto_assign_or_auc_is_poor",
        },
    }


def add_open_world_groups(predictions: list[dict[str, Any]], train_counts: Counter[str]) -> list[dict[str, Any]]:
    output = []
    for row in predictions:
        expected = str(row.get("expected_assignee") or "").strip().lower()
        frequency = int(train_counts.get(expected, 0))
        if frequency >= 10:
            group = "frequent_seen"
        elif frequency > 0:
            group = "low_frequency_seen"
        else:
            group = "unseen_assignee"
        output.append(
            {
                **row,
                "train_frequency": frequency,
                "open_set_group": group,
            }
        )
    return output


def false_auto_assign_rate(rows: list[dict[str, Any]], group: str) -> float:
    if group == "unseen_assignee":
        return 1.0 if rows else 0.0
    if not rows:
        return 0.0
    wrong = sum(1 for row in rows if not row.get("is_top1_correct"))
    return round(wrong / len(rows), 6)


def manual_triage_recall(rows: list[dict[str, Any]], probabilities: list[float], threshold: float, group: str) -> float:
    if group != "unseen_assignee" or not rows:
        return 0.0
    rejected = sum(1 for probability in probabilities if probability < threshold)
    return round(rejected / len(rows), 6)


def group_by_predictions(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return dict(grouped)


def verify_ranking_unchanged(metrics: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        key: round(float(metrics.get(key, 0.0)) - expected, 6)
        for key, expected in EXPECTED_CLOSED_SET_METRICS.items()
    }
    unchanged = all(abs(value) <= 1e-6 for value in deltas.values())
    return {
        "ranking_metrics_unchanged": unchanged,
        "expected": EXPECTED_CLOSED_SET_METRICS,
        "observed": {key: metrics.get(key) for key in EXPECTED_CLOSED_SET_METRICS},
        "deltas": deltas,
    }


def build_phase3_summary(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    output_dir: Path,
    validation_ranking_metrics: dict[str, Any],
    test_ranking_metrics: dict[str, Any],
    ranking_check: dict[str, Any],
    confidence_distribution: dict[str, Any],
    method_metrics: dict[str, Any],
    best_method: str,
    selective: dict[str, Any],
    routing: dict[str, Any],
    multi_window: dict[str, Any],
    open_world: dict[str, Any],
) -> dict[str, Any]:
    raw_test = method_metrics["test"]["raw_confidence"]
    best_test = method_metrics["test"][best_method]
    raw_val = method_metrics["validation"]["raw_confidence"]
    best_val = method_metrics["validation"][best_method]
    targets = selective["metrics"]["targets"]
    best_demo_policy = select_demo_policy(routing["rows"])
    deployment_policy = select_deployment_policy(routing["rows"])
    unseen_group = next(
        (row for row in open_world["group_rows"] if row["open_set_group"] == "unseen_assignee"),
        {},
    )
    final = {
        "A_best_calibration_method": best_method,
        "B_raw_confidence_directly_usable": "no",
        "C_ece_reduction": {
            "validation_raw_ece": raw_val["ece"],
            "validation_best_ece": best_val["ece"],
            "validation_reduction": round(raw_val["ece"] - best_val["ece"], 6),
            "test_raw_ece": raw_test["ece"],
            "test_best_ece": best_test["ece"],
            "test_reduction": round(raw_test["ece"] - best_test["ece"], 6),
        },
        "D_accuracy_thresholds_found": {
            target: targets[target]["threshold_found"] for target in targets
        },
        "E_accuracy_threshold_coverages": {
            target: {
                "threshold": targets[target]["threshold"],
                "test_coverage": targets[target]["test_coverage"],
                "test_selective_accuracy": targets[target]["test_selective_accuracy"],
            }
            for target in targets
        },
        "F_best_demo_policy": best_demo_policy,
        "G_best_deployment_policy": deployment_policy,
        "H_unseen_false_auto_assign_rate_acceptable": "not_without_extra_open_set_control",
        "I_confidence_sufficient_as_open_set_detector": "no",
        "J_next_step": "open_set_detector",
    }
    return {
        "experiment_name": "phase3_confidence_calibration_selective_prediction",
        "dataset": args.dataset,
        "command": " ".join(sys.argv),
        "config_snapshot": {
            "data_dir": str(args.data_dir),
            "output_dir": str(output_dir),
            "top_k": args.top_k,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "input_paths": {key: str(value) for key, value in paths.items()},
        "validation_ranking_metrics": validation_ranking_metrics,
        "test_ranking_metrics": test_ranking_metrics,
        "ranking_check": ranking_check,
        "confidence_distribution": confidence_distribution,
        "calibration_selection": {
            "best_method": best_method,
            "selection_rule": "lowest validation ECE, tie-broken by validation Brier score and log loss",
            "temperature_scaling_not_used": confidence_source_summary()["temperature_scaling_note"],
        },
        "calibration_metrics": {
            "validation": method_metrics["validation"],
            "test": method_metrics["test"],
        },
        "selective_prediction": selective["metrics"],
        "routing_policies": routing["rows"],
        "multi_window_calibration": multi_window["aggregate"],
        "open_world": open_world["metrics"],
        "open_world_group_rows": open_world["group_rows"],
        "unseen_group": unseen_group,
        "final_recommendation": final,
        "artifacts": artifact_listing(output_dir),
    }


def select_demo_policy(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    preference = {"balanced": 0.03, "automation_first": 0.02, "safety_first": 0.01}
    return max(
        rows,
        key=lambda row: (
            row["test_auto_assign_coverage"]
            + row["test_top3_coverage"]
            + preference.get(row["policy"], 0.0)
            - row["test_wrong_auto_assign_rate"]
        ),
    )["policy"]


def select_deployment_policy(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return min(rows, key=lambda row: (row["test_wrong_auto_assign_count"], -row["test_auto_assign_accuracy"], -row["test_auto_assign_coverage"]))["policy"]


def artifact_listing(output_dir: Path) -> dict[str, str]:
    names = [
        "raw_confidence_audit.csv",
        "confidence_distribution.json",
        "raw_reliability_diagram.png",
        "reliability_diagram_raw.png",
        "reliability_diagram_calibrated.png",
        "calibration_method_comparison.csv",
        "calibrated_predictions.csv",
        "assignee_confidence_calibrator.json",
        "coverage_accuracy_curve.csv",
        "coverage_risk_curve.csv",
        "coverage_risk_curve.png",
        "selective_prediction_metrics.json",
        "routing_policy_comparison.csv",
        "multi_window_calibration_summary.csv",
        "calibration_drift_report.md",
        "open_world_group_calibration.csv",
        "open_world_calibration_metrics.json",
        "phase3_calibration_summary.md",
        "phase3_calibration_summary.json",
    ]
    return {name: str(output_dir / name) for name in names}


def write_phase3_markdown(path: Path, summary: dict[str, Any]) -> None:
    final = summary["final_recommendation"]
    lines = [
        "# Phase 3 Confidence Calibration Summary",
        "",
        "Phase 3 is additive only. `current_assignee_triager` ranking logic was not modified.",
        "",
        "## Ranking Check",
        "",
        f"- Ranking unchanged: `{summary['ranking_check']['ranking_metrics_unchanged']}`",
        f"- Observed Top-1 / Hit@3 / Hit@5 / Hit@10 / MRR: "
        f"`{summary['test_ranking_metrics']['top1_accuracy']}` / "
        f"`{summary['test_ranking_metrics']['hit_at_3']}` / "
        f"`{summary['test_ranking_metrics']['hit_at_5']}` / "
        f"`{summary['test_ranking_metrics']['hit_at_10']}` / "
        f"`{summary['test_ranking_metrics']['mrr']}`",
        "",
        "## Calibration",
        "",
        f"- Best method: `{final['A_best_calibration_method']}`",
        f"- Raw confidence directly usable: `{final['B_raw_confidence_directly_usable']}`",
        f"- Validation ECE raw -> best: `{final['C_ece_reduction']['validation_raw_ece']}` -> "
        f"`{final['C_ece_reduction']['validation_best_ece']}`",
        f"- Test ECE raw -> best: `{final['C_ece_reduction']['test_raw_ece']}` -> "
        f"`{final['C_ece_reduction']['test_best_ece']}`",
        "",
        "## Selective Prediction",
        "",
        "| Target | Found | Threshold | Test Coverage | Test Accuracy |",
        "|---:|---|---:|---:|---:|",
    ]
    for target, payload in final["E_accuracy_threshold_coverages"].items():
        lines.append(
            f"| {target} | {final['D_accuracy_thresholds_found'][target]} | "
            f"{payload['threshold']} | {payload['test_coverage']} | {payload['test_selective_accuracy']} |"
        )
    lines.extend(
        [
            "",
            "## Routing",
            "",
            f"- Best demo policy: `{final['F_best_demo_policy']}`",
            f"- Best deployment policy: `{final['G_best_deployment_policy']}`",
            "",
            "## Open World",
            "",
            f"- Unseen false auto-assign acceptable: `{final['H_unseen_false_auto_assign_rate_acceptable']}`",
            f"- Confidence sufficient as open-set detector: `{final['I_confidence_sufficient_as_open_set_detector']}`",
            f"- Recommended next step: `{final['J_next_step']}`",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, artifact_path in summary["artifacts"].items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_drift_report(path: Path, multi_window: dict[str, Any]) -> None:
    lines = [
        "# Calibration Drift Report",
        "",
        "Per-window calibration uses the selected Phase 3 calibrator fit on each window's validation set.",
        "Global calibration uses the first window's validation fit and transfers to later windows.",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in multi_window["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Windows",
            "",
            "| Window | Raw ECE | Per-window ECE | Global ECE | Coverage | Selective Accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in multi_window["rows"]:
        lines.append(
            f"| {row['window_name']} | {row['raw_ece']} | {row['per_window_calibrated_ece']} | "
            f"{row['global_calibrated_ece']} | {row['coverage']} | {row['selective_accuracy']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_reliability_diagram(
    path: Path,
    validation_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
    validation_probs: list[float],
    test_probs: list[float],
    *,
    title: str,
) -> None:
    configure_matplotlib(path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for split, predictions, probabilities in [
        ("validation", validation_predictions, validation_probs),
        ("test", test_predictions, test_probs),
    ]:
        labels = correctness_labels(predictions)
        if len(set(labels)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(labels, probabilities, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", label=split)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.set_title(title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_coverage_risk_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    configure_matplotlib(path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    coverage = [row["test_coverage"] for row in rows]
    risk = [row["test_risk"] for row in rows]
    accuracy = [row["test_accuracy"] for row in rows]
    ax.plot(coverage, risk, marker="o", label="risk")
    ax.plot(coverage, accuracy, marker="s", label="accuracy")
    ax.set_title("Selective Prediction Coverage/Risk")
    ax.set_xlabel("Test coverage")
    ax.set_ylabel("Rate")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def configure_matplotlib(output_dir: Path) -> None:
    mpl_dir = output_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def raw_audit_fields() -> list[str]:
    return [
        "split",
        "bin_low",
        "bin_high",
        "sample_count",
        "mean_confidence",
        "actual_accuracy",
        "calibration_gap",
    ]


def calibration_method_fields() -> list[str]:
    return [
        "method",
        "split",
        "feature_set",
        "rows",
        "mean_probability",
        "actual_accuracy",
        "ece",
        "mce",
        "brier_score",
        "log_loss",
        "roc_auc",
        "pr_auc",
        "calibration_slope",
        "calibration_intercept",
    ]


def calibrated_prediction_fields(methods: list[str]) -> list[str]:
    base = [
        "ticket_id",
        "expected_assignee",
        "predicted_assignee",
        "rank",
        "reciprocal_rank",
        "is_top1_correct",
        "is_hit_at_3",
        "is_hit_at_5",
        "is_hit_at_10",
        "confidence",
        "raw_confidence",
        "margin",
        "normalized_margin",
        "top1_score",
        "top2_score",
        "score_margin",
        "candidate_count",
        "signal_count",
        "assignee_train_frequency",
        "component_history_support",
        "product_component_history_support",
        "best_calibrator",
        "calibrated_probability",
    ]
    return base + [f"{method}_probability" for method in methods] + ["ranked_candidates", "candidate_scores", "reason", "title"]


def coverage_fields() -> list[str]:
    return [
        "target_validation_coverage",
        "threshold",
        "test_coverage",
        "test_accuracy",
        "test_risk",
        "test_error_count",
        "accepted_ticket_count",
        "rejected_ticket_count",
    ]


def routing_fields() -> list[str]:
    return [
        "policy",
        "t_low",
        "t_high",
        "validation_objective",
        "validation_auto_assign_count",
        "validation_auto_assign_coverage",
        "validation_auto_assign_accuracy",
        "validation_top3_count",
        "validation_top3_coverage",
        "validation_top3_hit_at_3",
        "validation_manual_triage_count",
        "validation_manual_triage_rate",
        "validation_wrong_auto_assign_count",
        "validation_wrong_auto_assign_rate",
        "test_auto_assign_count",
        "test_auto_assign_coverage",
        "test_auto_assign_accuracy",
        "test_top3_count",
        "test_top3_coverage",
        "test_top3_hit_at_3",
        "test_manual_triage_count",
        "test_manual_triage_rate",
        "test_wrong_auto_assign_count",
        "test_wrong_auto_assign_rate",
    ]


def multi_window_fields() -> list[str]:
    return [
        "window_name",
        "window_kind",
        "train_count",
        "validation_count",
        "test_count",
        "roster_size",
        "best_method",
        "raw_ece",
        "per_window_calibrated_ece",
        "global_calibrated_ece",
        "raw_brier",
        "per_window_brier",
        "global_brier",
        "threshold_target_0_85",
        "coverage",
        "selective_accuracy",
        "accepted_count",
        "rejected_count",
    ]


def open_world_fields() -> list[str]:
    return [
        "open_set_group",
        "rows",
        "mean_raw_confidence",
        "median_raw_confidence",
        "mean_calibrated_probability",
        "raw_abstention_rate_at_0_85_policy",
        "calibrated_abstention_rate_at_0_85_policy",
        "raw_false_auto_assign_rate",
        "calibrated_false_auto_assign_rate",
        "calibrated_manual_triage_recall",
    ]


def open_world_prediction_fields() -> list[str]:
    return [
        "ticket_id",
        "open_set_group",
        "train_frequency",
        "expected_assignee",
        "predicted_assignee",
        "is_top1_correct",
        "is_hit_at_3",
        "raw_confidence",
        "calibrated_probability",
        "raw_auto_assign_at_0_85_policy",
        "calibrated_auto_assign_at_0_85_policy",
        "title",
    ]


def prefix_keys(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in payload.items()}


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clip_probability(value: float) -> float:
    return min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, float(value)))


def logit(probability: float) -> float:
    p = clip_probability(probability)
    return math.log(p / (1.0 - p))


if __name__ == "__main__":
    main()
