from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OPEN_SET_FEATURES = (
    "calibrated_probability",
    "probability_margin",
    "one_minus_entropy",
    "component_support_log",
    "product_component_support_log",
    "predicted_owner_global_log_count",
    "predicted_owner_component_log_count",
    "predicted_owner_pair_log_count",
    "component_owner_share",
    "product_component_owner_share",
    "recent_component_owner_share",
    "component_token_owner_share",
    "reporter_owner_share",
    "file_owner_share",
    "owner_recency",
    "bm25_owner_score",
    "bm25_best_issue_score",
    "sbert_owner_score",
    "candidate_source_count",
    "strong_source_agreement",
    "component_recency_drift",
    "candidate_count",
)


@dataclass(frozen=True)
class LeaveAssigneeOutFold:
    fold: int
    history_rows: list[dict[str, Any]]
    query_rows: list[dict[str, Any]]
    hidden_assignees: tuple[str, ...]
    audit: dict[str, Any]


class PortableLogistic:
    def __init__(self, feature_names: Iterable[str], pipeline: Pipeline) -> None:
        self.feature_names = tuple(feature_names)
        self.pipeline = pipeline

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(feature_matrix(rows, self.feature_names))[:, 1])

    def to_artifact(self) -> dict[str, Any]:
        scaler = self.pipeline.named_steps["scale"]
        model = self.pipeline.named_steps["model"]
        return {
            "model_type": "standardized_logistic_regression",
            "feature_names": list(self.feature_names),
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "coefficients": [float(value) for value in model.coef_[0]],
            "intercept": float(model.intercept_[0]),
        }


def fit_portable_logistic(
    rows: list[dict[str, Any]],
    labels: list[int],
    feature_names: Iterable[str],
    *,
    seed: int,
    balanced: bool,
) -> PortableLogistic:
    if len(set(labels)) < 2:
        raise ValueError("logistic training requires both positive and negative examples")
    names = tuple(feature_names)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced" if balanced else None,
                    max_iter=1500,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(feature_matrix(rows, names), np.asarray(labels, dtype=int))
    return PortableLogistic(names, pipeline)


def predict_portable_logistic(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> np.ndarray:
    names = tuple(str(value) for value in artifact["feature_names"])
    matrix = feature_matrix(rows, names)
    mean = np.asarray(artifact["scaler_mean"], dtype=float)
    scale = np.asarray(artifact["scaler_scale"], dtype=float)
    coefficients = np.asarray(artifact["coefficients"], dtype=float)
    normalized = (matrix - mean) / np.where(scale == 0.0, 1.0, scale)
    logits = normalized @ coefficients + float(artifact["intercept"])
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def feature_matrix(rows: list[dict[str, Any]], feature_names: Iterable[str]) -> np.ndarray:
    return np.asarray(
        [[safe_float(row.get(name)) for name in feature_names] for row in rows],
        dtype=float,
    )


def build_leave_assignee_out_folds(
    history_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    *,
    folds: int,
    hidden_query_fraction: float,
    minimum_history: int,
    seed: int,
) -> list[LeaveAssigneeOutFold]:
    if not history_rows or not query_rows:
        raise ValueError("history and query rows are required")
    if folds < 1:
        raise ValueError("folds must be positive")
    if not 0.05 <= hidden_query_fraction <= 0.5:
        raise ValueError("hidden_query_fraction must be between 0.05 and 0.5")

    history = sorted(history_rows, key=_sort_key)
    queries = sorted(query_rows, key=_sort_key)
    if _timestamp(history[-1]) > _timestamp(queries[0]):
        raise ValueError("leave-assignee-out history must end before its query period")
    history_counts = Counter(_assignee(row) for row in history)
    width = max(1, math.ceil(len(queries) / folds))
    output: list[LeaveAssigneeOutFold] = []
    for fold_index in range(folds):
        chunk = queries[fold_index * width : min(len(queries), (fold_index + 1) * width)]
        if not chunk:
            break
        query_counts = Counter(_assignee(row) for row in chunk)
        eligible = [
            owner
            for owner in query_counts
            if owner and history_counts[owner] >= minimum_history
        ]
        natural_unknown_rows = sum(
            count for owner, count in query_counts.items() if not owner or history_counts[owner] == 0
        )
        target_total_unknown = max(1, math.ceil(len(chunk) * hidden_query_fraction))
        minimum_synthetic_unknown = max(1, math.ceil(len(chunk) * 0.05))
        target_hidden_rows = max(minimum_synthetic_unknown, target_total_unknown - natural_unknown_rows)
        ordered = sorted(
            eligible,
            key=lambda owner: (
                query_counts[owner] > max(2, target_hidden_rows),
                history_counts[owner],
                _stable_order(owner, seed + fold_index),
                -query_counts[owner],
                owner,
            ),
        )
        hidden: list[str] = []
        hidden_rows = 0
        for owner in ordered:
            if hidden_rows >= target_hidden_rows and hidden:
                break
            if len(eligible) > 1 and len(hidden) + 1 >= len(eligible):
                break
            hidden.append(owner)
            hidden_rows += query_counts[owner]
        if not hidden:
            raise ValueError(f"fold {fold_index + 1} has no eligible assignee to hide")

        hidden_set = set(hidden)
        reduced_history = [row for row in history if _assignee(row) not in hidden_set]
        reduced_owners = {_assignee(row) for row in reduced_history}
        unknown_rows = sum(_assignee(row) not in reduced_owners for row in chunk)
        leaked_hidden_rows = sum(_assignee(row) in hidden_set for row in reduced_history)
        audit = {
            "fold": fold_index + 1,
            "history_rows_before_hiding": len(history),
            "history_rows_after_hiding": len(reduced_history),
            "hidden_history_rows_removed": len(history) - len(reduced_history),
            "query_rows": len(chunk),
            "hidden_assignees": sorted(hidden),
            "hidden_assignee_count": len(hidden),
            "natural_unknown_query_rows": natural_unknown_rows,
            "synthetic_unknown_target_rows": target_hidden_rows,
            "synthetic_unknown_query_rows": sum(query_counts[owner] for owner in hidden),
            "unknown_query_rows": unknown_rows,
            "unknown_query_rate": round(unknown_rows / len(chunk), 6),
            "hidden_owner_history_leak_rows": leaked_hidden_rows,
            "history_end": _timestamp(history[-1]),
            "query_start": _timestamp(chunk[0]),
            "query_end": _timestamp(chunk[-1]),
            "leakage_free": leaked_hidden_rows == 0 and _timestamp(history[-1]) <= _timestamp(chunk[0]),
        }
        output.append(
            LeaveAssigneeOutFold(
                fold=fold_index + 1,
                history_rows=reduced_history,
                query_rows=chunk,
                hidden_assignees=tuple(sorted(hidden)),
                audit=audit,
            )
        )
    return output


def attach_calibrated_probability(rows: list[dict[str, Any]], calibrator: PortableLogistic) -> None:
    for row, probability in zip(rows, calibrator.predict(rows), strict=True):
        row["calibrated_probability"] = round(float(probability), 8)


def attach_open_set_probability(rows: list[dict[str, Any]], detector: PortableLogistic) -> None:
    for row, probability in zip(rows, detector.predict(rows), strict=True):
        row["open_set_probability"] = round(float(probability), 8)


def open_set_metrics(rows: list[dict[str, Any]], *, threshold: float | None = None) -> dict[str, Any]:
    labels = np.asarray([int(not row["known_owner"]) for row in rows], dtype=int)
    scores = np.asarray([safe_float(row.get("open_set_probability")) for row in rows], dtype=float)
    result: dict[str, Any] = {
        "rows": len(rows),
        "known_rows": int(sum(labels == 0)),
        "unknown_rows": int(sum(labels == 1)),
        "auroc": metric_or_none(roc_auc_score, labels, scores),
        "auprc": metric_or_none(average_precision_score, labels, scores),
    }
    result.update(unknown_recall_at_known_fpr(labels, scores, maximum_fpr=0.05))
    if threshold is not None:
        predicted_unknown = scores >= threshold
        unknown = labels == 1
        known = labels == 0
        result.update(
            {
                "threshold": round(float(threshold), 8),
                "unknown_recall": ratio_int(np.sum(predicted_unknown & unknown), np.sum(unknown)),
                "known_false_positive_rate": ratio_int(np.sum(predicted_unknown & known), np.sum(known)),
            }
        )
    return result


def unknown_recall_at_known_fpr(
    labels: np.ndarray, scores: np.ndarray, *, maximum_fpr: float
) -> dict[str, Any]:
    best = {"unknown_recall_at_5pct_known_fpr": 0.0, "threshold_at_5pct_known_fpr": 1.000001}
    for threshold in sorted(set(float(value) for value in scores), reverse=True):
        predicted = scores >= threshold
        known_fpr = ratio_int(np.sum(predicted & (labels == 0)), np.sum(labels == 0))
        recall = ratio_int(np.sum(predicted & (labels == 1)), np.sum(labels == 1))
        if known_fpr <= maximum_fpr and recall >= best["unknown_recall_at_5pct_known_fpr"]:
            best = {
                "unknown_recall_at_5pct_known_fpr": recall,
                "threshold_at_5pct_known_fpr": round(threshold, 8),
            }
    return best


def select_routing_policy(
    rows: list[dict[str, Any]],
    *,
    target_auto_accuracy: float,
    minimum_auto_coverage: float,
    maximum_unseen_auto_rate: float,
    target_review_accuracy: float,
    minimum_auto_rows: int = 1,
    minimum_accuracy_lower_bound: float = 0.0,
    minimum_candidate_source_count: int = 0,
    maximum_unseen_rate_upper_bound: float | None = None,
    minimum_component_auto_rows: int = 30,
    minimum_component_accuracy: float = 0.0,
    minimum_component_accuracy_lower_bound: float = 0.0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("routing policy selection requires predictions")
    risk_values = quantile_candidates(rows, "open_set_probability", include=(0.0, 1.000001))
    confidence_values = quantile_candidates(rows, "calibrated_probability", include=(1.000001,))
    unknown_total = sum(not row["known_owner"] for row in rows)
    best: dict[str, Any] | None = None
    for risk_threshold in risk_values:
        eligible = [
            row
            for row in rows
            if row["open_set_probability"] < risk_threshold
            and int(safe_float(row.get("candidate_source_count")))
            >= minimum_candidate_source_count
        ]
        for high_threshold in confidence_values:
            accepted = [row for row in eligible if row["calibrated_probability"] >= high_threshold]
            if not accepted:
                continue
            accuracy = sum(row["is_top1_correct"] for row in accepted) / len(accepted)
            coverage = len(accepted) / len(rows)
            unseen_auto = sum(not row["known_owner"] for row in accepted)
            unseen_rate = unseen_auto / unknown_total if unknown_total else 0.0
            unseen_interval = wilson_interval(unseen_auto, unknown_total)
            accuracy_interval = wilson_interval(
                sum(bool(row["is_top1_correct"]) for row in accepted), len(accepted)
            )
            accepted_ids = {id(row) for row in accepted}
            component_metrics = component_routing_metrics(
                [
                    {
                        **row,
                        "routing_status": (
                            "auto_assign"
                            if id(row) in accepted_ids
                            else "manual_triage_policy_selection"
                        ),
                    }
                    for row in rows
                ]
            )
            component_checks = _component_policy_checks(
                component_metrics,
                minimum_auto_rows=minimum_component_auto_rows,
                minimum_accuracy=minimum_component_accuracy,
                minimum_accuracy_lower_bound=minimum_component_accuracy_lower_bound,
            )
            if (
                accuracy >= target_auto_accuracy
                and coverage >= minimum_auto_coverage
                and unseen_rate < maximum_unseen_auto_rate
                and (
                    maximum_unseen_rate_upper_bound is None
                    or unseen_interval["upper"] < maximum_unseen_rate_upper_bound
                )
                and len(accepted) >= minimum_auto_rows
                and accuracy_interval["lower"] >= minimum_accuracy_lower_bound
                and component_checks["passed"]
            ):
                candidate = {
                    "found": True,
                    "open_set_threshold": round(risk_threshold, 8),
                    "t_high": round(high_threshold, 8),
                    "auto_assignment_rows": len(accepted),
                    "auto_assignment_accuracy": round(accuracy, 6),
                    "auto_assignment_accuracy_ci95": accuracy_interval,
                    "auto_assignment_coverage": round(coverage, 6),
                    "unseen_auto_assignment_rate": round(unseen_rate, 6),
                    "unseen_auto_assignment_rows": unseen_auto,
                    "unseen_rows": unknown_total,
                    "unseen_auto_assignment_rate_ci95": unseen_interval,
                    "minimum_auto_rows": minimum_auto_rows,
                    "minimum_accuracy_lower_bound": minimum_accuracy_lower_bound,
                    "minimum_candidate_source_count": minimum_candidate_source_count,
                    "maximum_unseen_rate_upper_bound": maximum_unseen_rate_upper_bound,
                    "component_metrics": component_metrics,
                    "component_safety": component_checks,
                }
                if best is None or (
                    candidate["auto_assignment_coverage"],
                    candidate["auto_assignment_accuracy"],
                    -candidate["unseen_auto_assignment_rate"],
                ) > (
                    best["auto_assignment_coverage"],
                    best["auto_assignment_accuracy"],
                    -best["unseen_auto_assignment_rate"],
                ):
                    best = candidate
    if best is None:
        return {
            "found": False,
            "open_set_threshold": 0.0,
            "t_high": 1.000001,
            "t_low": 1.000001,
            "blocker": "development_safety_constraints_not_met",
        }

    review_candidates = [
        row
        for row in rows
        if row["open_set_probability"] < best["open_set_threshold"]
        and row["calibrated_probability"] < best["t_high"]
        and int(safe_float(row.get("candidate_source_count")))
        >= minimum_candidate_source_count
    ]
    low = best["t_high"]
    review_rows = 0
    review_accuracy = 0.0
    for threshold in quantile_candidates(review_candidates, "calibrated_probability", include=(best["t_high"],)):
        accepted = [row for row in review_candidates if row["calibrated_probability"] >= threshold]
        if not accepted:
            continue
        accuracy = sum(row["is_top3_correct"] for row in accepted) / len(accepted)
        if accuracy >= target_review_accuracy and len(accepted) >= review_rows:
            low = threshold
            review_rows = len(accepted)
            review_accuracy = accuracy
    best.update(
        {
            "t_low": round(float(low), 8),
            "top3_confirmation_rows": review_rows,
            "top3_confirmation_accuracy": round(review_accuracy, 6),
        }
    )
    return best


def select_multi_window_routing_policy(
    windows: dict[str, list[dict[str, Any]]],
    *,
    target_auto_accuracy: float,
    minimum_auto_coverage: float,
    maximum_unseen_auto_rate: float,
    target_review_accuracy: float,
    minimum_high_confidence: float = 0.50,
    minimum_review_confidence: float = 0.20,
    minimum_auto_rows_per_window: int = 1,
    minimum_accuracy_lower_bound: float = 0.0,
    minimum_review_rows_per_window: int = 1,
    minimum_review_coverage: float = 0.0,
    minimum_candidate_source_count: int = 0,
    maximum_unseen_rate_upper_bound: float | None = None,
    minimum_component_auto_rows: int = 30,
    minimum_component_accuracy: float = 0.0,
    minimum_component_accuracy_lower_bound: float = 0.0,
) -> dict[str, Any]:
    if len(windows) < 2 or any(not rows for rows in windows.values()):
        raise ValueError("multi-window policy selection requires at least two non-empty windows")
    pooled = [row for rows in windows.values() for row in rows]
    risk_values = quantile_candidates(pooled, "open_set_probability", include=(0.0, 1.000001))
    confidence_values = [
        value
        for value in quantile_candidates(pooled, "calibrated_probability", include=(1.000001,))
        if value >= minimum_high_confidence
    ]
    arrays = {name: _routing_arrays(rows) for name, rows in windows.items()}
    best: dict[str, Any] | None = None
    for risk_threshold in risk_values:
        for high_threshold in confidence_values:
            metrics = {
                name: _auto_metrics(
                    values,
                    risk_threshold,
                    high_threshold,
                    minimum_candidate_source_count,
                )
                for name, values in arrays.items()
            }
            if not all(
                row["auto_assignment_accuracy"] >= target_auto_accuracy
                and row["auto_assignment_coverage"] >= minimum_auto_coverage
                and row["unseen_auto_assignment_rate"] < maximum_unseen_auto_rate
                and (
                    maximum_unseen_rate_upper_bound is None
                    or row["unseen_auto_assignment_rate_ci95"]["upper"]
                    < maximum_unseen_rate_upper_bound
                )
                and row["auto_assignment_rows"] >= minimum_auto_rows_per_window
                and row["auto_assignment_accuracy_ci95"]["lower"]
                >= minimum_accuracy_lower_bound
                for row in metrics.values()
            ):
                continue
            component_checks = {}
            for name, values in arrays.items():
                component_metrics = _component_metrics_for_threshold(
                    values,
                    risk_threshold=risk_threshold,
                    high_threshold=high_threshold,
                    minimum_candidate_source_count=minimum_candidate_source_count,
                )
                check = _component_policy_checks(
                    component_metrics,
                    minimum_auto_rows=minimum_component_auto_rows,
                    minimum_accuracy=minimum_component_accuracy,
                    minimum_accuracy_lower_bound=minimum_component_accuracy_lower_bound,
                )
                metrics[name]["component_metrics"] = component_metrics
                metrics[name]["component_safety"] = check
                component_checks[name] = check
            if not all(check["passed"] for check in component_checks.values()):
                continue
            coverages = [row["auto_assignment_coverage"] for row in metrics.values()]
            accuracies = [row["auto_assignment_accuracy"] for row in metrics.values()]
            unseen_rates = [row["unseen_auto_assignment_rate"] for row in metrics.values()]
            candidate = {
                "found": True,
                "open_set_threshold": round(float(risk_threshold), 8),
                "t_high": round(float(high_threshold), 8),
                "minimum_high_confidence": minimum_high_confidence,
                "minimum_auto_rows_per_window": minimum_auto_rows_per_window,
                "minimum_accuracy_lower_bound": minimum_accuracy_lower_bound,
                "minimum_candidate_source_count": minimum_candidate_source_count,
                "maximum_unseen_rate_upper_bound": maximum_unseen_rate_upper_bound,
                "minimum_component_auto_rows": minimum_component_auto_rows,
                "minimum_component_accuracy": minimum_component_accuracy,
                "minimum_component_accuracy_lower_bound": (
                    minimum_component_accuracy_lower_bound
                ),
                "selection_window_metrics": metrics,
                "minimum_window_coverage": round(min(coverages), 6),
                "mean_window_coverage": round(float(np.mean(coverages)), 6),
                "mean_window_accuracy": round(float(np.mean(accuracies)), 6),
                "maximum_window_unseen_auto_rate": round(max(unseen_rates), 6),
                "maximum_window_unseen_auto_rate_ci95_upper": round(
                    max(
                        row["unseen_auto_assignment_rate_ci95"]["upper"]
                        for row in metrics.values()
                    ),
                    6,
                ),
            }
            if best is None or (
                candidate["minimum_window_coverage"],
                candidate["mean_window_coverage"],
                candidate["mean_window_accuracy"],
                -candidate["maximum_window_unseen_auto_rate"],
            ) > (
                best["minimum_window_coverage"],
                best["mean_window_coverage"],
                best["mean_window_accuracy"],
                -best["maximum_window_unseen_auto_rate"],
            ):
                best = candidate
    if best is None:
        return {
            "found": False,
            "open_set_threshold": 0.0,
            "t_high": 1.000001,
            "t_low": 1.000001,
            "blocker": "no_single_policy_passed_every_development_window",
        }

    low_candidates = [
        value
        for value in quantile_candidates(pooled, "calibrated_probability", include=(best["t_high"],))
        if minimum_review_confidence <= value <= best["t_high"]
    ]
    review_risk_values = [value for value in risk_values if value >= best["open_set_threshold"]]
    best_review: dict[str, Any] | None = None
    for review_risk_threshold in review_risk_values:
        for threshold in low_candidates:
            metrics = {
                name: _review_metrics(
                    values,
                    best["open_set_threshold"],
                    review_risk_threshold,
                    best["t_high"],
                    threshold,
                    minimum_candidate_source_count,
                )
                for name, values in arrays.items()
            }
            if not all(
                row["top3_confirmation_rows"] >= minimum_review_rows_per_window
                and row["top3_confirmation_rate"] >= minimum_review_coverage
                and row["top3_confirmation_accuracy"] >= target_review_accuracy
                for row in metrics.values()
            ):
                continue
            rates = [row["top3_confirmation_rate"] for row in metrics.values()]
            candidate = {
                "open_set_review_threshold": round(float(review_risk_threshold), 8),
                "t_low": round(float(threshold), 8),
                "minimum_review_confidence": minimum_review_confidence,
                "minimum_review_rows_per_window": minimum_review_rows_per_window,
                "minimum_review_coverage": minimum_review_coverage,
                "review_policy_found": True,
                "review_window_metrics": metrics,
                "minimum_window_review_rate": round(min(rates), 6),
                "mean_window_review_rate": round(float(np.mean(rates)), 6),
            }
            if best_review is None or (
                candidate["minimum_window_review_rate"],
                candidate["mean_window_review_rate"],
            ) > (
                best_review["minimum_window_review_rate"],
                best_review["mean_window_review_rate"],
            ):
                best_review = candidate
    best.update(
        best_review
        or {
            "open_set_review_threshold": best["open_set_threshold"],
            "t_low": best["t_high"],
            "review_window_metrics": {},
            "minimum_window_review_rate": 0.0,
            "mean_window_review_rate": 0.0,
            "minimum_review_rows_per_window": minimum_review_rows_per_window,
            "minimum_review_coverage": minimum_review_coverage,
            "review_policy_found": False,
            "review_policy_blocker": "no_top3_policy_passed_every_development_window",
        }
    )
    return best


def select_multi_window_top5_assist_policy(
    windows: dict[str, list[dict[str, Any]]],
    *,
    target_accuracy: float = 0.85,
    minimum_coverage: float = 0.10,
    minimum_rows_per_window: int = 100,
    minimum_accuracy_lower_bound: float = 0.80,
    minimum_candidate_source_count: int = 2,
    minimum_confidence: float = 0.0,
    top_k: int = 5,
    correctness_field: str = "is_top5_correct",
) -> dict[str, Any]:
    """Freeze one selective Top-K policy that passes every development window.

    This policy can only expose candidates for user confirmation. It never grants
    automatic-assignment authority.
    """
    if top_k != 5:
        raise ValueError("the semi-automatic policy currently supports top_k=5 only")
    if len(windows) < 2 or any(not rows for rows in windows.values()):
        raise ValueError("Top-5 policy selection requires at least two non-empty windows")
    if not correctness_field.strip():
        raise ValueError("correctness_field is required")
    pooled = [row for rows in windows.values() for row in rows]
    risk_values = quantile_candidates(
        pooled, "open_set_probability", include=(0.0, 1.000001)
    )
    confidence_values = [
        value
        for value in quantile_candidates(
            pooled, "calibrated_probability", include=(1.000001,)
        )
        if value >= minimum_confidence
    ]
    arrays = {
        name: _routing_arrays(rows, top5_correctness_field=correctness_field)
        for name, rows in windows.items()
    }
    best: dict[str, Any] | None = None
    for risk_threshold in risk_values:
        for confidence_threshold in confidence_values:
            metrics = {
                name: _top5_assist_threshold_metrics(
                    values,
                    risk_threshold=risk_threshold,
                    confidence_threshold=confidence_threshold,
                    minimum_candidate_source_count=minimum_candidate_source_count,
                    top_k=top_k,
                    require_all_candidates_eligible=(
                        correctness_field == "is_top5_eligible_correct"
                    ),
                )
                for name, values in arrays.items()
            }
            if not all(
                row["top5_assist_accuracy"] >= target_accuracy
                and row["top5_assist_coverage"] >= minimum_coverage
                and row["top5_assist_rows"] >= minimum_rows_per_window
                and row["top5_assist_accuracy_ci95"]["lower"]
                >= minimum_accuracy_lower_bound
                for row in metrics.values()
            ):
                continue
            coverages = [row["top5_assist_coverage"] for row in metrics.values()]
            accuracies = [row["top5_assist_accuracy"] for row in metrics.values()]
            candidate = {
                "found": True,
                "mode": "top5_user_confirmation_only",
                "top_k": top_k,
                "open_set_threshold": round(float(risk_threshold), 8),
                "minimum_confidence": round(float(confidence_threshold), 8),
                "target_accuracy": target_accuracy,
                "correctness_field": correctness_field,
                "correctness_definition": _correctness_definition(correctness_field),
                "candidate_eligibility_requirement": (
                    "all_displayed_candidates"
                    if correctness_field == "is_top5_eligible_correct"
                    else "active_roster_only"
                ),
                "minimum_coverage": minimum_coverage,
                "minimum_rows_per_window": minimum_rows_per_window,
                "minimum_accuracy_lower_bound": minimum_accuracy_lower_bound,
                "minimum_candidate_source_count": minimum_candidate_source_count,
                "selection_window_metrics": metrics,
                "minimum_window_coverage": round(min(coverages), 6),
                "mean_window_coverage": round(float(np.mean(coverages)), 6),
                "minimum_window_accuracy": round(min(accuracies), 6),
                "mean_window_accuracy": round(float(np.mean(accuracies)), 6),
                "requires_user_selection": True,
                "auto_assignment_authorized": False,
            }
            if best is None or (
                candidate["minimum_window_coverage"],
                candidate["mean_window_coverage"],
                candidate["minimum_window_accuracy"],
            ) > (
                best["minimum_window_coverage"],
                best["mean_window_coverage"],
                best["minimum_window_accuracy"],
            ):
                best = candidate
    if best is not None:
        return best
    return {
        "found": False,
        "mode": "top5_user_confirmation_only",
        "top_k": top_k,
        "open_set_threshold": 0.0,
        "minimum_confidence": 1.000001,
        "target_accuracy": target_accuracy,
        "correctness_field": correctness_field,
        "correctness_definition": _correctness_definition(correctness_field),
        "candidate_eligibility_requirement": (
            "all_displayed_candidates"
            if correctness_field == "is_top5_eligible_correct"
            else "active_roster_only"
        ),
        "minimum_coverage": minimum_coverage,
        "minimum_rows_per_window": minimum_rows_per_window,
        "minimum_accuracy_lower_bound": minimum_accuracy_lower_bound,
        "minimum_candidate_source_count": minimum_candidate_source_count,
        "selection_window_metrics": {},
        "requires_user_selection": True,
        "auto_assignment_authorized": False,
        "blocker": "no_single_top5_policy_passed_every_development_window",
    }


def apply_top5_assist_policy(
    rows: list[dict[str, Any]], policy: dict[str, Any]
) -> None:
    """Annotate Top-5 availability without ever mutating the proposed assignee."""
    for row in rows:
        if policy.get("found") is not True:
            available = False
            status = "manual_triage_top5_policy_unavailable"
            reason = "top5_quality_policy_not_available"
        elif int(safe_float(row.get("candidate_count"))) < int(
            policy.get("top_k", 5)
        ):
            available = False
            status = "manual_triage_insufficient_candidates"
            reason = "fewer_than_five_ranked_candidates"
        elif (
            policy.get("correctness_field") == "is_top5_eligible_correct"
            and row.get("eligibility_label_available") is not True
        ):
            available = False
            status = "manual_triage_eligibility_unavailable"
            reason = "reviewed_time_valid_eligibility_unavailable"
        elif (
            policy.get("correctness_field") == "is_top5_eligible_correct"
            and int(safe_float(row.get("eligible_top5_count")))
            < int(policy.get("top_k", 5))
        ):
            available = False
            status = "manual_triage_invalid_owner"
            reason = "top5_contains_unqualified_owner"
        elif int(safe_float(row.get("candidate_source_count"))) < int(
            policy.get("minimum_candidate_source_count", 0)
        ):
            available = False
            status = "manual_triage_insufficient_evidence"
            reason = "insufficient_candidate_sources"
        elif safe_float(row.get("open_set_probability")) >= float(
            policy["open_set_threshold"]
        ):
            available = False
            status = "manual_triage_open_set"
            reason = "unseen_owner_risk"
        elif safe_float(row.get("calibrated_probability")) < float(
            policy["minimum_confidence"]
        ):
            available = False
            status = "manual_triage_low_confidence"
            reason = "top5_quality_gate_low_confidence"
        else:
            available = True
            status = "top5_user_confirmation"
            reason = "top5_candidates_require_user_selection"
        row["top5_assist_available"] = available
        row["top5_assist_status"] = status
        row["top5_assist_reason"] = reason


def top5_assist_metrics(
    rows: list[dict[str, Any]], *, correctness_field: str = "is_top5_correct"
) -> dict[str, Any]:
    if correctness_field != "is_top5_correct" and any(
        correctness_field not in row for row in rows
    ):
        raise ValueError(
            f"Top-5 correctness field is missing from one or more rows: {correctness_field}"
        )
    assisted = [row for row in rows if row.get("top5_assist_available") is True]
    correct = sum(bool(row.get(correctness_field)) for row in assisted)
    result = {
        "rows": len(rows),
        "top5_assist_rows": len(assisted),
        "top5_assist_correct_rows": correct,
        "top5_assist_coverage": rounded_ratio(len(assisted), len(rows)),
        "top5_assist_accuracy": rounded_ratio(correct, len(assisted)),
        "top5_assist_accuracy_ci95": wilson_interval(correct, len(assisted)),
        "manual_triage_rows": len(rows) - len(assisted),
        "auto_assignment_rows": 0,
        "correctness_field": correctness_field,
        "correctness_definition": _correctness_definition(correctness_field),
    }
    if correctness_field != "is_top5_correct":
        exact_correct = sum(bool(row.get("is_top5_correct")) for row in assisted)
        eligible_slots = sum(int(safe_float(row.get("eligible_top5_count"))) for row in assisted)
        candidate_slots = sum(
            min(
                5,
                max(
                    len(row.get("ranked_candidates") or []),
                    int(safe_float(row.get("candidate_count"))),
                ),
            )
            for row in assisted
        )
        eligible_slots = min(eligible_slots, candidate_slots)
        result.update(
            {
                "secondary_exact_owner_correct_rows": exact_correct,
                "secondary_exact_owner_accuracy": rounded_ratio(
                    exact_correct, len(assisted)
                ),
                "secondary_exact_owner_accuracy_ci95": wilson_interval(
                    exact_correct, len(assisted)
                ),
                "top5_candidate_eligibility_precision": rounded_ratio(
                    eligible_slots, candidate_slots
                ),
                "top5_candidate_eligibility_precision_ci95": wilson_interval(
                    eligible_slots, candidate_slots
                ),
            }
        )
    return result


def top5_assist_gate(
    metrics: dict[str, Any],
    *,
    target_accuracy: float = 0.85,
    minimum_coverage: float = 0.10,
    minimum_rows: int = 100,
    minimum_accuracy_lower_bound: float = 0.80,
) -> dict[str, Any]:
    checks = {
        "top5_assist_accuracy": metrics.get("top5_assist_accuracy", 0.0)
        >= target_accuracy,
        "top5_assist_coverage": metrics.get("top5_assist_coverage", 0.0)
        >= minimum_coverage,
        "minimum_top5_assist_rows": metrics.get("top5_assist_rows", 0)
        >= minimum_rows,
        "top5_accuracy_confidence_lower_bound": metrics.get(
            "top5_assist_accuracy_ci95", {"lower": 0.0}
        )["lower"]
        >= minimum_accuracy_lower_bound,
        "never_auto_assigns": metrics.get("auto_assignment_rows", 0) == 0,
    }
    if metrics.get("correctness_field") == "is_top5_eligible_correct":
        checks["all_displayed_candidates_reviewed_eligible"] = metrics.get(
            "top5_candidate_eligibility_precision", 0.0
        ) == 1.0
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "targets": {
            "target_top5_assist_accuracy": target_accuracy,
            "minimum_top5_assist_coverage": minimum_coverage,
            "minimum_top5_assist_rows": minimum_rows,
            "minimum_top5_assist_accuracy_lower_bound": (
                minimum_accuracy_lower_bound
            ),
        },
        "requires_user_selection": True,
        "auto_assignment_authorized": False,
    }


def route_predictions(rows: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    for row in rows:
        risk = row["open_set_probability"]
        confidence = row["calibrated_probability"]
        auto_risk_threshold = policy["open_set_threshold"]
        review_risk_threshold = policy.get("open_set_review_threshold", auto_risk_threshold)
        require_active = bool(policy.get("require_active_owner", False))
        minimum_sources = max(0, int(policy.get("minimum_candidate_source_count", 0)))
        source_count = int(safe_float(row.get("candidate_source_count")))
        if require_active and row.get("predicted_owner_active") is not True:
            status = "manual_triage_invalid_owner"
            reason = "predicted_owner_not_confirmed_active"
        elif source_count < minimum_sources:
            status = "manual_triage_insufficient_evidence"
            reason = "insufficient_candidate_sources"
        elif risk < auto_risk_threshold and confidence >= policy["t_high"]:
            status = "auto_assign"
            reason = "high_confidence_low_open_set_risk"
        elif risk < review_risk_threshold and confidence >= policy["t_low"]:
            status = "top3_confirmation"
            reason = "top3_confirmation_required"
        elif risk >= review_risk_threshold:
            status = "manual_triage_open_set"
            reason = "unseen_owner_risk"
        else:
            status = "manual_triage_low_confidence"
            reason = "low_ranking_confidence"
        row["routing_status"] = status
        row["fallback_reason"] = reason


def routing_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    auto = [row for row in rows if row["routing_status"] == "auto_assign"]
    review = [row for row in rows if row["routing_status"] == "top3_confirmation"]
    unknown = [row for row in rows if not row["known_owner"]]
    unknown_auto = [row for row in unknown if row["routing_status"] == "auto_assign"]
    correct_auto = sum(bool(row["is_top1_correct"]) for row in auto)
    breakdown = Counter(row["routing_status"] for row in rows)
    return {
        "rows": len(rows),
        "known_rows": len(rows) - len(unknown),
        "unseen_rows": len(unknown),
        "auto_assignment_rows": len(auto),
        "auto_assignment_correct_rows": correct_auto,
        "auto_assignment_coverage": rounded_ratio(len(auto), len(rows)),
        "auto_assignment_accuracy": rounded_ratio(correct_auto, len(auto)),
        "auto_assignment_accuracy_ci95": wilson_interval(correct_auto, len(auto)),
        "unseen_auto_assignment_rows": len(unknown_auto),
        "unseen_auto_assignment_rate": rounded_ratio(len(unknown_auto), len(unknown)),
        "unseen_auto_assignment_rate_ci95": wilson_interval(len(unknown_auto), len(unknown)),
        "top3_confirmation_rows": len(review),
        "top3_confirmation_rate": rounded_ratio(len(review), len(rows)),
        "top3_confirmation_accuracy": rounded_ratio(
            sum(row["is_top3_correct"] for row in review), len(review)
        ),
        "manual_triage_rate": rounded_ratio(
            sum(status.startswith("manual_triage") for status in (row["routing_status"] for row in rows)),
            len(rows),
        ),
        "routing_status_breakdown": dict(sorted(breakdown.items())),
        "fallback_reason_breakdown": dict(sorted(Counter(row["fallback_reason"] for row in rows).items())),
        "component_metrics": component_routing_metrics(rows),
    }


def deployment_gate(
    metrics: dict[str, Any],
    *,
    target_auto_accuracy: float,
    minimum_auto_coverage: float,
    maximum_unseen_auto_rate: float,
    minimum_holdout_rows: int = 0,
    minimum_auto_rows: int = 0,
    minimum_accuracy_lower_bound: float = 0.0,
    minimum_component_auto_rows: int = 30,
    minimum_component_accuracy: float = 0.0,
    minimum_component_accuracy_lower_bound: float = 0.0,
    maximum_unseen_rate_upper_bound: float | None = None,
) -> dict[str, Any]:
    component_failures = {
        name: values
        for name, values in metrics.get("component_metrics", {}).items()
        if values.get("auto_assignment_rows", 0) >= minimum_component_auto_rows
        and values.get("auto_assignment_accuracy", 0.0) < minimum_component_accuracy
    }
    component_confidence_failures = {
        name: values
        for name, values in metrics.get("component_metrics", {}).items()
        if values.get("auto_assignment_rows", 0) >= minimum_component_auto_rows
        and values.get("auto_assignment_accuracy_ci95", {"lower": 0.0})["lower"]
        < minimum_component_accuracy_lower_bound
    }
    checks = {
        "auto_assignment_accuracy": metrics["auto_assignment_accuracy"] >= target_auto_accuracy,
        "auto_assignment_coverage": metrics["auto_assignment_coverage"] >= minimum_auto_coverage,
        "unseen_auto_assignment_rate": metrics["unseen_auto_assignment_rate"] < maximum_unseen_auto_rate,
        "minimum_holdout_rows": metrics.get("rows", 0) >= minimum_holdout_rows,
        "minimum_auto_rows": metrics.get("auto_assignment_rows", 0) >= minimum_auto_rows,
        "auto_accuracy_confidence_lower_bound": metrics.get(
            "auto_assignment_accuracy_ci95", {"lower": 0.0}
        )["lower"]
        >= minimum_accuracy_lower_bound,
        "component_accuracy_floor": not component_failures,
        "component_accuracy_confidence_floor": not component_confidence_failures,
    }
    if maximum_unseen_rate_upper_bound is not None:
        checks["unseen_auto_assignment_rate_confidence_upper_bound"] = (
            metrics.get("unseen_auto_assignment_rate_ci95", {"upper": 1.0})["upper"]
            < maximum_unseen_rate_upper_bound
        )
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": failed_checks,
        "component_failures": component_failures,
        "component_confidence_failures": component_confidence_failures,
        "targets": {
            "target_auto_accuracy": target_auto_accuracy,
            "minimum_auto_coverage": minimum_auto_coverage,
            "maximum_unseen_auto_rate": maximum_unseen_auto_rate,
            "minimum_holdout_rows": minimum_holdout_rows,
            "minimum_auto_rows": minimum_auto_rows,
            "minimum_accuracy_lower_bound": minimum_accuracy_lower_bound,
            "minimum_component_auto_rows": minimum_component_auto_rows,
            "minimum_component_accuracy": minimum_component_accuracy,
            "minimum_component_accuracy_lower_bound": minimum_component_accuracy_lower_bound,
            "maximum_unseen_rate_upper_bound": maximum_unseen_rate_upper_bound,
        },
        "blocker": (
            "explicit_operator_and_roster_approval_required"
            if all(checks.values())
            else "routing_safety_gate_failed:" + ",".join(failed_checks)
        ),
    }


def component_routing_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("component") or "unknown")].append(row)
    result: dict[str, Any] = {}
    for component, members in sorted(groups.items()):
        auto = [row for row in members if row.get("routing_status") == "auto_assign"]
        correct = sum(bool(row.get("is_top1_correct")) for row in auto)
        result[component] = {
            "rows": len(members),
            "auto_assignment_rows": len(auto),
            "auto_assignment_coverage": rounded_ratio(len(auto), len(members)),
            "auto_assignment_accuracy": rounded_ratio(correct, len(auto)),
            "auto_assignment_accuracy_ci95": wilson_interval(correct, len(auto)),
        }
    return result


def routing_score_diagnostics(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    calibrated = np.asarray([safe_float(row.get("calibrated_probability")) for row in rows])
    open_set = np.asarray([safe_float(row.get("open_set_probability")) for row in rows])
    high = float(policy["t_high"])
    open_threshold = float(policy["open_set_threshold"])
    minimum_sources = max(0, int(policy.get("minimum_candidate_source_count", 0)))
    source_count = np.asarray(
        [
            max(0, int(safe_float(row.get("candidate_source_count"))))
            for row in rows
        ]
    )
    below_open_gate = open_set < open_threshold
    enough_sources = source_count >= minimum_sources
    joint = below_open_gate & (calibrated >= high) & enough_sources
    return {
        "rows": len(rows),
        "calibrated_probability": distribution_summary(calibrated),
        "open_set_probability": distribution_summary(open_set),
        "frozen_t_high": high,
        "frozen_open_set_threshold": open_threshold,
        "rows_above_high_confidence": int(np.sum(calibrated >= high)),
        "rows_below_open_set_gate": int(np.sum(below_open_gate)),
        "minimum_candidate_source_count": minimum_sources,
        "rows_with_minimum_candidate_sources": int(np.sum(enough_sources)),
        "rows_passing_both_auto_assignment_gates": int(np.sum(joint)),
        "maximum_calibrated_probability_below_open_set_gate": (
            round(float(np.max(calibrated[below_open_gate])), 8) if np.any(below_open_gate) else None
        ),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": round(float(np.min(values)), 8),
        "p25": round(float(np.quantile(values, 0.25)), 8),
        "median": round(float(np.quantile(values, 0.50)), 8),
        "p75": round(float(np.quantile(values, 0.75)), 8),
        "maximum": round(float(np.max(values)), 8),
    }


def calibration_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "brier": None, "ece": None}
    labels = np.asarray([int(row["is_top1_correct"]) for row in rows], dtype=int)
    probabilities = np.asarray([safe_float(row.get("calibrated_probability")) for row in rows])
    return {
        "rows": len(rows),
        "positive_rate": round(float(np.mean(labels)), 6),
        "mean_probability": round(float(np.mean(probabilities)), 6),
        "brier": round(float(brier_score_loss(labels, probabilities)), 6),
        "ece": round(expected_calibration_error(labels, probabilities), 6),
    }


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = (probabilities >= low) & (
            (probabilities < high) | ((index == bins - 1) & (probabilities == 1.0))
        )
        if not np.any(members):
            continue
        result += float(np.mean(members)) * abs(
            float(np.mean(labels[members])) - float(np.mean(probabilities[members]))
        )
    return result


def build_drift_reference(
    rows: list[dict[str, Any]],
    feature_names: Iterable[str],
    *,
    bins: int = 10,
    maximum_psi: float = 0.25,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("drift reference requires rows")
    features: dict[str, Any] = {}
    for name in feature_names:
        values = np.asarray([safe_float(row.get(name)) for row in rows])
        boundaries = sorted(
            set(float(value) for value in np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
        )
        counts = np.bincount(np.digitize(values, boundaries), minlength=len(boundaries) + 1)
        proportions = (counts + 0.5) / (len(values) + 0.5 * len(counts))
        features[name] = {
            "boundaries": boundaries,
            "expected_proportions": [float(value) for value in proportions],
            "reference_distribution": distribution_summary(values),
        }
    return {
        "method": "population_stability_index_v1",
        "maximum_psi": maximum_psi,
        "reference_rows": len(rows),
        "features": features,
    }


def evaluate_score_drift(rows: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "severe_drift": True, "blocker": "empty_evaluation_window"}
    feature_metrics: dict[str, Any] = {}
    for name, payload in reference["features"].items():
        values = np.asarray([safe_float(row.get(name)) for row in rows])
        boundaries = np.asarray(payload["boundaries"], dtype=float)
        counts = np.bincount(np.digitize(values, boundaries), minlength=len(boundaries) + 1)
        actual = (counts + 0.5) / (len(values) + 0.5 * len(counts))
        expected = np.asarray(payload["expected_proportions"], dtype=float)
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        feature_metrics[name] = {
            "psi": round(psi, 6),
            "current_distribution": distribution_summary(values),
        }
    maximum = max(row["psi"] for row in feature_metrics.values())
    threshold = float(reference.get("maximum_psi", 0.25))
    return {
        "rows": len(rows),
        "maximum_psi": maximum,
        "threshold": threshold,
        "severe_drift": maximum > threshold,
        "features": feature_metrics,
    }


def quantile_candidates(
    rows: list[dict[str, Any]], key: str, *, include: tuple[float, ...] = ()
) -> list[float]:
    if not rows:
        return sorted(set(include))
    values = np.asarray([safe_float(row.get(key)) for row in rows], dtype=float)
    quantiles = np.linspace(0.0, 1.0, min(101, len(values)))
    return sorted({float(value) for value in np.quantile(values, quantiles)} | set(include))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _assignee(row: dict[str, Any]) -> str:
    return str(row.get("assignee") or row.get("assigned_to") or "").strip().lower()


def _timestamp(row: dict[str, Any]) -> str:
    return str(row.get("created_at") or row.get("creation_time") or "")


def _sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (_timestamp(row), str(row.get("ticket_id") or row.get("id") or ""))


def _stable_order(owner: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{owner}".encode("utf-8")).hexdigest()


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def ratio_int(numerator: Any, denominator: Any) -> float:
    denominator_value = int(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


def rounded_ratio(numerator: Any, denominator: Any) -> float:
    return round(ratio_int(numerator, denominator), 6)


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> dict[str, float]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return {"lower": 0.0, "upper": 1.0}
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "lower": round(max(0.0, center - margin), 6),
        "upper": round(min(1.0, center + margin), 6),
    }


def metric_or_none(metric: Any, labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(int(value) for value in labels)) < 2:
        return None
    return round(float(metric(labels, scores)), 6)


def _routing_arrays(
    rows: list[dict[str, Any]], *, top5_correctness_field: str = "is_top5_correct"
) -> dict[str, np.ndarray]:
    if top5_correctness_field != "is_top5_correct" and any(
        top5_correctness_field not in row for row in rows
    ):
        raise ValueError(
            "Top-5 correctness field is missing from one or more rows: "
            + top5_correctness_field
        )
    return {
        "risk": np.asarray([safe_float(row.get("open_set_probability")) for row in rows]),
        "confidence": np.asarray([safe_float(row.get("calibrated_probability")) for row in rows]),
        "correct": np.asarray([bool(row["is_top1_correct"]) for row in rows]),
        "top3": np.asarray([bool(row["is_top3_correct"]) for row in rows]),
        "top5": np.asarray(
            [
                bool(
                    row.get(
                        top5_correctness_field,
                        row.get("is_top3_correct")
                        if top5_correctness_field == "is_top5_correct"
                        else False,
                    )
                )
                for row in rows
            ]
        ),
        "exact_top5": np.asarray(
            [bool(row.get("is_top5_correct", row.get("is_top3_correct"))) for row in rows]
        ),
        "unknown": np.asarray([not bool(row["known_owner"]) for row in rows]),
        "source_count": np.asarray(
            [
                max(
                    0,
                    int(
                        safe_float(
                            row.get("candidate_source_count", row.get("source_count"))
                        )
                    ),
                )
                for row in rows
            ]
        ),
        "candidate_count": np.asarray(
            [
                max(
                    0,
                    int(
                        safe_float(
                            row.get(
                                "candidate_count",
                                len(row.get("ranked_candidates", []))
                                if isinstance(row.get("ranked_candidates"), list)
                                else 0,
                            )
                        )
                    ),
                )
                for row in rows
            ]
        ),
        "eligibility_label_available": np.asarray(
            [bool(row.get("eligibility_label_available")) for row in rows]
        ),
        "eligible_top5_count": np.asarray(
            [max(0, int(safe_float(row.get("eligible_top5_count")))) for row in rows]
        ),
        "component": np.asarray(
            [str(row.get("component") or "unknown") for row in rows], dtype=object
        ),
    }


def _correctness_definition(field: str) -> str:
    if field == "is_top5_eligible_correct":
        return "at_least_one_candidate_in_reviewed_time_valid_capability_set"
    if field == "is_top5_correct":
        return "historical_exact_assignee_in_top5"
    return f"custom_boolean_field:{field}"


def _top5_assist_threshold_metrics(
    values: dict[str, np.ndarray],
    *,
    risk_threshold: float,
    confidence_threshold: float,
    minimum_candidate_source_count: int,
    top_k: int,
    require_all_candidates_eligible: bool = False,
) -> dict[str, Any]:
    assisted = (
        (values["risk"] < risk_threshold)
        & (values["confidence"] >= confidence_threshold)
        & (values["source_count"] >= minimum_candidate_source_count)
        & (values["candidate_count"] >= top_k)
    )
    if require_all_candidates_eligible:
        assisted &= values["eligibility_label_available"] & (
            values["eligible_top5_count"] >= top_k
        )
    assisted_rows = int(np.sum(assisted))
    correct_rows = int(np.sum(values["top5"] & assisted))
    exact_correct_rows = int(np.sum(values["exact_top5"] & assisted))
    eligible_candidate_slots = int(
        np.sum(values["eligible_top5_count"][assisted])
    )
    candidate_slots = assisted_rows * top_k
    eligible_candidate_slots = min(eligible_candidate_slots, candidate_slots)
    total = len(values["risk"])
    result = {
        "rows": total,
        "top5_assist_rows": assisted_rows,
        "top5_assist_correct_rows": correct_rows,
        "top5_assist_coverage": rounded_ratio(assisted_rows, total),
        "top5_assist_accuracy": rounded_ratio(correct_rows, assisted_rows),
        "top5_assist_accuracy_ci95": wilson_interval(correct_rows, assisted_rows),
    }
    if require_all_candidates_eligible:
        result.update(
            {
                "secondary_exact_owner_correct_rows": exact_correct_rows,
                "secondary_exact_owner_accuracy": rounded_ratio(
                    exact_correct_rows, assisted_rows
                ),
                "secondary_exact_owner_accuracy_ci95": wilson_interval(
                    exact_correct_rows, assisted_rows
                ),
                "top5_candidate_eligibility_precision": rounded_ratio(
                    eligible_candidate_slots, candidate_slots
                ),
                "top5_candidate_eligibility_precision_ci95": wilson_interval(
                    eligible_candidate_slots, candidate_slots
                ),
            }
        )
    return result


def _auto_metrics(
    values: dict[str, np.ndarray],
    risk_threshold: float,
    high_threshold: float,
    minimum_candidate_source_count: int = 0,
) -> dict[str, Any]:
    accepted = (
        (values["risk"] < risk_threshold)
        & (values["confidence"] >= high_threshold)
        & (values["source_count"] >= minimum_candidate_source_count)
    )
    accepted_rows = int(np.sum(accepted))
    correct_rows = int(np.sum(values["correct"] & accepted))
    unknown_rows = int(np.sum(values["unknown"]))
    unknown_auto_rows = int(np.sum(values["unknown"] & accepted))
    return {
        "auto_assignment_rows": accepted_rows,
        "auto_assignment_correct_rows": correct_rows,
        "auto_assignment_accuracy": ratio_int(correct_rows, accepted_rows),
        "auto_assignment_accuracy_ci95": wilson_interval(correct_rows, accepted_rows),
        "auto_assignment_coverage": ratio_int(accepted_rows, len(values["risk"])),
        "unseen_auto_assignment_rate": ratio_int(
            unknown_auto_rows, unknown_rows
        ),
        "unseen_auto_assignment_rows": unknown_auto_rows,
        "unseen_rows": unknown_rows,
        "unseen_auto_assignment_rate_ci95": wilson_interval(
            unknown_auto_rows, unknown_rows
        ),
    }


def _component_metrics_for_threshold(
    values: dict[str, np.ndarray],
    *,
    risk_threshold: float,
    high_threshold: float,
    minimum_candidate_source_count: int,
) -> dict[str, Any]:
    accepted = (
        (values["risk"] < risk_threshold)
        & (values["confidence"] >= high_threshold)
        & (values["source_count"] >= minimum_candidate_source_count)
    )
    result: dict[str, Any] = {}
    for component in sorted(set(str(value) for value in values["component"])):
        members = values["component"] == component
        selected = members & accepted
        selected_rows = int(np.sum(selected))
        correct_rows = int(np.sum(values["correct"] & selected))
        result[component] = {
            "rows": int(np.sum(members)),
            "auto_assignment_rows": selected_rows,
            "auto_assignment_coverage": ratio_int(selected_rows, np.sum(members)),
            "auto_assignment_accuracy": ratio_int(correct_rows, selected_rows),
            "auto_assignment_accuracy_ci95": wilson_interval(
                correct_rows, selected_rows
            ),
        }
    return result


def _component_policy_checks(
    component_metrics: dict[str, Any],
    *,
    minimum_auto_rows: int,
    minimum_accuracy: float,
    minimum_accuracy_lower_bound: float,
) -> dict[str, Any]:
    eligible = {
        name: values
        for name, values in component_metrics.items()
        if int(values.get("auto_assignment_rows", 0)) >= minimum_auto_rows
    }
    point_failures = {
        name: values
        for name, values in eligible.items()
        if float(values.get("auto_assignment_accuracy", 0.0)) < minimum_accuracy
    }
    confidence_failures = {
        name: values
        for name, values in eligible.items()
        if float(
            values.get("auto_assignment_accuracy_ci95", {"lower": 0.0}).get(
                "lower", 0.0
            )
        )
        < minimum_accuracy_lower_bound
    }
    return {
        "passed": not point_failures and not confidence_failures,
        "eligible_components": len(eligible),
        "point_failures": point_failures,
        "confidence_failures": confidence_failures,
        "targets": {
            "minimum_auto_rows": minimum_auto_rows,
            "minimum_accuracy": minimum_accuracy,
            "minimum_accuracy_lower_bound": minimum_accuracy_lower_bound,
        },
    }


def _review_metrics(
    values: dict[str, np.ndarray],
    auto_risk_threshold: float,
    review_risk_threshold: float,
    high_threshold: float,
    low_threshold: float,
    minimum_candidate_source_count: int = 0,
) -> dict[str, Any]:
    source_eligible = values["source_count"] >= minimum_candidate_source_count
    auto = (
        (values["risk"] < auto_risk_threshold)
        & (values["confidence"] >= high_threshold)
        & source_eligible
    )
    accepted = (
        ~auto
        & (values["risk"] < review_risk_threshold)
        & (values["confidence"] >= low_threshold)
        & source_eligible
    )
    accepted_rows = int(np.sum(accepted))
    return {
        "top3_confirmation_rows": accepted_rows,
        "top3_confirmation_accuracy": ratio_int(np.sum(values["top3"] & accepted), accepted_rows),
        "top3_confirmation_rate": ratio_int(accepted_rows, len(values["risk"])),
    }
