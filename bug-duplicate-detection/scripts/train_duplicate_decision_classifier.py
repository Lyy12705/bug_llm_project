#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_NUMERIC_FEATURES = ("score", "title_score", "content_score", "second_score", "score_margin")
METADATA_FEATURES = ("component_match", "product_match", "severity_match", "priority_match")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight binary duplicate decision classifier from decision-feature CSVs.")
    parser.add_argument("--decisions-csv", required=True, help="Primary decisions CSV produced by tune-threshold --output-decisions-csv")
    parser.add_argument(
        "--extra-decisions-csv",
        action="append",
        default=[],
        help="Optional extra score source as prefix=path, e.g. tfidf=reports/tfidf_decisions.csv",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-json", default="reports/decision_classifier_metrics.json")
    parser.add_argument("--output-model", default="models/duplicate_decision_classifier.joblib")
    parser.add_argument("--output-predictions-csv", default="reports/decision_classifier_predictions.csv")
    args = parser.parse_args()

    primary_rows = read_rows(args.decisions_csv)
    merged_rows = merge_extra_sources(primary_rows, args.extra_decisions_csv)
    feature_names, x, y, query_ids = build_feature_matrix(merged_rows)
    if len(set(y.tolist())) < 2:
        raise SystemExit("Need both duplicate and non-duplicate rows to train a classifier")

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed),
    )
    model.fit(x[train_idx], y[train_idx])
    probabilities = model.predict_proba(x[test_idx])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = evaluate_predictions(y[test_idx], predictions)
    metrics.update(
        {
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "features": feature_names,
            "feature_weights": feature_weights(model, feature_names),
        }
    )

    write_json(metrics, args.output_json)
    write_model(model, args.output_model)
    write_predictions(args.output_predictions_csv, query_ids, test_idx, y[test_idx], predictions, probabilities)
    print(render_metrics(metrics))
    print(f"output_json={args.output_json}")
    print(f"output_model={args.output_model}")
    print(f"output_predictions_csv={args.output_predictions_csv}")
    return 0


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_extra_sources(primary_rows: list[dict[str, str]], specs: list[str]) -> list[dict[str, str]]:
    merged = [dict(row) for row in primary_rows]
    by_query = {row.get("query_id", ""): row for row in merged}
    for spec in specs:
        prefix, path = parse_extra_spec(spec)
        for row in read_rows(path):
            query_id = row.get("query_id", "")
            target = by_query.get(query_id)
            if target is None:
                continue
            for feature in BASE_NUMERIC_FEATURES:
                if row.get(feature):
                    target[f"{prefix}_{feature}"] = row[feature]
    return merged


def parse_extra_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise SystemExit("--extra-decisions-csv must use prefix=path")
    prefix, path = value.split("=", maxsplit=1)
    prefix = prefix.strip()
    if not prefix:
        raise SystemExit("Extra decision prefix cannot be empty")
    return prefix, path


def build_feature_matrix(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    extra_numeric = sorted({
        key for row in rows for key in row
        if key not in BASE_NUMERIC_FEATURES
        if any(key.endswith(f"_{feature}") for feature in BASE_NUMERIC_FEATURES)
    })
    feature_names = list(BASE_NUMERIC_FEATURES) + list(METADATA_FEATURES) + extra_numeric
    values: list[list[float]] = []
    labels: list[int] = []
    query_ids: list[str] = []
    for row in rows:
        labels.append(1 if row.get("actual_is_duplicate") == "true" else 0)
        query_ids.append(row.get("query_id", ""))
        values.append([feature_value(row, feature) for feature in feature_names])
    return feature_names, np.asarray(values, dtype=float), np.asarray(labels, dtype=int), query_ids


def feature_value(row: dict[str, str], feature: str) -> float:
    if feature in METADATA_FEATURES:
        return 1.0 if row.get(feature) == "true" else 0.0
    value = row.get(feature, "")
    if not value:
        return 0.0
    return float(value)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def feature_weights(model, feature_names: list[str]) -> dict[str, float]:
    classifier = model.named_steps["logisticregression"]
    weights = classifier.coef_[0]
    return {
        feature: float(weight)
        for feature, weight in sorted(zip(feature_names, weights), key=lambda item: abs(item[1]), reverse=True)
    }


def write_json(metrics: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_model(model, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


def write_predictions(
    output_path: str | Path,
    query_ids: list[str],
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "actual_is_duplicate", "predicted_is_duplicate", "duplicate_probability"])
        writer.writeheader()
        for row_index, actual, predicted, probability in zip(test_idx, y_true, y_pred, probabilities):
            writer.writerow(
                {
                    "query_id": query_ids[int(row_index)],
                    "actual_is_duplicate": str(bool(actual)).lower(),
                    "predicted_is_duplicate": str(bool(predicted)).lower(),
                    "duplicate_probability": f"{probability:.6f}",
                }
            )


def render_metrics(metrics: dict) -> str:
    lines = [
        "Duplicate Decision Classifier",
        f"precision={metrics['precision']:.4f}",
        f"recall={metrics['recall']:.4f}",
        f"f1={metrics['f1']:.4f}",
        f"accuracy={metrics['accuracy']:.4f}",
        f"true_positives={metrics['true_positives']}",
        f"false_positives={metrics['false_positives']}",
        f"true_negatives={metrics['true_negatives']}",
        f"false_negatives={metrics['false_negatives']}",
        "top_feature_weights",
    ]
    for feature, weight in list(metrics["feature_weights"].items())[:10]:
        lines.append(f"{feature}={weight:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
