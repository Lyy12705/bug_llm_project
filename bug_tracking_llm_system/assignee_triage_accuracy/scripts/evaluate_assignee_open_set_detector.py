from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from assignee_phase1_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_PHASE1_ROOT,
    build_metrics,
    evaluate_current_triager,
    normalize_assignee,
    normalize_component,
    read_jsonl,
    ratio,
    ticket_text,
    tokens,
    write_csv,
    write_json,
    write_jsonl,
)
from assignee_phase2_common import DEFAULT_RAW_PATH, build_window_view, default_windows, load_normalized_raw_records
from evaluate_assignee_calibration import (
    Calibrator,
    fit_calibrators,
    select_best_method,
)


DATASET = "bmo_paper_2024_3k"
DEFAULT_PHASE4_ROOT = Path(__file__).resolve().parents[1] / "phase4_open_set_detection"
OPEN_WORLD_DATA_DIR = DEFAULT_PHASE1_ROOT / "reports" / DATASET / "prefilter_open_world_eval" / "data"
CALIBRATION_TARGET_DATA_DIR = DEFAULT_DATA_DIR
FEATURE_NAMES = [
    "raw_confidence",
    "calibrated_probability",
    "top1_score",
    "top2_score",
    "score_margin",
    "normalized_margin",
    "score_entropy",
    "effective_candidate_count",
    "candidate_count",
    "predicted_assignee_train_frequency",
    "component_history_count",
    "product_component_history_count",
    "product_history_count",
    "distinct_assignees_component",
    "component_owner_entropy",
    "component_top1_owner_share",
    "predicted_component_owner_share",
    "best_bm25_similarity",
    "bm25_margin",
    "high_similarity_ticket_count",
    "retrieval_density",
    "unseen_component",
    "unseen_product",
    "unseen_product_component_pair",
    "rare_component",
    "rare_product_component_pair",
    "metadata_missingness",
    "component_top1_agrees_final",
    "product_component_top1_agrees_final",
    "bm25_top1_agrees_final",
    "signal_count",
    "disagreement_count",
]
EXPECTED_OPEN_WORLD_METRICS = {
    "top1_accuracy": 0.517857,
    "hit_at_3": 0.710714,
    "hit_at_5": 0.771429,
    "hit_at_10": 0.825,
    "mrr": 0.624089,
}


@dataclass
class Detector:
    name: str
    threshold: float
    model: Any | None = None
    feature_names: list[str] | None = None

    def scores(self, rows: list[dict[str, Any]]) -> list[float]:
        if self.name == "negative_raw_confidence":
            return [1.0 - safe_float(row.get("raw_confidence")) for row in rows]
        if self.name == "negative_calibrated_probability":
            return [1.0 - safe_float(row.get("calibrated_probability")) for row in rows]
        if self.name == "negative_margin":
            return [1.0 - safe_float(row.get("normalized_margin")) for row in rows]
        if self.name == "low_historical_support":
            return [low_support_score(row) for row in rows]
        if self.name == "rule_based_novelty":
            return [rule_based_novelty_score(row) for row in rows]
        if self.model is None or not self.feature_names:
            raise SystemExit(f"Detector model is missing for {self.name}")
        matrix = feature_matrix(rows, self.feature_names)
        return [float(value) for value in self.model.predict_proba(matrix)[:, 1]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate open-set assignee detection and unknown routing."
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--approve-rule-based-artifact",
        action="store_true",
        help="Mark the exported rule-based artifact approved after explicit target-domain review.",
    )
    args = parser.parse_args()

    if args.dataset != DATASET:
        raise SystemExit(f"Phase 4 currently expects {DATASET}, got {args.dataset}")

    output_dir = args.output_dir or DEFAULT_PHASE4_ROOT / "reports" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_train, calibration_validation, calibrator, calibrator_name = fit_closed_set_calibrator(args)
    protocol = build_leave_assignee_out_protocol(args, calibrator, output_dir)
    validation_rows = protocol["validation_features"]
    test_bundle = build_open_world_test_bundle(args, calibrator, output_dir)
    test_rows = test_bundle["test_features"]

    detectors = fit_detectors(validation_rows, args.seed)
    comparison_rows, comparison_metrics = compare_detectors(detectors, validation_rows, test_rows, args)
    best_detector_name = select_best_detector(comparison_metrics)
    best_detector = detectors[best_detector_name]
    best_validation_scores = best_detector.scores(validation_rows)
    best_test_scores = best_detector.scores(test_rows)
    best_detector.threshold = select_threshold(best_validation_scores, labels(validation_rows))["threshold"]
    rule_threshold = select_threshold(detectors["rule_based_novelty"].scores(validation_rows), labels(validation_rows))
    write_json(
        output_dir / "assignee_open_set_rule_based_novelty.json",
        build_runtime_open_set_artifact(
            dataset=args.dataset,
            threshold=rule_threshold,
            validation_metrics=comparison_metrics["validation"]["rule_based_novelty"],
            test_metrics=comparison_metrics["test"]["rule_based_novelty"],
            approved=args.approve_rule_based_artifact,
        ),
    )

    prediction_rows = attach_detector_scores(test_rows, detectors, best_detector_name)
    write_csv(output_dir / "open_set_feature_audit.csv", prediction_rows, feature_audit_fields(detectors))
    write_csv(output_dir / "feature_distribution_by_group.csv", feature_distribution_rows(test_rows), feature_distribution_fields())
    write_csv(output_dir / "open_set_detector_comparison.csv", comparison_rows, detector_comparison_fields())
    write_csv(output_dir / "open_set_predictions.csv", prediction_rows, open_set_prediction_fields(detectors))

    routing = compare_routing_policies(
        test_rows=test_rows,
        validation_rows=validation_rows,
        best_detector=best_detector,
        confidence_detector=detectors["negative_calibrated_probability"],
    )
    write_csv(output_dir / "routing_policy_open_set_comparison.csv", routing["rows"], routing_fields())

    failure = failure_analysis(test_rows, best_test_scores, best_detector.threshold)
    write_csv(output_dir / "open_set_false_negative.csv", failure["false_negative"], failure_fields())
    write_csv(output_dir / "open_set_false_positive.csv", failure["false_positive"], failure_fields())
    write_csv(output_dir / "high_confidence_unseen_cases.csv", failure["high_confidence_unseen"], failure_fields())

    multi_window = evaluate_multi_window_open_set(args, calibrator, best_detector_name)
    write_csv(output_dir / "multi_window_open_set_summary.csv", multi_window["rows"], multi_window_fields())
    write_open_set_drift_report(output_dir / "open_set_drift_report.md", multi_window)

    metrics = build_metrics_payload(
        args=args,
        calibration_train=calibration_train,
        calibration_validation=calibration_validation,
        calibrator_name=calibrator_name,
        protocol=protocol,
        test_bundle=test_bundle,
        detectors=detectors,
        comparison_metrics=comparison_metrics,
        best_detector_name=best_detector_name,
        routing=routing,
        failure=failure,
        multi_window=multi_window,
        output_dir=output_dir,
    )
    write_json(output_dir / "open_set_metrics.json", metrics)
    write_json(output_dir / "phase4_open_set_summary.json", metrics)
    write_phase4_summary(output_dir / "phase4_open_set_summary.md", metrics)
    print(json.dumps(metrics["final_recommendation"], ensure_ascii=False, indent=2, sort_keys=True))


def fit_closed_set_calibrator(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Calibrator, str]:
    train_path = CALIBRATION_TARGET_DATA_DIR / f"{DATASET}_history_train.jsonl"
    validation_path = CALIBRATION_TARGET_DATA_DIR / f"{DATASET}_validation_set.jsonl"
    roster_path = CALIBRATION_TARGET_DATA_DIR / f"{DATASET}_candidate_roster.json"
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    roster = read_roster(roster_path)
    validation_predictions, _ = evaluate_current_triager(
        train_path=train_path,
        train_rows=train_rows,
        test_rows=validation_rows,
        roster=roster,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    calibrators = fit_calibrators(validation_predictions, train_rows, args.seed)
    method_metrics = {}
    for name, calibrator in calibrators.items():
        probs = calibrator.predict(validation_predictions, train_rows)
        method_metrics[name] = {"ece": ece(labels_from_correctness(validation_predictions), probs)}
    best_name = select_best_method(
        {
            "validation": {
                name: {
                    "ece": row["ece"],
                    "brier_score": 0.0,
                    "log_loss": 0.0,
                }
                for name, row in method_metrics.items()
            }
        }
    )
    return train_rows, validation_predictions, calibrators[best_name], best_name


def build_leave_assignee_out_protocol(args: argparse.Namespace, calibrator: Calibrator, output_dir: Path) -> dict[str, Any]:
    train_path = OPEN_WORLD_DATA_DIR / "prefilter_history_train.jsonl"
    validation_path = OPEN_WORLD_DATA_DIR / "prefilter_validation_set.jsonl"
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    train_counts = Counter(normalize_assignee(row.get("assignee")) for row in train_rows)
    validation_counts = Counter(normalize_assignee(row.get("assignee")) for row in validation_rows)
    hidden_assignees = select_hidden_assignees(train_counts, validation_counts, args.seed)
    reduced_train = [
        row for row in train_rows if normalize_assignee(row.get("assignee")) not in hidden_assignees
    ]
    reduced_roster = {assignee for assignee in train_counts if assignee and assignee not in hidden_assignees}
    reduced_train_path = output_dir / "validation_protocol_data" / "leave_assignee_out_history_train.jsonl"
    write_jsonl(reduced_train_path, reduced_train)
    validation_predictions, validation_ranking_metrics = evaluate_current_triager(
        train_path=reduced_train_path,
        train_rows=reduced_train,
        test_rows=validation_rows,
        roster=reduced_roster,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    validation_features = build_feature_rows(
        predictions=validation_predictions,
        train_rows=reduced_train,
        calibrator=calibrator,
        group_train_counts=Counter(normalize_assignee(row.get("assignee")) for row in reduced_train),
    )
    full_train_assignees = {assignee for assignee in train_counts if assignee}
    temporal_newcomer_rows = [
        row for row in validation_rows if normalize_assignee(row.get("assignee")) not in full_train_assignees
    ]
    return {
        "protocol": "leave_assignee_out_validation",
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "reduced_train_path": str(reduced_train_path),
        "train_rows": len(train_rows),
        "reduced_train_rows": len(reduced_train),
        "validation_rows": len(validation_rows),
        "hidden_assignee_count": len(hidden_assignees),
        "hidden_assignees": sorted(hidden_assignees),
        "pseudo_unknown_rows": sum(1 for row in validation_features if row["open_set_group"] == "unseen_assignee"),
        "temporal_newcomer_validation_rows": len(temporal_newcomer_rows),
        "validation_ranking_metrics": validation_ranking_metrics,
        "validation_features": validation_features,
    }


def build_open_world_test_bundle(args: argparse.Namespace, calibrator: Calibrator, output_dir: Path) -> dict[str, Any]:
    train_path = OPEN_WORLD_DATA_DIR / "prefilter_history_train.jsonl"
    test_path = OPEN_WORLD_DATA_DIR / "prefilter_test_set.jsonl"
    roster_path = OPEN_WORLD_DATA_DIR / "prefilter_candidate_roster.json"
    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    roster = read_roster(roster_path)
    predictions, ranking_metrics = evaluate_current_triager(
        train_path=train_path,
        train_rows=train_rows,
        test_rows=test_rows,
        roster=roster,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    feature_rows = build_feature_rows(
        predictions=predictions,
        train_rows=train_rows,
        calibrator=calibrator,
        group_train_counts=Counter(normalize_assignee(row.get("assignee")) for row in train_rows),
    )
    ranking_check = verify_open_world_ranking(ranking_metrics)
    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "roster_path": str(roster_path),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "roster_size": len(roster),
        "ranking_metrics": ranking_metrics,
        "ranking_check": ranking_check,
        "test_features": feature_rows,
    }


def build_feature_rows(
    *,
    predictions: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    calibrator: Calibrator,
    group_train_counts: Counter[str],
) -> list[dict[str, Any]]:
    context = FeatureContext(train_rows)
    calibrated = calibrator.predict(predictions, train_rows)
    rows = []
    for prediction, calibrated_probability in zip(predictions, calibrated, strict=True):
        expected = normalize_assignee(prediction.get("expected_assignee"))
        frequency = int(group_train_counts.get(expected, 0))
        if frequency >= 10:
            group = "frequent_seen"
        elif frequency > 0:
            group = "low_frequency_seen"
        else:
            group = "unseen_assignee"
        feature_values = context.features(prediction)
        rows.append(
            {
                **prediction,
                **feature_values,
                "calibrated_probability": round(float(calibrated_probability), 6),
                "expected_train_frequency": frequency,
                "open_set_group": group,
                "unknown_label": 1 if group == "unseen_assignee" else 0,
                "known_label": 0 if group == "unseen_assignee" else 1,
            }
        )
    return rows


class FeatureContext:
    def __init__(self, train_rows: list[dict[str, Any]]) -> None:
        self.train_rows = train_rows
        self.global_counts: Counter[str] = Counter()
        self.component_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.product_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.product_component_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.component_ticket_counts: Counter[str] = Counter()
        self.product_ticket_counts: Counter[str] = Counter()
        self.product_component_ticket_counts: Counter[str] = Counter()
        for row in train_rows:
            assignee = normalize_assignee(row.get("assignee"))
            product = normalize_component(row.get("product"))
            component = normalize_component(row.get("component"))
            key = f"{product}::{component}"
            self.global_counts[assignee] += 1
            self.component_counts[component][assignee] += 1
            self.product_counts[product][assignee] += 1
            self.product_component_counts[key][assignee] += 1
            self.component_ticket_counts[component] += 1
            self.product_ticket_counts[product] += 1
            self.product_component_ticket_counts[key] += 1
        self.bm25 = BM25FeatureIndex(train_rows)

    def features(self, row: dict[str, Any]) -> dict[str, Any]:
        predicted = normalize_assignee(row.get("predicted_assignee"))
        product = normalize_component(row.get("product"))
        component = normalize_component(row.get("component"))
        product_component = f"{product}::{component}"
        scores = score_values(row.get("candidate_scores"))
        top1_score = safe_float(scores.get(predicted, max(scores.values()) if scores else 0.0))
        sorted_scores = sorted(scores.values(), reverse=True)
        top2_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        score_total = sum(max(value, 0.0) for value in sorted_scores)
        probabilities = [value / score_total for value in sorted_scores if score_total > 0 and value > 0]
        entropy = normalized_entropy(probabilities)
        effective_count = math.exp(-sum(prob * math.log(prob) for prob in probabilities)) if probabilities else 0.0
        component_counter = self.component_counts.get(component, Counter())
        product_counter = self.product_counts.get(product, Counter())
        product_component_counter = self.product_component_counts.get(product_component, Counter())
        bm25 = self.bm25.query(row)
        component_top1 = top_assignee(component_counter)
        product_component_top1 = top_assignee(product_component_counter)
        agreement_values = [
            component_top1 == predicted if component_top1 else None,
            product_component_top1 == predicted if product_component_top1 else None,
            bm25["top_assignee"] == predicted if bm25["top_assignee"] else None,
        ]
        disagreement_count = sum(1 for value in agreement_values if value is False)
        signal_count = count_signals(row.get("reason"))
        component_total = sum(component_counter.values())
        product_component_total = sum(product_component_counter.values())
        return {
            "raw_confidence": safe_float(row.get("confidence")),
            "top1_score": round(top1_score, 6),
            "top2_score": round(top2_score, 6),
            "score_margin": round(top1_score - top2_score, 6),
            "normalized_margin": safe_float(row.get("margin")),
            "score_entropy": round(entropy, 6),
            "effective_candidate_count": round(effective_count, 6),
            "candidate_count": len(row.get("ranked_candidates", []) or []),
            "predicted_assignee_train_frequency": int(self.global_counts.get(predicted, 0)),
            "component_history_count": int(component_counter.get(predicted, 0)),
            "product_component_history_count": int(product_component_counter.get(predicted, 0)),
            "product_history_count": int(product_counter.get(predicted, 0)),
            "distinct_assignees_component": len(component_counter),
            "component_owner_entropy": round(counter_entropy(component_counter), 6),
            "component_top1_owner_share": round(max(component_counter.values()) / component_total, 6) if component_total else 0.0,
            "predicted_component_owner_share": round(component_counter.get(predicted, 0) / component_total, 6) if component_total else 0.0,
            "best_bm25_similarity": round(bm25["best_score"], 6),
            "bm25_margin": round(bm25["margin"], 6),
            "high_similarity_ticket_count": bm25["high_similarity_count"],
            "retrieval_density": round(bm25["density"], 6),
            "unseen_component": 1 if component_total == 0 else 0,
            "unseen_product": 1 if self.product_ticket_counts.get(product, 0) == 0 else 0,
            "unseen_product_component_pair": 1 if product_component_total == 0 else 0,
            "rare_component": 1 if 0 < self.component_ticket_counts.get(component, 0) < 5 else 0,
            "rare_product_component_pair": 1 if 0 < self.product_component_ticket_counts.get(product_component, 0) < 5 else 0,
            "metadata_missingness": int(product == "unknown") + int(component == "unknown"),
            "component_top1_agrees_final": int(component_top1 == predicted) if component_top1 else 0,
            "product_component_top1_agrees_final": int(product_component_top1 == predicted) if product_component_top1 else 0,
            "bm25_top1_agrees_final": int(bm25["top_assignee"] == predicted) if bm25["top_assignee"] else 0,
            "signal_count": signal_count,
            "disagreement_count": disagreement_count,
            "bm25_top_assignee": bm25["top_assignee"],
            "component_top_assignee": component_top1,
            "product_component_top_assignee": product_component_top1,
        }


class BM25FeatureIndex:
    def __init__(self, rows: list[dict[str, Any]], *, neighbors: int = 25) -> None:
        self.rows = rows
        self.neighbors = neighbors
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        token_counts = []
        df: Counter[str] = Counter()
        for row in self.rows:
            counts = Counter(tokens(ticket_text(row)))
            token_counts.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            df.update(counts.keys())
        total = len(token_counts)
        self.avgdl = sum(self.doc_lengths) / total if total else 0.0
        for token, freq in df.items():
            self.idf[token] = math.log(1 + (total - freq + 0.5) / (freq + 0.5))
        for doc_index, counts in enumerate(token_counts):
            for token, frequency in counts.items():
                self.postings[token].append((doc_index, frequency))

    def query(self, row: dict[str, Any]) -> dict[str, Any]:
        query_tokens = Counter(tokens(ticket_text(row)))
        doc_scores: defaultdict[int, float] = defaultdict(float)
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            if token not in self.postings:
                continue
            idf = self.idf.get(token, 0.0)
            for doc_index, frequency in self.postings[token]:
                doc_len = self.doc_lengths[doc_index] or 1
                denominator = frequency + k1 * (1 - b + b * doc_len / (self.avgdl or 1.0))
                doc_scores[doc_index] += idf * ((frequency * (k1 + 1)) / denominator)
        ranked = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[: self.neighbors]
        if not ranked:
            return {"best_score": 0.0, "margin": 0.0, "high_similarity_count": 0, "density": 0.0, "top_assignee": ""}
        best_score = ranked[0][1]
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        high_count = sum(1 for _, score in ranked if score >= max(1.0, best_score * 0.50))
        top_assignee = normalize_assignee(self.rows[ranked[0][0]].get("assignee"))
        return {
            "best_score": best_score,
            "margin": best_score - top2_score,
            "high_similarity_count": high_count,
            "density": sum(score for _, score in ranked) / len(ranked),
            "top_assignee": top_assignee,
        }


def fit_detectors(validation_rows: list[dict[str, Any]], seed: int) -> dict[str, Detector]:
    y = np.array(labels(validation_rows), dtype=int)
    if len(set(y.tolist())) < 2:
        raise SystemExit("Open-set validation protocol needs both known and unknown rows.")
    detectors = {
        "negative_raw_confidence": Detector("negative_raw_confidence", 0.0),
        "negative_calibrated_probability": Detector("negative_calibrated_probability", 0.0),
        "negative_margin": Detector("negative_margin", 0.0),
        "low_historical_support": Detector("low_historical_support", 0.0),
        "rule_based_novelty": Detector("rule_based_novelty", 0.0),
    }
    matrix = feature_matrix(validation_rows, FEATURE_NAMES)
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced", C=0.5)),
        ]
    )
    logistic.fit(matrix, y)
    detectors["logistic_regression"] = Detector("logistic_regression", 0.0, logistic, FEATURE_NAMES)
    forest = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=4,
        max_depth=5,
        class_weight="balanced_subsample",
        random_state=seed,
    )
    forest.fit(matrix, y)
    detectors["random_forest"] = Detector("random_forest", 0.0, forest, FEATURE_NAMES)
    for detector in detectors.values():
        score_values_ = detector.scores(validation_rows)
        detector.threshold = select_threshold(score_values_, labels(validation_rows))["threshold"]
    return detectors


def compare_detectors(
    detectors: dict[str, Detector],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows = []
    metrics: dict[str, Any] = {"validation": {}, "test": {}}
    for detector_name, detector in detectors.items():
        for split, rows in [("validation", validation_rows), ("test", test_rows)]:
            scores = detector.scores(rows)
            split_metrics = detector_metrics(labels(rows), scores, detector.threshold, args.bootstrap_samples, args.seed)
            metrics[split][detector_name] = split_metrics
            output_rows.append(
                {
                    "detector": detector_name,
                    "split": split,
                    "threshold": detector.threshold,
                    **split_metrics,
                }
            )
    return output_rows, metrics


def detector_metrics(y_true: list[int], scores: list[float], threshold: float, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    y_pred = [1 if score >= threshold else 0 for score in scores]
    unknown_total = sum(y_true)
    known_total = len(y_true) - unknown_total
    tp = sum(1 for y, pred in zip(y_true, y_pred, strict=True) if y == 1 and pred == 1)
    tn = sum(1 for y, pred in zip(y_true, y_pred, strict=True) if y == 0 and pred == 0)
    fp = sum(1 for y, pred in zip(y_true, y_pred, strict=True) if y == 0 and pred == 1)
    fn = sum(1 for y, pred in zip(y_true, y_pred, strict=True) if y == 1 and pred == 0)
    unknown_recall = ratio(tp, unknown_total)
    known_recall = ratio(tn, known_total)
    balanced_accuracy = round((unknown_recall + known_recall) / 2, 6)
    return {
        "rows": len(y_true),
        "unknown_rows": unknown_total,
        "known_rows": known_total,
        "auroc": safe_metric(lambda: roc_auc_score(y_true, scores), y_true),
        "auprc": safe_metric(lambda: average_precision_score(y_true, scores), y_true),
        "fpr_at_95_tpr": fpr_at_tpr(y_true, scores, 0.95),
        "tpr_at_05_fpr": tpr_at_fpr(y_true, scores, 0.05),
        "tpr_at_10_fpr": tpr_at_fpr(y_true, scores, 0.10),
        "tpr_at_20_fpr": tpr_at_fpr(y_true, scores, 0.20),
        "unknown_recall": unknown_recall,
        "known_recall": known_recall,
        "balanced_accuracy": balanced_accuracy,
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6) if len(set(y_true)) > 1 else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "unknown_recall_95ci": bootstrap_ci(y_true, scores, threshold, "unknown_recall", bootstrap_samples, seed),
        "known_recall_95ci": bootstrap_ci(y_true, scores, threshold, "known_recall", bootstrap_samples, seed),
        "balanced_accuracy_95ci": bootstrap_ci(y_true, scores, threshold, "balanced_accuracy", bootstrap_samples, seed),
    }


def compare_routing_policies(
    *,
    test_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    best_detector: Detector,
    confidence_detector: Detector,
) -> dict[str, Any]:
    high_confidence_threshold = confidence_threshold_for_known_accuracy(validation_rows, 0.85)
    low_confidence_threshold = confidence_threshold_for_known_accuracy(validation_rows, 0.80)
    policies = {
        "confidence_only": {
            "detector": confidence_detector,
            "mode": "confidence_only",
            "t_high": high_confidence_threshold,
            "t_low": low_confidence_threshold,
        },
        "open_set_detector_only": {
            "detector": best_detector,
            "mode": "detector_only",
            "t_high": 0.0,
            "t_low": 0.0,
        },
        "open_set_plus_calibrated_confidence": {
            "detector": best_detector,
            "mode": "detector_confidence",
            "t_high": high_confidence_threshold,
            "t_low": low_confidence_threshold,
        },
        "open_set_plus_top3_confirmation": {
            "detector": best_detector,
            "mode": "detector_top3",
            "t_high": 0.0,
            "t_low": 0.0,
        },
    }
    rows = []
    for policy_name, spec in policies.items():
        routed = route_rows(
            rows=test_rows,
            detector=spec["detector"],
            mode=spec["mode"],
            high_confidence_threshold=spec["t_high"],
            low_confidence_threshold=spec["t_low"],
        )
        rows.append({"policy": policy_name, **routing_metrics(routed)})
    return {"rows": rows, "confidence_thresholds": {"high": high_confidence_threshold, "low": low_confidence_threshold}}


def route_rows(
    *,
    rows: list[dict[str, Any]],
    detector: Detector,
    mode: str,
    high_confidence_threshold: float,
    low_confidence_threshold: float,
) -> list[dict[str, Any]]:
    scores = detector.scores(rows)
    routed = []
    for row, risk in zip(rows, scores, strict=True):
        if risk >= detector.threshold:
            route = "manual_triage"
        elif mode == "detector_only":
            route = "auto_assign"
        elif mode == "detector_top3":
            route = "top3_recommendation"
        else:
            probability = safe_float(row.get("calibrated_probability"))
            if probability >= high_confidence_threshold:
                route = "auto_assign"
            elif probability >= low_confidence_threshold:
                route = "top3_recommendation"
            else:
                route = "manual_triage"
        routed.append({**row, "unknown_risk": round(risk, 6), "route": route})
    return routed


def routing_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    auto = [row for row in rows if row["route"] == "auto_assign"]
    top3 = [row for row in rows if row["route"] == "top3_recommendation"]
    manual = [row for row in rows if row["route"] == "manual_triage"]
    unseen = [row for row in rows if row["unknown_label"] == 1]
    known = [row for row in rows if row["unknown_label"] == 0]
    frequent = [row for row in rows if row["open_set_group"] == "frequent_seen"]
    low_freq = [row for row in rows if row["open_set_group"] == "low_frequency_seen"]
    return {
        "total_coverage": ratio(len(auto) + len(top3), total),
        "auto_assign_coverage": ratio(len(auto), total),
        "auto_assign_accuracy": ratio(sum(1 for row in auto if row["is_top1_correct"]), len(auto)),
        "unseen_false_auto_assign_rate": ratio(sum(1 for row in auto if row["unknown_label"] == 1), len(unseen)),
        "unseen_manual_triage_recall": ratio(sum(1 for row in manual if row["unknown_label"] == 1), len(unseen)),
        "known_manual_triage_false_positive_rate": ratio(sum(1 for row in manual if row["unknown_label"] == 0), len(known)),
        "frequent_seen_coverage": ratio(sum(1 for row in frequent if row["route"] != "manual_triage"), len(frequent)),
        "low_frequency_seen_coverage": ratio(sum(1 for row in low_freq if row["route"] != "manual_triage"), len(low_freq)),
        "top3_coverage": ratio(len(top3), total),
        "top3_hit_at_3": ratio(sum(1 for row in top3 if row["is_hit_at_3"]), len(top3)),
        "manual_triage_rate": ratio(len(manual), total),
        "overall_routing_error": overall_routing_error(rows),
        "auto_assign_count": len(auto),
        "top3_count": len(top3),
        "manual_triage_count": len(manual),
    }


def evaluate_multi_window_open_set(args: argparse.Namespace, calibrator: Calibrator, best_detector_name: str) -> dict[str, Any]:
    records = load_normalized_raw_records(DEFAULT_RAW_PATH)
    rows = []
    global_detector: Detector | None = None
    for window in default_windows():
        view = build_window_view(records, window, min_train_assignee_count=10)
        train_rows = view["train_raw"]
        validation_rows = view["validation_raw"]
        test_rows = view["test_raw"]
        train_counts = Counter(normalize_assignee(row.get("assignee")) for row in train_rows)
        validation_counts = Counter(normalize_assignee(row.get("assignee")) for row in validation_rows)
        hidden = select_hidden_assignees(train_counts, validation_counts, args.seed)
        reduced_train = [row for row in train_rows if normalize_assignee(row.get("assignee")) not in hidden]
        reduced_roster = {assignee for assignee in train_counts if assignee and assignee not in hidden}
        window_data_dir = DEFAULT_PHASE4_ROOT / "window_data" / window.name
        reduced_train_path = window_data_dir / "leave_assignee_out_history_train.jsonl"
        full_train_path = window_data_dir / "open_world_history_train.jsonl"
        write_jsonl(reduced_train_path, reduced_train)
        write_jsonl(full_train_path, train_rows)
        validation_predictions, _ = evaluate_current_triager(
            train_path=reduced_train_path,
            train_rows=reduced_train,
            test_rows=validation_rows,
            roster=reduced_roster,
            top_k=args.top_k,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        validation_features = build_feature_rows(
            predictions=validation_predictions,
            train_rows=reduced_train,
            calibrator=calibrator,
            group_train_counts=Counter(normalize_assignee(row.get("assignee")) for row in reduced_train),
        )
        test_roster = {assignee for assignee in train_counts if assignee}
        test_predictions, _ = evaluate_current_triager(
            train_path=full_train_path,
            train_rows=train_rows,
            test_rows=test_rows,
            roster=test_roster,
            top_k=args.top_k,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        test_features = build_feature_rows(
            predictions=test_predictions,
            train_rows=train_rows,
            calibrator=calibrator,
            group_train_counts=train_counts,
        )
        detectors = fit_detectors(validation_features, args.seed)
        detector = detectors[best_detector_name]
        if global_detector is None:
            global_detector = detector
        scores = detector.scores(test_features)
        metric = detector_metrics(labels(test_features), scores, detector.threshold, args.bootstrap_samples, args.seed)
        assert global_detector is not None
        global_scores = global_detector.scores(test_features)
        global_metric = detector_metrics(labels(test_features), global_scores, global_detector.threshold, args.bootstrap_samples, args.seed)
        newcomer_assignees = {
            normalize_assignee(row.get("assignee"))
            for row in test_rows
            if train_counts.get(normalize_assignee(row.get("assignee")), 0) == 0
        }
        rows.append(
            {
                "window_name": window.name,
                "window_kind": window.kind,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "test_count": len(test_rows),
                "hidden_assignee_count": len(hidden),
                "test_unknown_count": sum(labels(test_features)),
                "test_known_count": len(test_features) - sum(labels(test_features)),
                "newcomer_assignee_count": len(newcomer_assignees),
                "threshold": detector.threshold,
                "auroc": metric["auroc"],
                "auprc": metric["auprc"],
                "unknown_recall": metric["unknown_recall"],
                "known_recall": metric["known_recall"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "mcc": metric["mcc"],
                "global_auroc": global_metric["auroc"],
                "global_unknown_recall": global_metric["unknown_recall"],
                "global_known_recall": global_metric["known_recall"],
                "global_balanced_accuracy": global_metric["balanced_accuracy"],
            }
        )
    aggregate = {
        "auroc_mean": mean_value(rows, "auroc"),
        "unknown_recall_mean": mean_value(rows, "unknown_recall"),
        "known_recall_mean": mean_value(rows, "known_recall"),
        "balanced_accuracy_mean": mean_value(rows, "balanced_accuracy"),
        "threshold_min": min(row["threshold"] for row in rows),
        "threshold_max": max(row["threshold"] for row in rows),
        "newcomer_assignee_total": sum(row["newcomer_assignee_count"] for row in rows),
    }
    return {"rows": rows, "aggregate": aggregate, "best_detector": best_detector_name}


def build_metrics_payload(
    *,
    args: argparse.Namespace,
    calibration_train: list[dict[str, Any]],
    calibration_validation: list[dict[str, Any]],
    calibrator_name: str,
    protocol: dict[str, Any],
    test_bundle: dict[str, Any],
    detectors: dict[str, Detector],
    comparison_metrics: dict[str, Any],
    best_detector_name: str,
    routing: dict[str, Any],
    failure: dict[str, list[dict[str, Any]]],
    multi_window: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    best = comparison_metrics["test"][best_detector_name]
    confidence = comparison_metrics["test"]["negative_calibrated_probability"]
    test_best_auroc_detector = max(
        comparison_metrics["test"],
        key=lambda name: safe_float(comparison_metrics["test"][name]["auroc"]),
    )
    test_best_balanced_detector = max(
        comparison_metrics["test"],
        key=lambda name: comparison_metrics["test"][name]["balanced_accuracy"],
    )
    test_best_unknown_recall_detector = max(
        comparison_metrics["test"],
        key=lambda name: comparison_metrics["test"][name]["unknown_recall"],
    )
    best_routing = {row["policy"]: row for row in routing["rows"]}
    detector_only = best_routing["open_set_detector_only"]
    confidence_only = best_routing["confidence_only"]
    final = {
        "A_validation_selected_detector": best_detector_name,
        "A_frozen_test_best_auroc_detector": test_best_auroc_detector,
        "A_frozen_test_best_balanced_detector": test_best_balanced_detector,
        "A_frozen_test_best_unknown_recall_detector": test_best_unknown_recall_detector,
        "B_confidence_alone_vs_validation_selected_detector": {
            "confidence_auroc": confidence["auroc"],
            "best_auroc": best["auroc"],
            "auroc_delta": round(safe_float(best["auroc"]) - safe_float(confidence["auroc"]), 6),
            "confidence_unknown_recall": confidence["unknown_recall"],
            "best_unknown_recall": best["unknown_recall"],
        },
        "B_confidence_alone_vs_test_best_balanced_detector": {
            "confidence_balanced_accuracy": confidence["balanced_accuracy"],
            "test_best_balanced_detector": test_best_balanced_detector,
            "test_best_balanced_accuracy": comparison_metrics["test"][test_best_balanced_detector]["balanced_accuracy"],
        },
        "C_unseen_assignee_recall_lift": round(best["unknown_recall"] - confidence["unknown_recall"], 6),
        "D_unseen_false_auto_assign_reduction": round(
            confidence_only["unseen_false_auto_assign_rate"] - detector_only["unseen_false_auto_assign_rate"],
            6,
        ),
        "E_frequent_seen_coverage_sacrifice": round(1.0 - detector_only["frequent_seen_coverage"], 6),
        "F_low_frequency_seen_over_rejection": {
            "low_frequency_seen_coverage": detector_only["low_frequency_seen_coverage"],
            "judgment": "over_rejected" if detector_only["low_frequency_seen_coverage"] < 0.5 else "not_severe",
        },
        "G_detector_temporal_stability": {
            "auroc_mean": multi_window["aggregate"]["auroc_mean"],
            "threshold_min": multi_window["aggregate"]["threshold_min"],
            "threshold_max": multi_window["aggregate"]["threshold_max"],
            "judgment": "requires_periodic_validation" if multi_window["aggregate"]["threshold_min"] != multi_window["aggregate"]["threshold_max"] else "stable_threshold",
        },
        "H_production_routing_readiness": "research_ready_not_full_production",
        "I_ready_for_literature_llm_reproduction": "yes_with_shared_open_set_protocol",
        "J_llm_should_share_open_set_detector_policy": "yes",
    }
    return {
        "experiment_name": "phase4_open_set_assignee_detection",
        "dataset": args.dataset,
        "config_snapshot": {
            "top_k": args.top_k,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "output_dir": str(output_dir),
        },
        "problem_definition": {
            "frequent_seen": "gold assignee is in train and train frequency >= 10",
            "low_frequency_seen": "gold assignee is in train and train frequency 1-9",
            "unseen_assignee": "gold assignee is absent from train roster",
            "open_set_target": "known = frequent_seen + low_frequency_seen; unknown = unseen_assignee",
            "closed_set_vs_open_set": "closed-set ranking selects among known roster; open-set detection predicts whether routing should abstain/manual_triage before trusting a roster candidate.",
        },
        "calibration": {
            "calibrator": calibrator_name,
            "calibration_train_rows": len(calibration_train),
            "calibration_validation_predictions": len(calibration_validation),
        },
        "validation_protocol": {
            key: value for key, value in protocol.items() if key != "validation_features"
        },
        "test_setting": {
            key: value for key, value in test_bundle.items() if key != "test_features"
        },
        "detector_selection": {
            "best_detector": best_detector_name,
            "selection_rule": "validation balanced accuracy, tie-broken by AUROC and known recall",
            "threshold": detectors[best_detector_name].threshold,
        },
        "detector_metrics": comparison_metrics,
        "routing_policies": routing["rows"],
        "failure_counts": {
            "false_negative": len(failure["false_negative"]),
            "false_positive": len(failure["false_positive"]),
            "high_confidence_unseen": len(failure["high_confidence_unseen"]),
        },
        "multi_window": multi_window,
        "final_recommendation": final,
        "artifacts": artifact_paths(output_dir),
    }


def write_phase4_summary(path: Path, metrics: dict[str, Any]) -> None:
    final = metrics["final_recommendation"]
    lines = [
        "# Phase 4 Open-Set Assignee Detection Summary",
        "",
        "Phase 4 is additive only. `current_assignee_triager` ranking logic was not modified.",
        "",
        "## Ranking Check",
        "",
        f"- Open-world ranking unchanged: `{metrics['test_setting']['ranking_check']['ranking_metrics_unchanged']}`",
        f"- Observed open-world Top-1 / Hit@3 / Hit@5 / Hit@10 / MRR: "
        f"`{metrics['test_setting']['ranking_metrics']['top1_accuracy']}` / "
        f"`{metrics['test_setting']['ranking_metrics']['hit_at_3']}` / "
        f"`{metrics['test_setting']['ranking_metrics']['hit_at_5']}` / "
        f"`{metrics['test_setting']['ranking_metrics']['hit_at_10']}` / "
        f"`{metrics['test_setting']['ranking_metrics']['mrr']}`",
        "",
        "## Detector",
        "",
        f"- Validation-selected detector: `{final['A_validation_selected_detector']}`",
        f"- Frozen-test best AUROC detector: `{final['A_frozen_test_best_auroc_detector']}`",
        f"- Frozen-test best balanced detector: `{final['A_frozen_test_best_balanced_detector']}`",
        f"- Confidence AUROC vs validation-selected AUROC: `{final['B_confidence_alone_vs_validation_selected_detector']['confidence_auroc']}` -> "
        f"`{final['B_confidence_alone_vs_validation_selected_detector']['best_auroc']}`",
        f"- Unseen recall lift vs confidence: `{final['C_unseen_assignee_recall_lift']}`",
        f"- Frequent-seen coverage sacrifice under detector-only routing: `{final['E_frequent_seen_coverage_sacrifice']}`",
        "",
        "## Routing Recommendation",
        "",
        f"- Production readiness: `{final['H_production_routing_readiness']}`",
        f"- LLM reproduction readiness: `{final['I_ready_for_literature_llm_reproduction']}`",
        f"- LLM should share open-set routing: `{final['J_llm_should_share_open_set_detector_policy']}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact_path in metrics["artifacts"].items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_open_set_drift_report(path: Path, multi_window: dict[str, Any]) -> None:
    lines = [
        "# Open-Set Drift Report",
        "",
        f"- Best detector: `{multi_window['best_detector']}`",
        f"- AUROC mean: `{multi_window['aggregate']['auroc_mean']}`",
        f"- Threshold min/max: `{multi_window['aggregate']['threshold_min']}` / `{multi_window['aggregate']['threshold_max']}`",
        f"- Newcomer assignee total across windows: `{multi_window['aggregate']['newcomer_assignee_total']}`",
        "",
        "| Window | Unknown | Newcomer Assignees | Threshold | AUROC | Unknown Recall | Known Recall | Global AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in multi_window["rows"]:
        lines.append(
            f"| {row['window_name']} | {row['test_unknown_count']} | {row['newcomer_assignee_count']} | "
            f"{row['threshold']} | {row['auroc']} | {row['unknown_recall']} | {row['known_recall']} | {row['global_auroc']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def failure_analysis(rows: list[dict[str, Any]], scores: list[float], threshold: float) -> dict[str, list[dict[str, Any]]]:
    false_negative = []
    false_positive = []
    high_confidence_unseen = []
    for row, score in zip(rows, scores, strict=True):
        predicted_unknown = score >= threshold
        failure_row = failure_row_payload(row, score, threshold)
        if row["unknown_label"] == 1 and not predicted_unknown:
            false_negative.append(failure_row)
        if row["unknown_label"] == 0 and predicted_unknown:
            false_positive.append(failure_row)
        if row["unknown_label"] == 1 and safe_float(row.get("raw_confidence")) >= 0.80:
            high_confidence_unseen.append(failure_row)
    return {
        "false_negative": false_negative,
        "false_positive": false_positive,
        "high_confidence_unseen": high_confidence_unseen,
    }


def failure_row_payload(row: dict[str, Any], score: float, threshold: float) -> dict[str, Any]:
    return {
        "ticket_id": row.get("ticket_id"),
        "open_set_group": row.get("open_set_group"),
        "expected_assignee": row.get("expected_assignee"),
        "predicted_assignee": row.get("predicted_assignee"),
        "raw_confidence": row.get("raw_confidence"),
        "calibrated_probability": row.get("calibrated_probability"),
        "unknown_risk": round(score, 6),
        "threshold": threshold,
        "product": row.get("product"),
        "component": row.get("component"),
        "component_history_count": row.get("component_history_count"),
        "product_component_history_count": row.get("product_component_history_count"),
        "best_bm25_similarity": row.get("best_bm25_similarity"),
        "bm25_top1_agrees_final": row.get("bm25_top1_agrees_final"),
        "title": row.get("title"),
    }


def select_hidden_assignees(train_counts: Counter[str], validation_counts: Counter[str], seed: int) -> set[str]:
    eligible = [
        assignee
        for assignee in validation_counts
        if assignee in train_counts and assignee and validation_counts[assignee] >= 1
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    target_count = max(8, int(round(len(eligible) * 0.15)))
    hidden = set(eligible[:target_count])
    return hidden


def select_best_detector(metrics: dict[str, Any]) -> str:
    rows = []
    for detector, values in metrics["validation"].items():
        rows.append(
            (
                values["balanced_accuracy"],
                safe_float(values["auroc"]),
                values["known_recall"],
                -complexity_rank(detector),
                detector,
            )
        )
    rows.sort(reverse=True)
    return rows[0][4]


def complexity_rank(detector: str) -> int:
    order = {
        "negative_raw_confidence": 1,
        "negative_calibrated_probability": 2,
        "negative_margin": 3,
        "low_historical_support": 4,
        "rule_based_novelty": 5,
        "logistic_regression": 6,
        "random_forest": 7,
    }
    return order.get(detector, 99)


def select_threshold(scores: list[float], y_true: list[int]) -> dict[str, Any]:
    candidates = sorted(set(scores))
    if not candidates:
        return {"threshold": 1.000001, "balanced_accuracy": 0.0}
    candidates = [min(candidates) - 1e-6, *candidates, max(candidates) + 1e-6]
    best = None
    for threshold in candidates:
        unknown_recall = threshold_metric(y_true, scores, threshold, "unknown_recall")
        known_recall = threshold_metric(y_true, scores, threshold, "known_recall")
        balanced_accuracy = threshold_metric(y_true, scores, threshold, "balanced_accuracy")
        candidate = {
            "threshold": round(float(threshold), 6),
            "balanced_accuracy": balanced_accuracy,
            "unknown_recall": unknown_recall,
            "known_recall": known_recall,
        }
        key = (candidate["balanced_accuracy"], candidate["unknown_recall"], candidate["known_recall"], -candidate["threshold"])
        if best is None or key > best["key"]:
            best = {**candidate, "key": key}
    assert best is not None
    return {key: value for key, value in best.items() if key != "key"}


def confidence_threshold_for_known_accuracy(rows: list[dict[str, Any]], target: float) -> float:
    known = [row for row in rows if row["unknown_label"] == 0]
    if not known:
        return 1.000001
    probabilities = sorted(set(safe_float(row.get("calibrated_probability")) for row in known), reverse=True)
    best = None
    for threshold in probabilities:
        accepted = [row for row in known if safe_float(row.get("calibrated_probability")) >= threshold]
        if not accepted:
            continue
        accuracy = ratio(sum(1 for row in accepted if row["is_top1_correct"]), len(accepted))
        coverage = ratio(len(accepted), len(known))
        if accuracy >= target and (best is None or coverage > best["coverage"]):
            best = {"threshold": threshold, "coverage": coverage, "accuracy": accuracy}
    return round(best["threshold"], 6) if best else 1.000001


def attach_detector_scores(rows: list[dict[str, Any]], detectors: dict[str, Detector], best_detector: str) -> list[dict[str, Any]]:
    scores_by_detector = {name: detector.scores(rows) for name, detector in detectors.items()}
    output = []
    for index, row in enumerate(rows):
        copy = dict(row)
        for name, scores in scores_by_detector.items():
            copy[f"{name}_unknown_risk"] = round(scores[index], 6)
            copy[f"{name}_predict_unknown"] = scores[index] >= detectors[name].threshold
        copy["best_detector"] = best_detector
        copy["unknown_risk"] = copy[f"{best_detector}_unknown_risk"]
        copy["predict_unknown"] = copy[f"{best_detector}_predict_unknown"]
        output.append(copy)
    return output


def feature_distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = sorted({row["open_set_group"] for row in rows})
    for feature in FEATURE_NAMES:
        for group in groups:
            values = [safe_float(row.get(feature)) for row in rows if row["open_set_group"] == group]
            if not values:
                continue
            arr = np.array(values, dtype=float)
            output.append(
                {
                    "feature": feature,
                    "open_set_group": group,
                    "count": len(values),
                    "mean": round(float(arr.mean()), 6),
                    "median": round(float(np.median(arr)), 6),
                    "std": round(float(arr.std()), 6),
                    "min": round(float(arr.min()), 6),
                    "max": round(float(arr.max()), 6),
                }
            )
    return output


def detector_comparison_fields() -> list[str]:
    return [
        "detector",
        "split",
        "threshold",
        "rows",
        "unknown_rows",
        "known_rows",
        "auroc",
        "auprc",
        "fpr_at_95_tpr",
        "tpr_at_05_fpr",
        "tpr_at_10_fpr",
        "tpr_at_20_fpr",
        "unknown_recall",
        "known_recall",
        "balanced_accuracy",
        "mcc",
        "tp",
        "tn",
        "fp",
        "fn",
        "unknown_recall_95ci",
        "known_recall_95ci",
        "balanced_accuracy_95ci",
    ]


def feature_audit_fields(detectors: dict[str, Detector]) -> list[str]:
    return open_set_prediction_fields(detectors)


def open_set_prediction_fields(detectors: dict[str, Detector]) -> list[str]:
    detector_fields = []
    for name in detectors:
        detector_fields.extend([f"{name}_unknown_risk", f"{name}_predict_unknown"])
    return [
        "ticket_id",
        "open_set_group",
        "unknown_label",
        "expected_assignee",
        "predicted_assignee",
        "rank",
        "is_top1_correct",
        "is_hit_at_3",
        "raw_confidence",
        "calibrated_probability",
        *FEATURE_NAMES,
        "bm25_top_assignee",
        "component_top_assignee",
        "product_component_top_assignee",
        "best_detector",
        "unknown_risk",
        "predict_unknown",
        *detector_fields,
        "title",
    ]


def feature_distribution_fields() -> list[str]:
    return ["feature", "open_set_group", "count", "mean", "median", "std", "min", "max"]


def routing_fields() -> list[str]:
    return [
        "policy",
        "total_coverage",
        "auto_assign_coverage",
        "auto_assign_accuracy",
        "unseen_false_auto_assign_rate",
        "unseen_manual_triage_recall",
        "known_manual_triage_false_positive_rate",
        "frequent_seen_coverage",
        "low_frequency_seen_coverage",
        "top3_coverage",
        "top3_hit_at_3",
        "manual_triage_rate",
        "overall_routing_error",
        "auto_assign_count",
        "top3_count",
        "manual_triage_count",
    ]


def failure_fields() -> list[str]:
    return [
        "ticket_id",
        "open_set_group",
        "expected_assignee",
        "predicted_assignee",
        "raw_confidence",
        "calibrated_probability",
        "unknown_risk",
        "threshold",
        "product",
        "component",
        "component_history_count",
        "product_component_history_count",
        "best_bm25_similarity",
        "bm25_top1_agrees_final",
        "title",
    ]


def multi_window_fields() -> list[str]:
    return [
        "window_name",
        "window_kind",
        "train_count",
        "validation_count",
        "test_count",
        "hidden_assignee_count",
        "test_unknown_count",
        "test_known_count",
        "newcomer_assignee_count",
        "threshold",
        "auroc",
        "auprc",
        "unknown_recall",
        "known_recall",
        "balanced_accuracy",
        "mcc",
        "global_auroc",
        "global_unknown_recall",
        "global_known_recall",
        "global_balanced_accuracy",
    ]


def artifact_paths(output_dir: Path) -> dict[str, str]:
    names = [
        "open_set_feature_audit.csv",
        "feature_distribution_by_group.csv",
        "open_set_detector_comparison.csv",
        "open_set_predictions.csv",
        "assignee_open_set_rule_based_novelty.json",
        "open_set_metrics.json",
        "routing_policy_open_set_comparison.csv",
        "multi_window_open_set_summary.csv",
        "open_set_drift_report.md",
        "open_set_false_negative.csv",
        "open_set_false_positive.csv",
        "high_confidence_unseen_cases.csv",
        "phase4_open_set_summary.md",
        "phase4_open_set_summary.json",
    ]
    return {name: str(output_dir / name) for name in names}


def verify_open_world_ranking(metrics: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        key: round(float(metrics.get(key, 0.0)) - expected, 6)
        for key, expected in EXPECTED_OPEN_WORLD_METRICS.items()
    }
    return {
        "ranking_metrics_unchanged": all(abs(value) <= 1e-6 for value in deltas.values()),
        "expected": EXPECTED_OPEN_WORLD_METRICS,
        "observed": {key: metrics.get(key) for key in EXPECTED_OPEN_WORLD_METRICS},
        "deltas": deltas,
    }


def low_support_score(row: dict[str, Any]) -> float:
    support = (
        safe_float(row.get("component_history_count"))
        + safe_float(row.get("product_component_history_count"))
        + 0.05 * safe_float(row.get("predicted_assignee_train_frequency"))
    )
    return round(1.0 / (1.0 + support), 6)


def rule_based_novelty_score(row: dict[str, Any]) -> float:
    score = 0.0
    score += 0.25 * safe_float(row.get("unseen_component"))
    score += 0.25 * safe_float(row.get("unseen_product_component_pair"))
    score += 0.15 * safe_float(row.get("rare_component"))
    score += 0.15 * safe_float(row.get("rare_product_component_pair"))
    score += 0.10 if safe_float(row.get("component_history_count")) == 0 else 0.0
    score += 0.05 if safe_float(row.get("best_bm25_similarity")) == 0 else 0.0
    score += 0.05 * min(1.0, safe_float(row.get("disagreement_count")) / 3.0)
    return round(min(1.0, score), 6)


def build_runtime_open_set_artifact(
    *,
    dataset: str,
    threshold: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    approved: bool,
) -> dict[str, Any]:
    """Export an inspectable detector; learned Phase 4 models stay research-only."""
    return {
        "schema_version": 1,
        "artifact_type": "assignee_open_set_detector",
        "name": "rule_based_novelty",
        "deployment_status": "approved" if approved else "research_only",
        "ranker_confidence_version": "hybrid_ranker_v1",
        "dataset": dataset,
        "threshold": threshold["threshold"],
        "threshold_selection": threshold,
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
        "validation_metrics": validation_metrics,
        "frozen_test_metrics": test_metrics,
        "approval_note": (
            "Explicitly approved by the operator; revalidate when the roster or component distribution changes."
            if approved
            else "Research artifact. Runtime will reject it until an operator explicitly approves a target-domain artifact."
        ),
    }


def labels(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row.get("unknown_label", 0)) for row in rows]


def labels_from_correctness(rows: list[dict[str, Any]]) -> list[int]:
    return [1 if row.get("is_top1_correct") else 0 for row in rows]


def feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.array([[safe_float(row.get(feature)) for feature in feature_names] for row in rows], dtype=float)


def fpr_at_tpr(y_true: list[int], scores: list[float], target_tpr: float) -> float | None:
    if len(set(y_true)) < 2:
        return None
    best = None
    for threshold in sorted(set(scores)):
        pred = [1 if score >= threshold else 0 for score in scores]
        tpr, fpr = tpr_fpr(y_true, pred)
        if tpr >= target_tpr:
            best = fpr if best is None else min(best, fpr)
    return None if best is None else round(best, 6)


def tpr_at_fpr(y_true: list[int], scores: list[float], max_fpr: float) -> float | None:
    if len(set(y_true)) < 2:
        return None
    best = 0.0
    for threshold in sorted(set(scores)):
        pred = [1 if score >= threshold else 0 for score in scores]
        tpr, fpr = tpr_fpr(y_true, pred)
        if fpr <= max_fpr:
            best = max(best, tpr)
    return round(best, 6)


def tpr_fpr(y_true: list[int], y_pred: list[int]) -> tuple[float, float]:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    tp = sum(1 for y, pred in zip(y_true, y_pred, strict=True) if y == 1 and pred == 1)
    fp = sum(1 for y, pred in zip(y_true, y_pred, strict=True) if y == 0 and pred == 1)
    return ratio(tp, positives), ratio(fp, negatives)


def bootstrap_ci(
    y_true: list[int],
    scores: list[float],
    threshold: float,
    metric_name: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if samples <= 0 or not y_true:
        return {"low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        indexes = [rng.randrange(len(y_true)) for _ in y_true]
        sampled_y = [y_true[index] for index in indexes]
        sampled_scores = [scores[index] for index in indexes]
        values.append(threshold_metric(sampled_y, sampled_scores, threshold, metric_name))
    values.sort()
    return {"low": percentile(values, 0.025), "high": percentile(values, 0.975)}


def threshold_metric(y_true: list[int], scores: list[float], threshold: float, metric_name: str) -> float:
    y_pred = [1 if score >= threshold else 0 for score in scores]
    unknown_total = sum(y_true)
    known_total = len(y_true) - unknown_total
    tp = sum(1 for y, pred in zip(y_true, y_pred, strict=True) if y == 1 and pred == 1)
    tn = sum(1 for y, pred in zip(y_true, y_pred, strict=True) if y == 0 and pred == 0)
    unknown_recall = ratio(tp, unknown_total)
    known_recall = ratio(tn, known_total)
    if metric_name == "unknown_recall":
        return unknown_recall
    if metric_name == "known_recall":
        return known_recall
    if metric_name == "balanced_accuracy":
        return round((unknown_recall + known_recall) / 2, 6)
    raise SystemExit(f"Unsupported bootstrap metric: {metric_name}")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return round(float(values[index]), 6)


def safe_metric(fn: Any, y_true: list[int]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    try:
        return round(float(fn()), 6)
    except Exception:
        return None


def read_roster(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("candidates", []) if isinstance(payload, dict) else payload
    return {normalize_assignee(value) for value in values if normalize_assignee(value)}


def score_values(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {normalize_assignee(key): safe_float(item) for key, item in value.items()}
    return {}


def count_signals(reason: Any) -> int:
    text = str(reason or "")
    marker = "strongest signals:"
    if marker not in text:
        return 0
    signal_text = text.split(marker, 1)[1].split(".", 1)[0]
    return len([part for part in signal_text.split(",") if part.strip()])


def top_assignee(counter: Counter[str]) -> str:
    return counter.most_common(1)[0][0] if counter else ""


def counter_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    probs = [count / total for count in counter.values()]
    return -sum(prob * math.log(prob) for prob in probs if prob > 0)


def normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(prob * math.log(prob) for prob in probabilities if prob > 0)
    return entropy / math.log(len(probabilities))


def overall_routing_error(rows: list[dict[str, Any]]) -> float:
    errors = 0
    for row in rows:
        if row["route"] == "auto_assign" and not row["is_top1_correct"]:
            errors += 1
        elif row["route"] == "top3_recommendation" and not row["is_hit_at_3"]:
            errors += 1
        elif row["route"] != "manual_triage" and row["unknown_label"] == 1:
            errors += 1
    return ratio(errors, len(rows))


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    values = [safe_float(row.get(key)) for row in rows if row.get(key) is not None]
    return round(float(np.mean(values)), 6) if values else 0.0


def ece(labels_: list[int], probabilities: list[float]) -> float:
    total = len(labels_)
    if total == 0:
        return 0.0
    value = 0.0
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        selected = [
            (prob, label)
            for prob, label in zip(probabilities, labels_, strict=True)
            if low <= prob < high or (index == 9 and prob == 1.0)
        ]
        if not selected:
            continue
        mean_prob = sum(prob for prob, _ in selected) / len(selected)
        acc = sum(label for _, label in selected) / len(selected)
        value += len(selected) / total * abs(mean_prob - acc)
    return round(value, 6)


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
