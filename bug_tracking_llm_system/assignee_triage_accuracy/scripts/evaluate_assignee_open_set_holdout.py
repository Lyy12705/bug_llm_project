from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from assignee_open_set_common import (
    deployment_gate,
    open_set_metrics,
    predict_portable_logistic,
    route_predictions,
    routing_metrics,
    routing_score_diagnostics,
    write_json,
    write_jsonl,
)
from calibrate_assignee_ltr import scored_predictions
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
    safe_macro_f1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "phase6_candidate_ltr"
    / "reports"
    / "bmo_public_q4_holdout_source_quota_sbert"
)
DEFAULT_ROUTING_ARTIFACT = (
    PROJECT_ROOT
    / "assignee_triage_accuracy"
    / "phase6_candidate_ltr"
    / "reports"
    / "bmo_public_q3_open_set_shallow_v3"
    / "open_set_routing_artifact.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen assignee ranker and open-set policy once on a new temporal holdout."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--routing-artifact", type=Path, default=DEFAULT_ROUTING_ARTIFACT)
    parser.add_argument(
        "--train", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_history_train.jsonl"
    )
    parser.add_argument(
        "--validation", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_validation_set.jsonl"
    )
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confirm-untouched-holdout",
        action="store_true",
        help="Required acknowledgement that labels were not used to fit models or thresholds.",
    )
    parser.add_argument("--minimum-holdout-rows", type=int, default=2500)
    parser.add_argument("--minimum-auto-rows", type=int, default=250)
    parser.add_argument("--minimum-auto-accuracy-lower-bound", type=float, default=0.80)
    parser.add_argument("--maximum-unseen-rate-upper-bound", type=float, default=0.05)
    parser.add_argument("--minimum-component-auto-rows", type=int, default=30)
    parser.add_argument("--minimum-component-auto-accuracy", type=float, default=0.75)
    parser.add_argument("--minimum-component-auto-accuracy-lower-bound", type=float, default=0.60)
    parser.add_argument("--target-top3-confirmation-accuracy", type=float, default=0.90)
    args = parser.parse_args()

    if not args.confirm_untouched_holdout:
        raise SystemExit("--confirm-untouched-holdout is required")
    artifact = json.loads(args.routing_artifact.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "assignee_open_set_routing_bundle":
        raise SystemExit("Unsupported open-set routing artifact")
    if artifact.get("deployment_status") != "research_only":
        raise SystemExit("Holdout evaluation expects a frozen research-only artifact")
    policy = artifact.get("routing_policy") or {}
    if not policy.get("found"):
        raise SystemExit("The development routing policy did not pass its selection constraints")

    ranker_path = args.model_dir / "candidate_ltr_model.joblib"
    ranker_artifact_path = args.model_dir / "candidate_ltr_artifact.json"
    ranker_artifact = json.loads(ranker_artifact_path.read_text(encoding="utf-8"))
    if ranker_artifact.get("name") != artifact.get("ranker_name"):
        raise SystemExit("Ranker artifact does not match the frozen open-set routing bundle")
    ranker = joblib.load(ranker_path)

    label_map = read_label_map(args.assignee_label_map)
    train_rows = load_rows(args.train, label_map)
    validation_rows = load_rows(args.validation, label_map)
    holdout_rows = load_rows(args.holdout, label_map)
    holdout_rows, overlap_rows_excluded = exclude_cross_split_overlap(
        holdout_rows, [*train_rows, *validation_rows]
    )
    if holdout_rows and row_sort_key(holdout_rows[0]) < row_sort_key(validation_rows[-1]):
        raise SystemExit("Holdout is not temporally later than validation")

    requirements = artifact["ranker_requirements"]
    semantic = build_semantic_backend(
        str(requirements.get("embedding_backend") or "none"),
        str(requirements.get("sbert_model") or "sentence-transformers/all-MiniLM-L6-v2"),
    )
    if requirements.get("embedding_backend") == "sbert" and not semantic.status.startswith("sbert_local:"):
        raise SystemExit(f"The frozen ranker requires a local SBERT model: {semantic.status}")
    index = CandidateIndex(
        train_rows,
        half_life_days=float(requirements["half_life_days"]),
        smoothing_alpha=float(requirements["smoothing_alpha"]),
        semantic=semantic,
    )
    predictions = scored_predictions(
        holdout_rows, index, ranker, int(requirements["candidate_pool_size"])
    )
    calibrated = predict_portable_logistic(predictions, artifact["calibrator"])
    for row, probability in zip(predictions, calibrated, strict=True):
        row["calibrated_probability"] = round(float(probability), 8)
    open_set_scores = predict_portable_logistic(predictions, artifact["open_set_detector"])
    for row, probability in zip(predictions, open_set_scores, strict=True):
        row["open_set_probability"] = round(float(probability), 8)
    route_predictions(predictions, policy)

    routing = routing_metrics(predictions)
    targets = artifact["development_gate"]["targets"]
    gate = deployment_gate(
        routing,
        target_auto_accuracy=float(targets["target_auto_accuracy"]),
        minimum_auto_coverage=float(targets["minimum_auto_coverage"]),
        maximum_unseen_auto_rate=float(targets["maximum_unseen_auto_rate"]),
        minimum_holdout_rows=max(0, args.minimum_holdout_rows),
        minimum_auto_rows=max(0, args.minimum_auto_rows),
        minimum_accuracy_lower_bound=args.minimum_auto_accuracy_lower_bound,
        minimum_component_auto_rows=max(1, args.minimum_component_auto_rows),
        minimum_component_accuracy=args.minimum_component_auto_accuracy,
        minimum_component_accuracy_lower_bound=args.minimum_component_auto_accuracy_lower_bound,
        maximum_unseen_rate_upper_bound=args.maximum_unseen_rate_upper_bound,
    )
    gate["advisory_checks"] = {
        "top3_confirmation_accuracy": routing["top3_confirmation_accuracy"]
        >= args.target_top3_confirmation_accuracy,
    }
    gate["advisory_targets"] = {
        "target_top3_confirmation_accuracy": args.target_top3_confirmation_accuracy,
    }
    gate["passed"] = bool(gate["passed"] and ranker_artifact.get("validation_candidate"))
    gate["ranker_validation_candidate"] = bool(ranker_artifact.get("validation_candidate"))
    gate["production_integration_allowed"] = False
    gate["production_blocker"] = (
        "reviewed_active_roster_and_explicit_operator_approval_required"
        if gate["passed"]
        else "untouched_holdout_safety_gate_failed"
    )

    report = {
        "schema_version": 1,
        "method": "frozen_open_set_routing_holdout_evaluation_v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "deployment_status": "research_only",
        "protocol": {
            "holdout": str(args.holdout),
            "holdout_rows": len(predictions),
            "cross_split_overlap_rows_excluded": overlap_rows_excluded,
            "holdout_used_for_fitting": False,
            "holdout_used_for_threshold_selection": False,
            "ranker_selection": "fixed_before_holdout",
            "routing_policy_selection": "fixed_before_holdout",
        },
        "recommendation_metrics": recommendation_metrics(predictions),
        "recommendation_metrics_known_owner": recommendation_metrics(
            [row for row in predictions if row["known_owner"]]
        ),
        "open_set_metrics": open_set_metrics(
            predictions, threshold=float(policy["open_set_threshold"])
        ),
        "routing_metrics": routing,
        "routing_score_diagnostics": routing_score_diagnostics(predictions, policy),
        "deployment_gate": gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "open_set_holdout_report.json", report)
    write_jsonl(args.output_dir / "open_set_holdout_predictions.jsonl", predictions)
    write_summary(args.output_dir / "open_set_holdout_summary.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def recommendation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    expected = [str(row["expected_assignee"]) for row in rows]
    predicted = [str(row["predicted_assignee"]) for row in rows]
    ranks = []
    pool_hits = 0
    for row in rows:
        owner = str(row["expected_assignee"])
        candidates = list(row.get("ranked_candidates") or [])
        pool = list(row.get("candidate_pool") or candidates)
        rank = candidates.index(owner) + 1 if owner in candidates else None
        ranks.append(rank)
        pool_hits += owner in pool
    return {
        "rows": len(rows),
        "top1_accuracy": round(sum(rank == 1 for rank in ranks) / len(rows), 6),
        "top3_accuracy": round(sum(rank is not None and rank <= 3 for rank in ranks) / len(rows), 6),
        "top5_accuracy": round(sum(rank is not None and rank <= 5 for rank in ranks) / len(rows), 6),
        "top10_accuracy": round(sum(rank is not None and rank <= 10 for rank in ranks) / len(rows), 6),
        "candidate_pool_recall": round(pool_hits / len(rows), 6),
        "mrr": round(sum(1.0 / rank if rank else 0.0 for rank in ranks) / len(rows), 6),
        "macro_f1": round(safe_macro_f1(expected, predicted), 6),
    }


def load_rows(path: Path, label_map: dict[str, str]) -> list[dict[str, Any]]:
    return sorted(
        apply_label_map(canonicalize_source_rows(read_jsonl(path)), label_map),
        key=row_sort_key,
    )


def write_summary(path: Path, report: dict[str, Any]) -> None:
    routing = report["routing_metrics"]
    gate = report["deployment_gate"]
    lines = [
        "# Frozen open-set holdout evaluation",
        "",
        f"- Auto-assignment accuracy: {routing['auto_assignment_accuracy']}",
        f"- Auto-assignment coverage: {routing['auto_assignment_coverage']}",
        f"- Unseen-owner auto-assignment rate: {routing['unseen_auto_assignment_rate']}",
        f"- Safety gate passed: {gate['passed']}",
        f"- Production integration allowed: {gate['production_integration_allowed']}",
        f"- Blocker: {gate['production_blocker']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
