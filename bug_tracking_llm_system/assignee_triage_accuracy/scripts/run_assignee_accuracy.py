from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import PipelineConfig  # noqa: E402
from modules.assignee_triager import AssigneeTriager  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate assignee triage accuracy.")
    parser.add_argument("--dataset", default="controlled", help="Dataset prefix, e.g. controlled or eclipse.")
    parser.add_argument("--history", type=Path, default=None, help="History JSONL path.")
    parser.add_argument("--test", type=Path, default=None, help="Gold test JSONL path.")
    parser.add_argument("--roster", type=Path, default=None, help="Candidate roster JSON path.")
    parser.add_argument("--active-roster", type=Path, default=None, help="Production active assignee roster JSON path.")
    parser.add_argument("--inactive-assignees", type=Path, default=None, help="Inactive/departed assignee JSON/list path.")
    parser.add_argument("--component-ownership", type=Path, default=None, help="Component ownership JSON map.")
    parser.add_argument("--file-ownership", type=Path, default=None, help="File/module ownership JSON map.")
    parser.add_argument("--feedback", type=Path, default=None, help="Confirmed assignee feedback JSONL path.")
    parser.add_argument("--routing-policy", type=Path, default=None, help="Calibration routing policy CSV/JSON.")
    parser.add_argument("--routing-policy-name", default="", help="Policy row/name to apply.")
    parser.add_argument("--calibration-artifact", type=Path, default=None, help="Approved confidence calibration artifact JSON.")
    parser.add_argument(
        "--allow-uncalibrated-auto-assignment",
        action="store_true",
        help="Research-only comparison that permits auto-assignment without approved calibration.",
    )
    parser.add_argument("--open-set", action="store_true", help="Enable production open-set risk gate.")
    parser.add_argument("--open-set-risk-threshold", type=float, default=0.75)
    parser.add_argument("--open-set-artifact", type=Path, default=None, help="Approved open-set detector artifact JSON.")
    parser.add_argument("--output-dir", type=Path, default=EVAL_ROOT / "reports", help="Report output directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum candidate list length.")
    parser.add_argument(
        "--write-details",
        action="store_true",
        help="Also write row-level predictions and CSV error analysis; metrics JSON is always written.",
    )
    args = parser.parse_args()

    data_dir = EVAL_ROOT / "data"
    history_path = args.history or data_dir / f"{args.dataset}_history_train.jsonl"
    test_path = args.test or data_dir / f"{args.dataset}_test_set.jsonl"
    roster_path = args.roster or data_dir / f"{args.dataset}_candidate_roster.json"

    history_rows = read_jsonl(history_path)
    test_rows = read_jsonl(test_path)
    roster = read_roster(roster_path)
    roster_set = set(roster)

    component_counts = build_history_counts(history_rows)
    config = PipelineConfig(
        project_root=SYSTEM_ROOT,
        assignee_dataset_path=history_path,
        assignee_active_roster_path=args.active_roster,
        assignee_inactive_path=args.inactive_assignees,
        assignee_component_ownership_path=args.component_ownership,
        assignee_file_ownership_path=args.file_ownership,
        assignee_feedback_path=args.feedback,
        assignee_routing_policy_path=args.routing_policy,
        assignee_routing_policy_name=args.routing_policy_name,
        assignee_calibration_artifact_path=args.calibration_artifact,
        assignee_allow_uncalibrated_auto_assignment=args.allow_uncalibrated_auto_assignment,
        assignee_open_set_enabled=args.open_set,
        assignee_open_set_risk_threshold=args.open_set_risk_threshold,
        assignee_open_set_artifact_path=args.open_set_artifact,
        save_checkpoints=False,
    )
    triager = AssigneeTriager(config=config)

    predictions = []
    for row in test_rows:
        priority_result = {"predicted_priority": row.get("priority", "P3")}
        result = triager.assign(row, priority_result)
        candidates = production_candidates(result, roster_set=roster_set, top_k=args.top_k)
        expected = normalize_assignee(row.get("assignee"))
        predicted = candidates[0] if candidates else "manual_triage"
        routed = normalize_assignee(result.get("assignee")) or "manual_triage"
        reciprocal = reciprocal_rank(expected, candidates)
        predictions.append(
            {
                "ticket_id": row.get("ticket_id"),
                "title": row.get("title"),
                "component": normalize_component(row.get("component")),
                "priority": row.get("priority", "P3"),
                "expected_assignee": expected,
                "predicted_assignee": predicted,
                "routed_assignee": routed,
                "ranked_candidates": candidates,
                "rank": rank_of(expected, candidates),
                "reciprocal_rank": reciprocal,
                "is_top1_correct": expected == predicted,
                "is_routed_correct": expected == routed,
                "is_hit_at_3": expected in candidates[:3],
                "is_hit_at_5": expected in candidates[:5],
                "recommendation_eligible": expected != "manual_triage",
                "confidence": result.get("confidence"),
                "raw_confidence": result.get("raw_confidence"),
                "calibration_status": result.get("calibration_status", ""),
                "routing_status": result.get("routing_status", ""),
                "needs_manual_triage": bool(result.get("needs_manual_triage")),
                "fallback_used": bool(result.get("fallback_used")),
                "fallback_reason": result.get("fallback_reason", ""),
                "suggested_assignee": normalize_assignee(result.get("suggested_assignee")),
                "open_set_risk": result.get("open_set_risk"),
                "open_set_status": result.get("open_set_status", ""),
                "open_set_detector": result.get("open_set_detector", ""),
                "routing_policy": result.get("routing_policy", ""),
                "source": source_from_reason(str(result.get("reason", ""))),
                "reason": result.get("reason"),
                "error_type": error_type(
                    expected=expected,
                    predicted=predicted,
                    component=normalize_component(row.get("component")),
                    reciprocal=reciprocal,
                    source=source_from_reason(str(result.get("reason", ""))),
                    component_counts=component_counts,
                ),
            }
        )

    metrics = build_metrics(
        dataset=args.dataset,
        history_path=history_path,
        test_path=test_path,
        roster_path=roster_path,
        history_rows=history_rows,
        test_rows=test_rows,
        roster=roster,
        predictions=predictions,
        top_k=args.top_k,
        config=config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.dataset}_metrics.json", metrics)
    if args.write_details:
        write_jsonl(args.output_dir / f"{args.dataset}_predictions.jsonl", predictions)
        write_error_analysis(args.output_dir / f"{args.dataset}_error_analysis.csv", predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing JSONL file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def read_roster(path: Path) -> list[str]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        candidates = value.get("candidates", [])
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []
    return sorted({normalize_assignee(candidate) for candidate in candidates if normalize_assignee(candidate)})


def build_history_counts(history_rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    component_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in history_rows:
        component = normalize_component(row.get("component"))
        assignee = normalize_assignee(row.get("assignee"))
        if not component or not assignee:
            continue
        component_counts[component][assignee] += 1
    return dict(component_counts)


def production_candidates(prediction: dict[str, Any], *, roster_set: set[str], top_k: int) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        candidate = normalize_assignee(value)
        if not candidate or candidate == "manual_triage" or candidate in candidates:
            return
        if roster_set and candidate not in roster_set:
            return
        candidates.append(candidate)

    for candidate in prediction.get("ranked_candidates", []):
        add(candidate)
    return candidates[:top_k]


def build_metrics(
    *,
    dataset: str,
    history_path: Path,
    test_path: Path,
    roster_path: Path,
    history_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    roster: list[str],
    predictions: list[dict[str, Any]],
    top_k: int,
    config: PipelineConfig,
) -> dict[str, Any]:
    total = len(predictions)
    recommendation_rows = [row for row in predictions if row["recommendation_eligible"]]
    recommendation_total = len(recommendation_rows)
    auto_assignment_rows = [row for row in predictions if row["routed_assignee"] != "manual_triage"]
    source_counts = Counter(row["source"] for row in predictions)
    error_counts = Counter(row["error_type"] for row in predictions)

    per_component: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["component"]].append(row)
    for component, rows in sorted(grouped.items()):
        eligible_rows = [row for row in rows if row["recommendation_eligible"]]
        per_component[component] = {
            "rows": len(rows),
            "recommendation_rows": len(eligible_rows),
            "top1_accuracy": ratio(sum(1 for row in eligible_rows if row["is_top1_correct"]), len(eligible_rows)),
            "hit_at_3": ratio(sum(1 for row in eligible_rows if row["is_hit_at_3"]), len(eligible_rows)),
            "mrr": ratio(sum(float(row["reciprocal_rank"]) for row in eligible_rows), len(eligible_rows)),
        }

    return {
        "dataset": dataset,
        "history_path": str(history_path),
        "test_path": str(test_path),
        "roster_path": str(roster_path),
        "history_rows": len(history_rows),
        "test_rows": len(test_rows),
        "candidate_roster_size": len(roster),
        "developer_recommendation_rows": recommendation_total,
        "top_k": top_k,
        "production_routing_config": {
            "active_roster_path": str(config.assignee_active_roster_path) if config.assignee_active_roster_path else "",
            "inactive_path": str(config.assignee_inactive_path) if config.assignee_inactive_path else "",
            "component_ownership_path": str(config.assignee_component_ownership_path) if config.assignee_component_ownership_path else "",
            "file_ownership_path": str(config.assignee_file_ownership_path) if config.assignee_file_ownership_path else "",
            "feedback_path": str(config.assignee_feedback_path) if config.assignee_feedback_path else "",
            "routing_policy_path": str(config.assignee_routing_policy_path) if config.assignee_routing_policy_path else "",
            "routing_policy_name": config.assignee_routing_policy_name,
            "calibration_artifact_path": str(config.assignee_calibration_artifact_path) if config.assignee_calibration_artifact_path else "",
            "allow_uncalibrated_auto_assignment": config.assignee_allow_uncalibrated_auto_assignment,
            "open_set_enabled": config.assignee_open_set_enabled,
            "open_set_risk_threshold": config.assignee_open_set_risk_threshold,
            "open_set_artifact_path": str(config.assignee_open_set_artifact_path) if config.assignee_open_set_artifact_path else "",
        },
        "top1_accuracy": ratio(sum(1 for row in recommendation_rows if row["is_top1_correct"]), recommendation_total),
        "top3_accuracy": ratio(sum(1 for row in recommendation_rows if row["is_hit_at_3"]), recommendation_total),
        "top5_accuracy": ratio(sum(1 for row in recommendation_rows if row["is_hit_at_5"]), recommendation_total),
        "hit_at_3": ratio(sum(1 for row in recommendation_rows if row["is_hit_at_3"]), recommendation_total),
        "hit_at_5": ratio(sum(1 for row in recommendation_rows if row["is_hit_at_5"]), recommendation_total),
        "mrr": ratio(sum(float(row["reciprocal_rank"]) for row in recommendation_rows), recommendation_total),
        "macro_f1": macro_f1(
            [row["expected_assignee"] for row in recommendation_rows],
            [row["predicted_assignee"] for row in recommendation_rows],
        ),
        "unable_to_decide_rate": ratio(
            sum(1 for row in predictions if row["needs_manual_triage"]),
            total,
        ),
        "fallback_trigger_rate": ratio(sum(1 for row in predictions if row["fallback_used"]), total),
        "manual_triage_rate": ratio(sum(1 for row in predictions if row["routed_assignee"] == "manual_triage"), total),
        "auto_assignment_coverage": ratio(len(auto_assignment_rows), total),
        "auto_assignment_accuracy": ratio(
            sum(1 for row in auto_assignment_rows if row["is_routed_correct"]), len(auto_assignment_rows)
        ),
        "end_to_end_routing_accuracy": ratio(sum(1 for row in predictions if row["is_routed_correct"]), total),
        "source_breakdown": dict(sorted(source_counts.items())),
        "routing_status_breakdown": dict(sorted(Counter(row["routing_status"] or "unknown" for row in predictions).items())),
        "fallback_reason_breakdown": dict(sorted(Counter(row["fallback_reason"] or "none" for row in predictions).items())),
        "error_breakdown": dict(sorted(error_counts.items())),
        "class_imbalance": class_imbalance_report(history_rows, test_rows, roster),
        "leakage_audit": leakage_audit(history_rows, test_rows),
        "per_component": per_component,
    }


def source_from_reason(reason: str) -> str:
    lowered = reason.lower()
    if "hybrid ranking" in lowered:
        return "hybrid"
    if "component-owner mapping" in lowered:
        return "mapping"
    if "manual triage" in lowered:
        return "manual"
    return "unknown"


def error_type(
    *,
    expected: str,
    predicted: str,
    component: str,
    reciprocal: float,
    source: str,
    component_counts: dict[str, Counter[str]],
) -> str:
    if expected == predicted:
        return "correct"
    if predicted == "manual_triage":
        return "manual_triage_predicted"
    if component not in component_counts:
        return "cold_start_component"
    if component_counts.get(component, Counter()).get(expected, 0) <= 1:
        return "long_tail_assignee"
    if reciprocal > 0:
        return "misranked_candidate"
    if source == "mapping":
        return "mapping_mismatch"
    return "wrong_assignee"


def rank_of(expected: str, candidates: list[str]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate == expected:
            return index
    return None


def reciprocal_rank(expected: str, candidates: list[str]) -> float:
    rank = rank_of(expected, candidates)
    return 0.0 if rank is None else 1.0 / rank


def ratio(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def macro_f1(expected: list[str], predicted: list[str]) -> float:
    labels = sorted({label for label in expected + predicted if label})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        true_positive = sum(1 for gold, pred in zip(expected, predicted, strict=True) if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in zip(expected, predicted, strict=True) if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in zip(expected, predicted, strict=True) if gold == label and pred != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 6)


def class_imbalance_report(
    history_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], roster: list[str]
) -> dict[str, Any]:
    train_counts = Counter(
        normalize_assignee(row.get("assignee")) for row in history_rows if normalize_assignee(row.get("assignee"))
    )
    test_counts = Counter(
        normalize_assignee(row.get("assignee")) for row in test_rows if normalize_assignee(row.get("assignee"))
    )
    train_values = list(train_counts.values())
    max_count = max(train_values) if train_values else 0
    min_count = min(train_values) if train_values else 0
    test_labels = set(test_counts)
    train_labels = set(train_counts)
    roster_set = set(roster)
    return {
        "train_assignee_count": len(train_counts),
        "test_assignee_count": len(test_counts),
        "train_max_count": max_count,
        "train_min_count": min_count,
        "train_imbalance_ratio": round(max_count / min_count, 6) if min_count else None,
        "train_top_assignee_share": ratio(max_count, len(history_rows)),
        "train_singleton_assignees": sum(1 for count in train_values if count == 1),
        "test_assignees_absent_from_train": sorted(test_labels - train_labels),
        "test_assignees_absent_from_roster": sorted(test_labels - roster_set) if roster_set else [],
        "roster_coverage_rate": ratio(sum(count for assignee, count in test_counts.items() if assignee in roster_set), len(test_rows))
        if roster_set
        else 0.0,
        "top_train_assignees": [
            {"assignee": assignee, "count": count} for assignee, count in train_counts.most_common(10)
        ],
    }


def leakage_audit(history_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {str(row.get("ticket_id") or "").strip() for row in history_rows if str(row.get("ticket_id") or "").strip()}
    test_ids = {str(row.get("ticket_id") or "").strip() for row in test_rows if str(row.get("ticket_id") or "").strip()}
    train_titles = {_normalized_text_key(row.get("title")) for row in history_rows if _normalized_text_key(row.get("title"))}
    test_titles = {_normalized_text_key(row.get("title")) for row in test_rows if _normalized_text_key(row.get("title"))}
    train_title_desc = {
        _normalized_text_key(f"{row.get('title', '')} {row.get('description', '')}")
        for row in history_rows
        if _normalized_text_key(f"{row.get('title', '')} {row.get('description', '')}")
    }
    test_title_desc = {
        _normalized_text_key(f"{row.get('title', '')} {row.get('description', '')}")
        for row in test_rows
        if _normalized_text_key(f"{row.get('title', '')} {row.get('description', '')}")
    }
    train_dates = [_parse_date(row.get("created_at")) for row in history_rows]
    test_dates = [_parse_date(row.get("created_at")) for row in test_rows]
    train_dates = [value for value in train_dates if value is not None]
    test_dates = [value for value in test_dates if value is not None]
    test_start = min(test_dates) if test_dates else None
    return {
        "ticket_id_overlap_count": len(train_ids & test_ids),
        "exact_title_overlap_count": len(train_titles & test_titles),
        "exact_title_description_overlap_count": len(train_title_desc & test_title_desc),
        "created_at_available": bool(train_dates and test_dates),
        "test_start": test_start.isoformat() if test_start else "",
        "history_rows_after_test_start": sum(1 for value in train_dates if test_start and value >= test_start),
        "risk_flags": [
            flag
            for flag, triggered in (
                ("ticket_id_overlap", bool(train_ids & test_ids)),
                ("exact_title_overlap", bool(train_titles & test_titles)),
                ("history_created_after_test_start", any(test_start and value >= test_start for value in train_dates)),
            )
            if triggered
        ],
    }


def _normalized_text_key(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def _parse_date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_component(value: Any) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_error_analysis(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = [
        "ticket_id",
        "component",
        "priority",
        "expected_assignee",
        "predicted_assignee",
        "routed_assignee",
        "rank",
        "reciprocal_rank",
        "is_top1_correct",
        "is_routed_correct",
        "is_hit_at_3",
        "is_hit_at_5",
        "recommendation_eligible",
        "source",
        "error_type",
        "confidence",
        "raw_confidence",
        "calibration_status",
        "routing_status",
        "needs_manual_triage",
        "fallback_used",
        "fallback_reason",
        "suggested_assignee",
        "open_set_risk",
        "open_set_status",
        "open_set_detector",
        "routing_policy",
        "ranked_candidates",
        "reason",
        "title",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            output = dict(row)
            output["ranked_candidates"] = "|".join(row.get("ranked_candidates", []))
            writer.writerow({field: output.get(field, "") for field in fields})


if __name__ == "__main__":
    main()
