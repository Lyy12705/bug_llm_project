from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from assignee_phase1_common import (
    DEFAULT_PHASE1_ROOT,
    DEFAULT_RAW_PATH,
    build_metrics,
    build_prefilter_views,
    evaluate_current_triager,
    frequency_bucket,
    grouped_metrics,
    read_jsonl,
    ratio,
    rows_to_prediction_csv,
    write_csv,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate current assignee triage on pre-filter open-set views.")
    parser.add_argument("--dataset", default="bmo_paper_2024_3k")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-train-assignee-count", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_PHASE1_ROOT / "reports" / args.dataset / "prefilter_open_world_eval"
    data_dir = output_dir / "data"
    views = build_prefilter_views(args.raw, min_train_assignee_count=args.min_train_assignee_count)
    train_rows = views["pre_train"]
    validation_rows = views["pre_validation"]
    test_rows = views["pre_test"]
    train_counts = views["train_counts"]
    train_assignees = set(views["train_assignees"])

    train_path = data_dir / "prefilter_history_train.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(data_dir / "prefilter_validation_set.jsonl", validation_rows)
    write_jsonl(data_dir / "prefilter_test_set.jsonl", test_rows)
    write_json(data_dir / "prefilter_candidate_roster.json", {"candidates": sorted(train_assignees)})

    validation_predictions, validation_metrics = evaluate_current_triager(
        train_path=train_path,
        train_rows=train_rows,
        test_rows=validation_rows,
        roster=train_assignees,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    threshold_report = choose_confidence_threshold(validation_predictions, train_assignees)

    test_predictions, base_metrics = evaluate_current_triager(
        train_path=train_path,
        train_rows=train_rows,
        test_rows=test_rows,
        roster=train_assignees,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    enriched = enrich_open_set_predictions(
        test_predictions,
        train_counts=train_counts,
        threshold=threshold_report["selected_threshold"],
    )
    validation_enriched = enrich_open_set_predictions(
        validation_predictions,
        train_counts=train_counts,
        threshold=threshold_report["selected_threshold"],
    )

    metrics = {
        "experiment_name": "prefilter_open_world_eval",
        "dataset": args.dataset,
        "task_setting": "pre-filter open-world view with evaluation-only manual_triage threshold",
        "raw_path": str(args.raw),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_count": len(test_rows),
        "train_assignee_count": len(train_assignees),
        "frequent_assignee_threshold": args.min_train_assignee_count,
        "top_k": args.top_k,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "validation_metrics_before_abstention": validation_metrics,
        "selected_abstention_policy": threshold_report,
        "test_metrics_before_abstention": base_metrics,
        "group_metrics_before_abstention": grouped_metrics(
            enriched, "open_set_group", bootstrap_samples=args.bootstrap_samples, seed=args.seed
        ),
        "frequency_bucket_metrics_before_abstention": grouped_metrics(
            enriched, "train_frequency_bucket", bootstrap_samples=args.bootstrap_samples, seed=args.seed
        ),
        "open_set_abstention_metrics": abstention_metrics(enriched),
        "validation_abstention_metrics": abstention_metrics(validation_enriched),
        "notes": [
            "Threshold is selected on validation only and is evaluation-only; production current_assignee_triager is unchanged.",
            "Unseen assignees are evaluated via manual_triage/abstention metrics, not as ordinary closed-set classes.",
        ],
    }

    excluded_rows = excluded_ticket_audit(enriched)
    long_tail_rows = long_tail_bucket_rows(enriched)
    write_json(output_dir / "open_set_metrics.json", metrics)
    write_csv(output_dir / "excluded_ticket_audit.csv", excluded_rows, excluded_fields())
    write_csv(output_dir / "long_tail_bucket_metrics.csv", long_tail_rows, long_tail_fields())
    write_csv(output_dir / "per_ticket_open_set_predictions.csv", prediction_rows(enriched), prediction_fields())
    write_summary(output_dir / "prefilter_open_world_summary.md", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


def choose_confidence_threshold(predictions: list[dict[str, Any]], train_assignees: set[str]) -> dict[str, Any]:
    thresholds = [round(0.35 + index * 0.01, 2) for index in range(61)]
    best: dict[str, Any] | None = None
    unseen_count = sum(1 for row in predictions if row["expected_assignee"] not in train_assignees)
    for threshold in thresholds:
        enriched = [
            {
                **row,
                "should_manual_triage": float(row.get("confidence") or 0.0) < threshold,
                "is_unseen_assignee": row["expected_assignee"] not in train_assignees,
            }
            for row in predictions
        ]
        stats = abstention_metrics(enriched)
        objective = (
            stats["manual_triage_recall_unseen"]
            + stats["selective_accuracy_seen"]
            + 0.25 * stats["coverage"]
            - stats["false_auto_assign_rate_unseen"]
        )
        if unseen_count == 0:
            objective = stats["selective_accuracy_seen"] + 0.25 * stats["coverage"]
        candidate = {
            "selected_threshold": threshold,
            "selection_objective": round(objective, 6),
            "validation_unseen_count": unseen_count,
            **stats,
        }
        if best is None:
            best = candidate
            continue
        key = (
            candidate["selection_objective"],
            candidate["selective_accuracy_seen"],
            candidate["coverage"],
            -candidate["selected_threshold"],
        )
        best_key = (
            best["selection_objective"],
            best["selective_accuracy_seen"],
            best["coverage"],
            -best["selected_threshold"],
        )
        if key > best_key:
            best = candidate
    assert best is not None
    best["policy"] = "manual_triage_if_confidence_below_threshold"
    best["policy_scope"] = "evaluation_only_validation_selected"
    return best


def enrich_open_set_predictions(
    predictions: list[dict[str, Any]], *, train_counts: dict[str, int], threshold: float
) -> list[dict[str, Any]]:
    enriched = []
    for row in predictions:
        expected = row["expected_assignee"]
        train_frequency = int(train_counts.get(expected, 0))
        if train_frequency >= 10:
            group = "frequent_seen"
        elif train_frequency > 0:
            group = "low_frequency_seen"
        else:
            group = "unseen_assignee"
        should_manual = float(row.get("confidence") or 0.0) < threshold
        auto_prediction = "manual_triage" if should_manual else row["predicted_assignee"]
        enriched.append(
            {
                **row,
                "train_frequency": train_frequency,
                "train_frequency_bucket": frequency_bucket(train_frequency),
                "open_set_group": group,
                "is_unseen_assignee": train_frequency == 0,
                "should_manual_triage": should_manual,
                "auto_prediction": auto_prediction,
                "is_selective_auto_correct": (not should_manual) and train_frequency > 0 and row["is_top1_correct"],
            }
        )
    return enriched


def abstention_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    auto_rows = [row for row in rows if not row.get("should_manual_triage")]
    seen_rows = [row for row in rows if not row.get("is_unseen_assignee")]
    unseen_rows = [row for row in rows if row.get("is_unseen_assignee")]
    auto_seen = [row for row in auto_rows if not row.get("is_unseen_assignee")]
    unseen_manual = [row for row in unseen_rows if row.get("should_manual_triage")]
    seen_manual = [row for row in seen_rows if row.get("should_manual_triage")]
    selective_correct = [row for row in auto_seen if row.get("is_top1_correct")]
    selective_accuracy = ratio(len(selective_correct), len(auto_seen))
    return {
        "rows": total,
        "coverage": ratio(len(auto_rows), total),
        "manual_triage_rate": ratio(total - len(auto_rows), total),
        "seen_rows": len(seen_rows),
        "unseen_rows": len(unseen_rows),
        "manual_triage_recall_unseen": ratio(len(unseen_manual), len(unseen_rows)),
        "open_set_recall": ratio(len(unseen_manual), len(unseen_rows)),
        "false_auto_assign_rate_unseen": ratio(len(unseen_rows) - len(unseen_manual), len(unseen_rows)),
        "false_manual_triage_rate_seen": ratio(len(seen_manual), len(seen_rows)),
        "selective_accuracy_seen": selective_accuracy,
        "selective_risk_seen": round(1.0 - selective_accuracy, 6) if auto_seen else 0.0,
        "auto_assigned_seen_rows": len(auto_seen),
    }


def excluded_ticket_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["open_set_group"] == "frequent_seen":
            continue
        reason = "unseen_assignee" if row["open_set_group"] == "unseen_assignee" else "low_frequency_seen_assignee"
        output.append(
            {
                "ticket_id": row["ticket_id"],
                "exclusion_reason_from_closed_set": reason,
                "open_set_group": row["open_set_group"],
                "train_frequency": row["train_frequency"],
                "train_frequency_bucket": row["train_frequency_bucket"],
                "expected_assignee": row["expected_assignee"],
                "predicted_assignee": row["predicted_assignee"],
                "confidence": row.get("confidence"),
                "title": row.get("title"),
            }
        )
    return output


def long_tail_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for bucket, bucket_rows in sorted(group_by(rows, "train_frequency_bucket").items()):
        metrics = build_metrics(bucket_rows)
        output.append(
            {
                "train_frequency_bucket": bucket,
                "ticket_count": len(bucket_rows),
                "unique_assignee_count": len({row["expected_assignee"] for row in bucket_rows}),
                "top1_accuracy": metrics["top1_accuracy"],
                "hit_at_3": metrics["hit_at_3"],
                "hit_at_5": metrics["hit_at_5"],
                "hit_at_10": metrics["hit_at_10"],
                "mrr": metrics["mrr"],
            }
        )
    return output


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return grouped


def prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows_to_prediction_csv(rows):
        row["should_manual_triage"] = str(row.get("should_manual_triage"))
        row["is_unseen_assignee"] = str(row.get("is_unseen_assignee"))
        output.append(row)
    return output


def prediction_fields() -> list[str]:
    return [
        "ticket_id",
        "open_set_group",
        "train_frequency",
        "train_frequency_bucket",
        "expected_assignee",
        "predicted_assignee",
        "auto_prediction",
        "should_manual_triage",
        "is_unseen_assignee",
        "rank",
        "reciprocal_rank",
        "is_top1_correct",
        "is_hit_at_3",
        "is_hit_at_5",
        "is_hit_at_10",
        "confidence",
        "margin",
        "product",
        "component",
        "priority",
        "ranked_candidates",
        "reason",
        "title",
    ]


def excluded_fields() -> list[str]:
    return [
        "ticket_id",
        "exclusion_reason_from_closed_set",
        "open_set_group",
        "train_frequency",
        "train_frequency_bucket",
        "expected_assignee",
        "predicted_assignee",
        "confidence",
        "title",
    ]


def long_tail_fields() -> list[str]:
    return [
        "train_frequency_bucket",
        "ticket_count",
        "unique_assignee_count",
        "top1_accuracy",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "mrr",
    ]


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    group_metrics = metrics["group_metrics_before_abstention"]
    abstention = metrics["open_set_abstention_metrics"]
    lines = [
        "# Prefilter Open-World Assignee Evaluation",
        "",
        "- Production `current_assignee_triager` ranking logic was not modified.",
        "- This view uses the pre-filter temporal test split and keeps low-frequency/unseen assignees visible.",
        f"- Train / validation / test: `{metrics['train_count']}` / `{metrics['validation_count']}` / `{metrics['test_count']}`",
        f"- Train assignee count: `{metrics['train_assignee_count']}`",
        f"- Evaluation-only confidence threshold: `{metrics['selected_abstention_policy']['selected_threshold']}`",
        "",
        "## Group Metrics Before Abstention",
        "",
        "| Group | Rows | Top-1 | Hit@3 | Hit@5 | Hit@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in group_metrics.items():
        lines.append(
            f"| {group} | {row['rows']} | {row['top1_accuracy']:.6f} | {row['hit_at_3']:.6f} | "
            f"{row['hit_at_5']:.6f} | {row['hit_at_10']:.6f} | {row['mrr']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Abstention Metrics",
            "",
            f"- Coverage: `{abstention['coverage']}`",
            f"- Selective accuracy on seen auto-assigned rows: `{abstention['selective_accuracy_seen']}`",
            f"- Manual-triage recall on unseen assignees: `{abstention['manual_triage_recall_unseen']}`",
            f"- False auto-assign rate on unseen assignees: `{abstention['false_auto_assign_rate_unseen']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
