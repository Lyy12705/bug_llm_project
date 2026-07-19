from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from assignee_phase1_common import (
    DEFAULT_RAW_PATH,
    evaluate_current_triager,
    grouped_metrics,
    load_normalized_raw_records,
    normalize_assignee,
    rows_to_prediction_csv,
    write_csv,
    write_json,
    write_jsonl,
)
from assignee_phase2_common import DEFAULT_PHASE2_ROOT, build_window_view, default_windows, summarize_top1
from evaluate_assignee_open_set import (
    abstention_metrics,
    choose_confidence_threshold,
    enrich_open_set_predictions,
    prediction_fields as open_set_prediction_fields,
    prediction_rows as open_set_prediction_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run open-world temporal windows for current assignee triage.")
    parser.add_argument("--dataset", default="bmo_paper_2024_3k")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-train-assignee-count", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_PHASE2_ROOT / "reports" / args.dataset / "open_world_multi_window"
    prediction_dir = output_dir / "predictions"
    data_dir = output_dir / "window_data"
    records = load_normalized_raw_records(args.raw)
    windows = default_windows()
    metrics: dict[str, Any] = {
        "experiment_name": "open_world_multi_window_current_assignee_triager",
        "dataset": args.dataset,
        "raw_path": str(args.raw),
        "task_setting": "pre-filter open-world temporal windows; roster is all train assignees per window",
        "config_snapshot": {
            "min_train_assignee_count_for_bucket_only": args.min_train_assignee_count,
            "top_k": args.top_k,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "window_definitions": [window.__dict__ for window in windows],
        },
        "windows": {},
        "aggregate": {},
    }
    summary_rows = []

    for window in windows:
        view = build_window_view(records, window, min_train_assignee_count=args.min_train_assignee_count)
        train_rows = view["train_raw"]
        validation_rows = view["validation_raw"]
        test_rows = view["test_raw"]
        train_counts = Counter(normalize_assignee(row.get("assignee")) for row in train_rows)
        train_assignees = {assignee for assignee in train_counts if assignee}

        window_data_dir = data_dir / window.name
        train_path = window_data_dir / "open_world_history_train.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(window_data_dir / "open_world_validation_set.jsonl", validation_rows)
        write_jsonl(window_data_dir / "open_world_test_set.jsonl", test_rows)
        write_json(window_data_dir / "open_world_candidate_roster.json", {"candidates": sorted(train_assignees)})

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
        test_predictions, test_metrics = evaluate_current_triager(
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

        pred_jsonl = prediction_dir / f"{window.name}_open_world_predictions.jsonl"
        pred_csv = prediction_dir / f"{window.name}_open_world_predictions.csv"
        write_jsonl(pred_jsonl, enriched)
        write_csv(pred_csv, open_set_prediction_rows(enriched), open_set_prediction_fields())

        window_metrics = {
            "window_name": window.name,
            "window_kind": window.kind,
            "train_date_range": [window.train_start, window.train_end],
            "validation_date_range": [window.validation_start, window.validation_end],
            "test_date_range": [window.test_start, window.test_end],
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "test_count": len(test_rows),
            "train_assignee_count": len(train_assignees),
            "unseen_assignee_count": sum(1 for row in test_rows if train_counts.get(normalize_assignee(row.get("assignee")), 0) == 0),
            "low_frequency_assignee_count": sum(
                1
                for row in test_rows
                if 0 < train_counts.get(normalize_assignee(row.get("assignee")), 0) < args.min_train_assignee_count
            ),
            "validation_metrics_before_abstention": validation_metrics,
            "selected_abstention_policy": threshold_report,
            "test_metrics_before_abstention": test_metrics,
            "group_metrics_before_abstention": grouped_metrics(
                enriched, "open_set_group", bootstrap_samples=args.bootstrap_samples, seed=args.seed
            ),
            "frequency_bucket_metrics_before_abstention": grouped_metrics(
                enriched, "train_frequency_bucket", bootstrap_samples=args.bootstrap_samples, seed=args.seed
            ),
            "open_set_abstention_metrics": abstention_metrics(enriched),
            "predictions_jsonl": str(pred_jsonl),
            "predictions_csv": str(pred_csv),
        }
        metrics["windows"][window.name] = window_metrics
        summary_rows.append(flatten_window_row(window_metrics))

    top1_values = [
        row["test_metrics_before_abstention"]["top1_accuracy"] for row in metrics["windows"].values()
    ]
    coverage_values = [
        row["open_set_abstention_metrics"]["coverage"] for row in metrics["windows"].values()
    ]
    unseen_recall_values = [
        row["open_set_abstention_metrics"]["manual_triage_recall_unseen"]
        for row in metrics["windows"].values()
    ]
    worst_window = min(
        (
            (name, row["test_metrics_before_abstention"]["top1_accuracy"])
            for name, row in metrics["windows"].items()
        ),
        key=lambda item: item[1],
        default=("", 0.0),
    )
    metrics["aggregate"] = {
        "top1_before_abstention": {
            **summarize_top1(top1_values),
            "worst_window": worst_window[0],
            "worst_window_top1": worst_window[1],
        },
        "coverage_after_validation_threshold": summarize_top1(coverage_values),
        "manual_triage_recall_unseen_after_validation_threshold": summarize_top1(unseen_recall_values),
    }

    write_json(output_dir / "open_world_multi_window_metrics.json", metrics)
    write_csv(output_dir / "open_world_multi_window_summary.csv", summary_rows, summary_fields())
    write_report(output_dir / "open_world_multi_window_report.md", metrics)
    print(json.dumps(metrics["aggregate"], ensure_ascii=False, indent=2, sort_keys=True))


def flatten_window_row(row: dict[str, Any]) -> dict[str, Any]:
    test_metrics = row["test_metrics_before_abstention"]
    abstention = row["open_set_abstention_metrics"]
    return {
        "window_name": row["window_name"],
        "window_kind": row["window_kind"],
        "train_range": f"{row['train_date_range'][0]}..{row['train_date_range'][1]}",
        "validation_range": f"{row['validation_date_range'][0]}..{row['validation_date_range'][1]}",
        "test_range": f"{row['test_date_range'][0]}..{row['test_date_range'][1]}",
        "train_count": row["train_count"],
        "validation_count": row["validation_count"],
        "test_count": row["test_count"],
        "train_assignee_count": row["train_assignee_count"],
        "unseen_assignee_count": row["unseen_assignee_count"],
        "low_frequency_assignee_count": row["low_frequency_assignee_count"],
        "top1_accuracy": test_metrics["top1_accuracy"],
        "hit_at_3": test_metrics["hit_at_3"],
        "hit_at_5": test_metrics["hit_at_5"],
        "hit_at_10": test_metrics["hit_at_10"],
        "mrr": test_metrics["mrr"],
        "top1_ci_low": test_metrics["top1_bootstrap_95ci"]["low"],
        "top1_ci_high": test_metrics["top1_bootstrap_95ci"]["high"],
        "selected_threshold": row["selected_abstention_policy"]["selected_threshold"],
        "coverage": abstention["coverage"],
        "manual_triage_recall_unseen": abstention["manual_triage_recall_unseen"],
        "selective_accuracy_seen": abstention["selective_accuracy_seen"],
    }


def summary_fields() -> list[str]:
    return [
        "window_name",
        "window_kind",
        "train_range",
        "validation_range",
        "test_range",
        "train_count",
        "validation_count",
        "test_count",
        "train_assignee_count",
        "unseen_assignee_count",
        "low_frequency_assignee_count",
        "top1_accuracy",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "mrr",
        "top1_ci_low",
        "top1_ci_high",
        "selected_threshold",
        "coverage",
        "manual_triage_recall_unseen",
        "selective_accuracy_seen",
    ]


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Open-World Multi-Window Assignee Evaluation",
        "",
        "This complements the closed-set multi-window benchmark by keeping low-frequency and unseen test assignees visible.",
        "",
        "## Aggregate",
        "",
        f"- Top-1 mean/std/min/max before abstention: "
        f"`{metrics['aggregate']['top1_before_abstention']['mean']}` / "
        f"`{metrics['aggregate']['top1_before_abstention']['std']}` / "
        f"`{metrics['aggregate']['top1_before_abstention']['min']}` / "
        f"`{metrics['aggregate']['top1_before_abstention']['max']}`",
        f"- Worst window: `{metrics['aggregate']['top1_before_abstention']['worst_window']}` "
        f"({metrics['aggregate']['top1_before_abstention']['worst_window_top1']})",
        f"- Coverage after validation-selected threshold mean: "
        f"`{metrics['aggregate']['coverage_after_validation_threshold']['mean']}`",
        f"- Manual-triage recall on unseen mean: "
        f"`{metrics['aggregate']['manual_triage_recall_unseen_after_validation_threshold']['mean']}`",
        "",
        "## Windows",
        "",
        "| Window | Test | Train Assignees | Unseen | Low-Freq | Top-1 | Hit@10 | Threshold | Coverage | Unseen Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics["windows"].items():
        test_metrics = row["test_metrics_before_abstention"]
        abstention = row["open_set_abstention_metrics"]
        lines.append(
            f"| {name} | {row['test_count']} | {row['train_assignee_count']} | "
            f"{row['unseen_assignee_count']} | {row['low_frequency_assignee_count']} | "
            f"{test_metrics['top1_accuracy']:.6f} | {test_metrics['hit_at_10']:.6f} | "
            f"{row['selected_abstention_policy']['selected_threshold']} | "
            f"{abstention['coverage']:.6f} | {abstention['manual_triage_recall_unseen']:.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
