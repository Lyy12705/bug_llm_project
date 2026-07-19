from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from assignee_phase1_common import (
    DEFAULT_RAW_PATH,
    file_sha256,
    load_normalized_raw_records,
    normalize_assignee,
    normalize_component,
    rows_to_prediction_csv,
    write_csv,
    write_json,
    write_jsonl,
)
from assignee_phase2_common import (
    DEFAULT_PHASE2_ROOT,
    default_windows,
    evaluate_baselines_for_window,
    summarize_top1,
    window_rows,
    build_window_view,
)


BASELINES = [
    "global_majority",
    "product_majority",
    "component_majority",
    "product_component_majority",
    "bm25_text_knn",
    "current_assignee_triager",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-window temporal assignee-triage evaluation.")
    parser.add_argument("--dataset", default="bmo_paper_2024_3k")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-train-assignee-count", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_PHASE2_ROOT / "reports" / args.dataset / "multi_window_temporal"
    prediction_dir = output_dir / "multi_window_predictions"
    data_dir = output_dir / "window_data"
    records = load_normalized_raw_records(args.raw)
    windows = default_windows()

    window_manifest = []
    summary_rows = []
    all_metrics: dict[str, Any] = {
        "experiment_name": "multi_window_temporal_evaluation",
        "dataset": args.dataset,
        "raw_path": str(args.raw),
        "raw_sha256": file_sha256(args.raw),
        "date_distribution": dataset_distribution(records),
        "config_snapshot": {
            "min_train_assignee_count": args.min_train_assignee_count,
            "top_k": args.top_k,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "baselines": BASELINES,
            "window_definitions": [window.__dict__ for window in windows],
        },
        "windows": {},
        "aggregate": {},
    }

    for window in windows:
        view = build_window_view(records, window, min_train_assignee_count=args.min_train_assignee_count)
        train_rows = view["train"]
        validation_rows = view["validation"]
        test_rows = view["test"]
        roster = set(view["roster"])
        window_data_dir = data_dir / window.name
        train_path = window_data_dir / "history_train.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(window_data_dir / "validation_set.jsonl", validation_rows)
        write_jsonl(window_data_dir / "test_set.jsonl", test_rows)
        write_json(window_data_dir / "candidate_roster.json", {"candidates": sorted(roster)})

        predictions, metrics = evaluate_baselines_for_window(
            train_rows=train_rows,
            test_rows=test_rows,
            roster=roster,
            train_path=train_path,
            baselines=BASELINES,
            top_k=args.top_k,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        pred_jsonl = prediction_dir / f"{window.name}_predictions.jsonl"
        pred_csv = prediction_dir / f"{window.name}_predictions.csv"
        write_jsonl(pred_jsonl, predictions)
        write_csv(pred_csv, prediction_csv_rows(window.name, predictions), prediction_fields())

        window_info = window_summary(window, view, metrics, predictions)
        all_metrics["windows"][window.name] = {
            **window_info,
            "metrics": metrics,
            "predictions_jsonl": str(pred_jsonl),
            "predictions_csv": str(pred_csv),
        }
        window_manifest.append(window_info)
        for baseline, baseline_metrics in metrics.items():
            summary_rows.append(flatten_summary_row(window, window_info, baseline, baseline_metrics))

    for baseline in BASELINES:
        values = [
            all_metrics["windows"][window.name]["metrics"][baseline]["top1_accuracy"]
            for window in windows
            if baseline in all_metrics["windows"][window.name]["metrics"]
        ]
        worst_window = min(
            (
                (window.name, all_metrics["windows"][window.name]["metrics"][baseline]["top1_accuracy"])
                for window in windows
                if baseline in all_metrics["windows"][window.name]["metrics"]
            ),
            key=lambda item: item[1],
            default=("", 0.0),
        )
        all_metrics["aggregate"][baseline] = {
            **summarize_top1(values),
            "worst_window": worst_window[0],
            "worst_window_top1": round(worst_window[1], 6),
        }

    write_json(output_dir / "multi_window_metrics.json", all_metrics)
    write_csv(output_dir / "multi_window_summary.csv", summary_rows, summary_fields())
    write_json(output_dir / "multi_window_manifest.json", {"windows": window_manifest})
    write_report(output_dir / "temporal_stability_report.md", all_metrics)
    print(json.dumps(all_metrics["aggregate"], ensure_ascii=False, indent=2, sort_keys=True))


def dataset_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "created_at_min": min((row["created_at"] for row in records), default=""),
        "created_at_max": max((row["created_at"] for row in records), default=""),
        "assignee_count": len({normalize_assignee(row.get("assignee")) for row in records}),
        "product_distribution": Counter(normalize_component(row.get("product")) for row in records).most_common(),
        "top_components": Counter(normalize_component(row.get("component")) for row in records).most_common(25),
    }


def window_summary(
    window: Any, view: dict[str, Any], metrics: dict[str, Any], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    train_raw = view["train_raw"]
    validation_raw = view["validation_raw"]
    test_raw = view["test_raw"]
    test = view["test"]
    train_counts = view["train_counts_all"]
    current_predictions = [row for row in predictions if row.get("baseline") == "current_assignee_triager"]
    head_tail = head_tail_metrics(current_predictions, train_counts)
    return {
        "window_name": window.name,
        "window_kind": window.kind,
        "train_date_range": [window.train_start, window.train_end],
        "validation_date_range": [window.validation_start, window.validation_end],
        "test_date_range": [window.test_start, window.test_end],
        "train_count_raw": len(train_raw),
        "validation_count_raw": len(validation_raw),
        "test_count_raw": len(test_raw),
        "train_count": len(view["train"]),
        "validation_count": len(view["validation"]),
        "test_count": len(test),
        "roster_size": len(view["roster"]),
        "unseen_assignee_count": len(view["unseen_test"]),
        "low_frequency_assignee_count": len(view["low_frequency_test"]),
        "product_count": len({normalize_component(row.get("product")) for row in test_raw}),
        "component_count": len({normalize_component(row.get("component")) for row in test_raw}),
        "seen_unseen_ratio": round(len(test) / len(view["unseen_test"]), 6) if view["unseen_test"] else None,
        "head_top1_current": head_tail["head_top1"],
        "tail_top1_current": head_tail["tail_top1"],
        "head_tail_gap_current": head_tail["head_tail_gap"],
    }


def head_tail_metrics(predictions: list[dict[str, Any]], train_counts: Counter[str]) -> dict[str, Any]:
    head = [
        row for row in predictions if train_counts.get(normalize_assignee(row.get("expected_assignee")), 0) >= 20
    ]
    tail = [
        row
        for row in predictions
        if 10 <= train_counts.get(normalize_assignee(row.get("expected_assignee")), 0) < 20
    ]
    head_top1 = sum(1 for row in head if row.get("is_top1_correct")) / len(head) if head else None
    tail_top1 = sum(1 for row in tail if row.get("is_top1_correct")) / len(tail) if tail else None
    gap = None if head_top1 is None or tail_top1 is None else round(head_top1 - tail_top1, 6)
    return {
        "head_top1": None if head_top1 is None else round(head_top1, 6),
        "tail_top1": None if tail_top1 is None else round(tail_top1, 6),
        "head_tail_gap": gap,
    }


def flatten_summary_row(window: Any, info: dict[str, Any], baseline: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_name": window.name,
        "window_kind": window.kind,
        "baseline": baseline,
        "train_range": f"{window.train_start}..{window.train_end}",
        "validation_range": f"{window.validation_start}..{window.validation_end}",
        "test_range": f"{window.test_start}..{window.test_end}",
        "train_count": info["train_count"],
        "validation_count": info["validation_count"],
        "test_count": info["test_count"],
        "test_count_raw": info["test_count_raw"],
        "roster_size": info["roster_size"],
        "unseen_assignee_count": info["unseen_assignee_count"],
        "low_frequency_assignee_count": info["low_frequency_assignee_count"],
        "product_count": info["product_count"],
        "component_count": info["component_count"],
        "head_top1_current": info["head_top1_current"],
        "tail_top1_current": info["tail_top1_current"],
        "head_tail_gap_current": info["head_tail_gap_current"],
        "top1_accuracy": metrics["top1_accuracy"],
        "hit_at_3": metrics["hit_at_3"],
        "hit_at_5": metrics["hit_at_5"],
        "hit_at_10": metrics["hit_at_10"],
        "mrr": metrics["mrr"],
        "macro_assignee_top1": metrics.get("macro_assignee_top1", ""),
        "top1_ci_low": metrics["top1_bootstrap_95ci"]["low"],
        "top1_ci_high": metrics["top1_bootstrap_95ci"]["high"],
    }


def prediction_csv_rows(window_name: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in rows_to_prediction_csv(predictions):
        row["window_name"] = window_name
        rows.append(row)
    return rows


def prediction_fields() -> list[str]:
    return [
        "window_name",
        "baseline",
        "ticket_id",
        "product",
        "component",
        "priority",
        "expected_assignee",
        "predicted_assignee",
        "rank",
        "reciprocal_rank",
        "is_top1_correct",
        "is_hit_at_3",
        "is_hit_at_5",
        "is_hit_at_10",
        "ranked_candidates",
        "title",
    ]


def summary_fields() -> list[str]:
    return [
        "window_name",
        "window_kind",
        "baseline",
        "train_range",
        "validation_range",
        "test_range",
        "train_count",
        "validation_count",
        "test_count",
        "test_count_raw",
        "roster_size",
        "unseen_assignee_count",
        "low_frequency_assignee_count",
        "product_count",
        "component_count",
        "head_top1_current",
        "tail_top1_current",
        "head_tail_gap_current",
        "top1_accuracy",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "mrr",
        "macro_assignee_top1",
        "top1_ci_low",
        "top1_ci_high",
    ]


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Multi-Window Temporal Assignee Evaluation",
        "",
        "All windows are temporal. Roster and history are rebuilt from each window's train range only.",
        "",
        "## Aggregate Top-1",
        "",
        "| Baseline | Mean | Std | Min | Max | Worst Window |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for baseline, row in metrics["aggregate"].items():
        lines.append(
            f"| {baseline} | {row['mean']:.6f} | {row['std']:.6f} | {row['min']:.6f} | "
            f"{row['max']:.6f} | {row['worst_window']} ({row['worst_window_top1']:.6f}) |"
        )
    lines.extend(["", "## Window Notes", ""])
    for name, window in metrics["windows"].items():
        lines.append(
            f"- `{name}`: train `{window['train_count']}` / validation `{window['validation_count']}` / "
            f"test `{window['test_count']}` from raw test `{window['test_count_raw']}`, roster `{window['roster_size']}`, "
            f"unseen `{window['unseen_assignee_count']}`, low-frequency `{window['low_frequency_assignee_count']}`."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
