from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from assignee_phase1_common import (
    build_history_counts,
    file_sha256,
    frequency_bucket,
    normalize_assignee,
    normalize_component,
    prediction_csv_fields,
    read_jsonl,
    rows_to_prediction_csv,
    write_csv,
    write_json,
    write_jsonl,
)
from assignee_phase2_common import DEFAULT_PHASE2_ROOT, evaluate_baselines_for_window


BASELINES = ["bm25_text_knn", "current_assignee_triager"]
FEATURE_CONFIGS: dict[str, dict[str, bool]] = {
    "title_only": {"title": True, "description": False, "metadata": False},
    "description_only": {"title": False, "description": True, "metadata": False},
    "metadata_only": {"title": False, "description": False, "metadata": True},
    "title_plus_metadata": {"title": True, "description": False, "metadata": True},
    "title_plus_description": {"title": True, "description": True, "metadata": False},
    "title_plus_description_plus_metadata": {"title": True, "description": True, "metadata": True},
}
METADATA_FIELDS = ("product", "component", "priority", "severity", "status", "resolution", "bug_type")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate title/description/metadata ablations for assignee triage."
    )
    parser.add_argument("--dataset", default="bmo_paper_2024_3k_description_enriched")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_PHASE2_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_PHASE2_ROOT / "reports" / args.dataset / "description_ablation"
    data_output_dir = output_dir / "feature_sliced_data"
    prediction_dir = output_dir / "predictions"

    paths = dataset_paths(args.data_dir, args.dataset)
    train_rows = read_jsonl(paths["train"])
    validation_rows = read_jsonl(paths["validation"])
    test_rows = read_jsonl(paths["test"])
    roster_payload = json.loads(paths["roster"].read_text(encoding="utf-8"))
    roster = set(str(value) for value in roster_payload.get("candidates", []))

    train_counts = Counter(normalize_assignee(row.get("assignee")) for row in train_rows)
    original_test_by_id = {str(row.get("ticket_id")): row for row in test_rows}

    metrics_by_config: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for feature_name, switches in FEATURE_CONFIGS.items():
        transformed_train = [transform_row(row, switches) for row in train_rows]
        transformed_validation = [transform_row(row, switches) for row in validation_rows]
        transformed_test = [transform_row(row, switches) for row in test_rows]

        feature_data_dir = data_output_dir / feature_name
        train_path = feature_data_dir / "history_train.jsonl"
        validation_path = feature_data_dir / "validation_set.jsonl"
        test_path = feature_data_dir / "test_set.jsonl"
        write_jsonl(train_path, transformed_train)
        write_jsonl(validation_path, transformed_validation)
        write_jsonl(test_path, transformed_test)
        write_json(feature_data_dir / "candidate_roster.json", {"candidates": sorted(roster)})

        predictions, metrics = evaluate_baselines_for_window(
            train_rows=transformed_train,
            test_rows=transformed_test,
            roster=roster,
            train_path=train_path,
            baselines=BASELINES,
            top_k=args.top_k,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )

        enriched_predictions = enrich_predictions(
            predictions,
            feature_name=feature_name,
            switches=switches,
            train_counts=train_counts,
            original_test_by_id=original_test_by_id,
        )
        metrics_by_config[feature_name] = metrics
        all_prediction_rows.extend(enriched_predictions)

        write_jsonl(prediction_dir / f"{feature_name}_predictions.jsonl", enriched_predictions)
        write_csv(
            prediction_dir / f"{feature_name}_predictions.csv",
            rows_to_prediction_csv(enriched_predictions),
            prediction_csv_fields(
                [
                    "feature_config",
                    "train_assignee_count",
                    "frequency_bucket",
                    "original_description_nonempty",
                    "uses_title",
                    "uses_description",
                    "uses_metadata",
                ]
            ),
        )

        for baseline, baseline_metrics in metrics.items():
            metric_rows.append(metric_row(args.dataset, feature_name, baseline, baseline_metrics))
            bucket_rows.extend(
                bucket_metric_rows(
                    dataset=args.dataset,
                    feature_config=feature_name,
                    baseline=baseline,
                    predictions=[
                        row for row in enriched_predictions if row.get("baseline") == baseline
                    ],
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed,
                )
            )

        manifest_rows.append(
            {
                "feature_config": feature_name,
                "uses_title": switches["title"],
                "uses_description": switches["description"],
                "uses_metadata": switches["metadata"],
                "train_path": str(train_path),
                "validation_path": str(validation_path),
                "test_path": str(test_path),
                "train_rows": len(transformed_train),
                "validation_rows": len(transformed_validation),
                "test_rows": len(transformed_test),
                "roster_size": len(roster),
            }
        )

    add_reference_deltas(metric_rows)
    summary = build_summary(
        dataset=args.dataset,
        paths=paths,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        roster=roster,
        metrics_by_config=metrics_by_config,
        metric_rows=metric_rows,
        manifest_rows=manifest_rows,
        args=serializable_args(vars(args)),
    )
    write_json(output_dir / "description_ablation_summary.json", summary)
    write_csv(output_dir / "description_ablation_metrics.csv", metric_rows, metric_fields())
    write_csv(output_dir / "description_ablation_frequency_buckets.csv", bucket_rows, bucket_fields())
    write_csv(
        output_dir / "description_ablation_predictions.csv",
        rows_to_prediction_csv(all_prediction_rows),
        prediction_csv_fields(
            [
                "feature_config",
                "train_assignee_count",
                "frequency_bucket",
                "original_description_nonempty",
                "uses_title",
                "uses_description",
                "uses_metadata",
            ]
        ),
    )
    write_json(output_dir / "description_ablation_manifest.json", {"feature_slices": manifest_rows})
    write_report(output_dir / "description_ablation_report.md", summary, metric_rows)
    print(json.dumps(summary["headline"], ensure_ascii=False, indent=2, sort_keys=True))


def dataset_paths(data_dir: Path, dataset: str) -> dict[str, Path]:
    paths = {
        "train": data_dir / f"{dataset}_history_train.jsonl",
        "validation": data_dir / f"{dataset}_validation_set.jsonl",
        "test": data_dir / f"{dataset}_test_set.jsonl",
        "roster": data_dir / f"{dataset}_candidate_roster.json",
        "summary": data_dir / f"{dataset}_dataset_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing processed dataset files:\n" + "\n".join(missing))
    return paths


def serializable_args(args: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in args.items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def transform_row(row: dict[str, Any], switches: dict[str, bool]) -> dict[str, Any]:
    copy = dict(row)
    if not switches["title"]:
        copy["title"] = ""
    if not switches["description"]:
        copy["description"] = ""
    if not switches["metadata"]:
        for field in METADATA_FIELDS:
            if field in {"product", "component"}:
                copy[field] = "unknown"
            else:
                copy[field] = ""
    return copy


def enrich_predictions(
    predictions: list[dict[str, Any]],
    *,
    feature_name: str,
    switches: dict[str, bool],
    train_counts: Counter[str],
    original_test_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in predictions:
        expected = normalize_assignee(row.get("expected_assignee"))
        original = original_test_by_id.get(str(row.get("ticket_id")), {})
        train_count = train_counts.get(expected, 0)
        copy = dict(row)
        copy["feature_config"] = feature_name
        copy["train_assignee_count"] = train_count
        copy["frequency_bucket"] = frequency_bucket(train_count)
        copy["original_description_nonempty"] = bool(str(original.get("description") or "").strip())
        copy["uses_title"] = switches["title"]
        copy["uses_description"] = switches["description"]
        copy["uses_metadata"] = switches["metadata"]
        output.append(copy)
    return output


def metric_row(dataset: str, feature_config: str, baseline: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "feature_config": feature_config,
        "baseline": baseline,
        "rows": metrics["rows"],
        "top1_accuracy": metrics["top1_accuracy"],
        "hit_at_3": metrics["hit_at_3"],
        "hit_at_5": metrics["hit_at_5"],
        "hit_at_10": metrics["hit_at_10"],
        "mrr": metrics["mrr"],
        "macro_assignee_top1": metrics.get("macro_assignee_top1", ""),
        "macro_component_top1": metrics.get("macro_component_top1", ""),
        "top1_ci_low": metrics["top1_bootstrap_95ci"]["low"],
        "top1_ci_high": metrics["top1_bootstrap_95ci"]["high"],
        "delta_top1_vs_title_plus_metadata": "",
        "delta_mrr_vs_title_plus_metadata": "",
    }


def add_reference_deltas(metric_rows: list[dict[str, Any]]) -> None:
    reference: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        if row["feature_config"] == "title_plus_metadata":
            reference[row["baseline"]] = row
    for row in metric_rows:
        ref = reference.get(row["baseline"])
        if not ref:
            continue
        row["delta_top1_vs_title_plus_metadata"] = round(
            float(row["top1_accuracy"]) - float(ref["top1_accuracy"]), 6
        )
        row["delta_mrr_vs_title_plus_metadata"] = round(float(row["mrr"]) - float(ref["mrr"]), 6)


def bucket_metric_rows(
    *,
    dataset: str,
    feature_config: str,
    baseline: str,
    predictions: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_bucket[str(row.get("frequency_bucket") or "unknown")].append(row)
    output = []
    for bucket, rows in sorted(by_bucket.items()):
        metrics = simple_metrics(rows, bootstrap_samples=bootstrap_samples, seed=seed)
        output.append(
            {
                "dataset": dataset,
                "feature_config": feature_config,
                "baseline": baseline,
                "frequency_bucket": bucket,
                "rows": metrics["rows"],
                "top1_accuracy": metrics["top1_accuracy"],
                "hit_at_3": metrics["hit_at_3"],
                "hit_at_5": metrics["hit_at_5"],
                "hit_at_10": metrics["hit_at_10"],
                "mrr": metrics["mrr"],
                "top1_ci_low": metrics["top1_bootstrap_95ci"]["low"],
                "top1_ci_high": metrics["top1_bootstrap_95ci"]["high"],
            }
        )
    return output


def simple_metrics(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    from assignee_phase1_common import build_metrics

    return build_metrics(rows, bootstrap_samples=bootstrap_samples, seed=seed)


def build_summary(
    *,
    dataset: str,
    paths: dict[str, Path],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    roster: set[str],
    metrics_by_config: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    best_by_baseline = {}
    description_effects = {}
    for baseline in BASELINES:
        baseline_rows = [row for row in metric_rows if row["baseline"] == baseline]
        best = max(baseline_rows, key=lambda row: float(row["top1_accuracy"]))
        best_by_baseline[baseline] = {
            "feature_config": best["feature_config"],
            "top1_accuracy": best["top1_accuracy"],
            "hit_at_10": best["hit_at_10"],
            "mrr": best["mrr"],
        }
        by_feature = {row["feature_config"]: row for row in baseline_rows}
        description_effects[baseline] = {
            "all_minus_title_plus_metadata_top1": diff(
                by_feature,
                "title_plus_description_plus_metadata",
                "title_plus_metadata",
                "top1_accuracy",
            ),
            "title_plus_description_minus_title_only_top1": diff(
                by_feature,
                "title_plus_description",
                "title_only",
                "top1_accuracy",
            ),
            "description_only_minus_title_only_top1": diff(
                by_feature,
                "description_only",
                "title_only",
                "top1_accuracy",
            ),
            "all_minus_title_plus_metadata_mrr": diff(
                by_feature,
                "title_plus_description_plus_metadata",
                "title_plus_metadata",
                "mrr",
            ),
        }
    return {
        "experiment_name": "description_feature_ablation",
        "dataset": dataset,
        "input_paths": {key: str(value) for key, value in paths.items()},
        "input_sha256": {key: file_sha256(value) for key, value in paths.items()},
        "config_snapshot": {
            **args,
            "baselines": BASELINES,
            "feature_configs": FEATURE_CONFIGS,
            "metadata_fields": METADATA_FIELDS,
        },
        "data_snapshot": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows": len(test_rows),
            "roster_size": len(roster),
            "test_description_nonempty": sum(1 for row in test_rows if str(row.get("description") or "").strip()),
            "test_description_nonempty_ratio": round(
                sum(1 for row in test_rows if str(row.get("description") or "").strip()) / len(test_rows),
                6,
            )
            if test_rows
            else 0.0,
        },
        "feature_slices": manifest_rows,
        "metrics": metrics_by_config,
        "headline": {
            "best_by_baseline": best_by_baseline,
            "description_effects": description_effects,
        },
    }


def diff(by_feature: dict[str, dict[str, Any]], left: str, right: str, key: str) -> float | None:
    if left not in by_feature or right not in by_feature:
        return None
    return round(float(by_feature[left][key]) - float(by_feature[right][key]), 6)


def metric_fields() -> list[str]:
    return [
        "dataset",
        "feature_config",
        "baseline",
        "rows",
        "top1_accuracy",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "mrr",
        "macro_assignee_top1",
        "macro_component_top1",
        "top1_ci_low",
        "top1_ci_high",
        "delta_top1_vs_title_plus_metadata",
        "delta_mrr_vs_title_plus_metadata",
    ]


def bucket_fields() -> list[str]:
    return [
        "dataset",
        "feature_config",
        "baseline",
        "frequency_bucket",
        "rows",
        "top1_accuracy",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "mrr",
        "top1_ci_low",
        "top1_ci_high",
    ]


def write_report(path: Path, summary: dict[str, Any], metric_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Description Feature Ablation",
        "",
        "This is an evaluation-only ablation on the description-enriched BMO split.",
        "The production `AssigneeTriager` ranking logic is unchanged.",
        "",
        "## Data",
        "",
        f"- Train / validation / test: `{summary['data_snapshot']['train_rows']}` / "
        f"`{summary['data_snapshot']['validation_rows']}` / `{summary['data_snapshot']['test_rows']}`",
        f"- Candidate roster: `{summary['data_snapshot']['roster_size']}`",
        f"- Test descriptions present: `{summary['data_snapshot']['test_description_nonempty']}` "
        f"({summary['data_snapshot']['test_description_nonempty_ratio']})",
        "",
        "## Metrics",
        "",
        "| Baseline | Feature Config | Top-1 | Hit@3 | Hit@5 | Hit@10 | MRR | Delta Top-1 vs title+metadata |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(metric_rows, key=lambda item: (item["baseline"], item["feature_config"])):
        lines.append(
            f"| {row['baseline']} | {row['feature_config']} | {float(row['top1_accuracy']):.6f} | "
            f"{float(row['hit_at_3']):.6f} | {float(row['hit_at_5']):.6f} | "
            f"{float(row['hit_at_10']):.6f} | {float(row['mrr']):.6f} | "
            f"{row['delta_top1_vs_title_plus_metadata']} |"
        )
    lines.extend(
        [
            "",
            "## Headline Effects",
            "",
        ]
    )
    for baseline, effects in summary["headline"]["description_effects"].items():
        lines.append(
            f"- `{baseline}`: all minus title+metadata Top-1 = "
            f"`{effects['all_minus_title_plus_metadata_top1']}`, "
            f"MRR = `{effects['all_minus_title_plus_metadata_mrr']}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation Cautions",
            "",
            "- For `current_assignee_triager`, removing metadata collapses product/component to `unknown`; this is a stress test of missing metadata, not a pure text-only variant of the production algorithm.",
            "- Descriptions come from first public comments and can include boilerplate, HTML, code, and logs; this is useful for LLM-input planning but may add noise to deterministic BM25-style ranking.",
            "- All outputs are stored independently under the Phase 2 report directory.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
