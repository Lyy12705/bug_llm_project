#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "candidate_hit_at_20",
    "candidate_recall_at_20",
    "file_top_1_accuracy",
    "file_top_3_accuracy",
    "file_top_5_accuracy",
    "file_mrr",
    "symbol_top_1_accuracy",
    "symbol_top_3_accuracy",
    "symbol_top_5_accuracy",
    "symbol_mrr",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fault-localization metric files.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=METRICS.json",
        help="Add one named run. Repeat for TF-IDF, SBERT, or LLM outputs.",
    )
    parser.add_argument("--baseline", default=None, help="Baseline label; defaults to the first --run.")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs: dict[str, dict[str, Any]] = {}
    for value in args.run:
        if "=" not in value:
            raise SystemExit(f"Invalid --run value {value!r}; expected LABEL=PATH.")
        label, raw_path = value.split("=", maxsplit=1)
        label = label.strip()
        if not label or label in runs:
            raise SystemExit(f"Run labels must be non-empty and unique: {label!r}.")
        runs[label] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    comparison = compare_runs(runs, baseline=args.baseline)
    text = json.dumps(comparison, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(output)
    print(text)


def compare_runs(
    runs: dict[str, dict[str, Any]],
    *,
    baseline: str | None = None,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one metrics run is required.")
    baseline_label = baseline or next(iter(runs))
    if baseline_label not in runs:
        raise ValueError(f"Unknown baseline label: {baseline_label}")
    baseline_metrics = runs[baseline_label]
    baseline_file_rows = int(baseline_metrics.get("rows_with_file_ground_truth") or 0)
    baseline_symbol_rows = int(baseline_metrics.get("rows_with_symbol_ground_truth") or 0)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for label, metrics in runs.items():
        file_rows = int(metrics.get("rows_with_file_ground_truth") or 0)
        symbol_rows = int(metrics.get("rows_with_symbol_ground_truth") or 0)
        if file_rows != baseline_file_rows or symbol_rows != baseline_symbol_rows:
            warnings.append(
                f"{label} uses different evaluation denominators "
                f"(file={file_rows}, symbol={symbol_rows}) than {baseline_label} "
                f"(file={baseline_file_rows}, symbol={baseline_symbol_rows})."
            )
        metric_values = {key: float(metrics.get(key) or 0.0) for key in METRIC_KEYS}
        diagnostics = metrics.get("run_diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        llm_fallbacks = int(diagnostics.get("llm_rerank_fallback_predictions") or 0)
        if diagnostics.get("llm_rerank_requested") and llm_fallbacks:
            warnings.append(
                f"{label} requested LLM reranking but fell back to retrieval for "
                f"{llm_fallbacks} prediction(s)."
            )
        rows.append(
            {
                "label": label,
                "is_baseline": label == baseline_label,
                "rows_with_file_ground_truth": file_rows,
                "rows_with_symbol_ground_truth": symbol_rows,
                "metrics": metric_values,
                "run_diagnostics": diagnostics,
                "delta_vs_baseline": {
                    key: round(metric_values[key] - float(baseline_metrics.get(key) or 0.0), 6)
                    for key in METRIC_KEYS
                },
            }
        )
    return {"baseline": baseline_label, "runs": rows, "warnings": warnings}


if __name__ == "__main__":
    main()
