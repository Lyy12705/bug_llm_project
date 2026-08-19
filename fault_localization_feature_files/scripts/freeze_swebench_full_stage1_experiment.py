#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    "src/utils/fault_localization.py",
    "src/modules/bug_localizer.py",
    "scripts/run_swebench_lite_fault_localization.py",
    "scripts/run_swebench_full_stage1_experiment.py",
    "scripts/evaluate_fault_localization.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare full SWE-bench validation methods and freeze the winner before holdout."
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--run-manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_manifest_path = Path(args.split_manifest)
    split_manifest = read_object(split_manifest_path)
    runs = [read_object(Path(path)) for path in args.run_manifest]
    if len(runs) < 2:
        raise ValueError("At least two validation methods are required before freezing.")
    for run in runs:
        validate_run(run)
    selected = max(
        runs,
        key=lambda run: (
            float(run["metrics"]["candidate_recall_at_20"]),
            float(run["metrics"]["candidate_hit_at_20"]),
            float(run["metrics"]["file_top_1_accuracy"]),
            float(run["metrics"]["file_mrr"]),
        ),
    )
    holdout_files = split_manifest["split_files"]["frozen_holdout"]
    validation_files = split_manifest["split_files"]["validation"]
    selected_ticket_hash = sha256_file(Path(selected["tickets_path"]))
    selected_gold_hash = sha256_file(Path(selected["gold_path"]))
    if selected_ticket_hash != validation_files["tickets_sha256"]:
        raise ValueError("Selected validation run does not match the frozen Validation ticket file.")
    if selected_gold_hash != validation_files["gold_sha256"]:
        raise ValueError("Selected validation run does not match the frozen Validation gold file.")

    comparison = []
    for run in sorted(runs, key=lambda row: str(row["method_label"])):
        metrics = run["metrics"]
        comparison.append(
            {
                "method_label": run["method_label"],
                "method_id": run["method_id"],
                "candidate_hit_at_20": metrics["candidate_hit_at_20"],
                "candidate_recall_at_20": metrics["candidate_recall_at_20"],
                "file_top_1_accuracy": metrics["file_top_1_accuracy"],
                "file_mrr": metrics["file_mrr"],
                "bootstrap_95_ci": metrics["candidate_bootstrap_95_ci"],
                "miss_rows": metrics["candidate_outcomes_at_20"]["miss_rows"],
                "run_manifest": artifact(Path(run["predictions_output"]).parent / "run_manifest.json"),
            }
        )

    freeze = {
        "protocol": "swebench-full-stage1-protocol-v1",
        "status": "frozen_before_holdout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": (
            "Highest Validation Recall@20; ties break by Hit@20, Top-1 accuracy, then MRR."
        ),
        "dataset_rows": split_manifest["source_rows"],
        "validation_rows": validation_files["tickets"],
        "frozen_holdout_rows": holdout_files["tickets"],
        "method_comparison": comparison,
        "selected_method_label": selected["method_label"],
        "selected_method_id": selected["method_id"],
        "selected_method": selected["method"],
        "selected_validation_metrics": {
            key: selected["metrics"][key]
            for key in (
                "candidate_hit_at_20",
                "candidate_recall_at_20",
                "candidate_bootstrap_95_ci",
                "file_top_1_accuracy",
                "file_top_3_accuracy",
                "file_top_5_accuracy",
                "file_mrr",
                "candidate_outcomes_at_20",
                "per_repository",
            )
        },
        "split_manifest": artifact(split_manifest_path),
        "frozen_holdout_artifacts": {
            "tickets": artifact(Path(holdout_files["tickets_path"])),
            "gold": artifact(Path(holdout_files["gold_path"])),
        },
        "implementation": {
            relative: artifact(ROOT / relative)
            for relative in IMPLEMENTATION_FILES
        },
        "holdout_policy": "Execute the selected method once; do not tune after observing holdout results.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze manifest: {output}")
    output.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


def validate_run(run: dict[str, Any]) -> None:
    if run.get("protocol") != "swebench-full-stage1-protocol-v1":
        raise ValueError("Validation run uses the wrong protocol.")
    if run.get("tickets") != 500 or run.get("predictions") != 500 or run.get("failures") != 0:
        raise ValueError(f"Incomplete validation run: {run.get('method_label')}")
    metrics = run.get("metrics") or {}
    if metrics.get("matched_prediction_rows") != 500 or metrics.get("rows_missing_stage1_output") != 0:
        raise ValueError(f"Invalid validation coverage: {run.get('method_label')}")
    if metrics.get("average_candidate_count_at_20") != 20.0:
        raise ValueError(f"Validation run does not output exactly 20 candidates: {run.get('method_label')}")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
