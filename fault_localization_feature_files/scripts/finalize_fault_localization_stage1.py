#!/usr/bin/env python3
"""Seal the one-time Stage-1 holdout result against its pre-holdout freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a frozen Stage-1 holdout run and write an immutable evaluation record."
    )
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--holdout-metrics", required=True)
    parser.add_argument("--holdout-tickets", required=True)
    parser.add_argument("--holdout-gold", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    final = finalize(
        freeze_manifest=Path(args.freeze_manifest),
        holdout_manifest=Path(args.holdout_manifest),
        holdout_metrics=Path(args.holdout_metrics),
        holdout_tickets=Path(args.holdout_tickets),
        holdout_gold=Path(args.holdout_gold),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def finalize(
    *,
    freeze_manifest: Path,
    holdout_manifest: Path,
    holdout_metrics: Path,
    holdout_tickets: Path,
    holdout_gold: Path,
) -> dict[str, Any]:
    for path in (
        freeze_manifest,
        holdout_manifest,
        holdout_metrics,
        holdout_tickets,
        holdout_gold,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    freeze = _read_object(freeze_manifest)
    run = _read_object(holdout_manifest)
    metrics = _read_object(holdout_metrics)
    if freeze.get("status") != "frozen_before_holdout":
        raise ValueError("Freeze manifest is not a valid pre-holdout seal.")
    if run.get("frozen_protocol") != freeze.get("protocol"):
        raise ValueError("Holdout protocol does not match the pre-holdout freeze.")
    if run.get("method_id") != freeze.get("method_id") or run.get("method") != freeze.get("method"):
        raise ValueError("Holdout method differs from the frozen Development method.")
    if run.get("failures") != 0 or run.get("predictions") != run.get("tickets_selected"):
        raise ValueError("Holdout run is incomplete or contains failures.")
    if metrics.get("matched_prediction_rows") != metrics.get("rows_with_file_ground_truth"):
        raise ValueError("Holdout metrics do not cover every row with file ground truth.")
    if metrics.get("rows_missing_stage1_output") != 0:
        raise ValueError("At least one Holdout prediction is missing explicit Stage-1 output.")
    if metrics.get("average_candidate_count_at_20") != 20.0:
        raise ValueError("Holdout output does not contain exactly 20 candidates per evaluated ticket.")

    for relative, artifact in (freeze.get("implementation") or {}).items():
        path = Path(str(artifact.get("path") or ""))
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"Implementation changed after the pre-holdout freeze: {relative}")

    return {
        "protocol": freeze.get("protocol"),
        "status": "holdout_evaluated_once",
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_id": freeze.get("method_id"),
        "method": freeze.get("method"),
        "pre_holdout_freeze": _artifact(freeze_manifest),
        "implementation": freeze.get("implementation"),
        "development_acceptance": freeze.get("development_acceptance"),
        "holdout_result": {
            "candidate_hit_at_20": metrics.get("candidate_hit_at_20"),
            "candidate_recall_at_20": metrics.get("candidate_recall_at_20"),
            "candidate_bootstrap_95_ci": metrics.get("candidate_bootstrap_95_ci"),
            "rows": metrics.get("rows_with_file_ground_truth"),
            "outcomes": metrics.get("candidate_outcomes_at_20"),
            "file_top_1_accuracy": metrics.get("file_top_1_accuracy"),
            "file_top_3_accuracy": metrics.get("file_top_3_accuracy"),
            "file_top_5_accuracy": metrics.get("file_top_5_accuracy"),
            "file_mrr": metrics.get("file_mrr"),
        },
        "holdout_artifacts": {
            "run_manifest": _artifact(holdout_manifest),
            "metrics": _artifact(holdout_metrics),
            "tickets": _artifact(holdout_tickets),
            "gold": _artifact(holdout_gold),
        },
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
