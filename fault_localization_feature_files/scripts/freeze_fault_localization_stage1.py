#!/usr/bin/env python3
"""Validate and seal a Stage-1 development run before opening the holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_swebench_lite_fault_localization import (
    FROZEN_STAGE1_METHOD_SPEC,
    FROZEN_STAGE1_PROTOCOL,
)


IMPLEMENTATION_FILES = (
    "src/utils/fault_localization.py",
    "src/modules/bug_localizer.py",
    "scripts/run_swebench_lite_fault_localization.py",
    "scripts/evaluate_fault_localization.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal an evaluated Stage-1 development run with data and code fingerprints."
    )
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--development-metrics", required=True)
    parser.add_argument("--development-tickets", required=True)
    parser.add_argument("--development-gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-development-recall", type=float, default=0.90)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    freeze = build_freeze_manifest(
        development_manifest=Path(args.development_manifest),
        development_metrics=Path(args.development_metrics),
        development_tickets=Path(args.development_tickets),
        development_gold=Path(args.development_gold),
        minimum_development_recall=args.minimum_development_recall,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


def build_freeze_manifest(
    *,
    development_manifest: Path,
    development_metrics: Path,
    development_tickets: Path,
    development_gold: Path,
    minimum_development_recall: float,
) -> dict[str, Any]:
    if not 0.0 <= minimum_development_recall <= 1.0:
        raise ValueError("minimum_development_recall must be between 0 and 1.")
    for path in (
        development_manifest,
        development_metrics,
        development_tickets,
        development_gold,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    run = _read_object(development_manifest)
    metrics = _read_object(development_metrics)
    method = run.get("method")
    if method != FROZEN_STAGE1_METHOD_SPEC:
        raise ValueError("Development method does not exactly match the frozen Stage-1 method spec.")
    if run.get("frozen_protocol") != FROZEN_STAGE1_PROTOCOL:
        raise ValueError("Development run was not executed with the frozen Stage-1 protocol flag.")
    if run.get("failures") != 0 or run.get("predictions") != run.get("tickets_selected"):
        raise ValueError("Development run is incomplete or contains failures.")
    if metrics.get("matched_prediction_rows") != metrics.get("rows_with_file_ground_truth"):
        raise ValueError("Development metrics do not cover every row with file ground truth.")
    if metrics.get("rows_missing_stage1_output") != 0:
        raise ValueError("At least one development prediction is missing explicit Stage-1 output.")
    if metrics.get("average_candidate_count_at_20") != 20.0:
        raise ValueError("Development output does not contain exactly 20 candidates per evaluated ticket.")
    recall = float(metrics.get("candidate_recall_at_20") or 0.0)
    if recall < minimum_development_recall:
        raise ValueError(
            f"Development Recall@20 {recall:.6f} is below the freeze threshold "
            f"{minimum_development_recall:.6f}."
        )

    return {
        "protocol": FROZEN_STAGE1_PROTOCOL,
        "status": "frozen_before_holdout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_id": run.get("method_id"),
        "method": method,
        "development_acceptance": {
            "minimum_candidate_recall_at_20": minimum_development_recall,
            "candidate_hit_at_20": metrics.get("candidate_hit_at_20"),
            "candidate_recall_at_20": recall,
            "candidate_bootstrap_95_ci": metrics.get("candidate_bootstrap_95_ci"),
            "rows": metrics.get("rows_with_file_ground_truth"),
            "miss_rows": (metrics.get("candidate_outcomes_at_20") or {}).get("miss_rows"),
            "average_candidate_count_at_20": metrics.get("average_candidate_count_at_20"),
        },
        "development_artifacts": {
            "run_manifest": _artifact(development_manifest),
            "metrics": _artifact(development_metrics),
            "tickets": _artifact(development_tickets),
            "gold": _artifact(development_gold),
        },
        "implementation": {
            relative: _artifact(ROOT / relative)
            for relative in IMPLEMENTATION_FILES
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
