from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_eligibility import (  # noqa: E402
    annotate_eligible_owner_labels_from_snapshots,
    eligible_owner_metrics,
    read_assignee_eligibility_index,
)

from assignee_open_set_common import (  # noqa: E402
    top5_assist_metrics,
    wilson_interval,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact historical-owner correctness and reviewed eligible-owner "
            "correctness side by side without relabeling historical observations."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--eligibility-roster",
        type=Path,
        action="append",
        required=True,
        help="Repeat for non-overlapping time-versioned eligibility snapshots.",
    )
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = read_prediction_rows(args.predictions)
    indexes = [read_assignee_eligibility_index(path) for path in args.eligibility_roster]
    annotate_eligible_owner_labels_from_snapshots(rows, indexes)
    labeled = [row for row in rows if row["eligibility_label_available"]]
    report = {
        "schema_version": 1,
        "method": "dual_exact_and_reviewed_eligible_owner_evaluation_v1",
        "exact_owner": exact_owner_metrics(rows),
        "eligible_owner": eligible_owner_metrics(rows),
        "top5_assist_exact_owner": (
            top5_assist_metrics(rows)
            if any("top5_assist_available" in row for row in rows)
            else None
        ),
        "top5_assist_eligible_owner": (
            top5_assist_metrics(
                labeled, correctness_field="is_top5_eligible_correct"
            )
            if labeled and any("top5_assist_available" in row for row in labeled)
            else None
        ),
        "eligible_policy_evaluation_ready": len(labeled) == len(rows) and bool(rows),
        "blockers": (
            []
            if len(labeled) == len(rows) and rows
            else ["one_or_more_predictions_lack_a_time_valid_eligibility_snapshot"]
        ),
        "interpretation": {
            "exact_owner": "historical recorded assignee",
            "eligible_owner": "any reviewed, permission-confirmed, time-valid capable assignee",
            "historical_components_are_qualification": False,
        },
    }
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_predictions, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            raise ValueError("prediction JSON must be a list of objects")
        return payload
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("prediction JSONL must contain objects")
    return rows


def exact_owner_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    top1 = sum(bool(row.get("is_top1_correct")) for row in rows)
    top5 = sum(bool(row.get("is_top5_correct")) for row in rows)
    return {
        "rows": total,
        "top1_correct_rows": top1,
        "top1_accuracy": round(top1 / total, 6) if total else 0.0,
        "top1_accuracy_ci95": wilson_interval(top1, total),
        "top5_correct_rows": top5,
        "top5_accuracy": round(top5 / total, 6) if total else 0.0,
        "top5_accuracy_ci95": wilson_interval(top5, total),
    }


if __name__ == "__main__":
    main()
