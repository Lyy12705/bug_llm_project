from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SYSTEM_ROOT.parent
DEFAULT_SOURCE = WORKSPACE_ROOT / "ToJson" / "dataset" / "raw" / "eclipse_issue_sample.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small Eclipse assignee-triage benchmark.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Raw Eclipse CSV with Assigned to labels.")
    parser.add_argument("--output-dir", type=Path, default=EVAL_ROOT / "data", help="Output data directory.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Temporal train split ratio.")
    args = parser.parse_args()

    records = load_eclipse_rows(args.source)
    if not records:
        raise SystemExit(f"No usable assignee rows found in {args.source}")

    records.sort(key=lambda row: (row.get("created_at") or "", row.get("ticket_id") or ""))
    split_index = max(1, min(len(records) - 1, int(len(records) * args.train_ratio)))
    train_rows = records[:split_index]
    test_rows = records[split_index:]

    roster = sorted({row["assignee"] for row in train_rows if row.get("assignee")})
    roster_set = set(roster)
    filtered_test_rows = [row for row in test_rows if row.get("assignee") in roster_set]
    dropped_test_rows = [row for row in test_rows if row.get("assignee") not in roster_set]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "eclipse_history_train.jsonl", train_rows)
    write_jsonl(args.output_dir / "eclipse_test_set.jsonl", filtered_test_rows)
    write_json(args.output_dir / "eclipse_candidate_roster.json", {"candidates": roster})

    summary = {
        "source": str(args.source),
        "total_usable_rows": len(records),
        "train_rows": len(train_rows),
        "test_rows_before_filter": len(test_rows),
        "test_rows_after_filter": len(filtered_test_rows),
        "test_rows_dropped_unseen_assignee": len(dropped_test_rows),
        "candidate_roster_size": len(roster),
        "train_assignee_distribution": Counter(row["assignee"] for row in train_rows).most_common(),
        "test_assignee_distribution": Counter(row["assignee"] for row in filtered_test_rows).most_common(),
        "component_distribution": Counter(row["component"] for row in records).most_common(),
    }
    write_json(args.output_dir / "eclipse_dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_eclipse_rows(path: Path) -> list[dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            ticket_id = clean(row.get("ID"))
            title = clean(row.get("Summary"))
            description = clean(row.get("Description"))
            component = clean(row.get("Component")).lower()
            priority = clean(row.get("Priority")) or "P3"
            assignee = clean(row.get("Assigned to")).lower()
            if not ticket_id or not title or not assignee:
                continue
            rows.append(
                {
                    "ticket_id": f"eclipse_{ticket_id}",
                    "title": title,
                    "description": description,
                    "component": component or "unknown",
                    "priority": priority,
                    "assignee": assignee,
                    "created_at": clean(row.get("Creation time")),
                    "source": "eclipse_issue_sample",
                }
            )
    return rows


def clean(value: Any) -> str:
    return str(value or "").strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

