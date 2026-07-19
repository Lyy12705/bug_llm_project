from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[1]
GENERIC_ASSIGNEE_RE = re.compile(
    r"(nobody|unassigned|triage|inbox|default|bugzilla|bugs@|noreply|do-not-reply|disabled)",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare temporal splits for a paper-grade assignee benchmark.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=PAPER_ROOT / "data" / "raw" / "bmo_bugs_raw.jsonl",
        help="Raw Bugzilla JSONL.",
    )
    parser.add_argument("--dataset", default="bmo_paper", help="Output dataset prefix.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_ROOT / "data" / "processed",
        help="Processed dataset output directory.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--min-train-assignee-count", type=int, default=10)
    parser.add_argument("--keep-generic-assignees", action="store_true")
    parser.add_argument("--keep-assignee-identifiers", action="store_true")
    parser.add_argument("--write-label-map", action="store_true")
    args = parser.parse_args()

    records = [record for record in (normalize_record(row) for row in read_jsonl(args.raw)) if record]
    if not args.keep_generic_assignees:
        records = [row for row in records if not GENERIC_ASSIGNEE_RE.search(row["assignee"])]
    records, duplicate_audit = collapse_duplicate_clusters(records)
    if len(records) < 3:
        raise SystemExit(f"Not enough usable records after filtering: {len(records)}")

    records.sort(key=lambda row: (row["created_at"], row["ticket_id"]))
    train_rows, validation_rows, test_rows = temporal_split(records, args.train_ratio, args.validation_ratio)
    train_counts = Counter(row["assignee"] for row in train_rows)
    roster_raw = sorted(
        assignee for assignee, count in train_counts.items() if count >= args.min_train_assignee_count
    )
    roster_raw_set = set(roster_raw)

    train_rows = [row for row in train_rows if row["assignee"] in roster_raw_set]
    validation_rows = [row for row in validation_rows if row["assignee"] in roster_raw_set]
    test_rows = [row for row in test_rows if row["assignee"] in roster_raw_set]

    label_map = {assignee: assignee for assignee in roster_raw}
    if not args.keep_assignee_identifiers:
        label_map = {assignee: f"dev_{index:04d}" for index, assignee in enumerate(roster_raw, start=1)}
        train_rows = anonymize_assignees(train_rows, label_map)
        validation_rows = anonymize_assignees(validation_rows, label_map)
        test_rows = anonymize_assignees(test_rows, label_map)
    roster = sorted(label_map.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / f"{args.dataset}_history_train.jsonl", train_rows)
    write_jsonl(args.output_dir / f"{args.dataset}_validation_set.jsonl", validation_rows)
    write_jsonl(args.output_dir / f"{args.dataset}_test_set.jsonl", test_rows)
    write_json(args.output_dir / f"{args.dataset}_candidate_roster.json", {"candidates": roster})

    summary = build_summary(
        raw_path=args.raw,
        dataset=args.dataset,
        all_records=records,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        roster=roster,
        min_train_assignee_count=args.min_train_assignee_count,
        anonymized=not args.keep_assignee_identifiers,
        duplicate_audit=duplicate_audit,
    )
    write_json(args.output_dir / f"{args.dataset}_dataset_summary.json", summary)
    if args.write_label_map:
        write_json(args.output_dir / f"{args.dataset}_assignee_label_map.json", label_map)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing raw input: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def normalize_record(row: dict[str, Any]) -> dict[str, Any] | None:
    bug_id = clean(row.get("id"))
    title = clean(row.get("summary"))
    assignee = clean(row.get("assigned_to")).lower()
    created_at = clean(row.get("creation_time"))
    if not bug_id or not title or not assignee or not created_at:
        return None
    if not valid_date(created_at):
        return None

    return {
        "ticket_id": f"bmo_{bug_id}",
        "title": title,
        "description": clean(row.get("description")),
        "product": clean(row.get("product")).lower() or "unknown",
        "component": clean(row.get("component")).lower() or "unknown",
        "priority": clean(row.get("priority")) or "P3",
        "severity": clean(row.get("severity")),
        "status": clean(row.get("status")),
        "resolution": clean(row.get("resolution")),
        "assignee": assignee,
        "created_at": created_at,
        "duplicate_of": f"bmo_{clean(row.get('dupe_of'))}" if clean(row.get("dupe_of")) else "",
        "source": "bugzilla_mozilla",
    }


def temporal_split(
    records: list[dict[str, Any]], train_ratio: float, validation_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_end = max(1, min(len(records) - 2, int(len(records) * train_ratio)))
    validation_end = max(train_end + 1, min(len(records) - 1, train_end + int(len(records) * validation_ratio)))
    return records[:train_end], records[train_end:validation_end], records[validation_end:]


def collapse_duplicate_clusters(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id = {str(row["ticket_id"]): row for row in records}
    parent = {ticket_id: ticket_id for ticket_id in by_id}

    def find(ticket_id: str) -> str:
        while parent[ticket_id] != ticket_id:
            parent[ticket_id] = parent[parent[ticket_id]]
            ticket_id = parent[ticket_id]
        return ticket_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    explicit_links = 0
    content_first: dict[str, str] = {}
    for ticket_id, row in by_id.items():
        duplicate_of = str(row.get("duplicate_of") or "")
        if duplicate_of and duplicate_of in by_id:
            union(ticket_id, duplicate_of)
            explicit_links += 1
        content_key = normalized_content_key(row)
        previous = content_first.get(content_key)
        if previous:
            union(ticket_id, previous)
        else:
            content_first[content_key] = ticket_id

    groups: dict[str, list[dict[str, Any]]] = {}
    for ticket_id, row in by_id.items():
        groups.setdefault(find(ticket_id), []).append(row)
    representatives = [
        min(group, key=lambda row: (row["created_at"], row["ticket_id"])) for group in groups.values()
    ]
    representatives.sort(key=lambda row: (row["created_at"], row["ticket_id"]))
    return representatives, {
        "input_rows": len(records),
        "independent_clusters": len(representatives),
        "rows_removed": len(records) - len(representatives),
        "multi_row_clusters": sum(1 for group in groups.values() if len(group) > 1),
        "explicit_duplicate_links": explicit_links,
    }


def normalized_content_key(row: dict[str, Any]) -> str:
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    return " ".join(re.findall(r"[a-z0-9_]+", text))


def anonymize_assignees(rows: list[dict[str, Any]], label_map: dict[str, str]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copy = dict(row)
        copy["assignee"] = label_map[copy["assignee"]]
        output.append(copy)
    return output


def build_summary(
    *,
    raw_path: Path,
    dataset: str,
    all_records: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    roster: list[str],
    min_train_assignee_count: int,
    anonymized: bool,
    duplicate_audit: dict[str, int],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "raw_path": str(raw_path),
        "usable_rows_before_frequency_filter": len(all_records),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "candidate_roster_size": len(roster),
        "min_train_assignee_count": min_train_assignee_count,
        "assignees_anonymized": anonymized,
        "duplicate_audit": duplicate_audit,
        "created_at_min": min((row["created_at"] for row in all_records), default=None),
        "created_at_max": max((row["created_at"] for row in all_records), default=None),
        "train_component_distribution": Counter(row["component"] for row in train_rows).most_common(20),
        "test_component_distribution": Counter(row["component"] for row in test_rows).most_common(20),
        "train_assignee_distribution": Counter(row["assignee"] for row in train_rows).most_common(20),
        "test_assignee_distribution": Counter(row["assignee"] for row in test_rows).most_common(20),
    }


def valid_date(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def clean(value: Any) -> str:
    return str(value or "").strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
