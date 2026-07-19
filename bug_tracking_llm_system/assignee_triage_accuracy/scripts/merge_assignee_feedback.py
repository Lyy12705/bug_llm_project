from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_feedback import merge_feedback_into_history, read_assignee_feedback  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge confirmed assignee feedback into history without silently leaking later labels."
    )
    parser.add_argument("--history", type=Path, required=True, help="Existing historical assignee JSONL.")
    parser.add_argument("--feedback", type=Path, required=True, help="Confirmed assignee feedback JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Merged history JSONL destination.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional merge manifest JSON destination.")
    parser.add_argument(
        "--as-of",
        default="",
        help="Only use feedback confirmed at or before this ISO-8601 timestamp.",
    )
    parser.add_argument(
        "--exclude-tickets",
        type=Path,
        default=None,
        help="Optional JSONL split whose ticket IDs must be excluded to prevent evaluation leakage.",
    )
    args = parser.parse_args()

    history_rows = read_jsonl(args.history)
    feedback_rows = read_assignee_feedback(args.feedback)
    excluded_ids = ticket_ids(read_jsonl(args.exclude_tickets)) if args.exclude_tickets else set()
    usable_feedback = [row for row in feedback_rows if feedback_available_as_of(row, args.as_of)]
    usable_feedback = [row for row in usable_feedback if str(row.get("ticket_id") or "") not in excluded_ids]

    merged_rows, summary = merge_feedback_into_history(history_rows, usable_feedback)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "history_path": str(args.history),
        "feedback_path": str(args.feedback),
        "output_path": str(args.output),
        "as_of": args.as_of,
        "excluded_ticket_count": len(excluded_ids),
        "feedback_events_read": len(feedback_rows),
        "feedback_events_after_as_of_filter": sum(
            1 for row in feedback_rows if feedback_available_as_of(row, args.as_of)
        ),
        "feedback_events_used": len(usable_feedback),
        **summary,
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def ticket_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("ticket_id") or row.get("id") or "").strip()
        for row in rows
        if str(row.get("ticket_id") or row.get("id") or "").strip()
    }


def feedback_available_as_of(row: dict[str, Any], as_of: str) -> bool:
    if not as_of:
        return True
    created_at = parse_timestamp(row.get("created_at"))
    cutoff = parse_timestamp(as_of)
    return bool(created_at is not None and cutoff is not None and created_at <= cutoff)


def parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    main()
