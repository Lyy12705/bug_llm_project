from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSIGNEE_FEEDBACK_SCHEMA_VERSION = 1
MANUAL_TRIAGE = "manual_triage"


def build_assignee_feedback_record(
    *,
    ticket_json: dict[str, Any],
    prediction: dict[str, Any],
    final_assignee: str,
    reviewer: str = "",
    source: str = "manual_triage",
    notes: str = "",
    created_at: str | None = None,
) -> dict[str, Any]:
    final_owner = _normalize_assignee(final_assignee)
    predicted_owner = _normalize_assignee(prediction.get("assignee"))
    return {
        "schema_version": ASSIGNEE_FEEDBACK_SCHEMA_VERSION,
        "event_type": "assignee_feedback",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "ticket_id": str(ticket_json.get("ticket_id") or ticket_json.get("id") or ""),
        "ticket_created_at": str(
            ticket_json.get("created_at") or ticket_json.get("creation_time") or ticket_json.get("reported_at") or ""
        ),
        "product": str(ticket_json.get("product") or ""),
        "component": str(ticket_json.get("component") or "unknown"),
        "title": str(ticket_json.get("title") or ticket_json.get("summary") or ""),
        "description": str(ticket_json.get("description") or ticket_json.get("body") or ""),
        "predicted_assignee": predicted_owner,
        "suggested_assignee": _normalize_assignee(prediction.get("suggested_assignee")),
        "ranked_candidates": [_normalize_assignee(owner) for owner in prediction.get("ranked_candidates", [])],
        "confidence": prediction.get("confidence"),
        "routing_status": str(prediction.get("routing_status") or ""),
        "fallback_reason": str(prediction.get("fallback_reason") or ""),
        "final_assignee": final_owner,
        "accepted_auto_assignment": bool(final_owner and final_owner == predicted_owner and final_owner != MANUAL_TRIAGE),
        "reviewer": reviewer,
        "source": source,
        "notes": notes,
    }


def append_assignee_feedback(path: Path, record: dict[str, Any]) -> None:
    _validate_feedback_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_assignee_feedback(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("event_type") == "assignee_feedback":
                rows.append(row)
    return rows


def feedback_rows_to_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the latest confirmed outcome per ticket into profile-safe history rows."""
    latest_by_ticket: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        ticket_id = str(row.get("ticket_id") or "").strip()
        if ticket_id:
            previous = latest_by_ticket.get(ticket_id)
            if previous is None or _feedback_order(row, index) >= _feedback_order(previous[1], previous[0]):
                latest_by_ticket[ticket_id] = (index, row)

    history_rows: list[dict[str, Any]] = []
    for _, row in latest_by_ticket.values():
        final_assignee = _normalize_assignee(row.get("final_assignee"))
        if not final_assignee or final_assignee == MANUAL_TRIAGE:
            continue
        history_rows.append(
            {
                "ticket_id": row.get("ticket_id", ""),
                "product": row.get("product", ""),
                "component": row.get("component", "unknown"),
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "assignee": final_assignee,
                "created_at": row.get("ticket_created_at", ""),
                "feedback_created_at": row.get("created_at", ""),
                "source": "assignee_feedback",
            }
        )
    return history_rows


def merge_feedback_into_history(
    history_rows: list[dict[str, Any]], feedback_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replace same-ticket history labels with confirmed feedback outcomes."""
    rows_by_ticket: dict[str, dict[str, Any]] = {}
    unkeyed_rows: list[dict[str, Any]] = []
    for row in history_rows:
        ticket_id = str(row.get("ticket_id") or row.get("id") or "").strip()
        if ticket_id:
            rows_by_ticket[ticket_id] = row
        else:
            unkeyed_rows.append(row)

    confirmed_rows = feedback_rows_to_history_rows(feedback_rows)
    replacements = 0
    additions = 0
    for row in confirmed_rows:
        ticket_id = str(row.get("ticket_id") or "").strip()
        if ticket_id and ticket_id in rows_by_ticket:
            replacements += 1
        else:
            additions += 1
        if ticket_id:
            rows_by_ticket[ticket_id] = row
        else:
            unkeyed_rows.append(row)

    return [*unkeyed_rows, *rows_by_ticket.values()], {
        "base_history_rows": len(history_rows),
        "confirmed_feedback_rows": len(confirmed_rows),
        "feedback_replacements": replacements,
        "feedback_additions": additions,
        "merged_history_rows": len(unkeyed_rows) + len(rows_by_ticket),
    }


def _validate_feedback_record(record: dict[str, Any]) -> None:
    if record.get("event_type") != "assignee_feedback":
        raise ValueError("assignee feedback record must use event_type='assignee_feedback'")
    if record.get("schema_version") != ASSIGNEE_FEEDBACK_SCHEMA_VERSION:
        raise ValueError("unsupported assignee feedback schema_version")
    if not _normalize_assignee(record.get("final_assignee")):
        raise ValueError("final_assignee is required")
    if not str(record.get("ticket_id") or "").strip():
        raise ValueError("ticket_id is required")


def _normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


def _feedback_order(row: dict[str, Any], index: int) -> tuple[float, int]:
    text = str(row.get("created_at") or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return (float("-inf"), index)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed.astimezone(timezone.utc).timestamp(), index)
