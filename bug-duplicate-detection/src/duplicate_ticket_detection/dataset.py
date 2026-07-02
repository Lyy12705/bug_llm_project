from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping


ID_COLUMNS = ("ticket_id", "bug_id", "issue_id", "id")
TITLE_COLUMNS = ("title", "summary", "subject")
DESCRIPTION_COLUMNS = ("description", "body", "content", "comment", "comments")
DUPLICATE_OF_COLUMNS = ("duplicate_of", "dupe_of", "duplicated_of", "master_id")
DUPLICATE_GROUP_COLUMNS = ("duplicate_group", "duplicate_cluster", "dup_group", "group_id")

DEFAULT_CONTENT_COLUMNS = (
    "description",
    "body",
    "content",
    "comment",
    "comments",
    "steps_to_reproduce",
    "actual_result",
    "expected_result",
    "environment",
    "log",
    "logs",
)


@dataclass(frozen=True)
class TicketRecord:
    ticket_id: str
    title: str
    content: str
    duplicate_of: str | None = None
    duplicate_group: str | None = None
    fields: Mapping[str, str] | None = None

    def text_for(self, field: str) -> str:
        if field == "title":
            return self.title
        if field in {"content", "description", "body"}:
            return self.content
        if self.fields and field in self.fields:
            return self.fields[field]
        raise ValueError(f"Unknown ticket text field: {field}")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        if value and value not in self.parent:
            self.parent[value] = value

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, child: str, parent: str) -> None:
        if not child or not parent:
            return
        self.parent[self.find(child)] = self.find(parent)


def load_tickets(
    path: str | Path,
    *,
    content_columns: Iterable[str] | None = None,
    derive_duplicate_groups: bool = True,
) -> list[TicketRecord]:
    """Load tickets from CSV or JSONL and normalize common bug tracker fields."""

    path = Path(path)
    rows = _read_rows(path)
    columns = tuple(content_columns or DEFAULT_CONTENT_COLUMNS)
    tickets = [_row_to_ticket(row, content_columns=columns) for row in rows]

    if derive_duplicate_groups:
        tickets = with_duplicate_groups(tickets)

    return tickets


def write_tickets_csv(tickets: Iterable[TicketRecord], path: str | Path) -> None:
    records = list(tickets)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ticket_id",
        "title",
        "description",
        "duplicate_of",
        "duplicate_group",
    ]
    extras: list[str] = []
    for ticket in records:
        for key in (ticket.fields or {}).keys():
            if key not in fieldnames and key not in extras:
                extras.append(key)
    fieldnames.extend(extras)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ticket in records:
            row = {key: "" for key in fieldnames}
            row.update({key: _stringify(value) for key, value in (ticket.fields or {}).items()})
            row["ticket_id"] = ticket.ticket_id
            row["title"] = ticket.title
            row["description"] = ticket.content
            row["duplicate_of"] = ticket.duplicate_of or ""
            row["duplicate_group"] = ticket.duplicate_group or ""
            writer.writerow(row)


def with_duplicate_groups(tickets: Iterable[TicketRecord]) -> list[TicketRecord]:
    """Derive duplicate clusters from duplicate_of and duplicate_group fields."""

    records = list(tickets)
    uf = UnionFind()
    explicit_groups: dict[str, list[str]] = {}

    for ticket in records:
        uf.add(ticket.ticket_id)
        if ticket.duplicate_of:
            uf.union(ticket.ticket_id, ticket.duplicate_of)
        if ticket.duplicate_group:
            explicit_groups.setdefault(ticket.duplicate_group, []).append(ticket.ticket_id)

    for group_ids in explicit_groups.values():
        if not group_ids:
            continue
        first = group_ids[0]
        for ticket_id in group_ids[1:]:
            uf.union(ticket_id, first)

    known_ids = {ticket.ticket_id for ticket in records}
    group_sizes: dict[str, int] = {}
    for ticket in records:
        root = uf.find(ticket.ticket_id)
        group_sizes[root] = group_sizes.get(root, 0) + 1

    normalized: list[TicketRecord] = []
    for ticket in records:
        root = uf.find(ticket.ticket_id)
        group = root if group_sizes[root] > 1 and root in uf.parent else None
        if ticket.duplicate_group and ticket.duplicate_group in explicit_groups:
            group = root if len([tid for tid in explicit_groups[ticket.duplicate_group] if tid in known_ids]) > 1 else group
        normalized.append(replace(ticket, duplicate_group=group))
    return normalized


def tickets_with_duplicates(tickets: Iterable[TicketRecord]) -> list[TicketRecord]:
    groups = _groups_by_duplicate_id(tickets)
    return [ticket for ticket in tickets if ticket.duplicate_group in groups]


def relevant_duplicate_ids(query: TicketRecord, tickets: Iterable[TicketRecord]) -> set[str]:
    if not query.duplicate_group:
        return set()
    return {
        ticket.ticket_id
        for ticket in tickets
        if ticket.ticket_id != query.ticket_id and ticket.duplicate_group == query.duplicate_group
    }


def _groups_by_duplicate_id(tickets: Iterable[TicketRecord]) -> dict[str, list[TicketRecord]]:
    groups: dict[str, list[TicketRecord]] = {}
    for ticket in tickets:
        if ticket.duplicate_group:
            groups.setdefault(ticket.duplicate_group, []).append(ticket)
    return {group: members for group, members in groups.items() if len(members) >= 2}


def _read_rows(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} must be an object")
                rows.append({str(key): _stringify(cell) for key, cell in value.items()})
        return rows

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(key): _stringify(value) for key, value in row.items()} for row in reader]


def _row_to_ticket(row: Mapping[str, str], *, content_columns: Iterable[str]) -> TicketRecord:
    normalized = {str(key).strip(): _stringify(value).strip() for key, value in row.items()}
    ticket_id = _first_present(normalized, ID_COLUMNS)
    if not ticket_id:
        raise ValueError(f"Ticket row is missing an id column. Tried: {', '.join(ID_COLUMNS)}")

    title = _first_present(normalized, TITLE_COLUMNS)
    duplicate_of = _first_present(normalized, DUPLICATE_OF_COLUMNS) or None
    duplicate_group = _first_present(normalized, DUPLICATE_GROUP_COLUMNS) or None
    content_parts = [normalized[column] for column in content_columns if normalized.get(column)]
    if not content_parts:
        content_parts = [_first_present(normalized, DESCRIPTION_COLUMNS)]

    return TicketRecord(
        ticket_id=ticket_id,
        title=title,
        content="\n\n".join(part for part in content_parts if part),
        duplicate_of=duplicate_of,
        duplicate_group=duplicate_group,
        fields=normalized,
    )


def _first_present(row: Mapping[str, str], candidates: Iterable[str]) -> str:
    for column in candidates:
        value = row.get(column)
        if value:
            return value
    return ""


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value)
