#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate duplicate-ticket dataset completeness.")
    parser.add_argument("--tickets", required=True)
    parser.add_argument("--missing-masters-output", default="reports/missing_duplicate_masters.csv")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)
    missing_masters = find_missing_duplicate_masters(tickets)
    group_sizes = duplicate_group_sizes(tickets)
    empty_titles = sum(1 for ticket in tickets if not ticket.title.strip())
    empty_contents = sum(1 for ticket in tickets if not ticket.content.strip())

    write_missing_masters(missing_masters, args.missing_masters_output)
    print(f"tickets={len(tickets)}")
    print(f"duplicate_of_rows={sum(1 for ticket in tickets if ticket.duplicate_of)}")
    print(f"duplicate_groups={len(group_sizes)}")
    print(f"duplicate_groups_with_at_least_2={sum(1 for size in group_sizes.values() if size >= 2)}")
    print(f"missing_duplicate_masters={len(missing_masters)}")
    print(f"empty_titles={empty_titles}")
    print(f"empty_contents={empty_contents}")
    print(f"missing_masters_output={args.missing_masters_output}")
    return 0


def find_missing_duplicate_masters(tickets: list[TicketRecord]) -> list[TicketRecord]:
    known_ids = {ticket.ticket_id for ticket in tickets}
    return [ticket for ticket in tickets if ticket.duplicate_of and ticket.duplicate_of not in known_ids]


def duplicate_group_sizes(tickets: list[TicketRecord]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for ticket in tickets:
        if ticket.duplicate_group:
            sizes[ticket.duplicate_group] = sizes.get(ticket.duplicate_group, 0) + 1
    return sizes


def write_missing_masters(tickets: list[TicketRecord], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticket_id", "duplicate_of", "title"])
        writer.writeheader()
        for ticket in tickets:
            writer.writerow(
                {
                    "ticket_id": ticket.ticket_id,
                    "duplicate_of": ticket.duplicate_of or "",
                    "title": ticket.title,
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
