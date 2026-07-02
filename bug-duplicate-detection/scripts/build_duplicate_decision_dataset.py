#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, write_tickets_csv


NON_DUPLICATE_RESOLUTIONS = {"", "---", "fixed", "worksforme", "invalid", "wontfix", "incomplete"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a mixed duplicate/non-duplicate dataset for threshold and confusion-matrix evaluation."
    )
    parser.add_argument("--duplicates", required=True, help="CSV with duplicate tickets and their masters")
    parser.add_argument("--non-duplicates", nargs="+", required=True, help="One or more CSV files containing candidate non-duplicate tickets")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Non-duplicate rows per duplicate-group row")
    parser.add_argument("--max-negatives", type=int, help="Maximum number of non-duplicate rows to include")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    duplicate_tickets = load_tickets(args.duplicates)
    source_non_duplicates = []
    for path in args.non_duplicates:
        source_non_duplicates.extend(load_tickets(path))
    mixed, sampled_negatives, available_negatives = build_decision_dataset(
        duplicate_tickets,
        source_non_duplicates,
        negative_ratio=args.negative_ratio,
        max_negatives=args.max_negatives,
        seed=args.seed,
    )
    write_tickets_csv(mixed, args.output)

    positive_rows = [ticket for ticket in duplicate_tickets if ticket.duplicate_group]
    print(f"duplicate_rows={len(duplicate_tickets)}")
    print(f"positive_duplicate_rows={len(positive_rows)}")
    print(f"non_duplicate_source_rows={len(source_non_duplicates)}")
    print(f"non_duplicate_candidates={len(available_negatives)}")
    print(f"sampled_non_duplicate_rows={len(sampled_negatives)}")
    print(f"output_rows={len(mixed)}")
    print(f"output={args.output}")
    if not sampled_negatives:
        print("warning=no_non_duplicate_rows_added")
    return 0


def build_decision_dataset(
    duplicate_tickets: list[TicketRecord],
    source_non_duplicates: list[TicketRecord],
    *,
    negative_ratio: float = 1.0,
    max_negatives: int | None = None,
    seed: int = 13,
) -> tuple[list[TicketRecord], list[TicketRecord], list[TicketRecord]]:
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    if max_negatives is not None and max_negatives < 0:
        raise ValueError("max_negatives must be non-negative")

    rng = random.Random(seed)
    duplicate_ids = {ticket.ticket_id for ticket in duplicate_tickets}
    positive_rows = [ticket for ticket in duplicate_tickets if ticket.duplicate_group]
    seen_negative_ids: set[str] = set()
    available_negatives = [
        ticket
        for ticket in source_non_duplicates
        if is_non_duplicate_candidate(ticket, duplicate_ids)
        and ticket.ticket_id not in seen_negative_ids
        and not seen_negative_ids.add(ticket.ticket_id)
    ]

    target_negatives = round(len(positive_rows) * negative_ratio)
    if max_negatives is not None:
        target_negatives = min(target_negatives, max_negatives)
    target_negatives = min(target_negatives, len(available_negatives))

    sampled_negatives = rng.sample(available_negatives, target_negatives) if target_negatives else []
    output = list(duplicate_tickets) + sampled_negatives
    rng.shuffle(output)
    return output, sampled_negatives, available_negatives


def is_non_duplicate_candidate(ticket: TicketRecord, duplicate_ids: set[str]) -> bool:
    if ticket.ticket_id in duplicate_ids:
        return False
    if ticket.duplicate_of or ticket.duplicate_group:
        return False
    resolution = ((ticket.fields or {}).get("resolution") or "").strip().lower()
    if resolution == "duplicate":
        return False
    return resolution in NON_DUPLICATE_RESOLUTIONS


if __name__ == "__main__":
    raise SystemExit(main())
