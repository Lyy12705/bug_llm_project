#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_duplicate_decision_dataset import build_decision_dataset
from duplicate_ticket_detection.dataset import TicketRecord, with_duplicate_groups, write_tickets_csv
from fetch_bugzilla import fetch_bugs, fetch_bugs_by_ids, fetch_first_comment, to_project_row


DEFAULT_BASE_URL = "https://bugzilla.mozilla.org/rest"
DEFAULT_PRODUCT = ("Firefox",)
DEFAULT_NON_DUPLICATE_RESOLUTIONS = ("FIXED", "WORKSFORME", "INVALID", "WONTFIX")
DEFAULT_NON_DUPLICATE_STATUS = ("RESOLVED",)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare project datasets directly from Bugzilla. This removes the need "
            "to manually prepare CSV/JSONL input files before running experiments."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--product", action="append", help="Bugzilla product. Can be repeated. Defaults to Firefox.")
    parser.add_argument("--component", action="append", help="Optional Bugzilla component filter. Can be repeated.")
    parser.add_argument("--duplicate-limit", type=int, default=500)
    parser.add_argument("--non-duplicate-limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--non-duplicate-resolution", action="append", help="Non-duplicate resolution filter. Can be repeated.")
    parser.add_argument("--non-duplicate-status", action="append", help="Non-duplicate status filter. Can be repeated.")
    parser.add_argument(
        "--with-comments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch the first Bugzilla comment as the ticket description.",
    )
    parser.add_argument(
        "--include-duplicate-masters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch tickets referenced by dupe_of so duplicate groups are complete.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--max-negatives", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--duplicates-output", default="data/mozilla_firefox_duplicates.csv")
    parser.add_argument("--non-duplicates-output", default="data/mozilla_firefox_non_duplicates.csv")
    parser.add_argument("--decision-output", default="data/mozilla_firefox_decision_eval.csv")
    args = parser.parse_args()

    products = tuple(args.product or DEFAULT_PRODUCT)
    duplicate_rows = fetch_project_rows(
        base_url=args.base_url,
        products=products,
        components=tuple(args.component or ()),
        resolutions=("DUPLICATE",),
        statuses=(),
        limit=args.duplicate_limit,
        offset=args.offset,
        with_comments=args.with_comments,
        include_duplicate_masters=args.include_duplicate_masters,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep,
    )
    duplicate_tickets = rows_to_tickets(duplicate_rows)
    write_tickets_csv(duplicate_tickets, args.duplicates_output)

    non_duplicate_rows = fetch_project_rows(
        base_url=args.base_url,
        products=products,
        components=tuple(args.component or ()),
        resolutions=tuple(args.non_duplicate_resolution or DEFAULT_NON_DUPLICATE_RESOLUTIONS),
        statuses=tuple(args.non_duplicate_status or DEFAULT_NON_DUPLICATE_STATUS),
        limit=args.non_duplicate_limit,
        offset=args.offset,
        with_comments=args.with_comments,
        include_duplicate_masters=False,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep,
    )
    non_duplicate_tickets = rows_to_tickets(non_duplicate_rows)
    write_tickets_csv(non_duplicate_tickets, args.non_duplicates_output)

    decision_tickets, sampled_negatives, available_negatives = build_decision_dataset(
        duplicate_tickets,
        non_duplicate_tickets,
        negative_ratio=args.negative_ratio,
        max_negatives=args.max_negatives,
        seed=args.seed,
    )
    write_tickets_csv(decision_tickets, args.decision_output)

    print(f"source=bugzilla")
    print(f"base_url={args.base_url}")
    print(f"products={','.join(products)}")
    print(f"duplicate_rows={len(duplicate_tickets)}")
    print(f"non_duplicate_rows={len(non_duplicate_tickets)}")
    print(f"available_non_duplicate_candidates={len(available_negatives)}")
    print(f"sampled_non_duplicate_rows={len(sampled_negatives)}")
    print(f"decision_rows={len(decision_tickets)}")
    print(f"duplicates_output={args.duplicates_output}")
    print(f"non_duplicates_output={args.non_duplicates_output}")
    print(f"decision_output={args.decision_output}")
    return 0


def fetch_project_rows(
    *,
    base_url: str,
    products: tuple[str, ...],
    components: tuple[str, ...],
    resolutions: tuple[str, ...],
    statuses: tuple[str, ...],
    limit: int,
    offset: int,
    with_comments: bool,
    include_duplicate_masters: bool,
    batch_size: int,
    sleep_seconds: float,
) -> list[dict[str, str]]:
    fetch_args = argparse.Namespace(
        base_url=base_url,
        product=list(products),
        component=list(components),
        resolution=list(resolutions),
        status=list(statuses),
        limit=limit,
        offset=offset,
    )
    bugs = fetch_bugs(fetch_args)
    if include_duplicate_masters:
        bugs = include_missing_duplicate_masters(base_url, bugs, batch_size=batch_size)
    bugs = dedupe_bugs_by_id(bugs)

    rows: list[dict[str, str]] = []
    for index, bug in enumerate(bugs, start=1):
        description = ""
        if with_comments:
            try:
                description = fetch_first_comment(base_url, bug["id"])
            except Exception as exc:  # noqa: BLE001 - keep long-running exports alive.
                print(f"warning: comments failed bug={bug.get('id')} error={exc}", file=sys.stderr)
            time.sleep(sleep_seconds)
            print(f"comments {index}/{len(bugs)} bug={bug.get('id')}", file=sys.stderr)
        rows.append(to_project_row(bug, description))
    return rows


def include_missing_duplicate_masters(base_url: str, bugs: list[dict[str, Any]], *, batch_size: int) -> list[dict[str, Any]]:
    existing_ids = {str(bug.get("id")) for bug in bugs}
    master_ids = sorted(
        {
            str(bug.get("dupe_of"))
            for bug in bugs
            if bug.get("dupe_of") and str(bug.get("dupe_of")) not in existing_ids
        }
    )
    if not master_ids:
        return bugs
    print(f"duplicate_masters={len(master_ids)}", file=sys.stderr)
    return [*bugs, *fetch_bugs_by_ids(base_url, master_ids, batch_size=batch_size)]


def dedupe_bugs_by_id(bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for bug in bugs:
        by_id[str(bug.get("id"))] = bug
    return list(by_id.values())


def rows_to_tickets(rows: list[dict[str, str]]) -> list[TicketRecord]:
    tickets = [
        TicketRecord(
            ticket_id=row["ticket_id"],
            title=row["title"],
            content=row["description"],
            duplicate_of=row.get("duplicate_of") or None,
            fields=row,
        )
        for row in rows
    ]
    return with_duplicate_groups(tickets)


if __name__ == "__main__":
    raise SystemExit(main())
