#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


FIELDS = (
    "id",
    "summary",
    "status",
    "resolution",
    "dupe_of",
    "creation_time",
    "last_change_time",
    "product",
    "component",
    "priority",
    "severity",
    "creator",
    "assigned_to",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Bugzilla tickets to the project CSV schema")
    parser.add_argument("--base-url", required=True, help="Bugzilla REST base URL, e.g. https://bugzilla.mozilla.org/rest")
    parser.add_argument("--product", action="append", help="Product filter. Can be repeated.")
    parser.add_argument("--component", action="append", help="Component filter. Can be repeated.")
    parser.add_argument("--resolution", action="append", help="Resolution filter. Can be repeated.")
    parser.add_argument("--status", action="append", help="Status filter. Can be repeated.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--with-comments", action="store_true", help="Fetch the first comment as description")
    parser.add_argument(
        "--include-duplicate-masters",
        action="store_true",
        help="Fetch bugs referenced by dupe_of so duplicate pairs are complete.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for fetching bug IDs")
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay between comment requests")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bugs = fetch_bugs(args)
    if args.include_duplicate_masters:
        existing_ids = {str(bug.get("id")) for bug in bugs}
        master_ids = sorted({str(bug.get("dupe_of")) for bug in bugs if bug.get("dupe_of") and str(bug.get("dupe_of")) not in existing_ids})
        if master_ids:
            print(f"duplicate_masters={len(master_ids)}", file=sys.stderr)
            bugs.extend(fetch_bugs_by_ids(args.base_url, master_ids, batch_size=args.batch_size))

    bugs_by_id: dict[str, dict[str, Any]] = {}
    for bug in bugs:
        bugs_by_id[str(bug.get("id"))] = bug

    rows = []
    bugs = list(bugs_by_id.values())
    for index, bug in enumerate(bugs, start=1):
        description = ""
        if args.with_comments:
            try:
                description = fetch_first_comment(args.base_url, bug["id"])
            except Exception as exc:  # noqa: BLE001 - keep long-running exports alive.
                print(f"warning: comments failed bug={bug['id']} error={exc}", file=sys.stderr)
            time.sleep(args.sleep)
            print(f"comments {index}/{len(bugs)} bug={bug['id']}", file=sys.stderr)
        rows.append(to_project_row(bug, description))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else output_fields())
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)}")
    print(f"output={output}")
    return 0


def fetch_bugs(args: argparse.Namespace) -> list[dict[str, Any]]:
    query: list[tuple[str, str]] = [
        ("include_fields", ",".join(FIELDS)),
        ("limit", str(args.limit)),
        ("offset", str(args.offset)),
    ]
    for name in ("product", "component", "resolution", "status"):
        values = getattr(args, name) or []
        query.extend((name, value) for value in values)

    url = f"{args.base_url.rstrip('/')}/bug?{urllib.parse.urlencode(query)}"
    data = read_json(url)
    return data.get("bugs", [])


def fetch_bugs_by_ids(base_url: str, bug_ids: list[str], *, batch_size: int = 100) -> list[dict[str, Any]]:
    bugs: list[dict[str, Any]] = []
    for start in range(0, len(bug_ids), batch_size):
        batch = bug_ids[start : start + batch_size]
        query = [
            ("include_fields", ",".join(FIELDS)),
            ("id", ",".join(batch)),
        ]
        url = f"{base_url.rstrip('/')}/bug?{urllib.parse.urlencode(query)}"
        data = read_json(url)
        bugs.extend(data.get("bugs", []))
    return bugs


def fetch_first_comment(base_url: str, bug_id: int | str) -> str:
    url = f"{base_url.rstrip('/')}/bug/{bug_id}/comment"
    data = read_json(url)
    comments = data.get("bugs", {}).get(str(bug_id), {}).get("comments", [])
    if not comments:
        return ""
    return str(comments[0].get("text", ""))


def read_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60, context=ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def to_project_row(bug: dict[str, Any], description: str) -> dict[str, str]:
    return {
        "ticket_id": str(bug.get("id", "")),
        "title": str(bug.get("summary", "")),
        "description": description,
        "duplicate_of": str(bug.get("dupe_of") or ""),
        "status": str(bug.get("status", "")),
        "resolution": str(bug.get("resolution", "")),
        "product": str(bug.get("product", "")),
        "component": str(bug.get("component", "")),
        "priority": str(bug.get("priority", "")),
        "severity": str(bug.get("severity", "")),
        "reporter": str(bug.get("creator", "")),
        "assignee": str(bug.get("assigned_to", "")),
        "created_at": str(bug.get("creation_time", "")),
        "updated_at": str(bug.get("last_change_time", "")),
        "source": "bugzilla",
    }


def output_fields() -> list[str]:
    return list(to_project_row({}, "").keys())


if __name__ == "__main__":
    raise SystemExit(main())
