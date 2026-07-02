from __future__ import annotations

import argparse

from common import read_records, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate commit message completeness.")
    parser.add_argument("--pred", required=True, help="Commit message JSON or pipeline result JSON.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = read_records(args.pred)
    complete = []
    for row in rows:
        message = row.get("commit_message") if isinstance(row.get("commit_message"), dict) else row
        title = str(message.get("commit_message") or "")
        body = str(message.get("commit_body") or "")
        complete.append((":" in title) and len(body.split()) >= 8)
    metrics = {"rows": len(rows), "completeness_rate": sum(complete) / len(complete) if complete else 0.0}
    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
