from __future__ import annotations

import argparse

from common import accuracy, add_common_args, nested, read_records, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate assignee triage outputs.")
    add_common_args(parser)
    args = parser.parse_args()
    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    metrics = {
        "rows": min(len(gold_rows), len(pred_rows)),
        "top1_accuracy": accuracy(
            [row.get("assignee") for row in gold_rows],
            [nested(row, "assignee", "assignee", default=row.get("assignee", "")) for row in pred_rows],
        ),
    }
    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
