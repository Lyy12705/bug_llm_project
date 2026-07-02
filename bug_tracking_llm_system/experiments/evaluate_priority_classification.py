from __future__ import annotations

import argparse

from common import accuracy, add_common_args, nested, read_records, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate priority classification outputs.")
    add_common_args(parser)
    args = parser.parse_args()
    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    metrics = {
        "rows": min(len(gold_rows), len(pred_rows)),
        "accuracy": accuracy(
            [row.get("priority") for row in gold_rows],
            [nested(row, "priority", "predicted_priority", default=row.get("predicted_priority", "")) for row in pred_rows],
        ),
    }
    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
