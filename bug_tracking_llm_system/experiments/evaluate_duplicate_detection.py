from __future__ import annotations

import argparse

from common import accuracy, add_common_args, nested, read_records, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate duplicate detection JSON outputs.")
    add_common_args(parser)
    args = parser.parse_args()

    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    gold_duplicate = [bool(row.get("is_duplicate")) for row in gold_rows]
    pred_duplicate = [bool(nested(row, "duplicate", "is_duplicate", default=row.get("is_duplicate", False))) for row in pred_rows]
    gold_of = [row.get("duplicate_of") or "" for row in gold_rows]
    pred_of = [nested(row, "duplicate", "duplicate_of", default=row.get("duplicate_of", "")) or "" for row in pred_rows]
    metrics = {
        "rows": min(len(gold_rows), len(pred_rows)),
        "duplicate_label_accuracy": accuracy(gold_duplicate, pred_duplicate),
        "duplicate_of_accuracy": accuracy(gold_of, pred_of),
    }
    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
