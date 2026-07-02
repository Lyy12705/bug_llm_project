from __future__ import annotations

import argparse

from common import accuracy, add_common_args, nested, read_records, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structured ticket JSON extraction.")
    add_common_args(parser)
    parser.add_argument("--fields", default="bug_type,component,os,version,priority,error_message")
    args = parser.parse_args()

    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    metrics = {"rows": min(len(gold_rows), len(pred_rows)), "fields": {}}
    exact = []
    for field in fields:
        gold_values = [nested(row, "json_ground_truth", field, default=row.get(field, "")) for row in gold_rows]
        pred_values = [nested(row, "predicted_json", field, default=row.get(field, "")) for row in pred_rows]
        metrics["fields"][field] = accuracy(gold_values, pred_values)
    for gold, pred in zip(gold_rows, pred_rows):
        exact.append(all(str(nested(gold, "json_ground_truth", field, default=gold.get(field, ""))) == str(nested(pred, "predicted_json", field, default=pred.get(field, ""))) for field in fields))
    metrics["exact_match"] = sum(exact) / len(exact) if exact else 0.0
    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
