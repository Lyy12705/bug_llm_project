from __future__ import annotations

import argparse

from common import read_records, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate regression testing outputs.")
    parser.add_argument("--pred", required=True, help="Regression JSON or pipeline result JSON.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = read_records(args.pred)
    passed = []
    for row in rows:
        regression = row.get("regression") if isinstance(row.get("regression"), dict) else row
        passed.append(regression.get("regression_result") == "passed")
    metrics = {"rows": len(rows), "regression_pass_rate": sum(passed) / len(passed) if passed else 0.0}
    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
