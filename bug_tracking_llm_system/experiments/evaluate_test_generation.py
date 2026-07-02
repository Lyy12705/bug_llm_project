from __future__ import annotations

import argparse

from common import read_records, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated test outputs.")
    parser.add_argument("--pred", required=True, help="Generated tests JSON or pipeline result JSON.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = read_records(args.pred)
    generated = []
    fib = []
    for row in rows:
        tests = row.get("tests") if isinstance(row.get("tests"), dict) else row
        generated.append(bool(tests.get("generated_tests")))
        fib.append(bool(tests.get("fib_passed_tests")))
    metrics = {
        "rows": len(rows),
        "generated_test_rate": sum(generated) / len(generated) if generated else 0.0,
        "fib_success_rate": sum(fib) / len(fib) if fib else 0.0,
    }
    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
