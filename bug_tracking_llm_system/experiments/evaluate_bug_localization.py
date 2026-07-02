from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import add_common_args, read_records, write_metrics


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_fault_localization import evaluate_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bug localization outputs.")
    add_common_args(parser)
    args = parser.parse_args()
    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    metrics = evaluate_records(gold_rows, pred_rows)
    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
