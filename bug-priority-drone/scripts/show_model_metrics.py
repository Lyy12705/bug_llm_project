"""Print the main natural_holdout metrics for the current best model.

這支程式只讀取已存在的 evaluation CSV，不會重新訓練或修改模型。
"""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "reports" / "recall_balanced_best_eval.csv"

METRICS = [
    ("Accuracy", "accuracy"),
    ("Macro F1", "macro_f1"),
    ("Off-by-one accuracy", "off_by_one_accuracy"),
    ("MAE", "mae"),
    ("P1 recall", "p1_recall"),
    ("P2 recall", "p2_recall"),
    ("P3 recall", "p3_recall"),
    ("P4 recall", "p4_recall"),
    ("P5 recall", "p5_recall"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show only the main priority-prediction metrics from an eval CSV."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Evaluation CSV path. Defaults to the current best model result.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimal places to print.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No rows found in {input_path}")

    if missing := [column for _, column in METRICS if column not in df.columns]:
        raise ValueError(f"Missing required metric columns in {input_path}: {missing}")

    row = df.iloc[0]
    result = pd.DataFrame(
        {
            "Metric": [label for label, _ in METRICS],
            "Value": [f"{float(row[column]):.{args.decimals}f}" for _, column in METRICS],
        }
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
