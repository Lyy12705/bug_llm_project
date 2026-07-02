"""Create train/validation/natural_holdout splits for the priority model.

流程上先保留 natural_holdout 作為最終測試集，再用剩餘資料建立較均衡的
train_balanced 與 validation_balanced，避免模型只偏向資料量較多的類別。
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "eclipse_bug_reports_clean.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "experiment_splits")
RANDOM_STATE = 42
PRIORITIES = ["P1", "P2", "P3", "P4", "P5"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create DRONE/GRAY priority prediction splits.")
    parser.add_argument("--input", default=INPUT_PATH, help="Cleaned bug report CSV with fetch_group column.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for split CSV outputs.")
    parser.add_argument("--natural-test-size", type=float, default=0.5, help="Fraction of base/natural rows reserved for natural_test.")
    parser.add_argument(
        "--balanced-train-per-class",
        type=int,
        default=700,
        help="Target rows per priority for train_balanced.",
    )
    parser.add_argument(
        "--balanced-validation-per-class",
        type=int,
        default=250,
        help="Target rows per priority for validation_balanced.",
    )
    parser.add_argument(
        "--balanced-test-per-class",
        type=int,
        default=250,
        help="Deprecated compatibility argument. Natural holdout is the retained test set.",
    )
    parser.add_argument(
        "--balanced-per-class",
        type=int,
        default=None,
        help="Deprecated compatibility alias. Natural holdout is the retained test set.",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed.")
    return parser


def safe_stratified_split(df: pd.DataFrame, test_size: float, random_state: int):
    # 若各 priority 數量足夠，保留分層抽樣以維持 natural_holdout 分布。
    counts = df["priority"].value_counts()
    stratify = df["priority"] if len(counts) > 1 and counts.min() >= 2 else None
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)


def allocate_class_counts(available: int, requested: list[int]) -> list[int]:
    requested_total = sum(requested)
    if available <= 0 or requested_total <= 0:
        return [0 for _ in requested]
    if available >= requested_total:
        return requested.copy()

    raw = [available * count / requested_total for count in requested]
    allocated = [int(value) for value in raw]

    for idx, requested_count in enumerate(requested):
        if requested_count > 0 and allocated[idx] == 0 and sum(allocated) < available:
            allocated[idx] = 1

    remaining = available - sum(allocated)
    fractions = sorted(
        ((raw[idx] - int(raw[idx]), idx) for idx in range(len(requested))),
        reverse=True,
    )
    for _, idx in fractions:
        if remaining <= 0:
            break
        allocated[idx] += 1
        remaining -= 1

    return allocated


def split_balanced_by_priority(
    df: pd.DataFrame,
    train_per_class: int,
    validation_per_class: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 每個 priority 獨立抽取固定上限，降低多數類別主導訓練的風險。
    train_parts = []
    validation_parts = []

    for priority in PRIORITIES:
        class_df = df[df["priority"] == priority].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        if class_df.empty:
            continue

        train_n, validation_n = allocate_class_counts(
            len(class_df),
            [train_per_class, validation_per_class],
        )

        train_end = train_n
        validation_end = train_end + validation_n

        train_parts.append(class_df.iloc[:train_end])
        validation_parts.append(class_df.iloc[train_end:validation_end])

    train_df = pd.concat(train_parts).sort_values("creation_time").reset_index(drop=True) if train_parts else df.iloc[0:0].copy()
    validation_df = (
        pd.concat(validation_parts).sort_values("creation_time").reset_index(drop=True)
        if validation_parts
        else df.iloc[0:0].copy()
    )
    return train_df, validation_df


def write_split(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved {len(df):>5} rows -> {path}")
    print(df["priority"].value_counts().sort_index().to_string())
    print()


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    if "fetch_group" not in df.columns:
        raise ValueError("Input must contain fetch_group. Re-fetch and clean data with the updated scripts.")

    df["fetch_group"] = df["fetch_group"].fillna("unknown").astype(str)
    base_df = df[df["fetch_group"] == "base"].copy()
    original_base_count = len(base_df)
    quota_df = df[df["fetch_group"].str.startswith("quota_")].copy()
    backfill_df = df[df["fetch_group"].str.startswith("backfill_")].copy()

    if base_df.empty:
        print("No base rows found. Using all cleaned rows as the natural-test candidate pool.")
        base_df = df.copy()

    _, natural_test = safe_stratified_split(
        base_df,
        test_size=args.natural_test_size,
        random_state=args.random_state,
    )

    remaining_after_natural_test = df[~df["id"].isin(natural_test["id"])].copy()
    train_balanced, validation_balanced = split_balanced_by_priority(
        remaining_after_natural_test,
        train_per_class=args.balanced_train_per_class,
        validation_per_class=args.balanced_validation_per_class,
        random_state=args.random_state,
    )

    split_paths = {
        "train_balanced": os.path.join(args.output_dir, "train_balanced_clean.csv"),
        "validation_balanced": os.path.join(args.output_dir, "validation_balanced_clean.csv"),
        "natural_test": os.path.join(args.output_dir, "natural_test_clean.csv"),
    }

    write_split(train_balanced, split_paths["train_balanced"])
    write_split(validation_balanced, split_paths["validation_balanced"])
    write_split(natural_test, split_paths["natural_test"])

    summary_rows = []
    for name, path in split_paths.items():
        split_df = pd.read_csv(path)
        counts = split_df["priority"].value_counts().to_dict()
        row = {"split": name, "rows": len(split_df)}
        for priority in PRIORITIES:
            row[priority] = int(counts.get(priority, 0))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "split_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"saved summary -> {summary_path}")
    print(summary.to_string(index=False))
    print(f"\nBase rows: {original_base_count} | Quota rows: {len(quota_df)} | Backfill rows: {len(backfill_df)}")


if __name__ == "__main__":
    main()
