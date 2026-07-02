"""Summarize where each true priority class is misclassified.

此程式用於報告中的 per-class error analysis：讀取目前最佳模型與
natural_holdout features，輸出每個 priority 的 recall、錯判方向與逐筆錯誤明細。
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from analyze_priority_errors import LABELS, PRIORITY_NAMES, compact_text, predict_model_or_bundle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze where each true priority class is misclassified.",
    )
    parser.add_argument(
        "--model",
        default=project_path("models/recall_balanced_priority_model.joblib"),
        help="Saved model bundle to evaluate.",
    )
    parser.add_argument(
        "--feature-dir",
        default=project_path("data/processed/features_bm25_p2_error_keywords_natural_test"),
        help="Feature directory to analyze.",
    )
    parser.add_argument(
        "--train-feature-dir",
        default=project_path("data/processed/features_bm25_p2_error_keywords_train"),
        help="Training feature dir used to remove accidental overlap.",
    )
    parser.add_argument(
        "--summary-csv",
        default=project_path("reports/per_class_error_summary_recall_balanced.csv"),
        help="Per-class summary CSV output.",
    )
    parser.add_argument(
        "--destinations-csv",
        default=project_path("reports/per_class_error_destinations_recall_balanced.csv"),
        help="Per-class wrong-prediction destination CSV output.",
    )
    parser.add_argument(
        "--details-csv",
        default=project_path("reports/per_class_error_details_recall_balanced.csv"),
        help="Row-level misclassification details CSV output.",
    )
    parser.add_argument(
        "--summary-md",
        default=project_path("reports/per_class_error_analysis_recall_balanced.md"),
        help="Markdown report output.",
    )
    parser.add_argument("--keep-overlap", action="store_true", help="Do not remove rows also present in training metadata.")
    return parser


def label_name(label: int) -> str:
    return PRIORITY_NAMES[label]


def markdown_table(df: pd.DataFrame, limit: int | None = None) -> str:
    view = df.copy()
    if limit is not None:
        view = view.head(limit)
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    if view.empty:
        return "_No rows._"
    try:
        return view.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(view.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(str(row[column]) for column in view.columns) + " |" for _, row in view.iterrows()]
        return "\n".join([header, divider, *rows])


def load_eval_data(args: argparse.Namespace):
    # 讀取同一個 feature_dir 中的 X/y/meta，meta 用來輸出可讀的錯誤明細。
    X = load_npz(os.path.join(args.feature_dir, "X_features.npz"))
    y = np.load(os.path.join(args.feature_dir, "y.npy"))
    meta = pd.read_csv(os.path.join(args.feature_dir, "feature_meta.csv"))

    removed_overlap = 0
    train_meta_path = os.path.join(args.train_feature_dir, "feature_meta.csv")
    if not args.keep_overlap and os.path.exists(train_meta_path) and "id" in meta.columns:
        train_ids = set(pd.read_csv(train_meta_path, usecols=["id"])["id"].astype(str))
        keep_mask = ~meta["id"].astype(str).isin(train_ids)
        removed_overlap = int((~keep_mask).sum())
        X = X[keep_mask.to_numpy()]
        y = y[keep_mask.to_numpy()]
        meta = meta.loc[keep_mask].reset_index(drop=True)
    return X, y, meta, removed_overlap


def build_summary(matrix: np.ndarray) -> pd.DataFrame:
    # confusion matrix 的每一列是真實 priority，每一欄是模型預測 priority。
    rows = []
    for i, true_label in enumerate(LABELS):
        counts = matrix[i]
        support = int(counts.sum())
        correct = int(counts[i])
        wrong = support - correct
        wrong_counts = {int(label): int(counts[j]) for j, label in enumerate(LABELS) if j != i}
        if wrong:
            most_common_wrong_label, most_common_wrong_count = max(wrong_counts.items(), key=lambda item: item[1])
            most_common_wrong = label_name(most_common_wrong_label)
        else:
            most_common_wrong_count = 0
            most_common_wrong = ""
        row = {
            "true_priority": label_name(true_label),
            "support": support,
            "correct": correct,
            "wrong": wrong,
            "recall": correct / support if support else 0.0,
            "error_rate": wrong / support if support else 0.0,
            "most_common_wrong_prediction": most_common_wrong,
            "most_common_wrong_count": most_common_wrong_count,
        }
        for j, predicted_label in enumerate(LABELS):
            row[f"predicted_{label_name(predicted_label).split()[0]}"] = int(counts[j])
        rows.append(row)
    return pd.DataFrame(rows)


def build_destinations(matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, true_label in enumerate(LABELS):
        support = int(matrix[i].sum())
        wrong_total = int(support - matrix[i, i])
        for j, predicted_label in enumerate(LABELS):
            if i == j:
                continue
            count = int(matrix[i, j])
            if count == 0:
                continue
            rows.append(
                {
                    "true_priority": label_name(true_label),
                    "predicted_priority": label_name(predicted_label),
                    "count": count,
                    "percent_of_true_class": count / support if support else 0.0,
                    "percent_of_wrong_errors": count / wrong_total if wrong_total else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["true_priority", "count"], ascending=[True, False])


def build_error_details(meta: pd.DataFrame, y: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    error_mask = y != y_pred
    details = meta.loc[error_mask].copy()
    details["true_num"] = y[error_mask]
    details["predicted_num"] = y_pred[error_mask]
    details["true_priority"] = details["true_num"].map(label_name)
    details["predicted_priority"] = details["predicted_num"].map(label_name)
    details["error_direction"] = details["true_priority"] + " -> " + details["predicted_priority"]
    details["absolute_priority_distance"] = (details["true_num"] - details["predicted_num"]).abs()
    if "description" in details.columns:
        details["description_snippet"] = details["description"].apply(compact_text)

    preferred = [
        "id",
        "true_priority",
        "predicted_priority",
        "error_direction",
        "absolute_priority_distance",
        "product",
        "component",
        "severity",
        "summary",
        "description_snippet",
        "related_top1_priority",
        "related_top3_avg_priority",
        "related_top5_avg_priority",
        "related_similarity_top1",
        "creator",
        "creation_time",
        "status",
    ]
    cols = [column for column in preferred if column in details.columns]
    return details[cols].sort_values(["true_priority", "predicted_priority", "id"])


def write_report(
    path: str,
    summary: pd.DataFrame,
    destinations: pd.DataFrame,
    matrix: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    removed_overlap: int,
) -> None:
    confusion_df = pd.DataFrame(
        matrix,
        index=[label_name(label) for label in LABELS],
        columns=[label_name(label) for label in LABELS],
    ).reset_index(names="true_priority")
    overall = pd.DataFrame(
        [
            {
                "rows": len(y),
                "removed_training_overlap": removed_overlap,
                "accuracy": accuracy_score(y, y_pred),
                "macro_f1": f1_score(y, y_pred, labels=LABELS, average="macro", zero_division=0),
                "off_by_one_accuracy": np.mean(np.abs(y - y_pred) <= 1),
                "mae": np.mean(np.abs(y - y_pred)),
            }
        ]
    )
    answer_cols = [
        "true_priority",
        "support",
        "correct",
        "wrong",
        "recall",
        "most_common_wrong_prediction",
        "most_common_wrong_count",
        "predicted_P1",
        "predicted_P2",
        "predicted_P3",
        "predicted_P4",
        "predicted_P5",
    ]
    lines = [
        "# Per-Class Error Analysis",
        "",
        "This report answers where each true priority class is predicted when the model is wrong.",
        "",
        "## Overall Metrics",
        "",
        markdown_table(overall),
        "",
        "## P1-P5 被錯判到哪裡",
        "",
        markdown_table(summary[answer_cols]),
        "",
        "## Wrong-Prediction Destinations",
        "",
        markdown_table(destinations),
        "",
        "## Confusion Matrix",
        "",
        "Rows are true priorities and columns are predicted priorities.",
        "",
        markdown_table(confusion_df),
        "",
        "## Interpretation",
        "",
        "- `recall` 表示該 priority 的真實資料中，有多少比例被模型正確抓到。",
        "- `most_common_wrong_prediction` 表示該 priority 最常被錯判成哪一類。",
        "- 如果錯誤集中在相鄰 priority，例如 P2 -> P1 / P3，代表主要問題是邊界模糊。",
        "- 如果錯誤跨很多級，例如 P1 -> P5，才代表模型有較嚴重的排序判斷問題。",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    X, y, meta, removed_overlap = load_eval_data(args)
    model = joblib.load(args.model)
    y_pred = predict_model_or_bundle(model, X)
    matrix = confusion_matrix(y, y_pred, labels=LABELS)

    summary = build_summary(matrix)
    destinations = build_destinations(matrix)
    details = build_error_details(meta, y, y_pred)

    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    summary.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    destinations.to_csv(args.destinations_csv, index=False, encoding="utf-8-sig")
    details.to_csv(args.details_csv, index=False, encoding="utf-8-sig")
    write_report(args.summary_md, summary, destinations, matrix, y, y_pred, removed_overlap)

    print("=== Per-Class Error Analysis ===")
    print(summary[[
        "true_priority",
        "support",
        "correct",
        "wrong",
        "recall",
        "most_common_wrong_prediction",
        "most_common_wrong_count",
    ]].to_string(index=False))
    print(f"\nsaved summary -> {args.summary_csv}")
    print(f"saved destinations -> {args.destinations_csv}")
    print(f"saved error details -> {args.details_csv}")
    print(f"saved report -> {args.summary_md}")


if __name__ == "__main__":
    main()
