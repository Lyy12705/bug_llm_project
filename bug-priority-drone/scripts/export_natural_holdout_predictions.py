"""Export row-level predictions for the natural_holdout demo set.

這支程式不重新訓練模型，只讀取目前最佳模型與 natural_holdout feature matrix，
輸出每一筆 ticket 的真實 priority、模型預測 priority、是否判對與摘要欄位。
用途是 demo 或報告時展示「模型實際怎麼判斷每一筆 bug report」。
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, f1_score, recall_score

from analyze_priority_errors import LABELS, PRIORITY_NAMES, compact_text, predict_model_or_bundle


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "recall_balanced_priority_model.joblib")
DEFAULT_FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features_bm25_p2_error_keywords_natural_test")
DEFAULT_TRAIN_FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features_bm25_p2_error_keywords_train")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "reports", "recall_balanced_predictions_natural_holdout.csv")
DEFAULT_SUMMARY = os.path.join(PROJECT_ROOT, "reports", "recall_balanced_predictions_natural_holdout_demo.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export true-vs-predicted priority for every natural_holdout ticket.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Saved model bundle.")
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR, help="Natural holdout feature directory.")
    parser.add_argument(
        "--train-feature-dir",
        default=DEFAULT_TRAIN_FEATURE_DIR,
        help="Training feature dir used to remove accidental id overlap.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Row-level prediction CSV output path.")
    parser.add_argument("--summary-md", default=DEFAULT_SUMMARY, help="Demo summary Markdown output path.")
    parser.add_argument("--keep-overlap", action="store_true", help="Do not remove rows also present in training metadata.")
    return parser


def load_eval_data(feature_dir: str, train_feature_dir: str, keep_overlap: bool):
    X = load_npz(os.path.join(feature_dir, "X_features.npz"))
    y = np.load(os.path.join(feature_dir, "y.npy"))
    meta = pd.read_csv(os.path.join(feature_dir, "feature_meta.csv"))

    removed_overlap = 0
    train_meta_path = os.path.join(train_feature_dir, "feature_meta.csv")
    if not keep_overlap and os.path.exists(train_meta_path) and "id" in meta.columns:
        train_ids = set(pd.read_csv(train_meta_path, usecols=["id"])["id"].astype(str))
        keep_mask = ~meta["id"].astype(str).isin(train_ids)
        removed_overlap = int((~keep_mask).sum())
        X = X[keep_mask.to_numpy()]
        y = y[keep_mask.to_numpy()]
        meta = meta.loc[keep_mask].reset_index(drop=True)

    return X, y, meta, removed_overlap


def label_code(label: int) -> str:
    return PRIORITY_NAMES[int(label)].split()[0]


def build_prediction_frame(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame()
    output["id"] = meta["id"].to_numpy() if "id" in meta.columns else np.arange(len(meta))
    output["true_priority"] = [label_code(label) for label in y_true]
    output["true_priority_name"] = [PRIORITY_NAMES[int(label)] for label in y_true]
    output["predicted_priority"] = [label_code(label) for label in y_pred]
    output["predicted_priority_name"] = [PRIORITY_NAMES[int(label)] for label in y_pred]
    output["is_correct"] = y_true == y_pred
    output["priority_distance"] = np.abs(y_true - y_pred)

    # Demo 時保留人看得懂的欄位，避免 feature matrix 的大量數值欄位干擾閱讀。
    for column in ["summary", "severity", "product", "component", "creator", "creation_time", "status", "resolution"]:
        if column in meta.columns:
            output[column] = meta[column]
    if "description" in meta.columns:
        output["description_snippet"] = meta["description"].apply(lambda value: compact_text(value, limit=220))

    return output.sort_values(["is_correct", "true_priority", "id"], ascending=[True, True, True])


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int) -> str:
    view = df[[column for column in columns if column in df.columns]].head(limit).copy()
    if view.empty:
        return "_No rows._"
    try:
        return view.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(view.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(str(row[column]) for column in view.columns) + " |" for _, row in view.iterrows()]
        return "\n".join([header, divider, *rows])


def write_summary(path: str, predictions: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, removed_overlap: int) -> None:
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    metric_rows = [
        ("Rows", len(predictions)),
        ("Removed training overlap", removed_overlap),
        ("Accuracy", accuracy_score(y_true, y_pred)),
        ("Macro F1", f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        ("Off-by-one accuracy", float(np.mean(np.abs(y_true - y_pred) <= 1))),
        ("MAE", float(np.mean(np.abs(y_true - y_pred)))),
    ]
    metric_rows.extend((f"P{label} recall", float(recall)) for label, recall in zip(LABELS, recalls))
    metrics = pd.DataFrame(metric_rows, columns=["Metric", "Value"])
    metrics["Value"] = metrics["Value"].map(lambda value: f"{value:.4f}" if isinstance(value, float) else str(value))

    correct_examples = predictions[predictions["is_correct"]]
    wrong_examples = predictions[~predictions["is_correct"]]
    demo_cols = ["id", "true_priority", "predicted_priority", "is_correct", "summary", "severity", "product", "component"]
    lines = [
        "# Natural Holdout Row-Level Prediction Demo",
        "",
        "這份檔案用來展示目前最佳模型對每一筆測試 ticket 的判斷結果。",
        "",
        "## Metrics",
        "",
        markdown_table(metrics, ["Metric", "Value"], limit=len(metrics)),
        "",
        "## Example Correct Predictions",
        "",
        markdown_table(correct_examples, demo_cols, limit=8),
        "",
        "## Example Wrong Predictions",
        "",
        markdown_table(wrong_examples, demo_cols, limit=8),
        "",
        "## Output CSV",
        "",
        "- `reports/recall_balanced_predictions_natural_holdout.csv`：全部測試資料逐筆預測結果。",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    X, y_true, meta, removed_overlap = load_eval_data(args.feature_dir, args.train_feature_dir, args.keep_overlap)
    model = joblib.load(args.model)
    y_pred = predict_model_or_bundle(model, X)

    predictions = build_prediction_frame(meta, y_true, y_pred)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    predictions.to_csv(args.output, index=False, encoding="utf-8-sig")
    write_summary(args.summary_md, predictions, y_true, y_pred, removed_overlap)

    print(f"saved row-level predictions -> {args.output}")
    print(f"saved demo summary -> {args.summary_md}")
    print(f"rows: {len(predictions)}")
    print(f"accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"macro_f1: {f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.4f}")


if __name__ == "__main__":
    main()
