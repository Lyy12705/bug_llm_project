"""Export selected priority prediction errors for manual diagnosis.

預設會匯出真實 P2 但被判成 P1/P3 的案例，方便觀察 P2 邊界錯誤。
也可以透過參數改成分析其他 true/predicted priority 組合。
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features_bm25_p2_error_keywords_natural_test")
TRAIN_FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features_bm25_p2_error_keywords_train")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "recall_balanced_priority_model.joblib")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "reports", "p2_error_analysis_recall_balanced.csv")

LABELS = np.array([1, 2, 3, 4, 5])
PRIORITY_NAMES = {
    1: "P1 Blocker",
    2: "P2 Critical",
    3: "P3 Major",
    4: "P4 Minor",
    5: "P5 Trivial",
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export priority prediction errors for manual DRONE-style diagnosis.",
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Saved DRONE/GRAY model bundle.")
    parser.add_argument("--feature-dir", default=FEATURE_DIR, help="Feature directory to analyze.")
    parser.add_argument("--train-feature-dir", default=TRAIN_FEATURE_DIR, help="Training feature dir used to remove overlaps.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="CSV output path for selected errors.")
    parser.add_argument(
        "--true-priorities",
        default="2",
        help="Comma-separated true labels to inspect. Default: 2 for P2.",
    )
    parser.add_argument(
        "--predicted-priorities",
        default="1,3",
        help="Comma-separated predicted labels to inspect. Default: 1,3 for P2 misclassified as P1/P3.",
    )
    parser.add_argument("--keep-overlap", action="store_true", help="Do not remove rows also present in training metadata.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional Markdown summary path with grouped boundary-error diagnostics.",
    )
    return parser


def parse_label_list(value: str) -> set[int]:
    labels = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        labels.add(int(item.removeprefix("P").removeprefix("p")))
    return labels


def apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(thresholds, scores, side="right") + 1


def predict_drone_gray(model_or_bundle: dict, X) -> np.ndarray:
    scores = model_or_bundle["model"].predict(X)
    thresholds = np.array(model_or_bundle["thresholds"], dtype=float)
    return apply_thresholds(scores, thresholds)


def predict_hierarchical_drone_gray(model_or_bundle: dict, X) -> np.ndarray:
    group_scores = model_or_bundle["group_model"].predict(X)
    group_pred = apply_thresholds(group_scores, np.array(model_or_bundle["group_thresholds"], dtype=float))
    y_pred = np.full(X.shape[0], 3, dtype=int)

    high_mask = group_pred == 1
    low_mask = group_pred == 3

    if high_mask.any():
        high_scores = model_or_bundle["high_model"].predict(X[high_mask])
        y_pred[high_mask] = apply_thresholds(high_scores, np.array(model_or_bundle["high_thresholds"], dtype=float))
    if low_mask.any():
        low_scores = model_or_bundle["low_model"].predict(X[low_mask])
        y_pred[low_mask] = apply_thresholds(low_scores, np.array(model_or_bundle["low_thresholds"], dtype=float)) + 3

    return y_pred


def predict_model_or_bundle(model_or_bundle, X) -> np.ndarray:
    # 兼容專案歷史上使用過的不同模型 bundle；目前主模型走 recall_balanced_cost_sensitive。
    if not isinstance(model_or_bundle, dict):
        raise TypeError("Expected a DRONE/GRAY model bundle.")

    model_type = model_or_bundle.get("model_type")
    if model_type == "drone_gray_regression":
        return predict_drone_gray(model_or_bundle, X)
    if model_type == "hierarchical_drone_gray":
        return predict_hierarchical_drone_gray(model_or_bundle, X)
    if model_type == "direct_classifier":
        return model_or_bundle["model"].predict(X).astype(int)
    if model_type == "boundary_refined_direct_classifier":
        y_pred = model_or_bundle["base_model"].predict(X).astype(int)
        apply_mode = model_or_bundle.get("boundary_apply", "base_1_2_3")
        if apply_mode == "base_1_2":
            refine_mask = np.isin(y_pred, [1, 2])
        elif apply_mode == "base_2_3":
            refine_mask = np.isin(y_pred, [2, 3])
        else:
            refine_mask = np.isin(y_pred, [1, 2, 3])
        if refine_mask.any():
            y_pred[refine_mask] = model_or_bundle["boundary_model"].predict(X[refine_mask]).astype(int)
        return y_pred
    if model_type == "recall_balanced_cost_sensitive":
        y_pred = model_or_bundle["base_model"].predict(X).astype(int)
        p1p2_model = model_or_bundle.get("p1p2_model")
        if p1p2_model is not None:
            p1p2_mask = np.isin(y_pred, [1, 2])
            if p1p2_mask.any():
                y_pred[p1p2_mask] = p1p2_model.predict(X[p1p2_mask]).astype(int)
        return y_pred

    raise TypeError(
        "Expected a model bundle produced by scripts/train_drone_gray.py "
        "or scripts/train_hierarchical_drone_gray.py."
    )


def compact_text(value: object, limit: int = 500) -> str:
    text = "" if pd.isna(value) else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def keyword_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("has_") or col.startswith("summary_has_")]


def grouped_counts(df: pd.DataFrame, column: str, limit: int = 8) -> str:
    if column not in df.columns or df.empty:
        return "- no data"
    counts = df[column].fillna("unknown").astype(str).value_counts().head(limit)
    return "\n".join(f"- `{idx}`: {count}" for idx, count in counts.items())


def write_summary(
    path: str,
    result: pd.DataFrame,
    meta: pd.DataFrame,
    y: np.ndarray,
    y_pred: np.ndarray,
    true_labels: set[int],
    predicted_labels: set[int],
) -> None:
    true_mask = np.isin(y, list(true_labels))
    selected_mask = true_mask & np.isin(y_pred, list(predicted_labels))
    true_df = meta.loc[true_mask].copy()
    selected_df = meta.loc[selected_mask].copy()
    selected_df["predicted_priority_name"] = pd.Series(y_pred[selected_mask]).map(PRIORITY_NAMES).to_numpy()

    lines = [
        "# Boundary Error Analysis Summary",
        "",
        f"- True labels inspected: `{sorted(true_labels)}`",
        f"- Predicted labels selected: `{sorted(predicted_labels)}`",
        f"- True-label rows: `{int(true_mask.sum())}`",
        f"- Selected boundary errors: `{int(selected_mask.sum())}`",
        "",
        "## Predicted Label Counts",
        "",
        grouped_counts(selected_df, "predicted_priority_name"),
        "",
        "## Top Products In Selected Errors",
        "",
        grouped_counts(selected_df, "product"),
        "",
        "## Top Components In Selected Errors",
        "",
        grouped_counts(selected_df, "component"),
        "",
        "## Top Severities In Selected Errors",
        "",
        grouped_counts(selected_df, "severity"),
        "",
    ]

    key_cols = keyword_columns(meta)
    if key_cols:
        rows = []
        for col in key_cols:
            selected_rate = float(selected_df[col].mean()) if col in selected_df.columns and len(selected_df) else 0.0
            true_rate = float(true_df[col].mean()) if col in true_df.columns and len(true_df) else 0.0
            rows.append({
                "keyword_feature": col,
                "selected_error_rate": selected_rate,
                "true_label_rate": true_rate,
                "difference": selected_rate - true_rate,
            })
        keyword_df = pd.DataFrame(rows).sort_values("difference", ascending=False).head(12)
        lines.extend([
            "## Keyword Signals With Higher Error Rates",
            "",
            keyword_df.to_markdown(index=False, floatfmt=".4f"),
            "",
        ])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()

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

    model = joblib.load(args.model)
    y_pred = predict_model_or_bundle(model, X)

    true_labels = parse_label_list(args.true_priorities)
    predicted_labels = parse_label_list(args.predicted_priorities)
    error_mask = np.isin(y, list(true_labels)) & np.isin(y_pred, list(predicted_labels))

    result = meta.loc[error_mask].copy()
    result["true_num"] = y[error_mask]
    result["predicted_num"] = y_pred[error_mask]
    result["true_priority_name"] = result["true_num"].map(PRIORITY_NAMES)
    result["predicted_priority_name"] = result["predicted_num"].map(PRIORITY_NAMES)
    result["description_snippet"] = result["description"].apply(compact_text) if "description" in result.columns else ""

    preferred_cols = [
        "id",
        "true_priority_name",
        "predicted_priority_name",
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
    output_cols = [col for col in preferred_cols if col in result.columns]
    result = result[output_cols].sort_values(["true_priority_name", "predicted_priority_name", "id"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    if args.summary_output:
        write_summary(args.summary_output, result, meta, y, y_pred, true_labels, predicted_labels)

    print("=== Priority Error Analysis ===")
    print(f"Rows analyzed: {len(y)}")
    print(f"Removed overlapping training ids: {removed_overlap}")
    print("Confusion matrix labels P1..P5:")
    print(confusion_matrix(y, y_pred, labels=LABELS))
    print(
        f"Selected errors true={sorted(true_labels)} predicted={sorted(predicted_labels)}: "
        f"{len(result)}"
    )
    print(f"saved error analysis -> {args.output}")
    if args.summary_output:
        print(f"saved summary -> {args.summary_output}")


if __name__ == "__main__":
    main()
