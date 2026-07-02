"""Analyze hand-picked error directions such as P1->P2 and P4->P2.

這支程式不是訓練模型，而是把指定錯誤方向抽出來，統計 severity、product、
component、keyword signals 和 related-report signals，幫助後續改良模型。
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import CountVectorizer

from analyze_priority_errors import LABELS, PRIORITY_NAMES, compact_text, predict_model_or_bundle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze specific misclassification directions such as P1->P2 and P4->P2.",
    )
    parser.add_argument("--model", default=project_path("models/recall_balanced_priority_model.joblib"))
    parser.add_argument("--feature-dir", default=project_path("data/processed/features_bm25_p2_error_keywords_natural_test"))
    parser.add_argument("--train-feature-dir", default=project_path("data/processed/features_bm25_p2_error_keywords_train"))
    parser.add_argument(
        "--directions",
        default="1:2,4:2",
        help="Comma-separated true:predicted directions, e.g. 1:2,4:2.",
    )
    parser.add_argument("--output-csv", default=project_path("reports/targeted_error_directions_recall_balanced.csv"))
    parser.add_argument("--summary-md", default=project_path("reports/targeted_error_directions_recall_balanced.md"))
    parser.add_argument("--keep-overlap", action="store_true")
    return parser


def parse_directions(value: str) -> list[tuple[int, int]]:
    directions = []
    for item in value.split(","):
        if not item.strip():
            continue
        left, right = item.split(":")
        directions.append((int(left.strip().removeprefix("P").removeprefix("p")), int(right.strip().removeprefix("P").removeprefix("p"))))
    return directions


def bool_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0) > 0


def num_col(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def load_data(args: argparse.Namespace):
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


def add_signal_flags(df: pd.DataFrame) -> pd.DataFrame:
    # 將多個原始欄位整理成較容易報告的 high/low impact signals。
    df = df.copy()
    severity = df.get("severity", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    product = df.get("product", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    component = df.get("component", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()

    df["severity_normal_or_lower"] = severity.isin(["normal", "minor", "trivial", "enhancement"])
    df["severity_major_or_higher"] = severity.isin(["major", "critical", "blocker"])
    df["product_platform_or_jdt"] = product.isin(["platform", "jdt"])
    df["component_ui_debug_core"] = component.isin(["ui", "debug", "core"])
    df["stack_exception_signal"] = (
        bool_col(df, "has_stack_trace")
        | bool_col(df, "summary_has_stack_trace")
        | bool_col(df, "has_exception_terms")
        | bool_col(df, "summary_has_exception_terms")
        | bool_col(df, "has_file_line_stack_terms")
        | bool_col(df, "summary_has_file_line_stack_terms")
    )
    df["high_impact_signal"] = (
        bool_col(df, "has_crash_terms")
        | bool_col(df, "summary_has_crash_terms")
        | bool_col(df, "has_blocking_terms")
        | bool_col(df, "summary_has_blocking_terms")
        | bool_col(df, "has_data_loss_terms")
        | bool_col(df, "summary_has_data_loss_terms")
        | bool_col(df, "has_security_terms")
        | bool_col(df, "summary_has_security_terms")
        | bool_col(df, "has_build_break_terms")
        | bool_col(df, "summary_has_build_break_terms")
    )
    df["low_impact_signal"] = (
        bool_col(df, "has_enhancement_request_terms")
        | bool_col(df, "summary_has_enhancement_request_terms")
        | bool_col(df, "has_minor_visual_terms")
        | bool_col(df, "summary_has_minor_visual_terms")
        | bool_col(df, "has_preferences_terms")
        | bool_col(df, "summary_has_preferences_terms")
        | bool_col(df, "has_workaround_terms")
        | bool_col(df, "summary_has_workaround_terms")
    )
    df["debug_signal"] = bool_col(df, "has_debug_terms") | bool_col(df, "summary_has_debug_terms")
    df["update_install_signal"] = (
        bool_col(df, "has_install_update_terms")
        | bool_col(df, "summary_has_install_update_terms")
        | bool_col(df, "has_update_manager_terms")
        | bool_col(df, "summary_has_update_manager_terms")
    )
    df["related_pull_to_high"] = (
        (num_col(df, "related_top1_priority") <= 2)
        | (num_col(df, "related_top3_high_priority_rate") >= 0.50)
        | (num_col(df, "related_top3_avg_priority") <= 2.25)
    )
    df["related_pull_to_low"] = (
        (num_col(df, "related_top1_priority") >= 4)
        | (num_col(df, "related_top3_low_priority_rate") >= 0.50)
        | (num_col(df, "related_top3_avg_priority") >= 3.75)
    )
    return df


def compact_join_text(df: pd.DataFrame) -> pd.Series:
    summary = df.get("summary", pd.Series("", index=df.index)).fillna("").astype(str)
    desc = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    return summary + " " + desc


def top_phrases(df: pd.DataFrame, max_features: int = 12) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["phrase", "count"])
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{2,}\b",
    )
    matrix = vectorizer.fit_transform(compact_join_text(df))
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(-counts)[:max_features]
    return pd.DataFrame({"phrase": names[order], "count": counts[order].astype(int)})


def count_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count", "percent"])
    counts = df[column].fillna("unknown").astype(str).value_counts().rename_axis(column).reset_index(name="count")
    counts["percent"] = counts["count"] / len(df) if len(df) else 0.0
    return counts


def signal_table(direction_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    signal_cols = [
        "severity_normal_or_lower",
        "severity_major_or_higher",
        "stack_exception_signal",
        "high_impact_signal",
        "low_impact_signal",
        "debug_signal",
        "update_install_signal",
        "product_platform_or_jdt",
        "component_ui_debug_core",
        "related_pull_to_high",
        "related_pull_to_low",
    ]
    rows = []
    for column in signal_cols:
        if column not in direction_df.columns:
            continue
        direction_rate = float(direction_df[column].fillna(False).astype(bool).mean()) if len(direction_df) else 0.0
        reference_rate = float(reference_df[column].fillna(False).astype(bool).mean()) if len(reference_df) else 0.0
        rows.append({
            "signal": column,
            "direction_error_rate": direction_rate,
            "true_class_reference_rate": reference_rate,
            "difference": direction_rate - reference_rate,
        })
    return pd.DataFrame(rows).sort_values(["difference", "direction_error_rate"], ascending=False)


def markdown_table(df: pd.DataFrame, limit: int | None = None) -> str:
    view = df.copy()
    if limit is not None:
        view = view.head(limit)
    if view.empty:
        return "_No rows._"
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    try:
        return view.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(view.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(str(row[column]) for column in view.columns) + " |" for _, row in view.iterrows()]
        return "\n".join([header, divider, *rows])


def write_summary(path: str, selected: pd.DataFrame, meta: pd.DataFrame, directions: list[tuple[int, int]], removed_overlap: int) -> None:
    lines = [
        "# Targeted Error Direction Analysis",
        "",
        f"- Rows analyzed after overlap removal: `{len(meta)}`",
        f"- Removed overlapping training rows: `{removed_overlap}`",
        f"- Directions: `{', '.join(f'P{a}->P{b}' for a, b in directions)}`",
        "",
        "## Direction Counts",
        "",
        markdown_table(count_table(selected, "error_direction")),
        "",
    ]
    for true_label, pred_label in directions:
        direction_name = f"{PRIORITY_NAMES[true_label]} -> {PRIORITY_NAMES[pred_label]}"
        direction_df = selected[selected["error_direction"] == direction_name]
        reference_df = meta[meta["true_num"] == true_label]
        lines.extend([
            f"## {direction_name}",
            "",
            f"- Error rows: `{len(direction_df)}`",
            f"- True-class rows: `{len(reference_df)}`",
            "",
            "### Severity",
            "",
            markdown_table(count_table(direction_df, "severity")),
            "",
            "### Product",
            "",
            markdown_table(count_table(direction_df, "product"), 10),
            "",
            "### Component",
            "",
            markdown_table(count_table(direction_df, "component"), 10),
            "",
            "### Signals Compared With True Class",
            "",
            markdown_table(signal_table(direction_df, reference_df), 12),
            "",
            "### Top Phrases",
            "",
            markdown_table(top_phrases(direction_df), 12),
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "- P1 -> P2 表示高優先級 bug 被模型判太輕，可用 P1/P2 boundary refinement 補強高優先級邊界。",
        "- P4 -> P2 表示低優先級 bug 被模型拉太高，本版改用 cost-sensitive / recall-balanced learning 讓 P4 recall 納入正式選模目標。",
        "",
    ])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    directions = parse_directions(args.directions)
    X, y, meta, removed_overlap = load_data(args)
    model = joblib.load(args.model)
    y_pred = predict_model_or_bundle(model, X)
    meta = add_signal_flags(meta.copy())
    meta["true_num"] = y
    meta["predicted_num"] = y_pred
    meta["true_priority"] = meta["true_num"].map(PRIORITY_NAMES)
    meta["predicted_priority"] = meta["predicted_num"].map(PRIORITY_NAMES)
    meta["error_direction"] = meta["true_priority"] + " -> " + meta["predicted_priority"]

    mask = np.zeros(len(y), dtype=bool)
    for true_label, pred_label in directions:
        mask |= (y == true_label) & (y_pred == pred_label)
    selected = meta.loc[mask].copy()
    if "description" in selected.columns:
        selected["description_snippet"] = selected["description"].apply(compact_text)

    preferred = [
        "id",
        "true_priority",
        "predicted_priority",
        "error_direction",
        "severity",
        "product",
        "component",
        "summary",
        "description_snippet",
        "stack_exception_signal",
        "high_impact_signal",
        "low_impact_signal",
        "related_pull_to_high",
        "related_pull_to_low",
        "related_top1_priority",
        "related_top3_avg_priority",
        "related_top3_high_priority_rate",
        "related_top3_low_priority_rate",
        "creator",
        "creation_time",
        "status",
    ]
    output = selected[[column for column in preferred if column in selected.columns]].copy()
    output = output.sort_values(["error_direction", "id"])
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    write_summary(args.summary_md, selected, meta, directions, removed_overlap)

    print("=== Targeted Error Direction Analysis ===")
    print(count_table(selected, "error_direction").to_string(index=False))
    print(f"saved csv -> {args.output_csv}")
    print(f"saved report -> {args.summary_md}")


if __name__ == "__main__":
    main()
