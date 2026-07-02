"""Train the current cost-sensitive recall-balanced priority model.

這是目前 GitHub 主模型訓練程式：先訓練 cost-sensitive base classifier，
再加上 P1/P2 boundary refinement，最後用 validation objective 選出最佳模型。
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, recall_score

from train_boundary_priority_model import class_report_rows, fit_candidate
from train_classifier_engine import build_candidates, parse_float_list

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = np.array([1, 2, 3, 4, 5])


def project_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train recall-balanced model with cost-sensitive class weights and P1/P2 boundary refinement.",
    )
    parser.add_argument("--train-feature-dir", default=project_path("data/processed/features_bm25_p2_error_keywords_train"))
    parser.add_argument("--validation-feature-dir", default=project_path("data/processed/features_bm25_p2_error_keywords_validation"))
    parser.add_argument("--natural-feature-dir", default=project_path("data/processed/features_bm25_p2_error_keywords_natural_test"))
    parser.add_argument("--model-path", default=project_path("models/recall_balanced_priority_model.joblib"))
    parser.add_argument("--candidate-csv", default=project_path("reports/recall_balanced_grid_search.csv"))
    parser.add_argument("--eval-csv", default=project_path("reports/recall_balanced_best_eval.csv"))
    parser.add_argument("--class-report-csv", default=project_path("reports/recall_balanced_best_class_report.csv"))
    parser.add_argument("--summary-md", default=project_path("reports/recall_balanced_priority_model.md"))
    parser.add_argument("--model-types", default="sgd_log")
    parser.add_argument("--alpha-values", default="0.03,0.05,0.1")
    parser.add_argument("--c-values", default="0.1,0.3,1.0")
    parser.add_argument("--base-p1-sample-weights", default="1.0")
    parser.add_argument("--base-p2-sample-weights", default="1.0,1.15")
    parser.add_argument(
        "--base-p4-sample-weights",
        "--p4-sample-weights",
        dest="base_p4_sample_weights",
        default="1.0,1.2,1.4",
    )
    parser.add_argument(
        "--boundary-p1-sample-weights",
        "--p1-sample-weights",
        dest="boundary_p1_sample_weights",
        default="1.0,1.2,1.4",
    )
    parser.add_argument("--boundary-p2-sample-weights", default="1.0")
    parser.add_argument("--macro-recall-weight", type=float, default=1.00)
    parser.add_argument("--min-recall-weight", type=float, default=0.60)
    parser.add_argument("--macro-f1-weight", type=float, default=0.60)
    parser.add_argument("--accuracy-weight", type=float, default=0.25)
    parser.add_argument("--mae-weight", type=float, default=0.20)
    parser.add_argument("--off-by-one-weight", type=float, default=0.05)
    parser.add_argument("--p1-floor", type=float, default=0.60)
    parser.add_argument("--p2-floor", type=float, default=0.62)
    parser.add_argument("--p3-floor", type=float, default=0.72)
    parser.add_argument("--p4-floor", type=float, default=0.62)
    parser.add_argument("--p5-floor", type=float, default=0.68)
    parser.add_argument("--floor-penalty-weight", type=float, default=0.70)
    return parser


def load_features(feature_dir: str):
    X = load_npz(os.path.join(feature_dir, "X_features.npz"))
    y = np.load(os.path.join(feature_dir, "y.npy"))
    meta = pd.read_csv(os.path.join(feature_dir, "feature_meta.csv"))
    return X, y, meta


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, eval_set: str) -> dict:
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    row = {
        "eval_set": eval_set,
        "rows": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "macro_recall": float(np.mean(recalls)),
        "min_recall": float(np.min(recalls)),
        "off_by_one_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }
    for label, value in zip(LABELS, recalls):
        row[f"p{int(label)}_recall"] = float(value)
    return row


def selection_objective(row: dict, args: argparse.Namespace) -> float:
    # 選模不只看 accuracy，也獎勵 macro/min recall 並懲罰 MAE 與低 recall。
    floors = {
        "p1_recall": args.p1_floor,
        "p2_recall": args.p2_floor,
        "p3_recall": args.p3_floor,
        "p4_recall": args.p4_floor,
        "p5_recall": args.p5_floor,
    }
    floor_penalty = sum(max(0.0, floor - float(row[column])) for column, floor in floors.items())
    return (
        args.macro_recall_weight * float(row["macro_recall"])
        + args.min_recall_weight * float(row["min_recall"])
        + args.macro_f1_weight * float(row["macro_f1"])
        + args.accuracy_weight * float(row["accuracy"])
        + args.off_by_one_weight * float(row["off_by_one_accuracy"])
        - args.mae_weight * float(row["mae"])
        - args.floor_penalty_weight * floor_penalty
    )


def predict_recall_balanced(bundle: dict, X) -> np.ndarray:
    # 先用五分類 base model 預測；只有 P1/P2 預測再交給 boundary model 細分。
    y_pred = bundle["base_model"].predict(X).astype(int)

    p1p2_model = bundle.get("p1p2_model")
    if p1p2_model is not None:
        p1p2_mask = np.isin(y_pred, [1, 2])
        if p1p2_mask.any():
            y_pred[p1p2_mask] = p1p2_model.predict(X[p1p2_mask]).astype(int)

    return y_pred


def candidate_row(stage: str, settings: dict, val_row: dict, nat_row: dict, objective: float) -> dict:
    row = {"stage": stage, **settings, "validation_objective": objective}
    row.update({f"validation_{key}": value for key, value in val_row.items() if key not in {"eval_set", "rows"}})
    row.update({f"natural_{key}": value for key, value in nat_row.items() if key not in {"eval_set", "rows"}})
    return row


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> str:
    view = df.copy()
    if columns is not None:
        view = view[[column for column in columns if column in view.columns]]
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


def write_summary(path: str, candidates: pd.DataFrame, eval_df: pd.DataFrame, matrix: np.ndarray) -> None:
    eval_cols = [
        "model_stage",
        "eval_set",
        "accuracy",
        "macro_f1",
        "macro_recall",
        "min_recall",
        "off_by_one_accuracy",
        "mae",
        "p1_recall",
        "p2_recall",
        "p3_recall",
        "p4_recall",
        "p5_recall",
    ]
    cand_cols = [
        "stage",
        "base_model_type",
        "base_params",
        "base_p1_sample_weight",
        "base_p2_sample_weight",
        "base_p4_sample_weight",
        "p1p2_model_type",
        "p1p2_params",
        "boundary_p1_sample_weight",
        "boundary_p2_sample_weight",
        "validation_objective",
        "validation_macro_recall",
        "validation_min_recall",
        "validation_p1_recall",
        "validation_p2_recall",
        "validation_p4_recall",
        "natural_accuracy",
        "natural_macro_recall",
        "natural_min_recall",
    ]
    lines = [
        "# Recall-Balanced Priority Model",
        "",
        "This experiment keeps the DRONE/REP- feature design and uses a more formal recall-balanced training strategy:",
        "",
        "1. Cost-sensitive class sample weights for the base classifier.",
        "2. P1/P2 boundary classifier for high-priority boundary refinement.",
        "",
        "Selection uses validation-set `macro recall + minimum recall + macro F1`, with MAE and low-recall penalties.",
        "",
        "## Best Natural-Holdout Result",
        "",
        markdown_table(eval_df, eval_cols),
        "",
        "## Top Validation Candidates",
        "",
        markdown_table(candidates.sort_values("validation_objective", ascending=False), cand_cols, limit=12),
        "",
        "## Confusion Matrix",
        "",
        "Rows are true P1-P5; columns are predicted P1-P5.",
        "",
        "```text",
        str(matrix),
        "```",
        "",
        "## Interpretation",
        "",
        "- This model should be compared with the current best model before replacing it.",
        "- If macro recall or minimum recall improves but accuracy drops, keep it as an improvement experiment rather than the main model.",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    X_train, y_train, _ = load_features(args.train_feature_dir)
    X_val, y_val, _ = load_features(args.validation_feature_dir)
    X_nat, y_nat, _ = load_features(args.natural_feature_dir)

    rows = []
    best = None
    best_score = -np.inf

    base_fits = []
    base_p1_weights = parse_float_list(args.base_p1_sample_weights)
    base_p2_weights = parse_float_list(args.base_p2_sample_weights)
    base_p4_weights = parse_float_list(args.base_p4_sample_weights)
    for base_p1_weight in base_p1_weights:
        for base_p2_weight in base_p2_weights:
            for base_p4_weight in base_p4_weights:
                base_sample_weight = np.ones(len(y_train), dtype=float)
                base_sample_weight[y_train == 1] = base_p1_weight
                base_sample_weight[y_train == 2] = base_p2_weight
                base_sample_weight[y_train == 4] = base_p4_weight
                for base_model_type, base_params, base_model in build_candidates(args):
                    fit_candidate(base_model, X_train, y_train, sample_weight=base_sample_weight)
                    base_bundle = {"base_model": base_model, "p1p2_model": None}
                    y_val_pred = predict_recall_balanced(base_bundle, X_val)
                    y_nat_pred = predict_recall_balanced(base_bundle, X_nat)
                    val_row = metric_row(y_val, y_val_pred, "validation")
                    nat_row = metric_row(y_nat, y_nat_pred, "natural_holdout")
                    objective = selection_objective(val_row, args)
                    settings = {
                        "base_model_type": base_model_type,
                        "base_params": base_params,
                        "base_p1_sample_weight": base_p1_weight,
                        "base_p2_sample_weight": base_p2_weight,
                        "base_p4_sample_weight": base_p4_weight,
                        "p1p2_model_type": "",
                        "p1p2_params": "",
                        "boundary_p1_sample_weight": 1.0,
                        "boundary_p2_sample_weight": 1.0,
                    }
                    rows.append(candidate_row("cost_sensitive_base", settings, val_row, nat_row, objective))
                    base_fits.append((base_model_type, base_params, base_model, settings))
                    if objective > best_score:
                        best_score = objective
                        best = {
                            "bundle": base_bundle,
                            "settings": settings,
                            "stage": "cost_sensitive_base",
                            "nat_pred": y_nat_pred,
                            "nat_row": nat_row,
                        }

    p1p2_mask = np.isin(y_train, [1, 2])
    X_p1p2 = X_train[p1p2_mask]
    y_p1p2 = y_train[p1p2_mask]

    boundary_p1_weights = parse_float_list(args.boundary_p1_sample_weights)
    boundary_p2_weights = parse_float_list(args.boundary_p2_sample_weights)
    for base_model_type, base_params, base_model, base_settings in base_fits:
        for boundary_p1_weight in boundary_p1_weights:
            for boundary_p2_weight in boundary_p2_weights:
                boundary_sample_weight = np.ones(len(y_p1p2), dtype=float)
                boundary_sample_weight[y_p1p2 == 1] = boundary_p1_weight
                boundary_sample_weight[y_p1p2 == 2] = boundary_p2_weight
                for p1p2_model_type, p1p2_params, p1p2_model in build_candidates(args):
                    fit_candidate(p1p2_model, X_p1p2, y_p1p2, sample_weight=boundary_sample_weight)
                    p1p2_bundle = {"base_model": base_model, "p1p2_model": p1p2_model}
                    y_val_pred = predict_recall_balanced(p1p2_bundle, X_val)
                    y_nat_pred = predict_recall_balanced(p1p2_bundle, X_nat)
                    val_row = metric_row(y_val, y_val_pred, "validation")
                    nat_row = metric_row(y_nat, y_nat_pred, "natural_holdout")
                    objective = selection_objective(val_row, args)
                    settings = {
                        **base_settings,
                        "p1p2_model_type": p1p2_model_type,
                        "p1p2_params": p1p2_params,
                        "boundary_p1_sample_weight": boundary_p1_weight,
                        "boundary_p2_sample_weight": boundary_p2_weight,
                    }
                    rows.append(candidate_row("cost_sensitive_p1p2_boundary", settings, val_row, nat_row, objective))
                    if objective > best_score:
                        best_score = objective
                        best = {
                            "bundle": p1p2_bundle,
                            "settings": settings,
                            "stage": "cost_sensitive_p1p2_boundary",
                            "nat_pred": y_nat_pred,
                            "nat_row": nat_row,
                        }

    if best is None:
        raise RuntimeError("No candidate selected.")

    candidates = pd.DataFrame(rows).sort_values("validation_objective", ascending=False)
    os.makedirs(os.path.dirname(args.candidate_csv), exist_ok=True)
    candidates.to_csv(args.candidate_csv, index=False, encoding="utf-8-sig")

    model_bundle = {
        "model_type": "recall_balanced_cost_sensitive",
        "feature_profile": "bm25_p2_error_keywords",
        "stage": best["stage"],
        "labels": LABELS,
        "base_model": best["bundle"]["base_model"],
        "p1p2_model": best["bundle"].get("p1p2_model"),
        "settings": best["settings"],
        "selection_objective": {
            "macro_recall_weight": args.macro_recall_weight,
            "min_recall_weight": args.min_recall_weight,
            "macro_f1_weight": args.macro_f1_weight,
            "accuracy_weight": args.accuracy_weight,
            "mae_weight": args.mae_weight,
            "off_by_one_weight": args.off_by_one_weight,
            "floors": {
                "p1": args.p1_floor,
                "p2": args.p2_floor,
                "p3": args.p3_floor,
                "p4": args.p4_floor,
                "p5": args.p5_floor,
            },
        },
        "feature_dirs": {
            "train": args.train_feature_dir,
            "validation": args.validation_feature_dir,
            "natural_holdout": args.natural_feature_dir,
        },
    }
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(model_bundle, args.model_path)

    eval_row = dict(best["nat_row"])
    eval_row["model_stage"] = "recall_balanced_best"
    eval_df = pd.DataFrame([eval_row])
    eval_df.to_csv(args.eval_csv, index=False, encoding="utf-8-sig")
    class_df = pd.DataFrame(class_report_rows(y_nat, best["nat_pred"], "natural_holdout"))
    class_df.to_csv(args.class_report_csv, index=False, encoding="utf-8-sig")
    matrix = confusion_matrix(y_nat, best["nat_pred"], labels=LABELS)
    write_summary(args.summary_md, candidates, eval_df, matrix)

    print("=== Recall-Balanced Grid Search ===")
    top_cols = [
        "stage",
        "base_model_type",
        "base_params",
        "base_p1_sample_weight",
        "base_p2_sample_weight",
        "base_p4_sample_weight",
        "p1p2_params",
        "boundary_p1_sample_weight",
        "boundary_p2_sample_weight",
        "validation_objective",
        "validation_macro_recall",
        "validation_min_recall",
        "validation_p1_recall",
        "validation_p2_recall",
        "validation_p4_recall",
        "natural_accuracy",
        "natural_macro_recall",
        "natural_min_recall",
    ]
    print(candidates[top_cols].head(10).to_string(index=False))
    print("\nBest natural_holdout:")
    print(eval_df[[
        "accuracy",
        "macro_f1",
        "macro_recall",
        "min_recall",
        "off_by_one_accuracy",
        "mae",
        "p1_recall",
        "p2_recall",
        "p3_recall",
        "p4_recall",
        "p5_recall",
    ]].to_string(index=False))
    print("\nConfusion matrix labels P1..P5:")
    print(matrix)
    print(f"saved model -> {args.model_path}")
    print(f"saved eval -> {args.eval_csv}")
    print(f"saved report -> {args.summary_md}")


if __name__ == "__main__":
    main()
