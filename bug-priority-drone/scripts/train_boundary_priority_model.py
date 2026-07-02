"""Utilities and legacy entry point for boundary-refined priority models.

目前主流程不直接執行本檔；`train_recall_balanced_priority_model.py`
會 import 其中的 `fit_candidate` 與 `class_report_rows` 作為訓練輔助工具。
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, recall_score

from train_classifier_engine import build_candidates, selection_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = np.array([1, 2, 3, 4, 5])
BOUNDARY_LABELS = np.array([1, 2, 3])
LABEL_NAMES = {
    1: "P1 Blocker",
    2: "P2 Critical",
    3: "P3 Major",
    4: "P4 Minor",
    5: "P5 Trivial",
}

FEATURE_PROFILES = {
    "literature": {
        "train": "data/processed/features_literature_dupe_train",
        "validation": "data/processed/features_literature_dupe_validation",
        "natural_holdout": "data/processed/features_literature_dupe_natural_test",
    },
    "enhanced": {
        "train": "data/processed/features_enhanced_dupe_train",
        "validation": "data/processed/features_enhanced_dupe_validation",
        "natural_holdout": "data/processed/features_enhanced_dupe_natural_test",
    },
    "improved": {
        "train": "data/processed/features_improved_dupe_train",
        "validation": "data/processed/features_improved_dupe_validation",
        "natural_holdout": "data/processed/features_improved_dupe_natural_test",
    },
    "bm25_summary_high": {
        "train": "data/processed/features_bm25_summary_high_train",
        "validation": "data/processed/features_bm25_summary_high_validation",
        "natural_holdout": "data/processed/features_bm25_summary_high_natural_test",
    },
}


def project_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a direct classifier plus a P1/P2/P3 boundary refiner."
    )
    parser.add_argument("--profile", choices=sorted(FEATURE_PROFILES), default="bm25_summary_high")
    parser.add_argument("--model-path", default=project_path("models/boundary_refined_priority_model.joblib"))
    parser.add_argument("--report-csv", default=project_path("reports/boundary_refined_priority_model_eval.csv"))
    parser.add_argument("--candidate-report-csv", default=project_path("reports/boundary_refined_priority_model_candidates.csv"))
    parser.add_argument("--class-report-csv", default=project_path("reports/boundary_refined_priority_model_class_report.csv"))
    parser.add_argument("--report-md", default=project_path("reports/boundary_refined_priority_model_report.md"))
    parser.add_argument("--model-types", default="linear_svm,ridge_classifier,sgd_log")
    parser.add_argument("--c-values", default="0.1,0.3,1.0")
    parser.add_argument("--alpha-values", default="0.0001,0.001,0.01,0.1,1.0")
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "accuracy", "guarded_macro"],
        default="guarded_macro",
        help="Metric for selecting the base classifier before boundary refinement.",
    )
    parser.add_argument(
        "--boundary-apply",
        choices=["base_1_2_3", "base_1_2", "base_2_3"],
        default="base_1_2_3",
        help="Rows whose base prediction will be refined by the P1/P2/P3 boundary model.",
    )
    parser.add_argument(
        "--p2-weight",
        type=float,
        default=0.10,
        help="Extra validation objective weight for P2 recall in boundary-model selection.",
    )
    parser.add_argument("--p1-weight", type=float, default=0.00, help="Extra validation objective weight for P1 recall.")
    parser.add_argument("--p3-weight", type=float, default=0.00, help="Extra validation objective weight for P3 recall.")
    parser.add_argument(
        "--off-by-one-weight",
        type=float,
        default=0.00,
        help="Extra validation objective weight for off-by-one accuracy.",
    )
    parser.add_argument(
        "--collapse-floor",
        type=float,
        default=0.55,
        help="Penalty floor for P1/P3 recall so P2 improvement does not collapse neighboring classes.",
    )
    parser.add_argument(
        "--accuracy-drop-weight",
        type=float,
        default=0.10,
        help="Penalty weight for validation accuracy drop versus the base classifier.",
    )
    parser.add_argument(
        "--collapse-penalty-weight",
        type=float,
        default=0.20,
        help="Penalty weight when P1/P3 recall falls below collapse-floor.",
    )
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=1.0,
        help="Additional guardrail; penalize if refined validation accuracy drops more than this value.",
    )
    parser.add_argument(
        "--boundary-p2-sample-weight",
        type=float,
        default=1.0,
        help="Sample weight multiplier for P2 rows in the P1/P2/P3 boundary classifier.",
    )
    return parser


def load_features(feature_dir: str):
    X = load_npz(os.path.join(feature_dir, "X_features.npz"))
    y = np.load(os.path.join(feature_dir, "y.npy"))
    meta = pd.read_csv(os.path.join(feature_dir, "feature_meta.csv"))
    return X, y, meta


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, eval_set: str) -> dict:
    row = {
        "eval_set": eval_set,
        "rows": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "off_by_one_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "high_priority_recall_p1_p2": recall_score(np.isin(y_true, [1, 2]), np.isin(y_pred, [1, 2]), zero_division=0),
        "low_priority_recall_p4_p5": recall_score(np.isin(y_true, [4, 5]), np.isin(y_pred, [4, 5]), zero_division=0),
    }
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    for label, value in zip(LABELS, recalls):
        row[f"p{int(label)}_recall"] = float(value)
    return row


def class_report_rows(y_true: np.ndarray, y_pred: np.ndarray, eval_set: str) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    return [
        {
            "eval_set": eval_set,
            "priority": f"P{int(label)}",
            "label": LABEL_NAMES[int(label)],
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "support": int(class_support),
        }
        for label, class_precision, class_recall, class_f1, class_support in zip(
            LABELS, precision, recall, f1, support
        )
    ]


def boundary_mask(base_pred: np.ndarray, mode: str) -> np.ndarray:
    if mode == "base_1_2":
        return np.isin(base_pred, [1, 2])
    if mode == "base_2_3":
        return np.isin(base_pred, [2, 3])
    return np.isin(base_pred, [1, 2, 3])


def predict_refined(base_model, boundary_model, X, apply_mode: str) -> np.ndarray:
    y_pred = base_model.predict(X).astype(int)
    mask = boundary_mask(y_pred, apply_mode)
    if mask.any():
        y_pred[mask] = boundary_model.predict(X[mask]).astype(int)
    return y_pred


def fit_candidate(model, X, y, sample_weight=None):
    # 有些 estimator 不支援 sample_weight；不支援時退回一般 fit。
    if sample_weight is None:
        model.fit(X, y)
        return model
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    return model


def boundary_selection_score(
    row: dict,
    base_row: dict,
    p2_weight: float,
    collapse_floor: float,
    p1_weight: float = 0.0,
    p3_weight: float = 0.0,
    off_by_one_weight: float = 0.0,
    accuracy_drop_weight: float = 0.10,
    collapse_penalty_weight: float = 0.20,
    max_accuracy_drop: float = 1.0,
) -> float:
    p1_penalty = max(0.0, collapse_floor - float(row["p1_recall"]))
    p3_penalty = max(0.0, collapse_floor - float(row["p3_recall"]))
    accuracy_drop = max(0.0, float(base_row["accuracy"]) - float(row["accuracy"]))
    excess_accuracy_drop = max(0.0, accuracy_drop - max_accuracy_drop)
    return (
        float(row["macro_f1"])
        + p1_weight * float(row["p1_recall"])
        + p2_weight * float(row["p2_recall"])
        + p3_weight * float(row["p3_recall"])
        + off_by_one_weight * float(row["off_by_one_accuracy"])
        - collapse_penalty_weight * (p1_penalty + p3_penalty)
        - accuracy_drop_weight * accuracy_drop
        - 1.0 * excess_accuracy_drop
    )


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for column in columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    try:
        return view.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in view.iterrows()]
        return "\n".join([header, divider, *rows])


def write_report(report_path: str, eval_df: pd.DataFrame, candidate_df: pd.DataFrame, class_df: pd.DataFrame, matrices: dict[str, np.ndarray]) -> None:
    eval_cols = ["model_stage", "eval_set", "rows", "accuracy", "macro_f1", "off_by_one_accuracy", "mae", "p1_recall", "p2_recall", "p3_recall", "p4_recall", "p5_recall"]
    cand_cols = ["boundary_model_type", "boundary_params", "selection_score", "accuracy", "macro_f1", "p1_recall", "p2_recall", "p3_recall", "off_by_one_accuracy", "mae"]
    class_cols = ["eval_set", "priority", "label", "precision", "recall", "f1", "support"]
    lines = [
        "# Boundary Refined Priority Model Report",
        "",
        "This experiment adds a P1/P2/P3 boundary classifier on top of the direct DRONE/REP- classifier.",
        "",
        "## Evaluation Results",
        "",
        markdown_table(eval_df, eval_cols),
        "",
        "## Boundary Candidate Ranking",
        "",
        markdown_table(candidate_df.sort_values("selection_score", ascending=False).head(8), cand_cols),
        "",
        "## Per-Priority Classification Report",
        "",
        markdown_table(class_df, class_cols),
        "",
        "## Confusion Matrices",
        "",
    ]
    for name, matrix in matrices.items():
        lines.extend([f"### {name}", "", "Rows are true P1-P5; columns are predicted P1-P5.", "", "```text", str(matrix), "```", ""])
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    profile_dirs = {name: project_path(path) for name, path in FEATURE_PROFILES[args.profile].items()}
    missing = [path for path in profile_dirs.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Feature directories do not exist: {missing}")

    X_train, y_train, train_meta = load_features(profile_dirs["train"])
    X_val, y_val, _ = load_features(profile_dirs["validation"])
    X_nat, y_nat, nat_meta = load_features(profile_dirs["natural_holdout"])

    base_rows = []
    best_base_model = None
    best_base_row = None
    best_base_score = -np.inf
    for model_type, params, model in build_candidates(args):
        fit_candidate(model, X_train, y_train)
        y_val_pred = model.predict(X_val).astype(int)
        row = metric_row(y_val, y_val_pred, "validation")
        row.update({
            "candidate_model_type": model_type,
            "candidate_params": params,
            "selection_score": selection_score(row, args.selection_metric),
        })
        base_rows.append(row)
        if row["selection_score"] > best_base_score:
            best_base_score = float(row["selection_score"])
            best_base_model = model
            best_base_row = row

    if best_base_model is None or best_base_row is None:
        raise RuntimeError("No base classifier was selected.")

    train_boundary_mask = np.isin(y_train, BOUNDARY_LABELS)
    X_train_boundary = X_train[train_boundary_mask]
    y_train_boundary = y_train[train_boundary_mask]
    boundary_sample_weight = np.ones(len(y_train_boundary), dtype=float)
    boundary_sample_weight[y_train_boundary == 2] = float(args.boundary_p2_sample_weight)

    boundary_rows = []
    best_boundary_model = None
    best_boundary_row = None
    best_boundary_score = -np.inf
    for model_type, params, model in build_candidates(args):
        fit_candidate(model, X_train_boundary, y_train_boundary, sample_weight=boundary_sample_weight)
        y_val_refined = predict_refined(best_base_model, model, X_val, args.boundary_apply)
        row = metric_row(y_val, y_val_refined, "validation")
        row.update({
            "boundary_model_type": model_type,
            "boundary_params": params,
            "boundary_p2_sample_weight": args.boundary_p2_sample_weight,
            "selection_score": boundary_selection_score(
                row,
                best_base_row,
                p2_weight=args.p2_weight,
                collapse_floor=args.collapse_floor,
                p1_weight=args.p1_weight,
                p3_weight=args.p3_weight,
                off_by_one_weight=args.off_by_one_weight,
                accuracy_drop_weight=args.accuracy_drop_weight,
                collapse_penalty_weight=args.collapse_penalty_weight,
                max_accuracy_drop=args.max_accuracy_drop,
            ),
        })
        boundary_rows.append(row)
        print(
            f"{model_type} {params}: score={row['selection_score']:.4f} "
            f"acc={row['accuracy']:.4f} macro={row['macro_f1']:.4f} "
            f"p1={row['p1_recall']:.4f} p2={row['p2_recall']:.4f} p3={row['p3_recall']:.4f}"
        )
        if row["selection_score"] > best_boundary_score:
            best_boundary_score = float(row["selection_score"])
            best_boundary_model = model
            best_boundary_row = row

    if best_boundary_model is None or best_boundary_row is None:
        raise RuntimeError("No boundary classifier was selected.")

    base_nat_pred = best_base_model.predict(X_nat).astype(int)
    refined_nat_pred = predict_refined(best_base_model, best_boundary_model, X_nat, args.boundary_apply)
    eval_rows = []
    base_nat_row = metric_row(y_nat, base_nat_pred, "natural_holdout")
    base_nat_row["model_stage"] = "base_direct"
    refined_nat_row = metric_row(y_nat, refined_nat_pred, "natural_holdout")
    refined_nat_row["model_stage"] = "boundary_refined"
    eval_rows.extend([base_nat_row, refined_nat_row])
    eval_df = pd.DataFrame(eval_rows)

    class_df = pd.DataFrame(class_report_rows(y_nat, refined_nat_pred, "natural_holdout"))
    candidate_df = pd.DataFrame(boundary_rows).sort_values("selection_score", ascending=False)
    matrices = {
        "base_direct_natural_holdout": confusion_matrix(y_nat, base_nat_pred, labels=LABELS),
        "boundary_refined_natural_holdout": confusion_matrix(y_nat, refined_nat_pred, labels=LABELS),
    }

    bundle = {
        "model_type": "boundary_refined_direct_classifier",
        "feature_profile": args.profile,
        "base_model": best_base_model,
        "base_settings": best_base_row,
        "boundary_model": best_boundary_model,
        "boundary_settings": best_boundary_row,
        "boundary_apply": args.boundary_apply,
        "boundary_p2_sample_weight": args.boundary_p2_sample_weight,
        "boundary_objective": {
            "p1_weight": args.p1_weight,
            "p2_weight": args.p2_weight,
            "p3_weight": args.p3_weight,
            "off_by_one_weight": args.off_by_one_weight,
            "collapse_floor": args.collapse_floor,
            "accuracy_drop_weight": args.accuracy_drop_weight,
            "collapse_penalty_weight": args.collapse_penalty_weight,
            "max_accuracy_drop": args.max_accuracy_drop,
        },
        "labels": LABELS,
        "label_names": LABEL_NAMES,
        "feature_dirs": profile_dirs,
    }
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(bundle, args.model_path)

    os.makedirs(os.path.dirname(args.report_csv), exist_ok=True)
    eval_df.to_csv(args.report_csv, index=False, encoding="utf-8-sig")
    candidate_df.to_csv(args.candidate_report_csv, index=False, encoding="utf-8-sig")
    class_df.to_csv(args.class_report_csv, index=False, encoding="utf-8-sig")
    write_report(args.report_md, eval_df, candidate_df, class_df, matrices)

    print("\n=== Boundary Refined Model ===")
    print(f"profile: {args.profile}")
    print(f"base selected: {best_base_row['candidate_model_type']} {best_base_row['candidate_params']}")
    print(f"boundary selected: {best_boundary_row['boundary_model_type']} {best_boundary_row['boundary_params']}")
    print(eval_df[["model_stage", "accuracy", "macro_f1", "off_by_one_accuracy", "mae", "p1_recall", "p2_recall", "p3_recall", "p4_recall", "p5_recall"]].to_string(index=False))
    print(f"saved model -> {args.model_path}")
    print(f"saved report -> {args.report_md}")


if __name__ == "__main__":
    main()
