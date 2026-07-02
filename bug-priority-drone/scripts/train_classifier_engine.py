"""Shared classifier factory for DRONE/REP- feature matrices.

主訓練程式會透過這裡建立 SGDClassifier、SVM、Logistic Regression、
XGBoost 或 LightGBM 候選模型，並用 validation set 選出較好的設定。
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.svm import LinearSVC

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features_literature_dupe_train")
CALIBRATION_FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features_literature_dupe_validation")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "direct_classifier_model.joblib")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "direct_classifier_report.csv")

LABELS = np.array([1, 2, 3, 4, 5])
LABEL_NAMES = ["P1 Blocker", "P2 Critical", "P3 Major", "P4 Minor", "P5 Trivial"]


class ZeroBasedLabelClassifier:
    """Adapter for libraries that expect multiclass labels to start at zero."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.labels_ = None

    def fit(self, X, y, sample_weight=None):
        self.labels_ = np.array(sorted(np.unique(y)))
        label_to_index = {label: idx for idx, label in enumerate(self.labels_)}
        y_zero = np.array([label_to_index[label] for label in y], dtype=int)
        if sample_weight is None:
            self.estimator.fit(X, y_zero)
        else:
            self.estimator.fit(X, y_zero, sample_weight=sample_weight)
        return self

    def predict(self, X):
        pred = self.estimator.predict(X).astype(int)
        return self.labels_[pred]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train direct classifiers on DRONE/REP feature matrices.")
    parser.add_argument("--feature-dir", default=FEATURE_DIR, help="Training feature directory.")
    parser.add_argument("--calibration-feature-dir", default=CALIBRATION_FEATURE_DIR, help="Validation feature directory.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Output model bundle.")
    parser.add_argument("--report-path", default=REPORT_PATH, help="Output report CSV.")
    parser.add_argument(
        "--model-types",
        default="linear_svm,ridge_classifier,sgd_log",
        help="Comma-separated candidates: linear_svm,ridge_classifier,sgd_log,logistic,xgboost,lightgbm.",
    )
    parser.add_argument("--c-values", default="0.1,0.3,1.0", help="Comma-separated C values for SVM/logistic.")
    parser.add_argument("--alpha-values", default="0.0001,0.001,0.01,0.1,1.0", help="Comma-separated alpha values.")
    parser.add_argument("--xgb-n-estimators", default="120", help="Comma-separated XGBoost estimator counts.")
    parser.add_argument("--xgb-max-depths", default="3,4", help="Comma-separated XGBoost max depths.")
    parser.add_argument("--xgb-learning-rates", default="0.05,0.1", help="Comma-separated XGBoost learning rates.")
    parser.add_argument("--lgbm-n-estimators", default="120", help="Comma-separated LightGBM estimator counts.")
    parser.add_argument("--lgbm-num-leaves", default="15,31", help="Comma-separated LightGBM num_leaves values.")
    parser.add_argument("--lgbm-learning-rates", default="0.05,0.1", help="Comma-separated LightGBM learning rates.")
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "accuracy", "guarded_macro"],
        default="macro_f1",
        help="Validation metric used to select the best classifier.",
    )
    return parser


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_features(feature_dir: str):
    X = load_npz(os.path.join(feature_dir, "X_features.npz"))
    y = np.load(os.path.join(feature_dir, "y.npy"))
    return X, y


def build_candidates(args: argparse.Namespace):
    # 依照命令列指定的 model_types 產生候選分類器與參數說明。
    for model_type in parse_str_list(args.model_types):
        if model_type == "linear_svm":
            for c_value in parse_float_list(args.c_values):
                yield model_type, f"C={c_value}", LinearSVC(C=c_value, class_weight="balanced", dual="auto", random_state=42)
        elif model_type == "ridge_classifier":
            for alpha in parse_float_list(args.alpha_values):
                yield model_type, f"alpha={alpha}", RidgeClassifier(alpha=alpha, class_weight="balanced")
        elif model_type == "sgd_log":
            for alpha in parse_float_list(args.alpha_values):
                yield model_type, f"alpha={alpha}", SGDClassifier(
                    loss="log_loss",
                    alpha=alpha,
                    class_weight="balanced",
                    max_iter=2000,
                    tol=1e-4,
                    random_state=42,
                )
        elif model_type == "logistic":
            for c_value in parse_float_list(args.c_values):
                yield model_type, f"C={c_value}", LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="saga",
                    n_jobs=-1,
                    random_state=42,
                )
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
            except Exception as exc:
                print(f"warning: xgboost is unavailable ({exc}); skipping xgboost candidates.")
                continue
            for n_estimators in parse_int_list(args.xgb_n_estimators):
                for max_depth in parse_int_list(args.xgb_max_depths):
                    for learning_rate in parse_float_list(args.xgb_learning_rates):
                        estimator = XGBClassifier(
                            objective="multi:softprob",
                            num_class=len(LABELS),
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=0.9,
                            colsample_bytree=0.8,
                            reg_lambda=1.0,
                            eval_metric="mlogloss",
                            tree_method="hist",
                            n_jobs=-1,
                            random_state=42,
                        )
                        yield (
                            model_type,
                            f"n_estimators={n_estimators},max_depth={max_depth},learning_rate={learning_rate}",
                            ZeroBasedLabelClassifier(estimator),
                        )
        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except Exception as exc:
                print(f"warning: lightgbm is unavailable ({exc}); skipping lightgbm candidates.")
                continue
            for n_estimators in parse_int_list(args.lgbm_n_estimators):
                for num_leaves in parse_int_list(args.lgbm_num_leaves):
                    for learning_rate in parse_float_list(args.lgbm_learning_rates):
                        estimator = LGBMClassifier(
                            objective="multiclass",
                            num_class=len(LABELS),
                            n_estimators=n_estimators,
                            num_leaves=num_leaves,
                            learning_rate=learning_rate,
                            subsample=0.9,
                            colsample_bytree=0.8,
                            reg_lambda=1.0,
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=42,
                            verbose=-1,
                        )
                        yield (
                            model_type,
                            f"n_estimators={n_estimators},num_leaves={num_leaves},learning_rate={learning_rate}",
                            ZeroBasedLabelClassifier(estimator),
                        )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "off_by_one_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "p5_false_positive_rate": float(np.mean((y_true != 5) & (y_pred == 5))),
        "predicted_p5_rate": float(np.mean(y_pred == 5)),
        "high_priority_recall_p1_p2": recall_score(np.isin(y_true, [1, 2]), np.isin(y_pred, [1, 2]), zero_division=0),
        "low_priority_recall_p4_p5": recall_score(np.isin(y_true, [4, 5]), np.isin(y_pred, [4, 5]), zero_division=0),
    }
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    for label, recall in zip(LABELS, recalls):
        row[f"p{label}_recall"] = float(recall)
    return row


def selection_score(row: dict[str, float], metric: str) -> float:
    if metric == "accuracy":
        return float(row["accuracy"])
    if metric == "guarded_macro":
        p1_penalty = max(0.0, 0.40 - float(row["p1_recall"]))
        p2_penalty = max(0.0, 0.40 - float(row["p2_recall"]))
        return float(row["macro_f1"]) - 0.25 * p1_penalty - 0.20 * p2_penalty
    return float(row["macro_f1"])


def main() -> None:
    args = build_parser().parse_args()
    X_train, y_train = load_features(args.feature_dir)
    X_calib, y_calib = load_features(args.calibration_feature_dir)

    rows = []
    best_model = None
    best_row = None
    best_score = -np.inf

    for model_type, params, model in build_candidates(args):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_calib).astype(int)
        row = metric_row(y_calib, y_pred)
        row.update({
            "candidate_model_type": model_type,
            "candidate_params": params,
            "selection_score": selection_score(row, args.selection_metric),
        })
        rows.append(row)
        print(
            f"{model_type} {params}: "
            f"macro={row['macro_f1']:.4f} acc={row['accuracy']:.4f} "
            f"p1r={row['p1_recall']:.4f} p2r={row['p2_recall']:.4f}"
        )
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            best_row = row
            best_model = model

    result = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    result.to_csv(args.report_path, index=False, encoding="utf-8-sig")

    bundle = {
        "model_type": "direct_classifier",
        "model": best_model,
        "labels": LABELS,
        "settings": best_row,
    }
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(bundle, args.model_path)

    print("\n=== Best Direct Classifier ===")
    print(pd.Series(best_row).to_string())
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_calib, best_model.predict(X_calib), labels=LABELS))
    print("\nClassification Report:")
    print(classification_report(y_calib, best_model.predict(X_calib), labels=LABELS, target_names=LABEL_NAMES, digits=4, zero_division=0))
    print(f"saved model  -> {args.model_path}")
    print(f"saved report -> {args.report_path}")


if __name__ == "__main__":
    main()
