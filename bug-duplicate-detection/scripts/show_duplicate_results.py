#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RESULTS_CSV = "reports/duplicate_experiment_results.csv"
DEFAULT_DECISION_JSON = "sbert_decision_threshold_p90.json"
DEFAULT_REPORTS_DIR = "reports"
REQUIRED_RESULT_COLUMNS = {
    "experiment",
    "method",
    "combine",
    "fold",
    "mean_average_precision",
    "top_1_accuracy",
    "top_k_hit_rate",
    "mean_reciprocal_rank",
}
SORT_COLUMNS = {
    "MAP": "mean_average_precision",
    "top1": "top_1_accuracy",
    "topk": "top_k_hit_rate",
    "recall": "recall_at_k",
    "MRR": "mean_reciprocal_rank",
}
AUTO_SELECT_MODES = ("best", "latest", "default")
DECISION_JSON_PATTERNS = ("*decision*.json", "*threshold*.json")


@dataclass(frozen=True)
class ResultRow:
    experiment: str
    method: str
    combine: str
    fold: str
    queries: int
    mean_average_precision: float
    top_1_accuracy: float
    top_k_hit_rate: float
    mean_reciprocal_rank: float
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    base_model: str = ""
    epochs: int = 0
    max_triplets: int = 0
    seconds: float = 0.0


@dataclass(frozen=True)
class DecisionResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    queries: int
    positive_queries: int
    correct_duplicate_links: int = 0
    method: str = ""
    combine: str = ""
    split: str = ""
    train: int = 0
    validation: int = 0
    min_precision: float | None = None
    min_recall: float | None = None
    rerank_fields: str = ""
    base_model: str = ""
    constraints_satisfied: bool | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Show saved duplicate-ticket experiment results without rerunning experiments.")
    parser.add_argument("--results-csv", help="CSV produced by scripts/run_duplicate_experiments.py")
    parser.add_argument("--decision-json", help="JSON produced by cli tune-threshold --output-json")
    parser.add_argument(
        "--show-decision-matrix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show a decision/threshold confusion matrix when one is available. "
            "Use --no-show-decision-matrix to print only the best ranking result."
        ),
    )
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR, help="Directory to auto-search when --results-csv is omitted")
    parser.add_argument(
        "--select",
        choices=AUTO_SELECT_MODES,
        default="best",
        help="When --results-csv is omitted: best=highest selected metric, latest=newest CSV, default=legacy default path",
    )
    parser.add_argument("--fold", default="mean", help="Fold to display. Defaults to the final mean rows.")
    parser.add_argument("--sort-by", choices=tuple(SORT_COLUMNS.keys()), default="MAP")
    args = parser.parse_args()

    sort_column = SORT_COLUMNS[args.sort_by]
    path = resolve_results_csv(
        args.results_csv,
        reports_dir=Path(args.reports_dir),
        selection=args.select,
        score_column=sort_column,
        fold=args.fold,
    )

    rows = read_results_csv(path)
    selected_rows = [row for row in rows if row.fold == args.fold]
    if not selected_rows and args.fold == "mean":
        selected_rows = rows
    if not selected_rows:
        available_folds = ", ".join(sorted({row.fold for row in rows})) or "(none)"
        raise SystemExit(f"No rows for fold={args.fold}. Available folds: {available_folds}")

    selected_rows = sorted(selected_rows, key=lambda row: sort_value(row, sort_column), reverse=True)
    best_row = selected_rows[0]

    print(f"results_csv={path}")
    print(render_best_ranking_result(best_row, title=f"Best Ranking Result ({args.fold}, sort_by={args.sort_by})"))

    decision_path = resolve_decision_json(
        args.decision_json,
        reports_dir=Path(args.reports_dir),
        auto=args.show_decision_matrix,
    )
    if decision_path is not None:
        print("")
        print(f"decision_json={decision_path}")
        print(render_decision_matrix(read_decision_json(decision_path)))
    return 0


def resolve_results_csv(
    results_csv: str | None,
    *,
    reports_dir: Path,
    selection: str = "best",
    score_column: str = "mean_average_precision",
    fold: str = "mean",
) -> Path:
    if results_csv:
        path = Path(results_csv)
        if path.exists():
            return path
        raise SystemExit(f"Results CSV not found: {path}")

    default_path = Path(DEFAULT_RESULTS_CSV)
    discovered = discover_result_csvs(reports_dir)

    if selection == "default":
        if default_path.exists():
            return default_path
        if discovered:
            return discovered[0]
    elif selection == "latest":
        if discovered:
            return discovered[0]
        if default_path.exists():
            return default_path
    elif selection == "best":
        best_path = best_result_csv(discovered, score_column=score_column, fold=fold)
        if best_path is not None:
            return best_path
        if discovered:
            return discovered[0]
        if default_path.exists():
            return default_path
    else:
        raise SystemExit(f"Unknown --select value: {selection}")

    if discovered:
        return discovered[0]

    raise SystemExit(
        f"Results CSV not found: {default_path}\n"
        f"也沒有在 {reports_dir} 找到可讀的實驗 CSV。先跑完整實驗產生 CSV，或用 --results-csv 指到已存在的結果檔。"
    )


def discover_result_csvs(reports_dir: Path) -> list[Path]:
    if not reports_dir.exists():
        return []
    candidates = [path for path in reports_dir.glob("*.csv") if is_result_csv(path)]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_decision_json(decision_json: str | None, *, reports_dir: Path, auto: bool) -> Path | None:
    if decision_json:
        path = Path(decision_json)
        if path.exists():
            return path
        raise SystemExit(f"Decision JSON not found: {path}")
    if not auto:
        return None
    preferred = reports_dir / DEFAULT_DECISION_JSON
    if preferred.exists() and is_decision_json(preferred):
        return preferred
    discovered = discover_decision_jsons(reports_dir)
    return discovered[0] if discovered else None


def discover_decision_jsons(reports_dir: Path) -> list[Path]:
    if not reports_dir.exists():
        return []
    candidates: list[Path] = []
    for pattern in DECISION_JSON_PATTERNS:
        candidates.extend(path for path in reports_dir.glob(pattern) if is_decision_json(path))
    unique_candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    return unique_candidates


def is_decision_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metrics = data.get("metrics", data) if isinstance(data, dict) else {}
    if not isinstance(metrics, dict):
        return False
    required = {"threshold", "true_positives", "false_positives", "true_negatives", "false_negatives"}
    return required.issubset(metrics)


def is_result_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
    except OSError:
        return False
    return REQUIRED_RESULT_COLUMNS.issubset(fieldnames)


def best_result_csv(paths: list[Path], *, score_column: str, fold: str) -> Path | None:
    best_path: Path | None = None
    best_score = float("-inf")
    for path in paths:
        try:
            rows = read_results_csv(path)
        except (OSError, ValueError):
            continue
        selected_rows = [row for row in rows if row.fold == fold]
        if not selected_rows and fold == "mean":
            selected_rows = rows
        scores = [sort_value(row, score_column) for row in selected_rows]
        scores = [score for score in scores if score != float("-inf")]
        if not scores:
            continue
        score = max(scores)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def read_results_csv(path: Path) -> list[ResultRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_row_from_csv(row) for row in reader]


def render_results_table(rows: list[ResultRow], *, title: str) -> str:
    header = (
        f"{'Rank':>4}  {'Method':<6}  {'Combine':<16}  {'Experiment':<26}  "
        f"{'MAP':>7}  {'Top-1':>7}  {'Top-k':>7}  {'Recall':>7}  {'MRR':>7}  {'Sec':>7}"
    )
    separator = "-" * len(header)
    lines = [title, header, separator]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index:>4}  {row.method:<6}  {shorten(row.combine, 16):<16}  {shorten(row.experiment, 26):<26}  "
            f"{row.mean_average_precision:>7.4f}  {row.top_1_accuracy:>7.4f}  "
            f"{row.top_k_hit_rate:>7.4f}  {format_optional(row.recall_at_k):>7}  "
            f"{row.mean_reciprocal_rank:>7.4f}  {row.seconds:>7.1f}"
        )
    return "\n".join(lines)


def render_best_ranking_result(row: ResultRow, *, title: str) -> str:
    rows = [
        ("Experiment", row.experiment),
        ("Method", row.method),
        ("Combine", row.combine),
        ("Base model", row.base_model or "-"),
        ("Epochs", str(row.epochs or "-")),
        ("Max triplets", str(row.max_triplets or "-")),
        ("Queries", str(row.queries)),
        ("MAP", f"{row.mean_average_precision:.4f}"),
        ("Top-1", f"{row.top_1_accuracy:.4f}"),
        ("Top-k", f"{row.top_k_hit_rate:.4f}"),
        ("Recall", format_optional(row.recall_at_k)),
        ("MRR", f"{row.mean_reciprocal_rank:.4f}"),
        ("Seconds", f"{row.seconds:.1f}"),
    ]
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(value) for _, value in rows)
    header = f"{'Metric':<{label_width}}  {'Value':<{value_width}}"
    separator = "-" * len(header)
    lines = [
        title,
        header,
        separator,
    ]
    for label, value in rows:
        lines.append(f"{label:<{label_width}}  {value:<{value_width}}")
    return "\n".join(lines)


def render_decision_matrix(result: DecisionResult) -> str:
    actual_non_duplicate = result.true_negatives + result.false_positives
    actual_duplicate = result.false_negatives + result.true_positives
    false_positive_rate = ratio(result.false_positives, actual_non_duplicate)
    false_negative_rate = ratio(result.false_negatives, actual_duplicate)
    metric_rows = [
        ("Method", result.method or "-"),
        ("Combine", result.combine or "-"),
        ("Rerank fields", result.rerank_fields or "-"),
        ("Base model", result.base_model or "-"),
        ("Split", result.split or "-"),
        ("Train", str(result.train)),
        ("Validation", str(result.validation)),
        ("Queries", str(result.queries)),
        ("Positive queries", str(result.positive_queries)),
        ("Min precision", format_optional(result.min_precision)),
        ("Min recall", format_optional(result.min_recall)),
        ("Constraints satisfied", format_bool(result.constraints_satisfied)),
        ("Threshold", f"{result.threshold:.6f}"),
        ("Precision", f"{result.precision:.4f}"),
        ("Recall", f"{result.recall:.4f}"),
        ("F1", f"{result.f1:.4f}"),
        ("Accuracy", f"{result.accuracy:.4f}"),
        ("False positive rate", f"{false_positive_rate:.4f}"),
        ("False negative rate", f"{false_negative_rate:.4f}"),
        ("Correct duplicate links", str(result.correct_duplicate_links)),
    ]
    label_width = max(len(label) for label, _ in metric_rows)
    value_width = max(len(value) for _, value in metric_rows)
    header = f"{'Metric':<{label_width}}  {'Value':<{value_width}}"
    separator = "-" * len(header)

    lines = [
        "Duplicate Decision Metrics",
        header,
        separator,
    ]
    for label, value in metric_rows:
        lines.append(f"{label:<{label_width}}  {value:<{value_width}}")
    lines.append("")
    lines.append("Duplicate Decision Confusion Matrix")
    lines.append(f"{'Actual \\ Predicted':<24} {'Non-duplicate':>14} {'Duplicate':>12}")
    lines.append("-" * 52)
    lines.append(f"{'Non-duplicate':<24} {result.true_negatives:>14} {result.false_positives:>12}")
    lines.append(f"{'Duplicate':<24} {result.false_negatives:>14} {result.true_positives:>12}")
    return "\n".join(lines)


def sort_value(row: ResultRow, column: str) -> float:
    value = getattr(row, column)
    if value is None:
        return float("-inf")
    return float(value)


def shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    keep = width - 3
    left = max(1, keep // 2)
    right = keep - left
    return f"{value[:left]}...{value[-right:]}"


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(value).lower()


def read_decision_json(path: Path) -> DecisionResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Decision JSON must be an object: {path}")
    metrics = data.get("metrics", data)
    if not isinstance(metrics, dict):
        raise ValueError(f"Decision JSON metrics must be an object: {path}")
    return DecisionResult(
        threshold=_float_cell(metrics.get("threshold")),
        precision=_float_cell(metrics.get("precision")),
        recall=_float_cell(metrics.get("recall")),
        f1=_float_cell(metrics.get("f1")),
        accuracy=_float_cell(metrics.get("accuracy")),
        true_positives=_int_cell(metrics.get("true_positives")),
        false_positives=_int_cell(metrics.get("false_positives")),
        true_negatives=_int_cell(metrics.get("true_negatives")),
        false_negatives=_int_cell(metrics.get("false_negatives")),
        queries=_int_cell(metrics.get("queries")),
        positive_queries=_int_cell(metrics.get("positive_queries")),
        correct_duplicate_links=_int_cell(metrics.get("correct_duplicate_links")),
        method=str(data.get("method", metrics.get("method", "")) or ""),
        combine=str(data.get("combine", metrics.get("combine", "")) or ""),
        split=str(data.get("split", metrics.get("split", "")) or ""),
        train=_int_cell(data.get("train", metrics.get("train"))),
        validation=_int_cell(data.get("validation", metrics.get("validation"))),
        min_precision=_optional_float_cell(data.get("min_precision", metrics.get("min_precision"))),
        min_recall=_optional_float_cell(data.get("min_recall", metrics.get("min_recall"))),
        rerank_fields=str(data.get("rerank_fields", metrics.get("rerank_fields", "")) or ""),
        base_model=str(data.get("base_model", metrics.get("base_model", "")) or ""),
        constraints_satisfied=_optional_bool_cell(data.get("constraints_satisfied", metrics.get("constraints_satisfied"))),
    )


def _row_from_csv(row: dict[str, str]) -> ResultRow:
    return ResultRow(
        experiment=row.get("experiment", ""),
        method=row.get("method", ""),
        combine=row.get("combine", ""),
        fold=row.get("fold", ""),
        queries=_int_cell(row.get("queries")),
        mean_average_precision=_float_cell(row.get("mean_average_precision")),
        top_1_accuracy=_float_cell(row.get("top_1_accuracy")),
        top_k_hit_rate=_float_cell(row.get("top_k_hit_rate")),
        precision_at_k=_optional_float_cell(row.get("precision_at_k")),
        recall_at_k=_optional_float_cell(row.get("recall_at_k")),
        mean_reciprocal_rank=_float_cell(row.get("mean_reciprocal_rank")),
        base_model=row.get("base_model", ""),
        epochs=_int_cell(row.get("epochs")),
        max_triplets=_int_cell(row.get("max_triplets")),
        seconds=_float_cell(row.get("seconds")),
    )


def _float_cell(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _optional_float_cell(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_bool_cell(value) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _int_cell(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return round(float(value))


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
