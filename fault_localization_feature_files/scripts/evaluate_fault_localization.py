#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fault localization predictions.")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL with fixed files or modified files.")
    parser.add_argument("--pred", required=True, help="Prediction JSON/JSONL from scripts/fault_localization.py.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    metrics = evaluate_records(gold_rows, pred_rows)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return [row for row in value["records"] if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def evaluate_records(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = _align_rows(gold_rows, pred_rows)
    file_evaluated = 0
    symbol_evaluated = 0
    file_hits_at = {1: 0, 3: 0, 5: 0}
    symbol_hits_at = {1: 0, 3: 0, 5: 0}
    file_reciprocal_rank_total = 0.0
    symbol_reciprocal_rank_total = 0.0
    missing_file_ground_truth: list[str] = []
    missing_symbol_ground_truth: list[str] = []

    for gold, pred in pairs:
        gold_files = _gold_files(gold)
        gold_symbols = _gold_symbols(gold)
        ticket_id = _ticket_id(gold) or _ticket_id(pred)
        if gold_files:
            file_evaluated += 1
            file_rank = _first_relevant_file_rank(_candidate_files(pred), gold_files)
            if file_rank is not None:
                file_reciprocal_rank_total += 1.0 / file_rank
                for k in file_hits_at:
                    if file_rank <= k:
                        file_hits_at[k] += 1
        else:
            missing_file_ground_truth.append(ticket_id)

        if gold_symbols:
            symbol_evaluated += 1
            symbol_rank = _first_relevant_symbol_rank(_candidate_symbols(pred), gold_symbols)
            if symbol_rank is not None:
                symbol_reciprocal_rank_total += 1.0 / symbol_rank
                for k in symbol_hits_at:
                    if symbol_rank <= k:
                        symbol_hits_at[k] += 1
        else:
            missing_symbol_ground_truth.append(ticket_id)

    metrics: dict[str, Any] = {
        "rows": len(pairs),
        "rows_with_file_ground_truth": file_evaluated,
        "rows_with_symbol_ground_truth": symbol_evaluated,
        "file_top_1_accuracy": _ratio(file_hits_at[1], file_evaluated),
        "file_top_3_accuracy": _ratio(file_hits_at[3], file_evaluated),
        "file_top_5_accuracy": _ratio(file_hits_at[5], file_evaluated),
        "file_mrr": _ratio(file_reciprocal_rank_total, file_evaluated),
        "symbol_top_1_accuracy": _ratio(symbol_hits_at[1], symbol_evaluated),
        "symbol_top_3_accuracy": _ratio(symbol_hits_at[3], symbol_evaluated),
        "symbol_top_5_accuracy": _ratio(symbol_hits_at[5], symbol_evaluated),
        "symbol_mrr": _ratio(symbol_reciprocal_rank_total, symbol_evaluated),
    }
    metrics["rows_with_ground_truth"] = file_evaluated
    metrics["top_1_accuracy"] = metrics["file_top_1_accuracy"]
    metrics["top_3_accuracy"] = metrics["file_top_3_accuracy"]
    metrics["top_5_accuracy"] = metrics["file_top_5_accuracy"]
    metrics["mrr"] = metrics["file_mrr"]
    if missing_file_ground_truth or missing_symbol_ground_truth:
        metrics["missing_file_ground_truth_rows"] = len(missing_file_ground_truth)
        metrics["missing_symbol_ground_truth_rows"] = len(missing_symbol_ground_truth)
        metrics["note"] = (
            "Rows without fixed_files/modified_files/file ground truth are skipped for file-level metrics. "
            "Rows without fixed_symbols/function_name/class_name ground truth are skipped for symbol-level metrics."
        )
    return metrics


def _align_rows(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pred_by_id = {_ticket_id(row): row for row in pred_rows if _ticket_id(row)}
    if pred_by_id:
        pairs = []
        for gold in gold_rows:
            ticket_id = _ticket_id(gold)
            if ticket_id and ticket_id in pred_by_id:
                pairs.append((gold, pred_by_id[ticket_id]))
        if pairs:
            return pairs
    return list(zip(gold_rows, pred_rows))


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("id") or row.get("bug_id") or row.get("query_id") or "")


def _gold_files(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("fixed_files", "modified_files", "changed_files", "bug_fix_files", "files", "file_paths"):
        values.extend(_as_list(row.get(key)))
    for key in ("file_path", "file"):
        values.extend(_as_list(row.get(key)))
    for nested_key in ("ground_truth", "json_ground_truth", "bug_location"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            values.extend(_gold_files(nested))
    return _unique_paths(values)


def _candidate_files(row: dict[str, Any]) -> list[str]:
    candidates = row.get("localized_candidates")
    if not isinstance(candidates, list):
        candidates = row.get("candidates")
    values: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict):
                values.extend(_as_list(candidate.get("file_path") or candidate.get("file")))
    if not values and isinstance(row.get("bug_location"), dict):
        location = row["bug_location"]
        if isinstance(location.get("bug_location"), dict):
            location = location["bug_location"]
        values.extend(_as_list(location.get("file_path") or location.get("file")))
    return _unique_paths(values)


def _gold_symbols(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "fixed_symbols",
        "modified_symbols",
        "changed_symbols",
        "bug_fix_symbols",
        "symbols",
        "symbol_names",
        "functions",
        "function_names",
        "methods",
        "method_names",
        "classes",
        "class_names",
    ):
        values.extend(_as_list(row.get(key)))
    for key in ("symbol_name", "symbol_qualified_name", "function_name", "function", "method", "class_name", "class"):
        values.extend(_as_list(row.get(key)))
    for nested_key in ("ground_truth", "json_ground_truth", "bug_location"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            values.extend(_gold_symbols(nested))
    return _unique_symbols(values)


def _candidate_symbols(row: dict[str, Any]) -> list[str]:
    candidates = row.get("localized_candidates")
    if not isinstance(candidates, list):
        candidates = row.get("candidates")
    values: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            values.extend(
                _as_list(
                    candidate.get("symbol_qualified_name")
                    or candidate.get("symbol_name")
                    or candidate.get("function_name")
                    or candidate.get("function")
                    or candidate.get("class_name")
                )
            )
    if not values and isinstance(row.get("bug_location"), dict):
        location = row["bug_location"]
        if isinstance(location.get("bug_location"), dict):
            location = location["bug_location"]
        values.extend(
            _as_list(
                location.get("symbol_qualified_name")
                or location.get("symbol_name")
                or location.get("function_name")
                or location.get("function")
                or location.get("class_name")
            )
        )
    return _unique_symbols(values)


def _first_relevant_file_rank(ranked_files: list[str], gold_files: list[str]) -> int | None:
    for index, predicted in enumerate(ranked_files, start=1):
        if any(_file_matches(predicted, gold) for gold in gold_files):
            return index
    return None


def _first_relevant_symbol_rank(ranked_symbols: list[str], gold_symbols: list[str]) -> int | None:
    for index, predicted in enumerate(ranked_symbols, start=1):
        if any(_symbol_matches(predicted, gold) for gold in gold_symbols):
            return index
    return None


def _file_matches(predicted: str, gold: str) -> bool:
    left = _normalize_path(predicted)
    right = _normalize_path(gold)
    if not left or not right:
        return False
    return left == right or left.endswith("/" + right) or right.endswith("/" + left)


def _symbol_matches(predicted: str, gold: str) -> bool:
    left = _normalize_symbol(predicted)
    right = _normalize_symbol(gold)
    if not left or not right:
        return False
    return left == right or left.endswith("." + right) or right.endswith("." + left)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _unique_paths(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        normalized = _normalize_path(value)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _unique_symbols(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        normalized = _normalize_symbol(value)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./")


def _normalize_symbol(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "::" in text:
        text = text.split("::", maxsplit=1)[1]
    text = text.split(":", maxsplit=1)[0]
    return text.replace("#", ".").strip().lstrip(".")


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


if __name__ == "__main__":
    main()
