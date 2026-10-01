#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_FILE_FIELDS = (
    "fixed_files",
    "modified_files",
    "changed_files",
    "bug_fix_files",
    "files",
    "file_paths",
    "file_path",
    "file",
)
DEFAULT_SYMBOL_FIELDS = (
    "fixed_symbols",
    "modified_symbols",
    "changed_symbols",
    "bug_fix_symbols",
    "symbols",
    "symbol_names",
    "function_names",
    "functions",
    "method_names",
    "methods",
    "class_names",
    "classes",
    "symbol_name",
    "symbol_qualified_name",
    "function_name",
    "function",
    "method",
    "class_name",
    "class",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize bug-fixing files/symbols into fault-localization gold JSONL."
    )
    parser.add_argument("--input", required=True, help="Input JSON/JSONL records with fixing files or symbols.")
    parser.add_argument("--output", required=True, help="Output gold JSONL path.")
    parser.add_argument(
        "--file-fields",
        default=",".join(DEFAULT_FILE_FIELDS),
        help="Comma-separated fields to read as file-level ground truth.",
    )
    parser.add_argument(
        "--symbol-fields",
        default=",".join(DEFAULT_SYMBOL_FIELDS),
        help="Comma-separated fields to read as symbol-level ground truth.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip rows that have neither file-level nor symbol-level ground truth.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = prepare_gold_records(
        read_records(args.input),
        file_fields=_split_fields(args.file_fields),
        symbol_fields=_split_fields(args.symbol_fields),
        skip_empty=args.skip_empty,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output)}, ensure_ascii=False, indent=2))


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


def prepare_gold_records(
    records: list[dict[str, Any]],
    *,
    file_fields: list[str] | None = None,
    symbol_fields: list[str] | None = None,
    skip_empty: bool = False,
) -> list[dict[str, Any]]:
    file_keys = file_fields or list(DEFAULT_FILE_FIELDS)
    symbol_keys = symbol_fields or list(DEFAULT_SYMBOL_FIELDS)
    rows: list[dict[str, Any]] = []
    for record in records:
        ticket_id = _ticket_id(record)
        fixed_files = _unique_paths(_collect_values(record, file_keys))
        fixed_symbols = _unique_symbols(_collect_values(record, symbol_keys))
        if skip_empty and not fixed_files and not fixed_symbols:
            continue
        rows.append(
            {
                "ticket_id": ticket_id,
                "fixed_files": fixed_files,
                "fixed_symbols": fixed_symbols,
                "source_fields": {
                    "file_fields": file_keys,
                    "symbol_fields": symbol_keys,
                },
            }
        )
    return rows


def _collect_values(row: dict[str, Any], keys: list[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        values.extend(_as_list(row.get(key)))
    for nested_key in ("ground_truth", "json_ground_truth", "bug_location", "patch"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            values.extend(_collect_values(nested, keys))
    return values


def _ticket_id(row: dict[str, Any]) -> str:
    direct = row.get("ticket_id") or row.get("id") or row.get("bug_id") or row.get("query_id")
    if direct:
        return str(direct)
    for nested_key in ("structured_ticket", "ticket", "raw_ticket", "ground_truth", "json_ground_truth"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            nested_id = nested.get("ticket_id") or nested.get("id") or nested.get("bug_id") or nested.get("query_id")
            if nested_id:
                return str(nested_id)
    return ""


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
        normalized = str(value).replace("\\", "/").strip().lstrip("./")
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


def _normalize_symbol(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "::" in text:
        text = text.split("::", maxsplit=1)[1]
    text = text.split(":", maxsplit=1)[0]
    return text.replace("#", ".").strip().lstrip(".")


def _split_fields(value: str) -> list[str]:
    return [field.strip() for field in value.split(",") if field.strip()]


if __name__ == "__main__":
    main()
