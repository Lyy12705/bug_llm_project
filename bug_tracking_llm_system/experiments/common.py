from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return [row for row in value["records"] if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def nested(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def accuracy(gold: list[Any], pred: list[Any]) -> float:
    pairs = list(zip(gold, pred))
    if not pairs:
        return 0.0
    return sum(1 for expected, actual in pairs if str(expected) == str(actual)) / len(pairs)


def write_metrics(metrics: dict[str, Any], output: str | Path | None) -> None:
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gold", required=True, help="Gold JSON or JSONL file.")
    parser.add_argument("--pred", required=True, help="Predicted JSON or JSONL file.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path.")
